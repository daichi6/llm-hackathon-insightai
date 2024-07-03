#final_amazon.py

import fitz  # PyMuPDF
import json
import os
import boto3
from botocore.exceptions import ClientError
from pdf2image import convert_from_bytes
from io import BytesIO
import random
import time
import pandas as pd 


# Initialize Boto3 clients with the correct region
s3_client = boto3.client('s3', region_name='us-east-2')
textract_client = boto3.client('textract', region_name='us-east-2')

def extract_tf_blocks(data):
    """Extracts data from a specific page number from the JSON data."""
    page_blocks = {}
    count = 0

    for block in data["Blocks"]:
        figure_block = {}
        if block["BlockType"] == "TABLE":
            print("Table found")
            figure_block["BBox"] = block["Geometry"]["BoundingBox"]
            page = block["Page"]
            figure_block["Page"] = page
            page_blocks[count] = figure_block
            count += 1

    return page_blocks

def extract_snippet(pdf_path, bounding_box, page_number, snippet_name):
    """Extracts a snippet from a PDF file given its bounding box."""

    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]

    page_width = page.rect.width
    page_height = page.rect.height

    left = bounding_box['Left'] * page_width
    top = bounding_box['Top'] * page_height
    width = bounding_box['Width'] * page_width
    height = bounding_box['Height'] * page_height

    width_b = 30
    height_b = 70

    right = left + width
    bottom = top + height

    top = top - height_b
    bottom = bottom + height_b
    left, right = left - width_b, right + width_b

    top = max(top, 0)
    left = max(left, 0)
    right = min(right, page_width)
    bottom = min(bottom, page_height)

    rect = fitz.Rect(left, top, right, bottom)
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)
    return pix, doc


def upload_to_s3(file_path, bucket_name, object_name):
    """Uploads a file to an S3 bucket."""
    try:
        # Check if file already exists in bucket
        try:
            s3_client.head_object(Bucket=bucket_name, Key=object_name)
            print(f"File {object_name} already exists in bucket {bucket_name}. Skipping upload.")
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # File does not exist, upload it
                s3_client.upload_file(file_path, bucket_name, object_name)
                print(f"Uploaded file {object_name} to bucket {bucket_name}.")
            else:
                print(f"Error checking if file exists in S3: {e}")
                raise
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
        raise

def start_document_analysis(bucket_name, document_name):
    """Starts the asynchronous analysis of a document using Amazon Textract and returns the job ID."""
    print(f"Starting document analysis for {document_name} from bucket {bucket_name}...")
    try:
        response = textract_client.start_document_analysis(
            DocumentLocation={
                'S3Object': {
                    'Bucket': bucket_name,
                    'Name': document_name
                }
            },
            FeatureTypes=['TABLES', 'FORMS'],
            # Ensure this token is unique per request
            ClientRequestToken=str(random.randint(100000, 999999))
        )
        job_id = response['JobId']
        print(f"Document analysis started. Job ID: {job_id}")
        return job_id
    except ClientError as e:
        print(f"Error starting document analysis for {document_name} from bucket {bucket_name}: {e}")
        raise

def get_document_analysis(job_id):
    """Retrieves the results of a document analysis operation using the job ID."""
    print(f"Retrieving results for job ID: {job_id}...")
    try:
        response = textract_client.get_document_analysis(JobId=job_id)
        return response
    except ClientError as e:
        print(f"Error retrieving results for job ID {job_id}: {e}")
        raise

def read_from_s3(bucket_name, object_name):
    """Reads an object from S3 to test if permissions are correctly set."""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_name)
        print(f"Successfully read object {object_name} from bucket {bucket_name}.")
        return response['Body'].read()
    except ClientError as e:
        print(f"Error reading object {object_name} from bucket {bucket_name}: {e}")
        raise

def get_tables(file_count, object_name):
    bucket_name = 'hackathon-jr'
    
    # Download PDF from S3
    pdf_path = f"{object_name}"
    try:
        s3_client.download_file(bucket_name, object_name, pdf_path)
        print(f"Downloaded {object_name} from S3.")
    except ClientError as e:
        print(f"Failed to download {object_name} from S3: {e}")

    # Test reading from S3
    read_from_s3(bucket_name, object_name)
    print(f"Successfully read {object_name} from S3.")

    # Start document analysis using Textract
    try:
        job_id = start_document_analysis(bucket_name, object_name)
        
        # Wait for the job to complete - simple polling mechanism will need to be improved
        while True:
            response = get_document_analysis(job_id)
            print(response['JobStatus'])
            if response['JobStatus'] == 'SUCCEEDED':
                break
            elif response['JobStatus'] == 'FAILED':
                print(f"Document analysis failed for job ID: {job_id}")
                return
            time.sleep(10)  # Wait before polling again

        data = response

        #get the name of the file but remove the .pdf extension
        file_name = object_name.split('.')[0]
        #save te json data to s3 bucket 
        with open(file_name + ".json", 'w') as f:
            json.dump(data, f)

        # Extract table and figure data
        box_data = extract_tf_blocks(data)

        print(box_data)

        figure_count = 1
        for i, figure_data in box_data.items():
            print("WE MADE IT BABY")
            pix, doc = extract_snippet(pdf_path, figure_data["BBox"], figure_data["Page"], f"table_{i}")
            # Upload snippet image to S3
            snippet_name = f"{file_count:03d}_{figure_count:03d}_000.png"
            snippet_path = f"{snippet_name}"
            pix.save(snippet_path)
            upload_to_s3(snippet_path, bucket_name, snippet_name)
            figure_count += 1

    except ClientError as e:
        print(f"Failed to analyze {object_name} with Textract: {e}")

    return figure_count, doc

    