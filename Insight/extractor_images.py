#extractor_images.py

import boto3
import pymupdf as fitz
from botocore.exceptions import ClientError
from final_amazon import upload_to_s3

s3 = boto3.client('s3')
bucket_name = 'hackathon-jr'

def read_in_pdf_s3(file_path):
    try:
        # Read the file from S3 bucket
        s3_object = s3.get_object(Bucket=bucket_name, Key=file_path)
        file_content = s3_object['Body'].read()
        
        # Open the PDF using pymupdf
        doc = fitz.open(stream=file_content, filetype="pdf")
        return doc
    except Exception as e:
        print(f"Error reading in PDF {file_path}: {str(e)}")
        return None

def extract_images(doc, page_num, figure_num, username):
    print("Extracting images...")  # Print "Extracting images...
    figures = figure_num

    # doc = doc.split('/')[-1]
    doc = read_in_pdf_s3(doc)

    print(doc)
    for i in range(len(doc)):
        print(f"Processing page {i + 1}...")  # Print "Processing page {i + 1}...
        page = doc[i]  # Get the pages 
        text_data = page.get_text("dict")  # Get the text data

        #print(text_data)
        image_bbox = []
        for block in text_data["blocks"]:
            if block["type"] == 1:  # Skip non-text blocks
                image_bbox.append(block['bbox'])

        print(image_bbox)
        #now use the image_bbox to extract the images
        if (len(image_bbox) > 0):
            height = 60  
            width = 70
            for j, bbox in enumerate(image_bbox):
                x0, y0, x1, y1 = bbox
                x0 = max(x0 - width, 0)
                y0 = max(y0 - height, 0)
                x1 = min(x1 + width, page.rect.width)
                y1 = min(y1 + height, page.rect.height)
                rect = fitz.Rect(x0, y0, x1, y1)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=rect)


                #Save pixmap to temporary local file
                temp_image_path = f"/tmp/{username}_temp_image.png"
                pix.save(temp_image_path)

                # Upload snippet image to S3
                snippet_name = f"{page_num:03d}_{figures:03d}_000.png"
                s3_object_name = f"{username}/{snippet_name}"
                try:
                    upload_to_s3(temp_image_path, bucket_name, s3_object_name)
                except ClientError as e:
                    print(f"Failed to upload {snippet_name} to S3: {e}")
                    continue
                else:
                    print(f"Uploaded {snippet_name} to S3 bucket {bucket_name} under {s3_object_name}")
        
                figures += 1
                if figures >= 49:
                    time.sleep(65) # Sleep for 65 seconds if we have uploaded 49 images