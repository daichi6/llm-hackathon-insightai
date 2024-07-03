#engine.py

import os
import boto3
import pandas as pd
from UserChat import extract_text
from final_amazon import get_tables
from extractor_images import extract_images

s3_client = boto3.client('s3')
textract_client = boto3.client('textract', region_name='us-east-2')

def start_engine(file):
    bucket_name = 'hackathon-jr'
    local_file_path = file

    print(f"Starting to process file: {file}")

    # Download file from S3
    try:
        s3_client.download_file(bucket_name, file, local_file_path)
        print(f"Successfully downloaded {file} from S3")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return 1
    
    print(f"Size of downloaded file: {os.path.getsize(local_file_path)} bytes")

    # Check if file exists and is not empty
    if not os.path.exists(local_file_path):
        print(f"File {local_file_path} does not exist.")
        return 1
    
    if os.path.getsize(local_file_path) == 0:
        print(f"File {local_file_path} is empty.")
        return 1

    print(f"Processing {file}...")
    print(f"Extracting text from {file}...")

    # Extract text from the PDF file
    try:
        text = extract_text(local_file_path)
        print(f"Successfully extracted text from {file}")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return 1
    

    # Delete existing paper_num.csv if it exists
    try:
        s3_client.delete_object(Bucket=bucket_name, Key='paper_num.csv')
        print("Deleted existing paper_num.csv")
    except Exception as e:
        print(f"Error deleting paper_num.csv (This might be normal if it didn't exist): {e}")

    # Create new paper_num.csv
    df = pd.DataFrame({"Paper_Num": [0]})
    file_count = df['Paper_Num'].iloc[-1] + 1

    # Save to S3
    try:
        df.to_csv('paper_num.csv', index=False)
        s3_client.upload_file('paper_num.csv', bucket_name, 'paper_num.csv')
        print("Uploaded new paper_num.csv to S3")
    except Exception as e:
        print(f"Error creating or uploading paper_num.csv: {e}")
        return 1

    # Write the extracted text to a text file
    txt_file_name = f"{file_count}_000.txt"
    text_file_path = txt_file_name
    try:
        with open(text_file_path, 'w') as f:
            f.write(text)
        print(f"Successfully wrote extracted text to {txt_file_name}")
    except Exception as e:
        print(f"Error writing text to file: {e}")
        return 1

    # Upload the PDF and text files to the S3 bucket
    try:
        s3_client.upload_file(local_file_path, bucket_name, file)
        s3_client.upload_file(text_file_path, bucket_name, txt_file_name)
        print(f"Uploaded {file} and {txt_file_name} to the S3 bucket.")
    except Exception as e:
        print(f"Error uploading files to S3: {e}")
        return 1


    # Process tables and figures
    try:
        figure_count, doc = get_tables(file_count, file)
        print(f"Processed tables for {file}")
        extract_images(doc, file_count, figure_count)
        print(f"Extracted images from {file}")
    except Exception as e:
        print(f"Error processing tables or extracting images: {e}")
        return 1

    # Update paper_num.csv
    df.loc[file_count] = file_count
    try:
        df.to_csv('paper_num.csv', index=False)
        s3_client.upload_file('paper_num.csv', bucket_name, 'paper_num.csv')
        print(f"Updated paper_num.csv with new entry: {file_count}")
    except Exception as e:
        print(f"Error updating paper_num.csv: {e}")
        return 1

    print(f"Successfully completed processing {file}")
    return 0
    

# def start_engine(file):
#     bucket_name = 'hackathon-jr'

#     #if file already in s3 bucket
#     # Check if file already exists in S3 bucket
#     # response = s3_client.list_objects_v2(Bucket=bucket_name)
#     # objects = response['Contents']
#     # for obj in objects:
#     #     object_name = obj['Key']
#     #     if object_name == file:
#     #         print(f"File {file} already exists in the S3 bucket.")
#     #         #do extraction of information part 
#     #         return 0

#     # response = s3_client.list_objects_v2(Bucket=bucket_name)
#     # objects = response['Contents']
#     # for obj in objects:
#     #     object_name = obj['Key']
#     #     if object_name == file:
#     #         #load the file
#     #         print(f"File {file} already exists in the S3 bucket.")
    
#     #get file from s3 bucket and set the contents to temp_pdf_path
#     s3_client.download_file(bucket_name, file, file)
#     temp_pdf_path = file
    
#     print(f"Processing {file}...")
#     print(f"Extracting text from {file}...")

#     # Extract text from the PDF file
#     text = extract_text(temp_pdf_path)

#     # Write the extracted text to a text file
#     # Remove .pdf from the file name
#     txt_file_name = file.split('.')[0] + '.txt'
#     text_file_path = txt_file_name
#     with open(text_file_path, 'w') as f:
#         f.write(text)

#     # Upload the PDF file to the S3 bucket
#     s3_client.upload_file(temp_pdf_path, bucket_name, file)

#     # Upload the text file to the S3 bucket
#     s3_client.upload_file(text_file_path, bucket_name, txt_file_name)

#     print(f"Uploaded {file} and {txt_file_name} to the S3 bucket.")

#     #grab the csv file called paper_num.csv
#     body = s3_client.delete_object(Bucket=bucket_name, Key='paper_num.csv')

#     df= {"Paper_Num": 0}
#     df = pd.DataFrame(df, index=[0])

#     print(df)
#     #grab the lowest part of Paper_Num column
#     file_count = df['Paper_Num'].iloc[-1] + 1

#     #save to s3
#     df.to_csv('paper_num.csv', index=False)
#     s3_client.upload_file('paper_num.csv', bucket_name, 'paper_num.csv')

#     # List objects in the S3 bucket
#     try:
#         response = s3_client.list_objects_v2(Bucket=bucket_name)
#         objects = response['Contents']
#     except ClientError as e:
#         print(f"Failed to list objects in bucket {bucket_name}: {e}")
#         return

#     object_name = file

#     print(f"Processing {object_name}...")
#     figure_count, doc = get_tables(file_count, object_name)

#     #add extract_images here
#     extract_images(doc, file_count, figure_count)

#     #update the csv file with the new Paper_Num
#     df.loc[file_count] = file_count
#     df.to_csv('paper_num.csv', index=False)
#     s3_client.upload_file('paper_num.csv', bucket_name, 'paper_num.csv')
  