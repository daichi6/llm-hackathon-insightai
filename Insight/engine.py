import os
import boto3
import pandas as pd
from UserChat import extract_text
from final_amazon import get_tables
from extractor_images import extract_images
import io
import tempfile
import time

# Initialize S3 and Textract clients
s3_client = boto3.client('s3')
textract_client = boto3.client('textract', region_name='us-east-2')

def start_engine(file, username, df):
    print("IN ENGINE")
    bucket_name = 'hackathon-jr'

    print(f"Starting to process file: {file}")

    # Download file from S3 into memory
    try:
        s3_object = s3_client.get_object(Bucket=bucket_name, Key=file)
        file_content = s3_object['Body'].read()
        print(f"Successfully downloaded {file} from S3")
    except Exception as e:
        print(f"Error downloading file from S3: {e}")
        return 1

    print(f"Size of downloaded file: {len(file_content)} bytes")

    # Create a temporary file to handle the PDF content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(file_content)
        temp_pdf_path = temp_pdf.name

    print(f"Processing {file}...")
    print(f"Extracting text from {file}...")

    # Extract text from the PDF file
    try:
        text = extract_text(temp_pdf_path)
        print(f"Successfully extracted text from {file}")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return 1
    finally:
        os.remove(temp_pdf_path)  # Clean up the temporary fil
    
   # Read in the csv file called paper_num.csv from S3 bucket
    try:
        csv_object = s3_client.get_object(Bucket=bucket_name, Key='paper_num.csv')
        csv_content = csv_object['Body'].read()
        df = pd.read_csv(io.BytesIO(csv_content))
        print("Successfully read paper_num.csv from S3")
    except Exception as e:
        print(f"Error reading paper_num.csv from S3: {e}")
        return 1

    # Grab the file_count from the Paper_Num column
    file_count = df['Paper_Num'].iloc[-1] + 1

    print("File Count: ", file_count)

    # Save to S3 
    # try:
    #     csv_buffer = io.StringIO()
    #     df.to_csv(csv_buffer, index=False)
    #     s3_client.put_object(Bucket=bucket_name, Key=f'{username}/paper_num.csv', Body=csv_buffer.getvalue())
        
    #     print("Uploaded new paper_num.csv to S3")
    # except Exception as e:
    #     print(f"Error creating or uploading paper_num.csv: {e}")
    #     return 1

    # Write the extracted text to a text file in-memory
    txt_file_name = f"{username}/{file_count}_000.txt"
    text_buffer = io.StringIO()
    try:
        text_buffer.write(text)
        text_buffer.seek(0)  # Move the cursor to the beginning of the buffer
        print(f"Successfully wrote extracted text to {txt_file_name}")
    except Exception as e:
        print(f"Error writing text to buffer: {e}")
        return 1

    # Upload the PDF and text files to the S3 bucket
    try:
        s3_client.put_object(Bucket=bucket_name, Key=f'{username}/{os.path.basename(file)}', Body=file_content)
        s3_client.put_object(Bucket=bucket_name, Key=txt_file_name, Body=text_buffer.getvalue())
        print(f"Uploaded {file} and {txt_file_name} to the S3 bucket.")
    except Exception as e:
        print(f"Error uploading files to S3: {e}")
        return 1

    # Process tables and figures
    try:
        print("File Count: ", file_count)
        print("File: ", file)
        
        print("Username: ", username)
        figure_count, doc = get_tables(file_count, file, username)
        if figure_count >= 49:
            time.sleep(65)
        print()
        print(f"Processed tables for {file}")
        print("Figure Count: ", figure_count)
        print("Doc: ", doc)
        doc = file
        extract_images(doc, file_count, figure_count, username)
        print(f"Extracted images from {file}")
    except Exception as e:
        print(f"Error processing tables or extracting images: {e}")
        return 1

    # Update paper_num.csv
    df.loc[file_count] = file_count
    print(df)
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=f'{username}/paper_num.csv', Body=csv_buffer.getvalue())
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
  