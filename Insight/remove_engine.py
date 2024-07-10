import os
import boto3
import pandas as pd
import io
#engine.py


s3_client = boto3.client('s3')
textract_client = boto3.client('textract', region_name='us-east-2')

def start_engine(file, username):
    print("IN ENGINE")
    bucket_name = 'hackathon-jr'
    local_file_path = file

    print(f"Processing {file}...")
    # print(f"Extracting text from {file}...")
    
    # Delete existing paper_num.csv if it exists
    try:
        s3_client.delete_object(Bucket=bucket_name, Key=f'{username}/paper_num.csv')
        print("Deleted existing paper_num.csv")
    except Exception as e:
        print(f"Error deleting paper_num.csv (This might be normal if it didn't exist): {e}")

    # Create new paper_num.csv
    df = pd.DataFrame({"Paper_Num": [0]})
    file_count = df['Paper_Num'].iloc[-1] + 1

    print(df)

    # Save to S3 
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=f'{username}/paper_num.csv', Body=csv_buffer.getvalue())
        
        print("Uploaded new paper_num.csv to S3")
    except Exception as e:
        print(f"Error creating or uploading paper_num.csv: {e}")
        return 1

file = 'attention.pdf'
username = 'josh'
start_engine(file, username) 