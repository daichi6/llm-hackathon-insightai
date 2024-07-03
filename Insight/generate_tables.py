#generate_tables.py

import os
import pandas as pd
import requests
import time
import backoff
import boto3
import io

# Set your API key here
api_key = "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"

# Set up S3 client
s3 = boto3.client('s3')

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def summarize_text(text):
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1000,
            "system": "You are a helpful assistant that is specialized in reading and analyzing research papers.",
            "messages": [
                {"role": "user", "content": f"""Please provide two summaries of the following text:
1. A short description of 2-3 lines.
2. A long description of minimum 500 tokens.
Format your response as follows:
Short Description: [Your short summary here]
Long Description: [Your long summary here]
Here's the text to summarize:
{text}"""}
            ]
        }
        
        response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)
        
        if response.status_code == 200:
            full_response = response.json()['content'][0]['text']
            short_desc = full_response.split("Short Description:")[1].split("Long Description:")[0].strip()
            long_desc = full_response.split("Long Description:")[1].strip()
            return short_desc, long_desc
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        if 'rate_limit_exceeded' in str(e):
            retry_after = 60  # Default to 60 seconds if retry time is not provided
            print(f"Rate limit exceeded. Waiting for {retry_after} seconds.")
            time.sleep(retry_after)
            raise e
        else:
            print(f"An error occurred: {e}")
            raise e

def process_file(bucket, key):
    thesis_num = key.split('/')[-1].split('_')[0]
    
    # Read file from S3
    response = s3.get_object(Bucket=bucket, Key=key)
    content = response['Body'].read().decode('utf-8')
    
    short_desc, long_desc = summarize_text(content)
    return thesis_num, short_desc, long_desc

def generate_table_summaries():
    # Create an empty list to store the results
    results = []

    # S3 bucket and file information
    bucket_name = 'hackathon-jr'
    output_key = 'summaries.csv'
    # Process all .txt files in the bucket
    try:
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix='', Delimiter='/')
        for obj in response['Contents']:
            if obj['Key'].endswith('.txt'):
                thesis_num, short_desc, long_desc = process_file(bucket_name, obj['Key'])
                results.append({'thesis_num': thesis_num, 'description': long_desc})
    except s3.exceptions.NoSuchKey:
        print(f"No .txt files found in bucket {bucket_name}.")
    except Exception as e:
        print(f"An error occurred: {e}")

    # Create a DataFrame from the results
    df = pd.DataFrame(results)

    # Write the DataFrame to a CSV file in memory
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)

    # Upload the CSV to S3
    s3.put_object(Bucket=bucket_name, Key=output_key, Body=csv_buffer.getvalue())

    print(f"Summaries have been written to s3://{bucket_name}/{output_key}")