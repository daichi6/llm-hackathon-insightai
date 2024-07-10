import boto3
import json
import time
import random
from botocore.exceptions import ClientError

# Initialize Boto3 clients with the correct region
s3_client = boto3.client('s3', region_name='us-east-2')
textract_client = boto3.client('textract', region_name='us-east-2')

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
            ClientRequestToken=str(random.randint(100000, 999999))
        )
        job_id = response['JobId']
        print(f"Document analysis started. Job ID: {job_id}")
        return job_id
    except ClientError as e:
        print(f"Error starting document analysis: {e}")
        raise

def check_job_status(job_id):
    """Checks the status of the Textract job until it completes."""
    print(f"Checking job status for Job ID: {job_id}...")
    while True:
        response = textract_client.get_document_analysis(JobId=job_id)
        status = response['JobStatus']
        print(f"Current job status: {status}")
        if status in ['SUCCEEDED', 'FAILED']:
            return status
        time.sleep(5)

def get_document_analysis(job_id):
    """Retrieves the results of a document analysis operation using the job ID."""
    print(f"Retrieving results for Job ID: {job_id}...")
    pages = []
    next_token = None
    while True:
        if next_token:
            response = textract_client.get_document_analysis(JobId=job_id, NextToken=next_token)
            print("Fetching next set of results with NextToken...")
        else:
            response = textract_client.get_document_analysis(JobId=job_id)
            print("Fetching initial set of results...")
        
        pages.append(response)
        print(f"Received {len(response['Blocks'])} blocks in current response.")

        next_token = response.get('NextToken', None)
        if not next_token:
            print("No more pages to fetch.")
            break
        else:
            print("NextToken found, continuing to fetch more pages.")
    return pages

def main():
    bucket_name = 'hackathon-jr'
    document_name = 'attention.pdf'
    print(f"Starting process for document: {document_name} in bucket: {bucket_name}")

    # Start document analysis using Textract
    try:
        job_id = start_document_analysis(bucket_name, document_name)
        
        # Wait for the job to complete
        job_status = check_job_status(job_id)
        if job_status == 'SUCCEEDED':
            print(f"Job ID: {job_id} succeeded. Retrieving full results...")
            result_pages = get_document_analysis(job_id)
            
            # Combine all pages into one data structure
            all_data = []
            for page in result_pages:
                all_data.extend(page['Blocks'])
            
            data = {'Blocks': all_data}
            
            # Save the JSON data locally
            file_name = document_name.split('.')[0] + ".json"
            with open(file_name, 'w') as f:
                json.dump(data, f)
            print(f"JSON data saved to {file_name}")
        else:
            print(f"Document analysis failed for job ID: {job_id}")

    except ClientError as e:
        print(f"Failed to analyze {document_name} with Textract: {e}")

if __name__ == "__main__":
    print("Script started.")
    main()
    print("Script finished.")

