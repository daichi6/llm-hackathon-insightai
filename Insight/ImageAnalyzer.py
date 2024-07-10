#ImageAnalyzer
import os
import tempfile
import re
from PIL import Image
import pandas as pd
import base64
from anthropic import Anthropic
import imghdr
import boto3
import io
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('API_KEY')

# Set up S3 client
s3 = boto3.client('s3')

# Set your Anthropic API key
client = Anthropic(api_key=api_key)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    image_type = imghdr.what(image_path)
    media_type = f"image/{image_type}"
    return encoded_string, media_type

def get_image_analysis(image_path):
    image_name = os.path.basename(image_path)
    
    with Image.open(image_path) as img:
        width, height = img.size
        channels = len(img.getbands())

    base64_image, media_type = encode_image(image_path)

    base_prompt = f"""Analyze the image titled '{image_name}' from a scientific paper. The image dimensions are {width}x{height} with {channels} channels.
Provide a comprehensive analysis structured as follows:
1. Image Type and Overview:
   - Identify the type of scientific figure (e.g., graph, diagram, microscopy image, flow chart).
   - Describe the general layout and main components of the image.
2. Key Scientific Concepts:
   - Identify and explain the main scientific concepts or principles illustrated in the image.
   - Discuss how these concepts relate to the likely field of study.
3. Data Representation:
   - If applicable, describe how data is represented (e.g., axes, scales, color coding).
   - Analyze any trends, patterns, or significant data points shown.
4. Methodology Illustration:
   - Identify any experimental setups, procedures, or methodologies depicted.
   - Explain how the image contributes to understanding the research methods used.
5. Results and Findings:
   - Interpret any results or findings presented in the image.
   - Discuss the potential implications of these results in the context of the research.
6. Technical Details:
   - Analyze any technical elements, such as equations, molecular structures, or specialized notations.
   - Explain their significance in the context of the research.
7. Visual Aids and Annotations:
   - Describe any labels, legends, or other annotations present in the image.
   - Explain how these elements enhance the understanding of the scientific content.
8. Interdisciplinary Connections:
   - Suggest potential connections to other scientific fields or disciplines.
   - Discuss how this research might impact or relate to broader scientific understanding.
9. Limitations and Considerations:
   - Identify any potential limitations or considerations in the data or methodology presented.
   - Suggest possible areas for further research or investigation based on the image.
10. Summary of Scientific Significance:
    - Provide a concise summary of the key scientific insights and importance of this image in research.
11. Summary:
    - Give a basic Summary of the Image
Please provide a detailed analysis for each section, ensuring a comprehensive understanding of the scientific content and significance of the image."""

    print(f"Sending image to Claude API for comprehensive analysis of {image_path}")
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            #model="claude-3-5-sonnet-20240620",
            max_tokens=2000,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": base_prompt
                        }
                    ]
                }
            ]
        )
        
        analysis = response.content[0].text.strip()
        print(f"Received comprehensive analysis from Claude API")
        return analysis
    except Exception as e:
        print(f"Error in get_image_analysis for {image_path}: {str(e)}")
        return f"Error: {str(e)}"

def get_figure_numbers(image_path):
    try:
        base64_image, media_type = encode_image(image_path)
        
        prompt = """### Instructions ###

You are an engineer skilled in Computer Vision and NLP. Your task is to identify the figure number or table number in the given image. 
Please respond using the specified format: lowercase "figure" or "table" followed by the lowercase number. Do not include a space between "table" and the number, or "figure" and the number.
Please provide your response strictly in the specified format, without including any additional text for formatting. I will use your response directly. If the number is unknown, provide an empty string "". Do not guess. 
If multiple figure or table numbers are found in the image, please find only the main figure or table number in the image(the one occupying the largest part of the image). Do not provide more than one response. 
The characters 1 and i can be easily confused in the image, but since only numbers should follow "table" or "figure," it should be interpreted as 1. There might be other confusing letters and numbers, but they should be interpreted as numbers.

### Example Output ###
"table1"

### Example Output ###
"table3"

### Example Output ###
"figure1"

### Example Output ###
"figure4"

### Example Output ###
""

### Output ###
"""
        
        print(f"Sending image to Claude API for figure/table number of {image_path}")
        response = client.messages.create(
            #model="claude-3-5-sonnet-20240620",
            model="claude-3-haiku-20240307",
            max_tokens=10,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        )
        
        result = response.content[0].text.strip().lower()
        print(f"Received response from Claude API: {result}")
        
        if result.startswith('figure'):
            figure_number = result[6:]
            table_number = ""
        elif result.startswith('table'):
            table_number = result[5:]
            figure_number = ""
        else:
            figure_number = ""
            table_number = ""
        
        return figure_number, table_number
    except Exception as e:
        print(f"Error in get_figure_numbers for {image_path}: {str(e)}")
        return "", ""

def extract_thesis_number(filename):
    match = re.match(r'(\d{3})_\d{3}_\d{3}', filename)
    if match:
        return match.group(1)
    return "Unknown"

def process_image(image_path):
    try:
        figure_number, table_number = get_figure_numbers(image_path)
        analysis = get_image_analysis(image_path)
        return figure_number, table_number, analysis
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return "", "", f"Error: {str(e)}"


def process_s3_bucket(bucket_name, prefix=''):
    print(f"Processing S3 bucket: {bucket_name}")
    results = []
    
    # List objects in the bucket
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    if 'Contents' in response:
        for obj in response['Contents']:
            if obj['Key'].lower().endswith('.png'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                    s3.download_fileobj(bucket_name, obj['Key'], temp_file)
                    temp_file_path = temp_file.name
                
                thesis_number = extract_thesis_number(os.path.basename(obj['Key']))
                figure_number, table_number, analysis = process_image(temp_file_path)
                
                results.append({
                    'thesis_num': thesis_number,
                    'figure_num': figure_number,
                    'table_num': table_number,
                    'description': analysis
                })
                
                os.unlink(temp_file_path)  # Delete the temporary file
    else:
        print("No objects found in the bucket.")
    
    return pd.DataFrame(results)

def save_df_to_s3(df, bucket_name, file_key, input_prefix):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    file_key = f"{input_prefix}/{file_key}"  # Add input_prefix to file_key
    s3.put_object(Bucket=bucket_name, Key=file_key, Body=csv_buffer.getvalue())
    print(f"Results saved to S3: s3://{bucket_name}/{file_key}")

# Usage example
def get_context(username):
    bucket_name = "hackathon-jr"
    input_prefix = username  # Folder containing input PNG files
    output_key = "image_analysis_results.csv"  # Output CSV file path in S3
    
    try:
        print(f"Starting processing of S3 bucket: {bucket_name}")
        df_results = process_s3_bucket(bucket_name, input_prefix)
        
        if not df_results.empty:
            print("Results DataFrame:")
            print(df_results)
            # Save results to S3
            save_df_to_s3(df_results, bucket_name, output_key, input_prefix)
        else:
            print("No results were returned.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")