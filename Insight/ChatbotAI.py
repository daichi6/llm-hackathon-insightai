#ChatbotAI.py

import streamlit as st
from PyPDF2 import PdfReader
import os
import pandas as pd
import requests
import backoff
import base64
import re
import boto3
from io import StringIO
from engine import start_engine
from ImageAnalyzer import get_context
from UserChat import chat_main, update_vectorbase
from generate_tables import generate_table_summaries
from io import BytesIO

# Set your API key here
api_key = "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"

# Initialize S3 client
s3 = boto3.client('s3')

# Set page config
st.set_page_config(page_title="PDF Chatbot", page_icon="📚", layout="wide", initial_sidebar_state="collapsed")

# Hide Streamlit default header and footer
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .center-button { display: flex; justify-content: center; }
    .stApp {
        background-color: #000000;
        color: white;
    }
    .fixed-header {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background-color: #000000;
        padding: 5px;
        border-bottom: 1px solid #333;
        display: flex;
        align-items: center;
    }
    .fixed-header img {
        width: 60px;
        margin-right: 10px;
        opacity: 0.7;
    }
    .fixed-header h1 {
        margin: 0;
        font-size: 24px;
    }
    .nav-bar {
        display: flex;
        align-items: center;
        margin-left: 20px;
    }
    .nav-item {
        color: white;
        text-decoration: none;
        margin: 0 10px;
        padding: 5px 10px;
        border-radius: 5px;
        transition: background-color 0.3s;
    }
    .nav-item:hover {
        background-color: #333;
    }
    .main-content {
        margin-top: 0px;
    }
    .streamlit-expanderHeader {
        background-color: #333333;
        color: white;
    }
    .streamlit-expanderContent {
        background-color: #1a1a1a;
        color: white;
    }
    .block-container {
        padding-top: 3rem;
        padding-bottom: 0rem;
    }
    .stChatFloatingInputContainer {
        position: fixed;
        bottom: 20px;
        left: 50%;
        transform: translateX(-50%);
        width: 80%;
        background-color: #000000;
        padding: 10px;
        z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

def get_s3_csv(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

def map_paper_reference(user_input, insights_df):
    pattern = r'(paper|thesis)\s+(\d+)'
    matches = re.finditer(pattern, user_input.lower())
    
    mapping = {}
    for match in matches:
        paper_num = int(match.group(2))
        if paper_num in insights_df.index:
            paper_info = insights_df.loc[paper_num]
            mapping[paper_num] = paper_info['Paper Name']
    
    mapping_df = pd.DataFrame(list(mapping.items()), columns=['thesis_num', 'Paper Name'])
    

    # Get S3 data
    summaries_df = get_s3_csv('hackathon-jr', 'summaries.csv')
    image_analysis_df = get_s3_csv('hackathon-jr', 'image_analysis_results.csv')
    
    # Merge with S3 data
    merged_df = mapping_df.merge(summaries_df, on='thesis_num', how='left')
    merged_df = merged_df.merge(image_analysis_df, on='thesis_num', how='left')
    
    # If there are columns with the same name, we keep only one instance
    merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
    
    return merged_df

@backoff.on_exception(backoff.expo, Exception, max_tries=5)
def analyze_text(text):
    try:
        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01"
        }
        
        data = {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1000,
            "system": "You are a helpful assistant that specializes in analyzing research papers.",
            "messages": [
                {"role": "user", "content": f"""Analyze the following text from a research paper and provide:
1. The paper's title
2. The authors' names
3. A short description of 2-3 lines summarizing the paper's main points.

Format your response as follows:
Title: [Paper title]
Authors: [Author names]
Summary: [Your short summary here]

If You are not able to extract the answer, Please Respond 'N/A'

Here's the text to analyze:
{text[:2000]}"""}  # Using first 2000 characters for analysis
            ]
        }
        
        response = requests.post("https://api.anthropic.com/v1/messages", json=data, headers=headers)

        print(response)
        
        if response.status_code == 200:
            full_response = response.json()['content'][0]['text']
            title = full_response.split("Title:")[1].split("Authors:")[0].strip()
            authors = full_response.split("Authors:")[1].split("Summary:")[0].strip()
            summary = full_response.split("Summary:")[1].strip()
            return title, authors, summary
        else:
            raise Exception(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise e

# def extract_pdf_info(pdf_path):
#     with open(pdf_path, 'rb') as file:
#         reader = PdfReader(file)
#         text = ""
#         for page in reader.pages:
#             text += page.extract_text()
    
#     title, authors, summary = analyze_text(text)
#     return title, authors, summary

def extract_pdf_info(s3_path):
    # Parse the S3 URI
    bucket_name = s3_path.split('/')[2]
    object_key = '/'.join(s3_path.split('/')[3:])

    # Download the file from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    file_content = response['Body'].read()

    # Use BytesIO to create a file-like object
    pdf_file = BytesIO(file_content)

    # Now use PdfReader with this file-like object
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    # Your existing code to analyze the text and extract title, authors, summary
    title, authors, summary = analyze_text(text)

    return title, authors, summary

# def save_uploaded_file(uploaded_file):
#     #save to s3 bucket
#     bucket_name = "hackathon-jr"
#     s3.upload_file(uploaded_file.name, bucket_name, uploaded_file)

def save_uploaded_file(uploaded_file):
    bucket_name = "hackathon-jr"
    
    # First, let's check if the uploaded file has content
    if uploaded_file.size == 0:
        print(f"Error: Uploaded file '{uploaded_file.name}' is empty.")
        return False

    # Create a temporary local file
    temp_file_path = uploaded_file.name
    
    try:
        # Save the uploaded file content to a temporary file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Check the size of the temporary file
        if os.path.getsize(temp_file_path) == 0:
            print(f"Error: Temporary file '{temp_file_path}' is empty.")
            return False
        
        # Upload the file to S3
        s3.upload_file(temp_file_path, bucket_name, uploaded_file.name)
        print(f"Successfully uploaded '{uploaded_file.name}' to S3 bucket '{bucket_name}'")
        
        # Remove the temporary file
        os.remove(temp_file_path)
        
        return True
    
    except IOError as e:
        print(f"IOError while handling file: {e}")
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    # If we've reached here, an error occurred
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    return False


def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
    
def purge_s3_bucket(bucket_name):
    response = s3.list_objects_v2(Bucket=bucket_name)
    objects = response['Contents']
    for obj in objects:
        object_name = obj['Key']
        s3.delete_object(Bucket=bucket_name, Key=object_name)

def show_home_page():
    upload_insights_container = st.container()
    chat_container = st.container()

    with upload_insights_container:
        with st.expander("Upload PDFs and View Insights", expanded=True):
            left_column, right_column = st.columns(2)

            with left_column:
                st.subheader("Upload and Select PDFs")
                uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
                
                #once the files have been uploaded we can start the main engine
                if uploaded_files:
                    if st.button(f"Submit"):
                        st.success("Files uploaded successfully!")
                        bucket_name = "hackathon-jr"
                        purge_s3_bucket(bucket_name)
                        #load the vector db
                        for uploaded_file in uploaded_files:
                            #update vd
                            save_uploaded_file(uploaded_file)

                # if uploaded_files:
                #     if st.button(f"Submit"):
                #         bucket_name = "hackathon-jr"
                #         purge_s3_bucket(bucket_name)
                #         for uploaded_file in uploaded_files:
                #             start_engine(uploaded_file)            

                #pull from s3 bucket and list pdf files
                bucket_name = "hackathon-jr"
                response = s3.list_objects_v2(Bucket=bucket_name)
                objects = response['Contents']
                pdfs = []
                for obj in objects:
                    object_name = obj['Key']
                    if object_name.endswith('.pdf') and object_name not in pdfs:
                        pdfs.append(object_name)
                    
                selected_pdfs = st.multiselect("Select PDFs for analysis", pdfs)

            with right_column:  
                st.subheader("Initial Insights")
                if selected_pdfs and st.button("Analyze Selected PDFs"):
                    insights = []
                    for pdf in selected_pdfs:
                        with st.spinner(f"Analyzing {pdf}..."):
                            # Process the PDF with start_engine
                            start_engine(pdf)
                            

                            try:
                                # Construct the S3 path
                                s3_pdf_path = f"s3://hackathon-jr/{pdf}"
                                
                                # Extract info from the PDF
                                title, authors, summary = extract_pdf_info(s3_pdf_path)
                                
                                insights.append({
                                    "Paper Name": title,
                                    "Author Names": authors,
                                    "Short Description": summary
                                })
                                
                                st.success(f"Successfully analyzed {pdf}")
                            except Exception as e:
                                st.error(f"Error extracting information from {pdf}: {str(e)}")


      

                    with st.spinner(f"Generating ..."):

                        generate_table_summaries()
                        get_context()

                        st.session_state.insights = pd.DataFrame(insights).reset_index(drop=True)
                        st.session_state.insights.index = st.session_state.insights.index + 1
                    
                        if st.session_state.insights is not None:
                            st.write(st.session_state.insights)
                        elif not selected_pdfs:
                            st.write("Select PDFs to see insights")

    with chat_container:
        st.header("Chat")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    prompt = st.chat_input("What is your question?")
    if prompt:
        if st.session_state.insights is not None:
            mapping_df = map_paper_reference(prompt, st.session_state.insights)
            st.session_state.messages.append({"role": "user", "content": prompt})
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if not mapping_df.empty:
                st.subheader("Paper References and Additional Information")
                st.dataframe(mapping_df)
            else:
                st.info("No specific paper references found in the question.")
            
            # Comment out or remove the AI response part for now
            with st.spinner("Thinking..."):
                response = chat_main(prompt, mapping_df)
            st.session_state.messages.append({"role": "assistant", "content": response})

        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.info("No papers have been analyzed yet. Please upload and analyze some PDFs first.")
        st.experimental_rerun()

def main():
    st.markdown(f"""
    <div class="fixed-header">
        <img src="data:image/png;base64,{get_base64_of_image("LogoBig.png")}" alt="Logo">
        <h1>Insight AI</h1>
        <div class="nav-bar">
            <a href="https://www.youtube.com/watch?v=BBJa32lCaaY" class="nav-item" target="_blank">Coming Soon</a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    if 'insights' not in st.session_state:
        st.session_state.insights = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    show_home_page()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()