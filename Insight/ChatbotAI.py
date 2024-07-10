import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import datetime
from PIL import Image
import io
import os
import pandas as pd
import requests
import backoff
import base64
import re
from io import StringIO, BytesIO
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from engine import start_engine
from ImageAnalyzer import get_context
from UserChat import chat_main, update_vector_store
from generate_tables import generate_table_summaries
import time
import groq

load_dotenv()

# Initialize S3 client
s3 = boto3.client('s3')

# S3 bucket name
BUCKET_NAME = 'hackathon-jr'
USERNAME_FILE = 'usernames.txt'
LOGO_FILE = 'LogoBig.png'

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
        width: 40px;
        margin-right: 10px;
        opacity: 0.7;
    }
    .fixed-header h1 {
        margin: 0;
        font-size: 20px;
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
        margin-top: 60px;
    }
    .login-container {
        width: 100%;
        margin: 0 auto;
    }
    .login-container .stButton>button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

def load_and_display_logo():
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=LOGO_FILE)
        image_data = response['Body'].read()
        image = Image.open(io.BytesIO(image_data))
        st.image(image, use_column_width=True)
    except Exception as e:
        st.error(f"Failed to load logo: {str(e)}")

def check_username_exists(username):
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=USERNAME_FILE)
        existing_content = response['Body'].read().decode('utf-8')
        usernames = [line.split(',')[0] for line in existing_content.splitlines()]
        return username in usernames
    except s3.exceptions.NoSuchKey:
        return False
    except Exception as e:
        st.error(f"An error occurred while checking username: {str(e)}")
        return False

def append_username_to_s3(username):
    try:
        try:
            response = s3.get_object(Bucket=BUCKET_NAME, Key=USERNAME_FILE)
            existing_content = response['Body'].read().decode('utf-8')
        except s3.exceptions.NoSuchKey:
            existing_content = ""

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_content = f"{existing_content}{username},{timestamp}\n"

        s3.put_object(Bucket=BUCKET_NAME, Key=USERNAME_FILE, Body=new_content)

        create_user_folder(username)

        return True
    except NoCredentialsError:
        st.error("S3 credentials not available")
        return False
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return False

def create_user_folder(username):
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=f"{username}/")
    except Exception as e:
        st.error(f"Failed to create folder for {username}: {str(e)}")

def get_s3_csv(bucket, key):
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))

def extract_pdf_info(s3_pdf_path):
    # Extract bucket name and object key from s3_pdf_path
    bucket_name, object_key = s3_pdf_path.replace("s3://", "").split("/", 1)
    
    # Download the PDF file from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    pdf_content = response['Body'].read()
    
    # Create a PDF reader object
    pdf_reader = PdfReader(io.BytesIO(pdf_content))
    
    # Extract text from the first page (assuming title and authors are on the first page)
    first_page = pdf_reader.pages[0]
    text = first_page.extract_text()
    
    # Simple extraction logic (this should be improved for better accuracy)
    lines = text.split('\n')
    title = lines[0].strip()
    authors = lines[1].strip() if len(lines) > 1 else "Unknown"
    
    # Generate a simple summary (first 100 words)
    full_text = ' '.join([page.extract_text() for page in pdf_reader.pages])
    summary = ' '.join(full_text.split()[:100]) + '...'
    
    return title, authors, summary

import groq
import re

def map_paper_reference(user_input, insights_df):
    # Initialize Groq client with hard-coded API key
    client = groq.Groq(api_key="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")
    
    # Prompt for the LLaMa model
    prompt = f"""

### Instructions ###
You are an NLP engineer. Your task is to extract the academic paper numbers from the user's query below.
The "numbers" mean which academic papers the user is referring to. Interpret "Academic Paper" for terms such as "Thesis," "Paper," or "Document."
The figure or table numbers may be included in the user's query, but please ignore them.
Please provide your response as a list format, without any additional text for formatting. I will use your response directly.
If it is unclear which thesis is being referred to, PLEASE return "Can you specify the academic peper you are referring to?"


### Output Format ###
Format: a list

### Example user's query1 ###
What is the main hypothesis or research question addressed in the first academic article?

### Example Output1 ###
[1]

### Example user's query2 ###
Summarize the methodology used in the third academic article. Highlight any unique approaches or techniques employed.

### Example Output2 ###
[3]

### Example user's query3 ###
From the images and figures in the second article, describe the trend shown in Figure 3. What does it indicate about the research findings?

### Example Output3 ###
[2]

### Example user's query4 ###
Please compare table 3 and chart 4 from the second and third theses, respectively.

### Example Output4 ###
[2,3]


### User’s query ###
{user_input}

### Output ###
"""
    
    # Call the Groq API
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that extracts paper references from text."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    
    # Extract numbers from the response
    content = response.choices[0].message.content.strip()
    numbers = re.findall(r'\d+', content)
    extracted_references = [int(num) for num in numbers]
    
    mapping = {}
    for paper_num in extracted_references:
        if paper_num in insights_df.index:
            paper_info = insights_df.loc[paper_num]
            mapping[paper_num] = paper_info['PDF Name']
    
    if mapping:
        mapping_df = pd.DataFrame(list(mapping.items()), columns=['thesis_num', 'PDF Name'])
        
        try:
            summaries_df = get_s3_csv('hackathon-jr', 'summaries.csv')
            image_analysis_df = get_s3_csv('hackathon-jr', 'image_analysis_results.csv')
            
            merged_df = mapping_df.merge(summaries_df, on='thesis_num', how='left')
            merged_df = merged_df.merge(image_analysis_df, on='thesis_num', how='left')
            
            merged_df = merged_df.loc[:,~merged_df.columns.duplicated()]
            
            return merged_df
        except Exception as e:
            print(f"Error occurred while processing S3 data: {str(e)}")
            return mapping_df
    else:
        return pd.DataFrame()
    
def save_uploaded_file(uploaded_file, username):
    bucket_name = "hackathon-jr"
    
    if uploaded_file.size == 0:
        print(f"Error: Uploaded file '{uploaded_file.name}' is empty.")
        return False

    temp_file_path = uploaded_file.name
    
    try:
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if os.path.getsize(temp_file_path) == 0:
            print(f"Error: Temporary file '{temp_file_path}' is empty.")
            return False
        
        user_file_key = f"{username}/{uploaded_file.name}"
        s3.upload_file(temp_file_path, bucket_name, user_file_key)
        print(f"Successfully uploaded '{uploaded_file.name}' to S3 bucket '{bucket_name}' in folder '{username}'")
        
        os.remove(temp_file_path)
        
        return True
    
    except IOError as e:
        print(f"IOError while handling file: {e}")
    except ClientError as e:
        print(f"Error uploading file to S3: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    return False

def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def purge_user_folder(bucket_name, username):
    prefix = f"{username}/"
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    objects = response.get('Contents', [])
    for obj in objects:
        object_name = obj['Key']
        s3.delete_object(Bucket=bucket_name, Key=object_name)
    print(f"Purged folder for user {username}")

def show_login_page():
    # Create three columns: left spacer, center (for content), right spacer
    left_spacer, center_col, right_spacer = st.columns([2,3,2])
    
    with center_col:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        
        try:
            response = s3.get_object(Bucket=BUCKET_NAME, Key=LOGO_FILE)
            image_data = response['Body'].read()
            image = Image.open(io.BytesIO(image_data))
            
            max_width = 120  # Further reduced to make the image smaller
            aspect_ratio = image.width / image.height
            new_height = int(max_width / aspect_ratio)
            resized_image = image.resize((max_width, new_height))
            
            st.image(resized_image, use_column_width=True)
        except Exception as e:
            st.error(f"Failed to load logo: {str(e)}")

        st.markdown("<h2 style='text-align: center;'>Login or Sign Up</h2>", unsafe_allow_html=True)

        username = st.text_input("Username")
        
        col1, col2 = st.columns(2)
        
        with col1:
            login_button = st.button("Login", use_container_width=True, key="login")
        
        with col2:
            signup_button = st.button("Sign Up", use_container_width=True, key="signup")

        if login_button:
            if username:
                if check_username_exists(username):
                    st.session_state.username = username
                    st.success(f"Welcome back, {username}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Username not found. Please sign up.")
            else:
                st.warning("Please enter a username.")

        if signup_button:
            if username:
                if check_username_exists(username):
                    st.error("Username already exists. Please choose a different one.")
                else:
                    if append_username_to_s3(username):
                        st.session_state.username = username
                        st.success(f"Welcome, {username}! Your account has been created.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Failed to create account. Please try again.")
            else:
                st.warning("Please enter a username.")

        st.markdown('</div>', unsafe_allow_html=True)

def show_chatbot_page():
    st.title("Insight Analyzer")

    # Initialize the session state variable if it doesn't exist
    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    upload_insights_container = st.container()
    chat_container = st.container()

    with upload_insights_container:
        with st.expander("Upload PDFs and View Insights", expanded=True):
            left_column, right_column = st.columns(2)

            with left_column:
                st.subheader("Upload and Select PDFs")
                uploaded_files = st.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
                
                if uploaded_files:
                    if st.button(f"Submit"):
                        with st.spinner(f"Submitting..."):
                            
                            bucket_name = "hackathon-jr"
                            purge_user_folder(bucket_name, st.session_state.username)

                            for uploaded_file in uploaded_files:
                                if save_uploaded_file(uploaded_file, st.session_state.username):
                                    st.success(f"File {uploaded_file.name} uploaded successfully!")
                                else:
                                    st.error(f"Failed to upload {uploaded_file.name}")

                            update_vector_store("vectordb_faiss", uploaded_files, st.session_state.username)

                bucket_name = "hackathon-jr"
                prefix = f"{st.session_state.username}/"
                response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
                objects = response.get('Contents', [])
                pdfs = [obj['Key'].split('/')[-1] for obj in objects if obj['Key'].endswith('.pdf')]
                    
                selected_pdfs = st.multiselect("Select PDFs for analysis", pdfs)

            with right_column:  
                st.subheader("Initial Insights")
                st.markdown("<small>Takes approx 3 minutes per paper</small>", unsafe_allow_html=True)
                if selected_pdfs and st.button("Analyze Selected PDFs"):
                    print("Getting Insights")
                     # Delete existing paper_num.csv if it exists

                    try:
                        s3.delete_object(Bucket=bucket_name, Key=f'{st.session_state.username}/paper_num.csv')
                        print("Deleted existing paper_num.csv")
                        # Create new paper_num.csv
                        df = pd.DataFrame({"Paper_Num": [0]})
                    except Exception as e:
                        print(f"Error deleting paper_num.csv (This might be normal if it didn't exist): {e}")

                    insights = []
                    for pdf in selected_pdfs:
                        print("Selected PDFs", selected_pdfs)
                        with st.spinner(f"Analyzing {pdf}..."):
                            print("Starting Engine")
                            start_engine(f"{st.session_state.username}/{pdf}", st.session_state.username, df)
                            try:
                                s3_pdf_path = f"s3://hackathon-jr/{st.session_state.username}/{pdf}"
                                title, authors, summary = extract_pdf_info(s3_pdf_path)
                                insights.append({
                                    "PDF Name": pdf,
                                    "Paper Name": title,
                                    "Author Names": authors,
                                    "Short Description": summary
                                })
                                st.success(f"Successfully analyzed {pdf}")
                            except Exception as e:
                                st.error(f"Error extracting information from {pdf}: {str(e)}")

                    with st.spinner(f"Generating ..."):
                        generate_table_summaries(st.session_state.username)
                        get_context(st.session_state.username)

                        st.session_state.insights = pd.DataFrame(insights).reset_index(drop=True)
                        st.session_state.insights.index = st.session_state.insights.index + 1
                    
                        if st.session_state.insights is not None:
                            st.write(st.session_state.insights)
                            # Set the analysis_done flag to True
                            st.session_state.analysis_done = True
                        elif not selected_pdfs:
                            st.write("Select PDFs to see insights")

    with chat_container:
        st.header("Chat")
        
        # Display all existing messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.session_state.analysis_done:
            prompt = st.chat_input("What is your question?")
            if prompt:
                if st.session_state.insights is not None:
                    mapping_df = map_paper_reference(prompt, st.session_state.insights)
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    
                    if not mapping_df.empty:
                        st.subheader("Paper References and Additional Information")
                        st.dataframe(mapping_df)
                    else:
                        st.info("No specific paper references found in the question. Please include information about which paper number your question is referring to.")
                    
                    with st.spinner("Thinking..."):
                        response = chat_main(prompt, mapping_df, st.session_state.username)
                    st.session_state.messages.append({"role": "assistant", "content": response[0]})

                else:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.info("No papers have been analyzed yet. Please upload and analyze some PDFs first.")
                st.rerun()
        else:
            st.info("Please analyze selected PDFs before using the chatbot.")
            st.chat_input("What is your question?", disabled=True)

    # with chat_container:
    #     st.header("Chat")
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):
    #             st.markdown(message["content"])

    #     if st.session_state.analysis_done:
    #         prompt = st.chat_input("What is your question?")
    #         if prompt:
    #             if st.session_state.insights is not None:
    #                 mapping_df = map_paper_reference(prompt, st.session_state.insights)
    #                 st.session_state.messages.append({"role": "user", "content": prompt})
    #                 for message in st.session_state.messages:
    #                     with st.chat_message(message["role"]):
    #                         st.markdown(message["content"])
                    
    #                 if not mapping_df.empty:
    #                     st.subheader("Paper References and Additional Information")
    #                     st.dataframe(mapping_df)
    #                 else:
    #                     st.info("No specific paper references found in the question. Please include information about which paper number your question is referring to.")
                    
    #                 with st.spinner("Thinking..."):
    #                     response = chat_main(prompt, mapping_df, st.session_state.username)
    #                 st.session_state.messages.append({"role": "assistant", "content": response[0]})

    #             else:
    #                 st.session_state.messages.append({"role": "user", "content": prompt})
    #                 st.info("No papers have been analyzed yet. Please upload and analyze some PDFs first.")
    #             st.rerun()
    #     else:
    #         st.info("Please analyze selected PDFs before using the chatbot.")
    #         st.chat_input("What is your question?", disabled=True)

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

    if 'username' not in st.session_state:
        st.session_state.username = None

    if 'insights' not in st.session_state:
        st.session_state.insights = None

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.username:
        show_login_page()
    else:
        show_chatbot_page()

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()