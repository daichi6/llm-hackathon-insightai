# UserChat.py

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from sentence_transformers import SentenceTransformer
import tempfile

import faiss
import os
import pickle
import pymupdf as fitz
import json
import pandas as pd
from groq import Groq
from io import StringIO
import boto3
from dotenv import load_dotenv
import re



class CustomHuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed(self, texts):
        return self.model.encode(texts)
    
    def embed_documents(self, texts):
        return self.embed(texts)
    
    def __call__(self, text):
        if isinstance(text, str):
            return self.embed([text])[0]
        return self.embed(text)

    
print("Loading embeddings") # BREAKS HERE
# Daichi Added - Load the vectorDB and create retreivers
#embeddings = HuggingFaceEmbeddings(model_name="allenai/scibert_scivocab_uncased")
#embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embeddings = CustomHuggingFaceEmbeddings()
print("Finished loading embeddings")

## 2. Functions
# Functions

def extract_text(pdf_path):
    """
    Extract text from a single PDF file.

    Args:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF.
    """
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text("text")
    return text

def extract_texts_from_pdfs(pdf_paths):
    """
    Extract text from each PDF file in the list and create Document objects.

    Args:
        pdf_paths (list of str): List of paths to PDF files.

    Returns:
        list of Document: List of Document objects containing the extracted text.
    """
    docs = []
    for pdf_path in pdf_paths:
        text = extract_text(pdf_path)
        doc = Document(page_content=text, metadata={"source": pdf_path})
        docs.append(doc)
    return docs

def split_documents_into_chunks(docs, chunk_size=500, chunk_overlap=100):
    """
    Splits the given documents into chunks of specified size with overlap.

    Args:
        docs (list): List of documents to split.
        chunk_size (int): Size of each chunk. Default is 500 characters.
        chunk_overlap (int): Overlap size between chunks. Default is 100 characters.

    Returns:
        dict: Dictionary of lists containing split documents with chunks per original document.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    doc_chunks = {}
    for doc in docs:
        doc_chunks[doc.metadata["source"]] = text_splitter.split_documents([doc])
    return doc_chunks

def add_chunk_numbers_to_metadata(doc_chunks):
    """
    Adds chunk numbers to the metadata of each split document.

    Args:
        doc_chunks (dict): Dictionary of lists containing split documents.

    Returns:
        dict: Dictionary of lists containing split documents with updated metadata.
    """
    for chunks in doc_chunks.values():
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk"] = idx
    return doc_chunks

def configure_faiss_vector_store(doc_splits, embeddings):
    """
    Configures FAISS as the vector store using the provided document splits and embeddings.

    Args:
        doc_splits (dict): Dictionary of lists containing split documents.
        embeddings (Embeddings): Embeddings to be used for FAISS.

    Returns:
        dict: Dictionary of FAISS vector stores per document.
    """
    vector_stores = {}
    for doc_source, chunks in doc_splits.items():
        vector_stores[doc_source] = FAISS.from_documents(chunks, embeddings)
    return vector_stores

### Change
# def save_faiss_vector_store(vector_db, directory):
#     print("Directory: ", directory)
#     if not os.path.exists(directory):
#         os.makedirs(directory)

#     for doc_source, vector_store in vector_db.items():
#         index_path = os.path.join(directory, f"{doc_source}.index")
#         print("Index Path", index_path)
#         faiss.write_index(vector_store.index, index_path)

#         # Save docstore and index_to_docstore_id
#         metadata_path = os.path.join(directory, f"{doc_source}.metadata")
#         print("Metadata Path", metadata_path)
#         with open(metadata_path, 'wb') as f:
#             pickle.dump({
#                 'docstore': vector_store.docstore,
#                 'index_to_docstore_id': vector_store.index_to_docstore_id
#             }, f)

def save_faiss_vector_store(vector_db, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for doc_source, vector_store in vector_db.items():
        # Extract just the filename from the full path
        filename = os.path.basename(doc_source)
        
        # Remove the .pdf extension if present
        filename = os.path.splitext(filename)[0]
        
        # Create a safe filename
        safe_filename = re.sub(r'[^\w\-_\. ]', '_', filename)
        
        # Create the index and metadata filenames
        index_filename = f"{safe_filename}.pdf.index"
        metadata_filename = f"{safe_filename}.pdf.metadata"
        
        # Create the full paths
        index_path = os.path.join(directory, index_filename)
        metadata_path = os.path.join(directory, metadata_filename)

        print("Index Path", index_path)
        faiss.write_index(vector_store.index, index_path)

        print("Metadata Path", metadata_path)
        with open(metadata_path, 'wb') as f:
            pickle.dump({
                'docstore': vector_store.docstore,
                'index_to_docstore_id': vector_store.index_to_docstore_id
            }, f)

def load_faiss_vector_store(directory, embedding):
    vector_db = {}

    ### Change
    # for filename in os.listdir(directory):
    #     if filename.endswith('.index'):
    #         doc_source = os.path.splitext(filename)[0]
    #         index_path = os.path.join(directory, filename)
    #         metadata_path = os.path.join(directory, f"{doc_source}.metadata")

    for filename in os.listdir(directory):
        if filename.endswith('.pdf.index'):
            doc_source = filename[:-10]  # Remove '.pdf.index'
            index_path = os.path.join(directory, filename)
            metadata_path = os.path.join(directory, f"{doc_source}.pdf.metadata")

            # Load FAISS index
            index = faiss.read_index(index_path)
            #index = FAISS.read_index(index_path)


            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)

            # Reconstruct FAISS vector store
            vector_store = FAISS(
                embedding_function=embedding,
                index=index,
                docstore=metadata['docstore'],
                index_to_docstore_id=metadata['index_to_docstore_id']
            )

            vector_db[doc_source] = vector_store

    return vector_db


s3 = boto3.client('s3')

# def update_vector_store(existing_vectordb_path, new_pdf_paths, embedding_model):
#     # Load existing vector store
#     vector_db = load_faiss_vector_store(existing_vectordb_path, embedding_model)

#     # grab from s3 buckets and extract the text
#     for pdf_path in new_pdf_paths:
#         s3.download_file('hackathon-jr', pdf_path, pdf_path)
#     new_docs = extract_texts_from_pdfs(new_pdf_paths)

#     # new_docs = extract_texts_from_pdfs(new_pdf_paths)

#     # Split the documents into chunks and add metadata
#     new_doc_splits = split_documents_into_chunks(new_docs)
#     new_doc_splits = add_chunk_numbers_to_metadata(new_doc_splits)

#     # Configure FAISS for new documents
#     new_vector_db = configure_faiss_vector_store(new_doc_splits, embedding_model)

#     # Update existing vector store with new documents
#     vector_db.update(new_vector_db)

#     # Save the updated vector store
#     save_faiss_vector_store(vector_db, existing_vectordb_path)

#     return vector_db
 

def create_retrievers(vector_stores, search_type="similarity", k=5):
    """
    Exposes the vector store index to retrievers for multiple documents.

    Args:
        vector_stores (dict): Dictionary of FAISS vector stores per document.
        search_type (str): The type of search to perform. Default is "similarity".
        k (int): The number of documents to return. Default is 5.

    Returns:
        dict: Dictionary of retrievers per document.
    """

    retrievers = {}
    for doc_source, vector_store in vector_stores.items():
        retrievers[doc_source] = vector_store.as_retriever(
            search_type=search_type, search_kwargs={"k": k}
        )
    return retrievers

def process_query(query: str, retriever):
    """
    Processes the query using the provided retriever to retrieve relevant document chunks.

    Args:
        query (str): The query string to search for relevant documents.
        retriever: The retriever object configured to use the vector store for document retrieval.

    Returns:
        str: A string containing the formatted content and metadata of the retrieved document chunks.
    """
    # Retrieve chunks based on the query
    docs = retriever.get_relevant_documents(query)

    # Initialize an empty string to collect all outputs
    full_output = ""

    for i, doc in enumerate(docs, 1):
        chunk_output = f"-----Chunk {i}------\n"
        chunk_output += f"Content: {doc.page_content}...\n"
        chunk_output += f"Metadata {doc.metadata}\n\n"

        # Append the chunk output to the full output
        full_output += chunk_output

    return full_output

def get_groq_response(client, prompt, model="llama3-70b-8192", max_tokens=2048, temperature=0.0):
    """
    Generates a response using the provided client, model, prompt, and specified parameters.

    Args:
        client: The client object to interact with the API.
        prompt (str): The prompt to generate a response for.
        model (str, optional): The model identifier to use for generating the response. Default is "llama3-70b-8192".
        max_tokens (int, optional): The maximum number of tokens for the generated response. Default is 2048.
        temperature (float, optional): The temperature setting for the response generation. Default is 0.0.

    Returns:
        tuple: The generated response content and usage statistics.
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return chat_completion.choices[0].message.content, chat_completion.usage
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None

def generate_prompt_hyde(instruction, user_query, context_hyde):
    """
    Generates a prompt for HyDE by replacing placeholders in the instruction template with the user's query and context.

    Args:
        instruction (str): The template instruction containing placeholders.
        user_query (str): The user's query to be inserted into the instruction.
        context_hyde (str): The context for creating a hypothetical answer to be inserted into the instruction.

    Returns:
        str: The generated instruction with the placeholders replaced by the user's query and context.
    """
    instruction = instruction.replace("{USER_QUERY}", user_query)
    instruction = instruction.replace("{CONTEXT_HYDE}", context_hyde)
    return instruction

def generate_prompt_final(instruction, user_query, context_figure_table, context_rag_hyde, context_rag_general):
    """
    Generates a final prompt by replacing placeholders in the instruction template with the user's query and various contexts.

    Args:
        instruction (str): The template instruction containing placeholders.
        user_query (str): The user's query to be inserted into the instruction.
        context_figure_table (str): The context(description) related to figure and table to be inserted into the instruction.
        context_rag_hyde (str): The context retreived from RAG HyDE to be inserted into the instruction.
        context_rag_general (str): The general context retreived from RAG to be inserted into the instruction.

    Returns:
        str: The generated instruction with the placeholders replaced by the user's query and contexts.
    """
    instruction = instruction.replace("{USER_QUERY}", user_query)
    instruction = instruction.replace("{CONTEXT_FIGURE_TABLE}", context_figure_table)
    instruction = instruction.replace("{CONTEXT_RAG_HYDE}", context_rag_hyde)
    instruction = instruction.replace("{CONTEXT_RAG_GENERAL}", context_rag_general)
    return instruction

def generate_prompt_extract_query(instruction, user_query):
    """
    Generates a prompt for extracting keys from the user's query by replacing placeholders in the instruction template.

    Args:
        instruction (str): The template instruction containing a placeholder for the user's query.
        user_query (str): The user's query to be inserted into the instruction.

    Returns:
        str: The generated instruction with the placeholder replaced by the user's query.
    """
    instruction = instruction.replace("{USER_QUERY}", user_query)
    return instruction

def parse_and_convert_keys(json_string):
    """
    Parse the JSON string and convert the string values in the keys list to their appropriate types.

    Args:
    json_string (str): A JSON string representing a list of dictionaries with string values for 'thesis', 'figure', and 'table'.

    Returns:
    list: A list of dictionaries with 'thesis' as int, and 'figure' and 'table' as int or None.
    """
    try:
        keys = json.loads(json_string)
        if not keys:
            return []

        converted_keys = []
        for key in keys:
            converted_key = {
                "thesis": int(key["thesis"]) if key["thesis"] else None,
                "figure": int(key["figure"]) if key["figure"] else None,
                "table": int(key["table"]) if key["table"] else None
            }
            converted_keys.append(converted_key)
        return converted_keys
    except json.JSONDecodeError as e:
        # print(f"JSON decoding error: {e}")
        return []
    except KeyError as e:
        # print(f"Missing key in JSON data: {e}")
        return []
    except ValueError as e:
        # print(f"Value error: {e}")
        return []
    except Exception as e:
        # print(f"An unexpected error occurred: {e}")
        return []

# def extract_descriptions(df, keys):
#     """
#     Extract and format descriptions from the dataframe based on the provided keys.

#     Args:
#     df (DataFrame): The dataframe containing thesis, figure, table, and description data.
#     keys (list): A list of dictionaries with 'thesis' as int, and 'figure' and 'table' as int or None.

#     Returns:
#     list: A list of formatted descriptions corresponding to the provided keys.
#     """
#     formatted_descriptions = []

#     for key in keys:
#         thesis_num = key["thesis"]
#         figure_num = key["figure"]
#         table_num = key["table"]

#         if figure_num is not None:
#             description = df[(df["thesis_num"] == thesis_num) & (df["figure_num"] == figure_num)]["description"].values
#             prefix = f"thesis{thesis_num} figure{figure_num} description: "
#         elif table_num is not None:
#             description = df[(df["thesis_num"] == thesis_num) & (df["table_num"] == table_num)]["description"].values
#             prefix = f"thesis{thesis_num} table{table_num} description: "
#         else:
#             description = []
#             prefix = ""

#         if len(description) > 0:
#             formatted_descriptions.append(prefix + description[0])
#         else:
#             formatted_descriptions.append(prefix + "Description not found")

#     return formatted_descriptions

def extract_descriptions(df, keys):
    """
    Extract and format descriptions from the dataframe based on the provided keys.

    Args:
    df (DataFrame): The dataframe containing thesis, figure, table, and description data.
    keys (list): A list of dictionaries with 'thesis' as int, and 'figure' and 'table' as int or None.

    Returns:
    list: A list of formatted descriptions corresponding to the provided keys.
    """
    formatted_descriptions = []

    # Check available columns
    available_columns = df.columns
    
    for key in keys:
        thesis_num = key["thesis"]
        figure_num = key.get("figure")
        table_num = key.get("table")

        description = df[df["thesis_num"] == thesis_num]

        if "figure_num" in available_columns and figure_num is not None:
            description = description[description["figure_num"] == figure_num]
            prefix = f"thesis{thesis_num} figure{figure_num} description: "
        elif "table_num" in available_columns and table_num is not None:
            description = description[description["table_num"] == table_num]
            prefix = f"thesis{thesis_num} table{table_num} description: "
        else:
            prefix = f"thesis{thesis_num} description: "

        if not description.empty and "description" in available_columns:
            formatted_descriptions.append(prefix + description["description"].values[0])
        else:
            formatted_descriptions.append(prefix + "Description not found")

    return formatted_descriptions

def extract_thesis_numbers(converted_keys):
    """
    Extracts the thesis numbers from a list of dictionaries.

    Args:
    converted_keys (list): A list of dictionaries with 'thesis', 'figure', and 'table' keys.

    Returns:
    list: A list of thesis numbers.
    """
    try:
        thesis_numbers = [item['thesis'] for item in converted_keys]
        return thesis_numbers
    except Exception as e:
        # print(f"An error occurred while extracting thesis numbers: {e}")
        return []

def get_descriptions_for_thesis_summary(thesis_numbers, table_summary):
    """
    Retrieves the descriptions for the given thesis numbers from the table_summary DataFrame.

    Args:
    thesis_numbers (list): A list of thesis numbers.
    table_summary (pd.DataFrame): The DataFrame containing thesis numbers and their descriptions.

    Returns:
    list: A list of descriptions corresponding to the thesis numbers, formatted to indicate which thesis each description belongs to.
    """
    try:
        result = []
        for thesis_num in thesis_numbers:
            description = table_summary.loc[table_summary['thesis_num'] == thesis_num, 'description'].values[0]
            result.append(f"Summary description for thesis {thesis_num}: '{description}'")
        return result
    except Exception as e:
        # print(f"An error occurred while retrieving descriptions: {e}")
        return []
## 3. Prompts
# Prompts(Instructions)

instruction_hyde = """
### Instructions ###
You are an expert in scientific academic papers. Your task is to answer to "Users' query" below.　If the information in the "Context" below seems relevant to "Users' query", please refer to it.

### User’s query ###
{USER_QUERY}

### Context ###
{CONTEXT_HYDE}

### Output ###
"""

instruction_final = """
### Instructions ###
You are an expert in scientific academic papers. Your task is to answer to "Users' query" below.
If the information in the "Figure/Table Context" and "Text Context" below seem relevant to "Users' query", please refer to them.
"Text Context" includes several chunks from different parts of an academic paper. "Figure/Table Context" includes the descriptions related to figures or tables in an academic paper.
Please refer only to the relevant contexts for your response. There is no need to include unrelated context in your response.
If the user asks about a specific figure or table and the information is contained in the Figure/Table Context, please ensure that this information is included in your response.
If you determine that the previous conversation history is relevant, please also refer to that information to answer the user's query, especially when the the contexts below are empty.
If the contexts and the previous conversation history do not contain the necessary information and it is difficult to answer even with general knowledge and previous context, please respond with 'The information provided is insufficient to answer your question.　Could you please clarify your question with the paper numbers?'.

##### User’s query #####
{USER_QUERY}


##### Figure/Table Context #####
{CONTEXT_FIGURE_TABLE}

##### Text Context #####
{CONTEXT_RAG_HYDE}

{CONTEXT_RAG_GENERAL}


##### Output #####
"""

instruction_extract_query = """
### Instructions ###
You are an NLP engineer. Your task is to extract the "numbers" from the user's query below.
The "numbers" mean which academic paper the user is referring to, 2) which figure the user is referring to, and 3) which table the user is referring to.
There may be cases where all, some, or none of these are specified. Enter the number only for the specified fields, and return an empty string "" for fields that are not specified.
Interpret "figure" for terms such as "Chart," "Diagram," or "Image." Interpret "thesis" for terms such as "Academic Paper," "Paper," or "Document."
Please provide your response as a list of objects, each containing thesis, figure, and table.　Please provide your response strictly in the specified format, without including any additional text for formatting. I will use your response directly.
If it is unclear which thesis, figure, or table is being referred to, it is okay to return an empty string. Please do not make any assumptions.

### Output Format ###
Format: a list of objects

### Example user's query1 ###
What is the main hypothesis or research question addressed in the first academic article?

### Example Output1 ###
[
  {
  "thesis": "1",
  "figure": "",
  "table": ""
  }
]

### Example user's query2 ###
Summarize the methodology used in the third academic article. Highlight any unique approaches or techniques employed.

### Example Output2 ###
[
  {
  "thesis": "3",
  "figure": "",
  "table": ""
  }
]


### Example user's query3 ###
Q. From the images and figures in the second article, describe the trend shown in Figure 2. What does it indicate about the research findings?

### Example Output3 ###
[
  {
  "thesis": "2",
  "figure": "2",
  "table": ""
  }
]

### Example user's query4 ###
Q. What can be understood from Image 3 in the third paper?

### Example Output4 ###
[
  {
  "thesis": "3",
  "figure": "3",
  "table": ""
  }
]

### Example user's query4 ###
Q. Please explain Figure 3 and Table 2 of the second academic paper. What do these indicate about the research findings?

### Example Output4 ###
[
  {
  "thesis": "2",
  "figure": "3",
  "table": ""
  },
  {
  "thesis": "2",
  "figure": "",
  "table": "2"
  }
]

### Example user's query5 ###
Q. Please compare table 3 and chart 4 from the second and third theses, respectively.

### Example Output5 ###
[
  {
  "thesis": "2",
  "figure": "",
  "table": "3"
  },
  {
  "thesis": "3",
  "figure": "4",
  "table": ""
  }
]

### Example user's query6 ###
Do you like an apple?

### Example Output6 ###
[
  {
  "thesis": "",
  "figure": "",
  "table": ""
  }
]

### Example user's query7 ###
Considering the previous conversations, please propose a new research direction or hypothesis.

### Example Output7 ###
[
  {
  "thesis": "",
  "figure": "",
  "table": ""
  }
]


### User’s query ###
{USER_QUERY}

### Output ###
"""

# def update_vectorbase(pdf_path): 
# #   global retrievers
  
#   pdf_paths = [pdf_path]  # Change it to your new PDF paths
#existing_vectordb_path = "vectordb_faiss"  # Change it to your existing VectorDB path

#   # Embedding
#   # SciBERT(Allen Institute for AI) - for academic(science) paper including computer science
# #   embeddings = HuggingFaceEmbeddings(model_name="allenai/scibert_scivocab_uncased")
#   # Update the vector store with new PDFs
#   vector_db = update_vector_store(existing_vectordb_path, pdf_paths, embeddings)
  
# #   retrievers = create_retrievers(vector_db)

# #   return retrievers


# ADDRESSSSSS
#ISSUE


# new_pdf_names = [".pdf",".pdf"]
# update_vector_store("vectordb_faiss", new_pdf_paths, embeddings)
print("Finished updating vectorDB")

#vector_db = load_faiss_vector_store("vectordb_faiss", embeddings)

### Change
vector_db = load_faiss_vector_store("vectordb_faiss", embeddings)
print("Finished loading vectorDB")
retrievers = create_retrievers(vector_db)
print("Finished creating retrievers")


## 6. Main function
## User selection and Mapping
# User thesis selection before asking questions

#pdf_paths_user_selected = ["attention.pdf", "Multimodal.pdf"]
# mapping for image/table description table
# YOUR CODES HERE

# mapping for summary table
# YOUR CODES HERE
# LLM for the main flow
load_dotenv()

client_main = Groq(
    api_key=os.getenv('CLIENT_MAIN_API_KEY')
)

client_extract = Groq(
    api_key=os.getenv('CLIENT_EXTRACT_API_KEY')
)

client_hyde = Groq(
    api_key=os.getenv('CLIENT_HYDE_API_KEY')
)


def update_vector_store(existing_vectordb_path, new_pdf_names, username):

    new_pdf_names = [pdf_name.name for pdf_name in new_pdf_names]

    print("New PDF Names: ", new_pdf_names)

    embedding_model = embeddings
    # Load existing vector store
    vector_db = load_faiss_vector_store(existing_vectordb_path, embedding_model)

    # Extract the base names of existing PDFs in the vector DB
    existing_keys = vector_db.keys()

    # Filter out the PDF files that are already processed
    new_files_to_process = [pdf_name for pdf_name in new_pdf_names if pdf_name not in existing_keys]

    # If no new files to process, return
    if not new_files_to_process:
        print("No new PDF files to process.")
        return

    ###### NEED TO CHECK ######
    # Download new files from S3 and extract the text - check this 
    
    for pdf_name in new_files_to_process:

        s3_pdf_path = f"{username}/{pdf_name}"
        # Create a temporary directory for this process
        #grab the current directory 
        temp_dir = os.getcwd()
        local_pdf_path = os.path.join(temp_dir, pdf_name)
        
        # Download PDF from S3
        try:
            s3.download_file('hackathon-jr', s3_pdf_path, local_pdf_path)
            print(f"Downloaded {pdf_name} from S3.")
        except Exception as e:
            print(f"Failed to download {pdf_name} from S3: {e}")
            return 0, None  # Return early if download fails

    # try: 
    #     s3.download_file('hackathon-jr', pdf_name, pdf_name)
    #     print("Downloaded PDF from S3")
    # except Exception as e:
    #     print(f"An error occurred while downloading the PDF from S3: {e}")
    #     return

        print([local_pdf_path])
        new_docs = extract_texts_from_pdfs([local_pdf_path])

        # Split the documents into chunks and add metadata
        new_doc_splits = split_documents_into_chunks(new_docs)

        print("New Doc Splits: ")
        print(new_doc_splits)
        print()
        new_doc_splits = add_chunk_numbers_to_metadata(new_doc_splits)

        print("New New Doc Splits: ")
        print(new_doc_splits)
        print()

        # Configure FAISS for new documents
        new_vector_db = configure_faiss_vector_store(new_doc_splits, embedding_model)

        print()
        print("New Vector DB: ")
        print(new_vector_db)

        # Update existing vector store with new documents
        vector_db.update(new_vector_db)

        # Save the updated vector store
        ### Change
        #save_faiss_vector_store(vector_db, existing_vectordb_path)
        save_faiss_vector_store(vector_db, "vectordb_faiss")

        global retrievers

        retrievers = create_retrievers(vector_db)

        print("Updated Retrievers: ", retrievers)

    print("Vector store updated successfully.")

# Main

def chat_main(user_query, mapping_df, username):
    global retrievers

    # Initialize the S3 client
    s3 = boto3.client('s3')

    # Specify the bucket name and object keys
    bucket_name = 'hackathon-jr'
    user_prefix = f"{username}/"
    object_key_image = f"{user_prefix}image_analysis_results.csv"
    object_key_table = f"{user_prefix}summaries.csv"

    try:
        # Read the content of the image analysis file from S3
        response = s3.get_object(Bucket=bucket_name, Key=object_key_image)
        content = response['Body'].read().decode('utf-8')

        # Read the content of the summaries file from S3
        response_summary = s3.get_object(Bucket=bucket_name, Key=object_key_table)
        content_summary = response_summary['Body'].read().decode('utf-8')

        # Create DataFrames from the content
        table_summary = pd.read_csv(StringIO(content))
        table_figure_table = pd.read_csv(StringIO(content_summary))

        # Now you have your data in the DataFrames called table_summary and table_figure_table
        print("Finished reading in the data")
        print(table_summary.head())
        print(table_figure_table.head())

        # Return the response
        # return response

    except s3.exceptions.NoSuchKey:
        print(f"One or more files not found in the user's folder: {user_prefix}")
        return "Error: Unable to retrieve necessary data. Please ensure all required files have been generated."

  
    ## 1. extract keys from user's query and find figure/table description and summary description ##
    # generate prompt to extract keys from user's query
    prompt_extract_query = generate_prompt_extract_query(instruction_extract_query, user_query)
    # get keys from Extraction LLM
    response_keys = get_groq_response(client_extract, prompt_extract_query)

    # parse keys
    keys = parse_and_convert_keys(response_keys[0])

    # extract figure/table descriptions
    descriptions_figure_table = extract_descriptions(table_figure_table, keys)

    # extract thesis numbers from keys
    keys_thesis = extract_thesis_numbers(keys)
    # get summary descriptions
    descriptions_summary = get_descriptions_for_thesis_summary(keys_thesis, table_summary)

    ## 2. get context for HyDE and general RAG ##
    # add summary of the thesis as a context for HyDE
    context_hyde = descriptions_summary
    # create prompt for HyDE
    prompt_hyde = generate_prompt_hyde(instruction_hyde, user_query, str(context_hyde))
    # get a hypothetical answer from HyDE LLM
    response_hyde = get_groq_response(client_hyde, prompt_hyde)

    # # create contexts
    # initialize empty strings for contexts
    context_rag_hyde = ""
    context_rag_general = ""

    print("OVERALL MAP: ", mapping_df)

    # search for documents based on keys_thesis(thesis number extracted from user's query)
    if keys_thesis and all(key is not None for key in keys_thesis):
        for key in keys_thesis:
            if isinstance(key, int):
                #   adjusted_key = key - 1  # adjust the key by subtracting 1
                #   doc_source = mapping_df["thesis_num"][adjusted_key]  # get the document source(FROM USER'S SELECTED LISTS) based on the adjusted key
                doc_source = mapping_df[mapping_df['thesis_num']==key]['PDF Name'].iloc[0]
                print(doc_source)

                # S3 bucket details
                bucket_name = 'hackathon-jr'
                file_name = 'doc_source.txt'

                try:
                    # Initialize the S3 client
                    s3 = boto3.client('s3')

                    # Convert the list to a string
                    content = '\n'.join(map(str, doc_source))

                    # Create a file-like object in memory
                    file_obj = StringIO(content)

                    # Upload the file to S3
                    s3.put_object(Bucket=bucket_name, Key=file_name, Body=file_obj.getvalue())

                    print(f"Successfully wrote list to s3://{bucket_name}/{file_name}")

                except Exception as e:
                    print(f"Error writing to S3: {str(e)}")

                #remove pdf from doc_source
                doc_source = doc_source.replace(".pdf", "")
                print("This is Doc Source: " + doc_source) 
                global retriever
                
                print("Chat Main Retrievers: ", retrievers)
                retriever = retrievers[doc_source]  # get the corresponding retriever

                # process query for RAG(Hyde)
                result_hyde = process_query(response_hyde[0], retriever)
                context_rag_hyde += f"Document {key}:\n{result_hyde}\n"

                # process query for RAG(General)
                result_general = process_query(user_query, retriever)
                context_rag_general += f"Document {key}:\n{result_general}\n"
    else:
        context_rag_hyde = ""
        context_rag_general = ""

    ## 3. get a final response ##
    # create prompt for a final response
    prompt_final = generate_prompt_final(instruction_final, user_query, str(descriptions_figure_table), context_rag_hyde, context_rag_general)
    # get final response from main LLM
    response_final = get_groq_response(client_main, prompt_final)

    return response_final