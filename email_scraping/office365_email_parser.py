import os
import argparse
import pickle
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Tuple
import tiktoken
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from langchain.schema import Document
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool
from email.utils import parsedate_to_datetime
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_email(file_path: str) -> Dict:
    """
    Parse a single email file using UnstructuredEmailLoader.
    
    Args:
    file_path (str): Path to the email file (.eml or .msg)
    
    Returns:
    Dict: Parsed email data
    """
    print(f"Attempting to parse email: {file_path}")
    try:
        loader = UnstructuredEmailLoader(file_path, mode="elements")
        print("UnstructuredEmailLoader initialized successfully")
        
        elements = loader.load()
        # print(f"Loaded {len(elements)} elements from the email")
        
        email_data = {
            'source': file_path,
            'content': [],
            'metadata': {}
        }
        
        for i, element in enumerate(elements):
            # print(f"Processing element {i+1}/{len(elements)}")
            email_data['content'].append(element.page_content)
            email_data['metadata'].update(element.metadata)
        
        # Combine content into a single string
        email_data['content'] = '\n'.join(email_data['content'])
        
        print("Email parsing completed successfully")
        return email_data
    except Exception as e:
        print(f"Error parsing email {file_path}: {str(e)}")
        return None

def process_email_directories(directories: List[str]) -> List[Dict]:
    """
    Process all email files in multiple directories.
    
    Args:
    directories (List[str]): List of paths to directories containing email files
    
    Returns:
    List[Dict]: List of parsed email data
    """
    parsed_emails = []
    total_files = sum(len([f for f in os.listdir(d) if f.endswith((".eml", ".msg"))]) for d in directories)
    
    with tqdm(total=total_files, desc="Processing emails") as pbar:
        for directory in directories:
            print(f"\nProcessing emails in directory: {directory}")
            if not os.path.exists(directory):
                print(f"Warning: Directory '{directory}' does not exist. Skipping.")
                continue
            for filename in os.listdir(directory):
                if filename.endswith((".eml", ".msg")):
                    file_path = os.path.join(directory, filename)
                    parsed_email = parse_email(file_path)
                    if parsed_email:
                        parsed_emails.append(parsed_email)
                        pbar.set_postfix({"Last processed": filename})
                    pbar.update(1)
    
    print(f"\nProcessed {len(parsed_emails)} emails successfully")
    return parsed_emails

def count_tokens(text: str) -> int:
    """
    Count the number of tokens in a given text using tiktoken.
    """
    encoding = tiktoken.encoding_for_model("text-embedding-3-small")
    return len(encoding.encode(text))

def calculate_embedding_tokens(documents: List[Document], chunk_size: int, chunk_overlap: int) -> Tuple[pd.DataFrame, int]:
    """
    Calculate the number of embedding tokens for each document and prepare a DataFrame.
    
    Args:
    documents (List[Document]): List of Langchain Document objects
    chunk_size (int): Size of each text chunk
    chunk_overlap (int): Overlap between chunks
    
    Returns:
    Tuple[pd.DataFrame, int]: DataFrame with token counts and estimated costs, and total number of embeddings
    """
    print("Calculating embedding tokens...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    data = []
    total_tokens = 0
    total_embeddings = 0
    
    for doc in tqdm(documents, desc="Calculating tokens"):
        full_text = f"Source: {doc.metadata.get('source', 'Unknown')}\n"
        full_text += f"Content: {doc.page_content}\n"
        for key, value in doc.metadata.items():
            if key != 'source':
                full_text += f"{key}: {value}\n"
        
        chunks = text_splitter.split_text(full_text)
        num_chunks = len(chunks)
        tokens = sum(count_tokens(chunk) for chunk in chunks)
        total_tokens += tokens
        total_embeddings += num_chunks
        
        data.append({
            'source': doc.metadata.get('source', 'Unknown'),
            'num_chunks': num_chunks,
            'tokens': tokens,
        })
    
    df = pd.DataFrame(data)
    
    # Calculate costs for each model
    df['cost_text_embedding_3_small'] = df['tokens'] * 0.02 / 1_000_000
    df['cost_text_embedding_3_large'] = df['tokens'] * 0.13 / 1_000_000
    df['cost_ada_v2'] = df['tokens'] * 0.10 / 1_000_000
    
    df.loc['Total'] = df.sum()
    df.loc['Total', 'source'] = 'TOTAL'
    
    print(f"Total tokens to be embedded: {total_tokens}")
    print(f"Total number of embeddings: {total_embeddings}")
    print("\nEstimated costs for each model:")
    print(f"text-embedding-3-small: ${df.loc['Total', 'cost_text_embedding_3_small']:.2f}")
    print(f"text-embedding-3-large: ${df.loc['Total', 'cost_text_embedding_3_large']:.2f}")
    print(f"ada v2: ${df.loc['Total', 'cost_ada_v2']:.2f}")
    
    return df, total_embeddings

def save_documents(documents: List[Document], output_file: str):
    """Save Langchain documents to a local file."""
    with open(output_file, 'wb') as f:
        pickle.dump(documents, f)
    logging.info(f"Saved {len(documents)} documents to {output_file}")

def load_documents(input_file: str) -> List[Document]:
    """Load Langchain documents from a local file."""
    with open(input_file, 'rb') as f:
        documents = pickle.load(f)
    logging.info(f"Loaded {len(documents)} documents from {input_file}")
    return documents

def parse_date(date_string: str) -> str:
    try:
        return datetime.fromisoformat(date_string).isoformat()
    except ValueError:
        return date_string  # Return original string if parsing fails

def clean_url(url: str) -> str:
    return url.replace('3D"', '').strip('"')

def prepare_documents(parsed_emails: List[Dict], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Prepare Langchain documents from parsed emails with comprehensive metadata."""
    logging.info("Preparing documents for embedding...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = []
    for email in tqdm(parsed_emails, desc="Preparing documents"):
        chunks = text_splitter.split_text(email['content'])
        
        # Extract relevant metadata
        metadata = {
            "source": email['source'],
            "subject": email['metadata'].get('subject', ''),
            "from": email['metadata'].get('sent_from', [''])[0],  # Take the first sender if multiple
            "to": ', '.join(email['metadata'].get('sent_to', [])),
            "message_id": email['metadata'].get('email_message_id', '').strip(),
            "date": parse_date(email['metadata'].get('last_modified', '')),
            "languages": email['metadata'].get('languages', []),
            "filetype": email['metadata'].get('filetype', ''),
            "file_directory": email['metadata'].get('file_directory', ''),
            "filename": email['metadata'].get('filename', ''),
            "category": email['metadata'].get('category', ''),
            "element_id": email['metadata'].get('element_id', ''),
            "category_depth": email['metadata'].get('category_depth', 0),
            "link_texts": email['metadata'].get('link_texts', []),
            "link_urls": [clean_url(url) for url in email['metadata'].get('link_urls', [])],
            "parent_id": email['metadata'].get('parent_id', ''),
            "text_as_html": email['metadata'].get('text_as_html', ''),
            "unique_id": f"{email['metadata'].get('filename', '')}_{email['metadata'].get('element_id', '')}"
        }
        
        # Handle multiple "Received" headers if present
        received_headers = email['metadata'].get('received', [])
        if isinstance(received_headers, str):
            received_headers = [received_headers]
        metadata['received'] = received_headers
        
        # Add all raw headers for potential future use
        metadata['headers'] = email['metadata']
        
        # Create a document for each chunk with the full metadata
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata['chunk_index'] = i
            chunk_metadata['total_chunks'] = len(chunks)
            documents.append(Document(
                page_content=chunk,
                metadata=chunk_metadata
            ))
    
    logging.info(f"Prepared {len(documents)} document chunks with metadata")
    
    # Log sample of prepared documents
    if documents:
        sample_doc = documents[0]
        logging.info("Sample of prepared document:")
        logging.info(f"Content preview: {sample_doc.page_content[:200]}...")
        logging.info(f"Metadata: {sample_doc.metadata}")
    
    return documents

def embed_and_store_emails(documents: List[Document], connection_string: str, model: str):
    total_documents = len(documents)
    logging.info(f"Embedding and storing {total_documents} document chunks")

    logging.info(f"Initializing embedding function with model: {model}")
    embedding_function = OpenAIEmbeddings(model=model, api_key=os.getenv('WF_INTERNAL_OPENAI'))

    logging.info("Starting embedding and storage process...")
    batch_size = 100  # Adjust this based on your needs and API limits

    # Create SQLAlchemy engine
    engine = create_engine(connection_string)

    # Create PGVector instance
    vector_store = PGVector(
        embedding_function=embedding_function,
        collection_name="email_embeddings",
        connection=engine,
        use_jsonb=True
    )

    for i in tqdm(range(0, total_documents, batch_size), desc="Embedding and storing"):
        batch = documents[i:i+batch_size]
        try:
            # Log sample data from the first document in each batch
            if batch:
                sample_doc = batch[0]
                logging.info(f"Sample document from batch {i//batch_size + 1}:")
                logging.info(f"Content preview: {sample_doc.page_content[:200]}...")
                logging.info(f"Metadata: {sample_doc.metadata}")

            vector_store.add_documents(batch)
            logging.info(f"Successfully embedded and stored batch {i//batch_size + 1}/{(total_documents-1)//batch_size + 1}")
        except Exception as e:
            logging.error(f"Error in batch {i//batch_size + 1}: {str(e)}")
            raise e

    logging.info("Embedding and storage process completed.")
    logging.info(f"Total documents processed: {total_documents}")

    # Add a verification step
    logging.info("Verifying stored embeddings...")
    try:
        results = vector_store.similarity_search("Test query", k=1)
        if results:
            logging.info("Successfully retrieved a document from the vector store.")
            logging.info(f"Retrieved document content preview: {results[0].page_content[:200]}...")
            logging.info(f"Retrieved document metadata: {results[0].metadata}")
        else:
            logging.warning("No documents retrieved from the vector store.")
    except Exception as e:
        logging.error(f"Error verifying stored embeddings: {str(e)}")

def get_connection_string():
    username = os.getenv('DB_USERNAME')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('RDS_ENDPOINT')
    port = os.getenv('RDS_PORT')
    database = os.getenv('DB_NAME')
    connection_string = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
    print(f"Debug - Connection string: {connection_string}")  # Add this line for debugging
    return connection_string

def get_db_engine(connection_string: str, pool_size: int = 5, max_overflow: int = 10):
    return create_engine(
        connection_string,
        poolclass=QueuePool,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
        pool_recycle=3600
    )

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process emails, save documents, and optionally embed them.")
    parser.add_argument("--parse", action="store_true", help="Parse emails and save documents")
    parser.add_argument("--embed", action="store_true", help="Embed and store documents in PGVector")
    parser.add_argument("--output", default="email_documents.pkl", help="Output file for saved documents")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Size of each text chunk")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Overlap between chunks")
    parser.add_argument("--connection-string", default=get_connection_string(), help="PostgreSQL connection string")
    parser.add_argument("--model", choices=['text-embedding-3-small', 'text-embedding-3-large', 'ada_v2'], 
                        default='text-embedding-3-small', help="Embedding model to use")
    args = parser.parse_args()

    if args.parse:
        email_directories = [
            "EmailsSent",
            "EmailsInbox",
            # Add more directories as needed
        ]
        print(f"Starting email parsing process for directories: {', '.join(email_directories)}")
        parsed_emails = process_email_directories(email_directories)
        if parsed_emails:
            documents = prepare_documents(parsed_emails, args.chunk_size, args.chunk_overlap)
            save_documents(documents, args.output)
            print(f"Documents saved to {args.output}")
        else:
            print("No emails were successfully parsed.")
    
    if args.embed:
        if os.path.exists(args.output):
            documents = load_documents(args.output)
            df, total_embeddings = calculate_embedding_tokens(documents, args.chunk_size, args.chunk_overlap)
            df.to_excel("embedding_tokens.xlsx", index=False)
            print(f"\nToken counts and cost estimates exported to embedding_tokens.xlsx")
            print(f"Total number of embeddings to be created: {total_embeddings}")
            
            model_cost_column = f"cost_{args.model.replace('-', '_')}"
            estimated_cost = df.loc['Total', model_cost_column]
            print(f"\nEstimated cost for selected model ({args.model}): ${estimated_cost:.2f}")
            user_input = input("Do you want to proceed with embedding? (y/n): ")
            if user_input.lower() == 'y':
                embed_and_store_emails(documents, args.connection_string, args.model)
            else:
                print("Embedding process cancelled.")
        else:
            print(f"Error: Document file {args.output} not found. Please run with --parse first.")

