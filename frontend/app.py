import streamlit as st
import os
from dotenv import load_dotenv
from langchain_postgres import PGVector, PostgresChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from datetime import datetime, timedelta
from PIL import Image
import requests
from io import BytesIO
import uuid
import json
from langchain.globals import set_verbose
import psycopg
import traceback
import sys
from langchain.memory import ConversationBufferMemory

set_verbose(True)

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(layout="wide", page_title="Getting Automated AI Email RAG Chatbot")

# Add this near the top of your file, after st.set_page_config
st.markdown("""
    <style>
    .stMarkdown, .stText {  /* Targets both markdown and text elements */
        font-size: 1.2rem !important;
    }
    .stChatMessage {
        font-size: 1.2rem !important;
    }
    .stExpander {
        font-size: 1rem !important;
    }
    .stTextArea textarea {
        font-size: 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


def debug_print(title, value):
    st.write(f"DEBUG - {title}:")
    if isinstance(value, dict):
        st.json(value)
    else:
        st.write(value)

# Helper functions
def get_connection_string():
    username = os.getenv('DB_USERNAME')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('RDS_ENDPOINT')
    port = os.getenv('RDS_PORT')
    database = os.getenv('DB_NAME')
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"

def get_or_create_session_id():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    return st.session_state.session_id

def rerank_documents(documents, query):
    if not documents:
        return documents
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]

# Initialize constants and resources
CONNECTION_STRING = get_connection_string()
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Define available models
openai_models = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]
anthropic_models = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]

# Custom prompt template
custom_template = """
You are an AI assistant with access to a database of emails.
Answer the user's question based on the provided email excerpts and previous conversation context.

Previous conversation and email context:
{chat_history}

Previously retrieved email content:
{context}

Instructions:
1. ALWAYS check the previous conversation and retrieved email content first
2. Only perform a new email search if the information cannot be found in the previous context
3. If the question is a follow-up about previously discussed emails, use that information instead of searching again
4. Be concise but thorough in your responses

Current Question: {question}

Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question", "context", "chat_history"],
    template=custom_template
)

# Vector store setup
@st.cache_resource
def get_vectorstore():
    return PGVector(
        embeddings=embeddings,
        collection_name="email_embeddings",
        connection=CONNECTION_STRING,
        use_jsonb=True
    )

vectorstore = get_vectorstore()

# Add this after vectorstore initialization
def check_vectorstore():
    try:
        # Try a simple search without filters
        results = vectorstore.similarity_search(
            "test",
            k=1,
            filter=None  # Remove all filters for this test
        )
        st.write("Debug - Sample document metadata:", results[0].metadata if results else "No documents found")
        st.write("Debug - Total documents in store:", len(results))
    except Exception as e:
        st.error(f"Vectorstore check failed: {str(e)}")

# # Call this function before the chat interface
# check_vectorstore()

# Retriever setup
def get_retriever(filter_by_sender, filter_by_subject, filter_date_range):
    search_kwargs = {"k": 5}
    metadata_filter = {}

    if filter_date_range:
        start_date, end_date = filter_date_range
        metadata_filter["date"] = {
            "$between": [
                start_date.isoformat() + "+00:00",
                end_date.isoformat() + "+00:00"
            ]
        }
    
    if filter_by_sender:
        metadata_filter["from"] = {"$ilike": f"%{filter_by_sender}%"}
    
    if filter_by_subject:
        metadata_filter["subject"] = {"$ilike": f"%{filter_by_subject}%"}

    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": search_kwargs["k"],
            "filter": metadata_filter if metadata_filter else None
        }
    )

# Memory setup
def create_table_if_not_exists(connection, table_name):
    with connection.cursor() as cursor:
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                message JSONB,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        connection.commit()

def create_index_if_not_exists(connection, table_name, index_name):
    with connection.cursor() as cursor:
        cursor.execute(f"""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_class c
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.relname = '{index_name}' AND n.nspname = 'public'
                ) THEN
                    CREATE INDEX {index_name} ON {table_name} (session_id);
                END IF;
            END
            $$;
        """)
        connection.commit()

# Use these functions in your get_memory function
def get_memory():
    session_id = st.session_state.session_id
    conn_info = CONNECTION_STRING

    # Create the table and index if they do not exist
    table_name = "message_store"
    index_name = "idx_message_store_session_id"
    sync_connection = psycopg.connect(conn_info)
    create_table_if_not_exists(sync_connection, table_name)
    create_index_if_not_exists(sync_connection, table_name, index_name)

    # Create message history store with correct parameter order
    message_history = PostgresChatMessageHistory(
        table_name,  # First positional argument
        session_id,  # Second positional argument
        sync_connection=sync_connection  # Keyword argument
    )
    
    # Create and return the runnable chain with message history
    def get_history(session_id: str) -> BaseChatMessageHistory:
        return PostgresChatMessageHistory(
            table_name,  # First positional argument
            session_id,  # Second positional argument
            sync_connection=sync_connection  # Keyword argument
        )
    
    return {
        "history_factory": get_history,
        "session_id": session_id,
        "input_messages_key": "question",
        "history_messages_key": "chat_history"
    }

# UI helper functions
def display_tips_and_examples():
    st.markdown("Here are some examples of questions you can ask:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info('"What was the last email I received from [specific person]?"')
        st.info('"Summarize my recent conversations about [specific topic]."')
        st.info('"Do I have any upcoming deadlines mentioned in my emails?"')
    
    with col2:
        st.info('"Find emails related to [specific project or event]."')
        st.info('"Who are the most frequent senders in my inbox this month?"')
        st.info('"Are there any unread important emails from last week?"')
    
    st.markdown("### Tips for better results:")
    
    st.markdown("""
    - Be specific in your questions
    - Mention dates or time frames when relevant
    - You can ask follow-up questions about previous responses
    - Use the sidebar filters to narrow down your search
    """)
    
    if st.button("Try an example question"):
        example_question = "What was the last email I received from John regarding the project update?"
        st.session_state.messages.append({"role": "user", "content": example_question})
        st.experimental_rerun()

# Sidebar
logo_url = "https://gettingautomated.com/wp-content/uploads/2024/07/logo.png"
response = requests.get(logo_url)
logo = Image.open(BytesIO(response.content))
st.sidebar.image(logo, width=200)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'sessions' not in st.session_state:
    st.session_state.sessions = {}
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True

# Session Management (moved up)
def manage_sessions():
    current_session_id = st.session_state.session_id
    
    st.sidebar.header("Session Management")
    
    # Display current session
    st.sidebar.info(f"Current Session ID: {current_session_id[:8]}...")
    
    # Option to save current session
    session_name = st.sidebar.text_input("Enter a name for this session:")
    if st.sidebar.button("Save Current Session") and session_name:
        st.session_state.sessions[current_session_id] = {
            "name": session_name,
            "messages": st.session_state.messages
        }
        st.sidebar.success(f"Session '{session_name}' saved!")
    
    # Option to load a previous session
    if st.session_state.sessions:
        session_to_load = st.sidebar.selectbox(
            "Load a previous session:",
            options=[f"{s['name']} ({k[:8]}...)" for k, s in st.session_state.sessions.items()],
            index=None,
            placeholder="Select a session to load..."
        )
        if session_to_load:
            selected_session_id = [k for k, s in st.session_state.sessions.items() if session_to_load.startswith(s['name'])][0]
            if st.sidebar.button("Load Selected Session"):
                st.session_state.session_id = selected_session_id
                st.session_state.messages = st.session_state.sessions[selected_session_id]["messages"]
                st.experimental_rerun()
    
    # Option to start a new session
    if st.sidebar.button("Start New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.experimental_rerun()

    # Option to export sessions
    if st.sidebar.button("Export Sessions"):
        sessions_data = json.dumps(st.session_state.sessions)
        st.sidebar.download_button(
            label="Download Sessions",
            data=sessions_data,
            file_name="chat_sessions.json",
            mime="application/json"
        )

    # Option to import sessions
    uploaded_file = st.sidebar.file_uploader("Import Sessions", type="json")
    if uploaded_file is not None:
        imported_sessions = json.load(uploaded_file)
        st.session_state.sessions.update(imported_sessions)
        st.sidebar.success("Sessions imported successfully!")

manage_sessions()

# Model Selection (moved down)
st.sidebar.title("Model Selection")
model_provider = st.sidebar.selectbox("Choose a model provider", ["OpenAI", "Anthropic"])

if model_provider == "OpenAI":
    model_name = st.sidebar.selectbox("Choose an OpenAI model", openai_models)
    llm = ChatOpenAI(model_name=model_name, temperature=0)
else:
    model_name = st.sidebar.selectbox("Choose an Anthropic model", anthropic_models)
    llm = ChatAnthropic(model=model_name, temperature=0)

st.sidebar.title("About")
st.sidebar.info("This chatbot uses RAG to answer questions about your emails.")
st.sidebar.info(f"Connected to database: {os.getenv('DB_NAME')}")
st.sidebar.info(f"Using model: {model_name} from {model_provider}")

st.sidebar.header("Metadata Filters")
filter_by_sender = st.sidebar.text_input("Filter by Sender")
filter_by_subject = st.sidebar.text_input("Filter by Subject")
filter_date_range = st.sidebar.date_input(
    "Filter by Date Range",
    value=(datetime.now() - timedelta(days=365), datetime.now()),
    key="date_range"
)

st.sidebar.header("Tips")
st.sidebar.info("""
- Be specific in your questions for better results.
- Use date ranges to narrow down your search.
- You can ask follow-up questions about previous responses.
- Use the metadata filters to focus on specific senders or subjects.
""")

# Main layout
st.title("Getting Automated AI Email Chatbot")

# Initialize chat history and first visit flag
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.first_visit = True

# Display tips and examples on first visit
if st.session_state.first_visit:
    display_tips_and_examples()
    st.session_state.first_visit = False

# Add this after your helper functions and before the chat interface
def format_chat_history(messages):
    formatted_history = []
    for i in range(0, len(messages) - 1):
        msg = messages[i]
        if msg["role"] == "user":
            if i + 1 < len(messages) and messages[i + 1]["role"] == "assistant":
                formatted_history.append(
                    f"Human: {msg['content']}\nAssistant: {messages[i + 1]['content']}"
                )
    return "\n\n".join(formatted_history)

# Also initialize previous_contexts in session state
if "previous_contexts" not in st.session_state:
    st.session_state.previous_contexts = []

# Chat interface
chat_container = st.container()
input_container = st.container()

# Initialize memory and QA chain
memory_config = get_memory()
retriever = get_retriever(filter_by_sender, filter_by_subject, filter_date_range)

# Create a single chain instance that persists
@st.cache_resource
def get_qa_chain():
    # Create memory with PostgresChatMessageHistory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        chat_memory=memory_config["history_factory"](memory_config["session_id"]),
        return_messages=True,
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,  # Use the wrapped memory
        combine_docs_chain_kwargs={
            "prompt": prompt_template,
            "document_prompt": PromptTemplate(
                input_variables=["page_content"],
                template="{page_content}"
            ),
            "document_separator": "\n\n"
        },
        return_source_documents=True,
        verbose=True
    )

# Get the persistent chain
qa_chain = get_qa_chain()

# Display chat history
with chat_container:
    # Display all messages from session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# User input and response
with input_container:
    if prompt := st.chat_input("What would you like to know about your emails?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            response = qa_chain({
                "question": prompt,
                "chat_history": st.session_state.messages[:-1]  # Exclude current message
            })
            
            # Add error checking for response structure
            if isinstance(response, dict):
                answer = response.get('answer', "I apologize, but I couldn't generate a response at this time.")
                source_documents = response.get('source_documents', [])
                reranked_documents = rerank_documents(source_documents, prompt)
                
                with chat_container:
                    st.chat_message("user").markdown(prompt)
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                        
                        if reranked_documents:
                            with st.expander("View Sources", expanded=False):
                                for i, doc in enumerate(reranked_documents[:3]):
                                    st.write(f"**Source {i+1}:**")
                                    st.write(f"From: {doc.metadata.get('from', 'Unknown')}")
                                    st.write(f"Subject: {doc.metadata.get('subject', 'No subject')}")
                                    st.write(f"Date: {doc.metadata.get('date', 'No date')}")
                                    st.write("Content:")
                                    st.text(doc.page_content)
                                    st.write("---")
                
                st.session_state.messages.append({"role": "assistant", "content": answer})
            else:
                st.error("Unexpected response format from the chain")
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add a debug expander to verify context persistence
with st.expander("Debug: Conversation History", expanded=False):
    st.write("Current Memory Contents:")
    message_history = memory_config["history_factory"](memory_config["session_id"])
    
    st.write(f"Session ID: {memory_config['session_id']}")
    st.write("Messages in PostgreSQL store:")
    for message in message_history.messages:
        st.write(f"{message.type}: {message.content}")
    
    st.write("\nMessages in Session State:")
    for message in st.session_state.messages:
        st.write(f"{message['role']}: {message['content']}")

