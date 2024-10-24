import streamlit as st
import os
from dotenv import load_dotenv
from langchain_postgres import PGVector, PostgresChatMessageHistory
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
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

set_verbose(True)

# Load environment variables
load_dotenv()

# Streamlit UI setup
st.set_page_config(layout="wide", page_title="Getting Automated AI Email RAG Chatbot")

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
Answer the user's question based on the provided email excerpts.

Provide as much information as possible in your answer while still being concise.

Previous conversation:
{chat_history}

Instructions:
1. Use information from the emails to provide a helpful, accurate answer.
2. If the information is not available in the emails, say
   "I couldn't find information related to your question in your emails."
3. Do not make up information or use outside knowledge.
4. Reference the email subject or sender when relevant.
5. Use the chat history to maintain context of the conversation.

Emails:
{context}

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

# Retriever setup
def get_retriever(filter_by_sender, filter_by_subject, filter_date_range):
    search_kwargs = {"k": 5}
    metadata_filter = {}

    if filter_by_sender:
        metadata_filter["from"] = {"$ilike": f"%{filter_by_sender}%"}
    if filter_by_subject:
        metadata_filter["subject"] = {"$ilike": f"%{filter_by_subject}%"}
    if filter_date_range:
        start_date, end_date = filter_date_range
        metadata_filter["date"] = {
            "$gte": start_date.isoformat(),
            "$lte": end_date.isoformat()
        }

    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs=search_kwargs,
        metadata_field_fn=lambda metadata: {
            "from": metadata.get("from", "Unknown sender"),
            "subject": metadata.get("subject", "No subject"),
            "date": metadata.get("date", "Unknown date"),
            "to": metadata.get("to", "Unknown recipient"),
            "cc": metadata.get("cc", ""),
            "bcc": metadata.get("bcc", ""),
        }
    )
    return retriever

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

    # Create message history store
    message_history = PostgresChatMessageHistory(
        table_name,
        session_id,
        sync_connection=sync_connection
    )
    
    # Create and return the runnable chain with message history
    def get_history(session_id: str) -> BaseChatMessageHistory:
        return message_history
    
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
    "Filter by Date Range", value=(datetime.now() - timedelta(days=30), datetime.now())
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

# Chat interface
chat_container = st.container()
input_container = st.container()

# Initialize memory and QA chain
memory_config = get_memory()
retriever = get_retriever(filter_by_sender, filter_by_subject, filter_date_range)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    verbose=True,
    rephrase_question=True
)

# Wrap the chain with message history
qa_chain_with_history = RunnableWithMessageHistory(
    qa_chain,
    memory_config["history_factory"],
    input_messages_key=memory_config["input_messages_key"],
    history_messages_key=memory_config["history_messages_key"]
)

# Display chat history
with chat_container:
    for message in memory_config["history_factory"](memory_config["session_id"]).messages:
        with st.chat_message(message.type):
            st.markdown(message.content)

# User input and response
with input_container:
    if prompt := st.chat_input("What would you like to know about your emails?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            response = qa_chain_with_history.invoke(
                {"question": prompt},
                {"configurable": {"session_id": memory_config["session_id"]}}
            )
            answer = response['answer']
            source_documents = response.get('source_documents', [])
            reranked_documents = rerank_documents(source_documents, prompt)

            with chat_container:
                st.chat_message("user").markdown(prompt)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    if reranked_documents:
                        with st.expander("View Sources", expanded=False):
                            st.write("Sources (ranked by relevance):")
                            for i, doc in enumerate(reranked_documents[:3]):
                                st.write(f"**Document {i+1}: {doc.metadata.get('subject', 'No subject')}**")
                                st.write(f"From: {doc.metadata.get('from', 'Unknown sender')}")
                                st.write(f"To: {doc.metadata.get('to', 'Unknown recipient')}")
                                st.write(f"Date: {doc.metadata.get('date', 'Unknown date')}")
                                if doc.metadata.get('cc'):
                                    st.write(f"CC: {doc.metadata['cc']}")
                                if doc.metadata.get('bcc'):
                                    st.write(f"BCC: {doc.metadata['bcc']}")
                                st.write("Content:")
                                st.text_area(f"Content of Document {i+1}", doc.page_content, height=150)
                                st.write("---")
                    else:
                        with st.expander("View Sources", expanded=False):
                            st.write("No source documents were returned.")

            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add a debug expander to verify context persistence
with st.expander("Debug: Conversation History", expanded=False):
    st.write("Current Memory Contents:")
    message_history = memory_config["history_factory"](memory_config["session_id"])
    for message in message_history.messages:
        st.write(f"{message.type}: {message.content}")

