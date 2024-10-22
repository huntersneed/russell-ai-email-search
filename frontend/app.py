import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from sentence_transformers import CrossEncoder
from datetime import datetime, timedelta

# Set page config to wide mode for better layout
st.set_page_config(layout="wide", page_title="Email RAG Chatbot")

# Load environment variables
load_dotenv()

# Define available models for each provider
openai_models = ["gpt-4o", "gpt-4o-mini", "o1-preview", "o1-mini"]
anthropic_models = ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229"]

# Sidebar for model selection
st.sidebar.title("Model Selection")
model_provider = st.sidebar.selectbox("Choose a model provider", ["OpenAI", "Anthropic"])

if model_provider == "OpenAI":
    model_name = st.sidebar.selectbox("Choose an OpenAI model", openai_models)
    llm = ChatOpenAI(model_name=model_name, temperature=0)
else:
    model_name = st.sidebar.selectbox("Choose an Anthropic model", anthropic_models)
    llm = ChatAnthropic(model=model_name, temperature=0)

# Sidebar for additional information
st.sidebar.title("About")
st.sidebar.info("This chatbot uses RAG to answer questions about your emails.")
st.sidebar.info(f"Connected to database: {os.getenv('DB_NAME')}")
st.sidebar.info(f"Using model: {model_name} from {model_provider}")

# Metadata Filters
st.sidebar.header("Metadata Filters")
filter_by_sender = st.sidebar.text_input("Filter by Sender")
filter_by_subject = st.sidebar.text_input("Filter by Subject")
filter_date_range = st.sidebar.date_input(
    "Filter by Date Range", value=(datetime.now() - timedelta(days=30), datetime.now())
)

# Main layout
st.title("Workflowsy AI Email Chatbot")

# Initialize chat history and first visit flag
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.first_visit = True





# Display tips and examples if there are no messages
def display_tips_and_examples():
    st.markdown("Here are some examples of questions you can ask:")
    
    # Create three columns for a more compact layout
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
    
    # Use bullet points for tips
    st.markdown("""
    - Be specific in your questions
    - Mention dates or time frames when relevant
    - You can ask follow-up questions about previous responses
    - Use the sidebar filters to narrow down your search
    """)
    
    # Add a button to try an example question
    if st.button("Try an example question"):
        example_question = "What was the last email I received from John regarding the project update?"
        st.session_state.messages.append({"role": "user", "content": example_question})
        st.experimental_rerun()

# Create a container for the chat history
chat_container = st.container()

# Create a container for the user input
input_container = st.container()

# Display tips and examples on first visit
if st.session_state.first_visit:
    display_tips_and_examples()
    st.session_state.first_visit = False

# Display chat messages from history on app rerun
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Define a custom prompt template
custom_template = """
You are an AI assistant with access to a database of emails.
Answer the user's question based on the provided email excerpts.

Instructions:
1. Use information from the emails to provide a helpful, accurate answer.
2. If the information is not available in the emails, say
   "I couldn't find information related to your question in your emails."
3. Do not make up information or use outside knowledge.
4. Reference the email subject or sender when relevant.

Emails:
{context}

Question: {question}

Answer:
"""

# Set up database connection
def get_connection_string():
    username = os.getenv('DB_USERNAME')
    password = os.getenv('DB_PASSWORD')
    host = os.getenv('RDS_ENDPOINT')
    port = os.getenv('RDS_PORT')
    database = os.getenv('DB_NAME')
    return f"postgresql://{username}:{password}@{host}:{port}/{database}"

CONNECTION_STRING = get_connection_string()

# Set up embeddings and vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource
def get_vectorstore():
    return PGVector(
        connection_string=CONNECTION_STRING,
        embedding_function=embeddings,
        collection_name="email_embeddings"
    )

vectorstore = get_vectorstore()

# Create a PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["question", "context"],
    template=custom_template
)

# Cross-encoder for reranking
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(documents, query):
    if not documents:
        return documents
    pairs = [(query, doc.page_content) for doc in documents]
    scores = cross_encoder.predict(pairs)
    reranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in reranked]

# Improved retriever setup
def get_retriever():
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

retriever = get_retriever()

# Set up retrieval chain with the custom prompt
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True,
    verbose=True  # Enable verbose logging
)

# User input
with input_container:
    if prompt := st.chat_input("What would you like to know about your emails?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            # Get response from QA chain
            response = qa_chain({"question": prompt})
            answer = response['answer']
            source_documents = response.get('source_documents', [])

            # Rerank the source documents
            reranked_documents = rerank_documents(source_documents, prompt)

            # Display user message and assistant response in chat container
            with chat_container:
                st.chat_message("user").markdown(prompt)
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    
                    if reranked_documents:
                        with st.expander("View Sources", expanded=False):
                            st.write("Sources (ranked by relevance):")
                            for i, doc in enumerate(reranked_documents[:3]):  # Show top 3 sources
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
                                st.write("---")  # Add a separator between documents
                    else:
                        with st.expander("View Sources", expanded=False):
                            st.write("No source documents were returned.")

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Add a "New Chat" button in the sidebar
if st.sidebar.button("New Chat"):
    st.session_state.messages = []
    st.session_state.first_visit = True
    st.rerun()

# Add a "Tips" section in the sidebar
st.sidebar.header("Tips")
st.sidebar.info("""
- Be specific in your questions for better results.
- Use date ranges to narrow down your search.
- You can ask follow-up questions about previous responses.
- Use the metadata filters to focus on specific senders or subjects.
""")
