import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()


# Define a custom prompt template
custom_template = """
You are an AI assistant with access to a database of emails. Your primary function is to answer questions about these emails and provide relevant information based on their content. Here are some key points to remember:

1. You have access to a vector database containing email content and metadata.
2. Your responses should be based on the information found in these emails.
3. If asked about specific emails, conversations, or topics, search the database and provide accurate information.
4. When mentioning dates, people, or events, ensure they are directly referenced in the email database.
5. If you're unsure or can't find relevant information in the emails, say so honestly.
6. Provide sources (email references) for your information when possible.
7. Respect privacy and confidentiality. Do not share sensitive information or personal details unless explicitly asked.
8. Your knowledge is limited to the content of the emails in the database. Do not make assumptions or provide information beyond what's in the emails.
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
embeddings = OpenAIEmbeddings()
vectorstore = PGVector(
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    collection_name="email_embeddings"
)

# Set up language model (OpenAI or Claude)
model_choice = st.sidebar.selectbox("Choose a model", ["OpenAI", "Claude"])

if model_choice == "OpenAI":
    llm = OpenAI(temperature=0)
else:
    llm = ChatAnthropic(temperature=0)

# Create a PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["chat_history", "question", "context"],
    template=custom_template + """

Chat History: {chat_history}
Human: {question}
AI: Let me search the email database for relevant information.
{context}
Based on this information, I can respond:
"""
)

# Set up retrieval chain with the custom prompt
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
qa_chain = ConversationalRetrievalChain.from_llm(
    llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

# Streamlit UI
st.title("Email RAG Chatbot")

# Informative prompt
st.markdown("""
This chatbot has access to a vector database containing your emails. It uses Retrieval-Augmented Generation (RAG) to provide informed responses based on the content of your emails.

You can ask questions about:
- Specific emails or conversations
- General topics discussed in your emails
- Dates, people, or events mentioned in your emails

The chatbot will search through the email database and provide relevant information along with the sources (email files) it used to generate the answer.

Example questions:
- "What was the last email I received about project X?"
- "Summarize my recent conversations with John Doe."
- "When is my next scheduled meeting according to my emails?"

Feel free to ask any question about your emails!
""")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about your emails?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get response from QA chain
    response = qa_chain({"question": prompt})
    answer = response['answer']
    source_documents = response.get('source_documents', [])

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(answer)
        st.write("Debug: Response keys:", list(response.keys()))  # Debug line
        st.write("Debug: Number of source documents:", len(source_documents))  # Debug line
        if source_documents:
            st.write("Sources:")
            for i, doc in enumerate(source_documents):
                st.write(f"- Document {i+1}:")
                st.write(f"  Source: {doc.metadata.get('source', 'Unknown source')}")
                st.write(f"  Content: {doc.page_content[:100]}...")  # Display first 100 characters of content
        else:
            st.write("No source documents were returned.")

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})

# Sidebar for additional information
st.sidebar.title("About")
st.sidebar.info("This chatbot uses RAG to answer questions about your emails.")
st.sidebar.info(f"Connected to database: {os.getenv('DB_NAME')}")
st.sidebar.info(f"Using model: {model_choice}")
