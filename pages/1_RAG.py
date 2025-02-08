import streamlit as st
import tempfile
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# Initialize vector database
vector_db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
# Load Mixtral model
llm = ChatOllama(model="mistral")

print(st.session_state.messages)

# Function to display chat history
def display_messages():
    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])

# Function to process uploaded files
def process_file():
    st.session_state.messages = []  # Clear previous messages
    for file in st.session_state["file_uploader"]:
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name
        
        # Process and store embeddings
        with st.session_state["feeder_spinner"], st.spinner("Uploading the file..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            with open(file_path, "rb") as f:  # Read in binary mode
                raw_content = f.read()

            # Try decoding with UTF-8, fallback to ISO-8859-1
            try:
                text_content = raw_content.decode("utf-8")
            except UnicodeDecodeError:
                text_content = raw_content.decode("ISO-8859-1")  # Fallback encoding

            chunks = text_splitter.create_documents([text_content])
            vector_db.add_documents(chunks)
        os.remove(file_path)

# Retrieve documents from vector store
def retrieve_docs(query):
    docs = vector_db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

# Function to process user input
def process_input():
    if prompt := st.chat_input("Ask me anything..."):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        retrieved_docs = retrieve_docs(prompt)
        
        # Generate response using Mixtral
        messages = [
            SystemMessage(content="Use the retrieved context to answer the user's query."),
            HumanMessage(content=f"Context: {retrieved_docs}\n\nQuery: {prompt}")
        ]
        response = llm.invoke(messages).content
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Main function to run the app
def main():
    st.title("🔍 Mixtral RAG Chatbot")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        key="file_uploader",
        on_change=process_file,
        accept_multiple_files=True
    )
    
    st.session_state["feeder_spinner"] = st.empty()
    display_messages()
    process_input()

if __name__ == "__main__":
    main()
