import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage

# Initialize Ollama with Mistral
llm = ChatOllama(model="mistral")

# Function to extract text from a webpage
def extract_text_from_url(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract readable text from paragraphs
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        text_content = "\n".join(paragraphs)

        if len(text_content) < 100:
            return "Error: Not enough text found on the page."

        return text_content

    except requests.exceptions.RequestException as e:
        return f"Error: {e}"

# Function to summarize text using Mistral
def summarize_text(text):
    system_prompt = "Summarize the following web page content into a short and clear summary."
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=text[:4000])  # Limit input to 4000 characters
    ]

    try:
        response = llm.invoke(messages)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("Web Page Summarizer")

url = st.text_input("Enter a URL to summarize:")
if url:
    with st.spinner("Extracting content..."):
        text_content = extract_text_from_url(url)
    
    if text_content.startswith("Error"):
        st.error(text_content)
    else:
        #st.write("Extracted text (first 1000 characters):", text_content[:1000] + "...")
        
        with st.spinner("Summarizing..."):
            summary = summarize_text(text_content)
        
        if summary.startswith("Error"):
            st.error(summary)
        else:
            st.subheader("Summary:")
            st.write(summary)
