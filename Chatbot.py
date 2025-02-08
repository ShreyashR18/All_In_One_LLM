import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# Set up Streamlit page
st.set_page_config(page_title="All in One LLM", layout="wide")
st.title("🗨️Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages with alignment
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage) or isinstance(msg, AIMessage):
        with st.chat_message("assistant"):  # AI messages on the left
            st.write(msg.content)
    else:
        with st.chat_message("user"):  # User messages on the right
            st.write(msg.content)

# Load Mistral model
chat_model = ChatOllama(model="mistral")

# User input
user_input = st.chat_input("Type your message...")

if user_input:
    # Add user message to session
    user_msg = HumanMessage(content=user_input)
    st.session_state.messages.append(user_msg)
    
    # Display user message on the right
    with st.chat_message("user"):
        st.write(user_input)
    
    # Display AI response container on the left
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Stream response from Mistral
        for chunk in chat_model.stream(st.session_state.messages):
            full_response += chunk.content
            response_placeholder.write(full_response)  # Update response in real-time

        # Add AI response to session state
        st.session_state.messages.append(AIMessage(content=full_response))
