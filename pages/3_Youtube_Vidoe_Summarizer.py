import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_community.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage
import re

# Load Mistral model using Ollama
llm = ChatOllama(model="mistral")

# Function to extract YouTube video ID from URL
def extract_video_id(url):
    pattern = r"(?:v=|\/)([0-9A-Za-z_-]{11}).*"
    match = re.search(pattern, url)
    return match.group(1) if match else None

# Function to fetch transcript from YouTube
def get_transcript(video_id):
    try:
        transcript_data = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry["text"] for entry in transcript_data])
        return transcript_text
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")
        return None

# Function to generate summary using Mistral
def summarize_transcript(transcript):
    messages = [
        SystemMessage(content="You are an AI assistant that summarizes YouTube video transcripts concisely."),
        HumanMessage(content=f"Summarize this transcript:\n{transcript}")
    ]
    return llm.invoke(messages).content

# Streamlit UI
def main():
    st.title("🎥 YouTube Video Summarizer")
    url = st.text_input("Enter YouTube Video URL:")
    
    if st.button("Summarize"):
        video_id = extract_video_id(url)
        if not video_id:
            st.error("Invalid YouTube URL! Please enter a valid URL.")
            return

        st.info("Fetching transcript...")
        transcript = get_transcript(video_id)

        if transcript:
            st.success("Transcript fetched successfully! Generating summary...")
            summary = summarize_transcript(transcript)
            st.subheader("📄 Summary:")
            st.write(summary)

if __name__ == "__main__":
    main()
