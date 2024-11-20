import streamlit as st
import google.generativeai as genai
import PyPDF2
import io
from typing import Dict, List, Optional
import os
from dataclasses import dataclass
import json

# Data structures
@dataclass
class Topic:
    title: str
    content: str
    subtopics: List['Topic']
    completed: bool = False

class TutorialState:
    def __init__(self):
        self.topics: List[Topic] = []
        self.current_topic_index: int = 0
        self.questions_asked: int = 0
        self.max_questions_per_topic: int = 3

    def reset(self):
        self.current_topic_index = 0
        for topic in self.topics:
            topic.completed = False
        self.questions_asked = 0

# Initialize Gemini
def init_gemini(api_key: str = None):
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    return None

# PDF Processing
def process_pdf(pdf_file) -> str:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def generate_tutorial_structure(content: str, model) -> List[Topic]:
    prompt = f"""
    Analyze the following educational content and create a structured tutorial outline.
    Format the response as JSON with the following structure:
    {{
        "topics": [
            {{
                "title": "Topic Title",
                "content": "Main content to teach",
                "subtopics": [
                    {{
                        "title": "Subtopic Title",
                        "content": "Subtopic content",
                        "subtopics": []
                    }}
                ]
            }}
        ]
    }}
    Content: {content[:10000]}  # Limit content length for API
    """
    
    response = model.generate_content(prompt)
    structure = json.loads(response.text)
    
    def create_topics(topic_data) -> Topic:
        return Topic(
            title=topic_data["title"],
            content=topic_data["content"],
            subtopics=[create_topics(st) for st in topic_data.get("subtopics", [])]
        )
    
    return [create_topics(t) for t in structure["topics"]]

def teach_topic(topic: Topic, model) -> str:
    prompt = f"""
    Act as a tutor teaching the following topic: {topic.title}
    
    Content to teach: {topic.content}
    
    Provide a clear explanation with examples. End with a relevant question to test understanding.
    """
    response = model.generate_content(prompt)
    return response.text

def evaluate_answer(answer: str, topic: Topic, model) -> tuple[bool, str]:
    prompt = f"""
    Topic: {topic.title}
    Content: {topic.content}
    Student's answer: {answer}
    
    Evaluate if the student understood the concept. Respond with either:
    UNDERSTOOD: [explanation] or
    NOT_UNDERSTOOD: [explanation with additional help]
    """
    response = model.generate_content(prompt)
    understood = response.text.startswith("UNDERSTOOD")
    explanation = response.text.split(": ", 1)[1]
    return understood, explanation

# Streamlit UI
def main():
    st.title("AI Educational Tutor")
    
    # Initialize session state
    if 'tutorial_state' not in st.session_state:
        st.session_state.tutorial_state = TutorialState()
    
    # API Key Management
    api_key = None
    
    # Try to get API key from secrets
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
        # If not in secrets, show input field
        st.warning("⚠️ Gemini API Key not found in secrets.")
        api_key_input = st.text_input(
            "Please enter your Gemini API Key:",
            type="password",
            help="Get your API key from Google AI Studio (https://makersuite.google.com/app/apikey)"
        )
        if api_key_input:
            api_key = api_key_input
            
    if not api_key:
        st.error("""
        No API key provided. To run this application, you need to either:
        1. Set up `.streamlit/secrets.toml` with your GEMINI_API_KEY, or
        2. Enter your API key in the field above
        
        To get an API key:
        1. Go to https://makersuite.google.com/app/apikey
        2. Create or select a project
        3. Generate an API key
        """)
        st.stop()
    
    # Initialize model with API key
    if 'model' not in st.session_state:
        st.session_state.model = init_gemini(api_key)
    
    # File upload
    pdf_file = st.file_uploader("Upload Educational PDF", type="pdf", accept_multiple_files=False)
    
    # Reset button
    if st.button("Reset Tutorial"):
        st.session_state.tutorial_state.reset()
        st.experimental_rerun()
    
    if pdf_file:
        # Check file size
        if pdf_file.size > 15 * 1024 * 1024:  # 15MB limit
            st.error("File size exceeds 15MB limit.")
            return
            
        # Process PDF if not already processed
        if not st.session_state.tutorial_state.topics:
            with st.spinner("Processing PDF content..."):
                try:
                    content = process_pdf(pdf_file)
                    st.session_state.tutorial_state.topics = generate_tutorial_structure(
                        content, st.session_state.model
                    )
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    return
        
        # Display current topic
        state = st.session_state.tutorial_state
        if state.current_topic_index < len(state.topics):
            current_topic = state.topics[state.current_topic_index]
            
            st.subheader(f"Topic: {current_topic.title}")
            
            if not current_topic.completed:
                try:
                    # Teaching phase
                    teaching_content = teach_topic(current_topic, st.session_state.model)
                    st.markdown(teaching_content)
                    
                    # Get student's answer
                    student_answer = st.text_area("Your Answer:")
                    if st.button("Submit"):
                        understood, explanation = evaluate_answer(
                            student_answer, current_topic, st.session_state.model
                        )
                        
                        if understood:
                            st.success(explanation)
                            current_topic.completed = True
                            state.current_topic_index += 1
                            state.questions_asked = 0
                            st.experimental_rerun()
                        else:
                            st.warning(explanation)
                            state.questions_asked += 1
                            if state.questions_asked >= state.max_questions_per_topic:
                                st.info("Let's move on to the next topic.")
                                current_topic.completed = True
                                state.current_topic_index += 1
                                state.questions_asked = 0
                                st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error during tutorial: {str(e)}")
        else:
            st.success("Congratulations! You've completed all topics!")

if __name__ == "__main__":
    main()
