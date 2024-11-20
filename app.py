import streamlit as st
import google.generativeai as genai
import PyPDF2
import io
from typing import Dict, List, Optional
import os
from dataclasses import dataclass
import json

# Data structures remain the same
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

def init_gemini(api_key: str = None):
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    return None

def process_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        raise

def generate_tutorial_structure(content: str, model) -> List[Topic]:
    prompt = """
    Analyze this educational content and create a structured tutorial outline.
    Return ONLY valid JSON matching this exact structure, with no additional text:
    {
        "topics": [
            {
                "title": "Topic Title",
                "content": "Main content to teach (2-3 sentences)",
                "subtopics": [
                    {
                        "title": "Subtopic Title",
                        "content": "Subtopic content (2-3 sentences)",
                        "subtopics": []
                    }
                ]
            }
        ]
    }

    Content to analyze:
    """ + content[:5000]
    
    try:
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Display the raw response for debugging
        st.write("Processing tutorial structure...")
        st.write("Raw AI Response (for debugging):")
        st.code(response_text)
        
        # Parse JSON
        try:
            # First attempt: direct JSON parsing
            structure = json.loads(response_text)
        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error: {str(e)}")
            # Second attempt: try to find JSON object within the text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                try:
                    structure = json.loads(json_str)
                except json.JSONDecodeError as e2:
                    st.error(f"Second JSON parsing attempt failed: {str(e2)}")
                    raise
            else:
                raise ValueError("Could not find valid JSON in the response")
        
        # Validate structure
        if "topics" not in structure:
            st.error("Response JSON missing 'topics' key")
            st.write("Received structure:", structure)
            raise ValueError("Response JSON missing 'topics' key")
        
        if not structure["topics"]:
            st.error("No topics found in the response")
            raise ValueError("No topics found in the response")
        
        def create_topics(topic_data) -> Topic:
            return Topic(
                title=topic_data["title"],
                content=topic_data["content"],
                subtopics=[create_topics(st) for st in topic_data.get("subtopics", [])]
            )
        
        topics = [create_topics(t) for t in structure["topics"]]
        
        # Validate we actually got topics
        if not topics:
            st.error("No topics were created from the structure")
            raise ValueError("No topics were created from the structure")
        
        st.success(f"Successfully generated {len(topics)} topics!")
        
        # Display topic structure for verification
        st.write("Generated Topic Structure:")
        for i, topic in enumerate(topics, 1):
            st.write(f"{i}. {topic.title}")
            for j, subtopic in enumerate(topic.subtopics, 1):
                st.write(f"   {i}.{j} {subtopic.title}")
        
        return topics
        
    except Exception as e:
        st.error(f"Error generating tutorial structure: {str(e)}")
        st.error("Raw AI Response:")
        st.code(response_text)
        raise

# Rest of the functions (teach_topic, evaluate_answer) remain the same

def main():
    st.title("AI Educational Tutor")
    
    # Initialize session state
    if 'tutorial_state' not in st.session_state:
        st.session_state.tutorial_state = TutorialState()
    
    # API Key Management
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except KeyError:
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
        """)
        st.stop()
    
    # Initialize model
    if 'model' not in st.session_state:
        st.session_state.model = init_gemini(api_key)
    
    # File upload
    pdf_file = st.file_uploader("Upload Educational PDF", type="pdf", accept_multiple_files=False)
    
    # Reset button with confirmation
    if st.button("Reset Tutorial"):
        if st.session_state.tutorial_state.topics:
            if st.button("Confirm Reset"):
                st.session_state.tutorial_state.reset()
                st.experimental_rerun()
        else:
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
                    if not content.strip():
                        st.error("No text could be extracted from the PDF. Please ensure the PDF contains readable text.")
                        return
                    
                    st.info("PDF processed successfully. Generating tutorial structure...")
                    # Display some of the content for verification
                    st.write("First 500 characters of PDF content:")
                    st.code(content[:500])
                    
                    try:
                        st.session_state.tutorial_state.topics = generate_tutorial_structure(
                            content, st.session_state.model
                        )
                        if not st.session_state.tutorial_state.topics:
                            st.error("No topics were generated. Please try again.")
                            st.stop()
                        st.success(f"Tutorial structure generated with {len(st.session_state.tutorial_state.topics)} topics!")
                    except Exception as e:
                        st.error(f"Error in tutorial structure generation: {str(e)}")
                        st.stop()
                except Exception as e:
                    st.error(f"Error processing content: {str(e)}")
                    st.stop()

        # Rest of the main function remains the same...

if __name__ == "__main__":
    main()
