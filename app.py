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
    # Improved prompt with explicit JSON formatting instructions
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
    """ + content[:5000]  # Reduced content length for better processing
    
    try:
        # Generate response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Debug logging
        st.debug(f"Raw AI Response:\n{response_text}")
        
        # Try to find JSON in the response
        try:
            # First attempt: direct JSON parsing
            structure = json.loads(response_text)
        except json.JSONDecodeError:
            # Second attempt: try to find JSON object within the text
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                structure = json.loads(json_str)
            else:
                raise ValueError("Could not find valid JSON in the response")
        
        # Validate structure
        if "topics" not in structure:
            raise ValueError("Response JSON missing 'topics' key")
        
        def create_topics(topic_data) -> Topic:
            return Topic(
                title=topic_data["title"],
                content=topic_data["content"],
                subtopics=[create_topics(st) for st in topic_data.get("subtopics", [])]
            )
        
        return [create_topics(t) for t in structure["topics"]]
        
    except Exception as e:
        st.error(f"Error generating tutorial structure: {str(e)}")
        st.error("Raw AI Response:")
        st.code(response.text)
        raise

def teach_topic(topic: Topic, model) -> str:
    prompt = f"""
    Act as a tutor teaching this topic. Provide a clear explanation with examples,
    and end with a relevant question to test understanding.
    
    TOPIC: {topic.title}
    CONTENT TO TEACH: {topic.content}
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error in teaching topic: {str(e)}")
        raise

def evaluate_answer(answer: str, topic: Topic, model) -> tuple[bool, str]:
    prompt = f"""
    Topic: {topic.title}
    Content: {topic.content}
    Student's answer: {answer}
    
    Evaluate if the student understood the concept.
    Reply with EXACTLY one of these two formats:
    UNDERSTOOD: [brief explanation]
    or
    NOT_UNDERSTOOD: [explanation with additional help]
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith("UNDERSTOOD:"):
            return True, response_text[11:].strip()
        elif response_text.startswith("NOT_UNDERSTOOD:"):
            return False, response_text[14:].strip()
        else:
            # Handle unexpected response format
            st.warning("Unexpected response format from AI. Treating as not understood.")
            return False, "Let's try to understand this topic better."
            
    except Exception as e:
        st.error(f"Error evaluating answer: {str(e)}")
        raise

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
                    if not content.strip():
                        st.error("No text could be extracted from the PDF. Please ensure the PDF contains readable text.")
                        return
                        
                    st.info("PDF processed successfully. Generating tutorial structure...")
                    st.session_state.tutorial_state.topics = generate_tutorial_structure(
                        content, st.session_state.model
                    )
                    st.success("Tutorial structure generated successfully!")
                except Exception as e:
                    st.error(f"Error processing content: {str(e)}")
                    return
        
        # Display current topic
        state = st.session_state.tutorial_state
        if state.topics and state.current_topic_index < len(state.topics):
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
        elif state.topics:
            st.success("Congratulations! You've completed all topics!")

if __name__ == "__main__":
    main()
