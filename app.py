import streamlit as st
from typing import Dict, List, Any
import json
import google.generativeai as genai
import PyPDF2
import docx
import io
from functools import lru_cache

# Configure page
st.set_page_config(page_title="AI Teaching Assistant", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .main > div {
        padding: 2rem;
    }
    .stButton > button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4A90E2;
        color: white;
    }
    .stTextArea > div > div > textarea {
        border-radius: 5px;
    }
    .upload-section {
        border: 2px dashed #4A90E2;
        border-radius: 10px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
    }
    .feedback-box {
        background-color: #f8f9fa;
        border-left: 4px solid #4A90E2;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 5px 5px 0;
    }
    .topic-header {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def initialize_model():
    """Initialize or get the Google Gemini model"""
    try:
        # Check if API key is in session state
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = st.secrets.get('GOOGLE_API_KEY', '')
        
        if not st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = st.text_input(
                'Enter Google API Key:', 
                type='password'
            )
            if not st.session_state.gemini_api_key:
                st.warning('Please enter your Google API key to continue.')
                return None
        
        # Configure the Gemini API
        genai.configure(api_key=st.session_state.gemini_api_key)
        
        # Initialize the model
        return genai.GenerativeModel('gemini-pro')
    
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        return None

def process_text_from_file(file_content, file_type) -> str:
    """Extract text from different file types"""
    try:
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            return " ".join(page.extract_text() for page in pdf_reader.pages)
            
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(file_content))
            return " ".join(paragraph.text for paragraph in doc.paragraphs)
            
        elif file_type == "text/markdown":
            return file_content.decode()
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        raise Exception(f"Error processing file content: {str(e)}")

def process_uploaded_file(uploaded_file) -> Dict:
    """Process uploaded file and extract content"""
    try:
        # Read file content
        file_content = uploaded_file.read()
        
        # Extract text based on file type
        text_content = process_text_from_file(file_content, uploaded_file.type)
        
        # Create basic structure
        return {
            "text": text_content,
            "structure": {
                "sections": create_sections(text_content)
            }
        }
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return {"text": "", "structure": {"sections": []}}

def create_sections(text: str) -> List[Dict]:
    """Create sections from text content"""
    sections = []
    current_section = {"title": "Introduction", "content": []}
    
    for paragraph in text.split('\n\n'):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        if paragraph.isupper() or paragraph.endswith(':'):
            if current_section["content"]:
                sections.append(dict(current_section))
            current_section = {"title": paragraph, "content": []}
        else:
            current_section["content"].append(paragraph)
    
    if current_section["content"]:
        sections.append(dict(current_section))
    
    return sections

class DynamicTeacher:
    def __init__(self, model):
        self.model = model

    def analyze_document(self, content: Dict[str, Any]) -> List[Dict]:
        """Initial document analysis to create learning structure"""
        try:
            text_content = content['text']
            
            prompt = f"""
            You are a helpful teaching assistant. Your task is to analyze the following educational content and create a learning structure.
            
            Content to analyze:
            {text_content[:2000]}...

            Instructions:
            Create a learning structure with clear topics and key points. You must respond with ONLY a JSON object in the following format:
            {{
                "topics": [
                    {{
                        "title": "Topic title",
                        "key_points": ["point 1", "point 2"],
                        "content": "Relevant content from the document",
                        "teaching_style": "conceptual",
                        "difficulty": "beginner"
                    }}
                ]
            }}

            Remember:
            1. Response must be valid JSON
            2. Do not include any other text or explanations
            3. Only include the JSON structure
            4. Ensure all JSON keys and values are properly quoted
            """

            # Get response from model
            response = self.model.generate_content(prompt, generation_config={
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 40
            })

            # Clean and parse the response
            response_text = response.text.strip()
            
            # Debug logging
            st.write("Raw response:", response_text)  # We'll remove this after confirming it works
            
            # Try to find JSON in the response
            try:
                # Find the first { and last }
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = response_text[start:end]
                    structure = json.loads(json_str)
                    
                    # Validate structure
                    if 'topics' not in structure:
                        structure = {'topics': []}
                    
                    # Ensure each topic has required fields
                    for topic in structure['topics']:
                        topic.setdefault('key_points', [])
                        topic.setdefault('content', '')
                        topic.setdefault('teaching_style', 'conceptual')
                        topic.setdefault('difficulty', 'beginner')
                    
                    return structure['topics']
                else:
                    raise ValueError("No JSON object found in response")
                    
            except json.JSONDecodeError as e:
                st.error(f"JSON parsing error: {str(e)}")
                # Attempt to create a basic structure from the response
                return [{
                    'title': 'Document Overview',
                    'key_points': ['Key points from the document'],
                    'content': text_content[:1000],
                    'teaching_style': 'conceptual',
                    'difficulty': 'beginner'
                }]

        except Exception as e:
            st.error(f"Error analyzing document: {str(e)}")
            return [{
                'title': 'Document Overview',
                'key_points': ['Document analysis needs review'],
                'content': text_content[:1000] if text_content else 'Content unavailable',
                'teaching_style': 'conceptual',
                'difficulty': 'beginner'
            }]

    def teach_topic(self, topic: Dict, user_progress: Dict) -> str:
        """Dynamically generate teaching content"""
        try:
            prompt = f"""
            Act as an expert teacher teaching this topic: {topic['title']}

            Key points to cover:
            {json.dumps(topic['key_points'], indent=2)}

            Content to teach from:
            {topic['content']}

            Teaching context:
            - Teaching style: {topic['teaching_style']}
            - Difficulty level: {topic['difficulty']}
            - Student progress: {user_progress.get('understanding_level', 'beginner')}

            Create an engaging lesson that:
            1. Introduces the topic clearly
            2. Explains concepts with examples
            3. Uses analogies when helpful
            4. Includes practice questions
            5. Checks for understanding

            Format your response as a clear lesson with markdown formatting.
            Include emoji for visual engagement 📚
            Make it conversational and encouraging 🎯
            Include practice questions at the end ✍️
            """

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            st.error(f"Error generating lesson: {str(e)}")
            return "Error generating lesson content."

    def evaluate_response(self, topic: Dict, user_response: str) -> Dict:
        """Evaluate user's response"""
        try:
            prompt = f"""
            As a teacher, evaluate this student's response to questions about: {topic['title']}

            Key points that should be understood:
            {json.dumps(topic['key_points'], indent=2)}

            Student's response:
            {user_response}

            Provide evaluation as JSON:
            {{
                "understanding_level": 0-100,
                "feedback": "Specific, encouraging feedback",
                "areas_to_review": ["area1", "area2"],
                "next_steps": "Recommendation for what to do next"
            }}
            """

            response = self.model.generate_content(prompt)
            return json.loads(response.text)

        except Exception as e:
            st.error(f"Error evaluating response: {str(e)}")
            return {
                "understanding_level": 50,
                "feedback": "Unable to evaluate response.",
                "areas_to_review": [],
                "next_steps": "Please try again."
            }

def main():
    # Initialize session state
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = 0
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = {'understanding_level': 'beginner'}
    
    # Initialize model
    model = initialize_model()
    if model is None:
        st.stop()
        return

    # Store model in session state
    st.session_state.model = model

    # Initialize teacher
    teacher = DynamicTeacher(st.session_state.model)

    # Sidebar navigation
    with st.sidebar:
        st.title("📚 Learning Progress")
        if st.session_state.topics:
            for i, topic in enumerate(st.session_state.topics):
                progress_status = "✅" if i < st.session_state.current_topic else "📍" if i == st.session_state.current_topic else "⭕️"
                if st.button(f"{progress_status} {topic['title']}", key=f"topic_{i}"):
                    st.session_state.current_topic = i
                    st.rerun()

    # Main content area
    if not st.session_state.topics:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1>📚 Welcome to AI Teaching Assistant</h1>
            <p style="font-size: 1.2rem; color: #666; margin: 1rem 0;">
                Upload your learning material and let AI help you master the content.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload your document",
            type=['pdf', 'docx', 'md'],
            help="Supported formats: PDF, DOCX, MD"
        )
        
        if uploaded_file:
            with st.spinner("Processing your document..."):
                try:
                    # Process document
                    content = process_uploaded_file(uploaded_file)
                    
                    if content["text"]:
                        # Analyze content
                        topics = teacher.analyze_document(content)
                        if topics:
                            st.session_state.topics = topics
                            st.rerun()
                        else:
                            st.error("Could not extract topics from the document. Please try a different document.")
                    else:
                        st.error("Could not extract text from the document. Please check if the file is readable.")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")

    else:
        # Display current topic
        current_topic = st.session_state.topics[st.session_state.current_topic]
        
        # Topic header
        st.markdown(f"""
        <div class="topic-header">
            <h2>{current_topic['title']}</h2>
            <p>Topic {st.session_state.current_topic + 1} of {len(st.session_state.topics)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate teaching content
        with st.spinner("Preparing your lesson..."):
            teaching_content = teacher.teach_topic(
                current_topic, 
                st.session_state.user_progress
            )
        
        # Display teaching content
        st.markdown(teaching_content)
        
        # Practice section
        st.markdown("### ✍️ Practice Time")
        user_response = st.text_area(
            "Your response to the practice questions:",
            height=150
        )
        
        if user_response:
            with st.spinner("Evaluating your response..."):
                evaluation = teacher.evaluate_response(current_topic, user_response)
            
            # Show feedback
            st.markdown(f"""
            <div class="feedback-box">
                <h3>Feedback</h3>
                <p>{evaluation['feedback']}</p>
                <div style="margin-top: 1rem;">
                    <div style="background-color: #e9ecef; height: 8px; border-radius: 4px;">
                        <div style="background-color: {'#2ECC71' if evaluation['understanding_level'] >= 70 else '#4A90E2'}; 
                             width: {evaluation['understanding_level']}%; 
                             height: 100%; 
                             border-radius: 4px;">
                        </div>
                    </div>
                    <p style="text-align: right; color: #666; margin-top: 0.25rem;">
                        Understanding Level: {evaluation['understanding_level']}%
                    </p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if evaluation['areas_to_review']:
                st.markdown("### Areas to Review")
                for area in evaluation['areas_to_review']:
                    st.markdown(f"- {area}")
            
            # Update progress if understanding is good
            if evaluation['understanding_level'] >= 70:
                st.success("✨ Great understanding! Ready to move on!")
                if st.session_state.current_topic < len(st.session_state.topics) - 1:
                    if st.button("Next Topic ➡️"):
                        st.session_state.current_topic += 1
                        st.rerun()
                else:
                    st.balloons()
                    st.success("🎉 Congratulations! You've completed all topics!")

if __name__ == "__main__":
    main()
