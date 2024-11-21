import streamlit as st
from typing import Dict, List, Any
import json
import google.generativeai as genai
import PyPDF2
import docx
import io
from functools import lru_cache

# Configure page with improved layout
st.set_page_config(
    page_title="AI Teaching Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #f8f9fa;
    }
    
    /* Main content area styling */
    .main > div {
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background-color: #4A90E2;
        color: white;
        font-weight: 500;
        transition: all 0.3s ease;
        border: none;
        margin: 0.5rem 0;
    }
    
    .stButton > button:hover {
        background-color: #357ABD;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* Reset button styling */
    .reset-button > button {
        background-color: #dc3545;
    }
    
    .reset-button > button:hover {
        background-color: #c82333;
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        padding: 0.75rem;
    }
    
    /* Upload section styling */
    .upload-section {
        border: 2px dashed #4A90E2;
        border-radius: 10px;
        padding: 3rem;
        margin: 2rem 0;
        text-align: center;
        background-color: #f8f9fa;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: #357ABD;
        background-color: #e9ecef;
    }
    
    /* Topic header styling */
    .topic-header {
        background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .topic-header h2 {
        margin: 0;
        color: white;
    }
    
    /* Navigation styling */
    .nav-button {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 1rem;
    }
    
    /* Progress indicators */
    .progress-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.25rem 0;
        transition: all 0.3s ease;
    }
    
    .progress-indicator:hover {
        background-color: #e9ecef;
    }
    
    /* Welcome screen styling */
    .welcome-screen {
        text-align: center;
        padding: 3rem;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .welcome-screen h1 {
        color: #4A90E2;
        margin-bottom: 1.5rem;
    }
    
    /* Loader styling */
    .stSpinner > div {
        border-color: #4A90E2;
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
                type='password',
                help="Enter your Google API key to access the Gemini model"
            )
            if not st.session_state.gemini_api_key:
                st.warning('⚠️ Please enter your Google API key to continue.')
                return None
        
        # Configure the Gemini API
        genai.configure(api_key=st.session_state.gemini_api_key)
        
        # Initialize the model
        return genai.GenerativeModel('gemini-pro')
    
    except Exception as e:
        st.error(f"❌ Error initializing Gemini model: {str(e)}")
        return None

def reset_application():
    """Reset the application state"""
    for key in list(st.session_state.keys()):
        if key != 'gemini_api_key':  # Preserve API key
            del st.session_state[key]
    st.rerun()

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
        file_content = uploaded_file.read()
        text_content = process_text_from_file(file_content, uploaded_file.type)
        
        return {
            "text": text_content,
            "structure": {
                "sections": create_sections(text_content)
            }
        }
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
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

            response = self.model.generate_content(prompt, generation_config={
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 40
            })

            response_text = response.text.strip()
            
            try:
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                if start != -1 and end != 0:
                    json_str = response_text[start:end]
                    structure = json.loads(json_str)
                    
                    if 'topics' not in structure:
                        structure = {'topics': []}
                    
                    for topic in structure['topics']:
                        topic.setdefault('key_points', [])
                        topic.setdefault('content', '')
                        topic.setdefault('teaching_style', 'conceptual')
                        topic.setdefault('difficulty', 'beginner')
                    
                    return structure['topics']
                else:
                    raise ValueError("No JSON object found in response")
                    
            except json.JSONDecodeError as e:
                st.error(f"❌ JSON parsing error: {str(e)}")
                return [{
                    'title': 'Document Overview',
                    'key_points': ['Key points from the document'],
                    'content': text_content[:1000],
                    'teaching_style': 'conceptual',
                    'difficulty': 'beginner'
                }]

        except Exception as e:
            st.error(f"❌ Error analyzing document: {str(e)}")
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
            4. Provides clear explanations of key points

            Format your response as a clear lesson with markdown formatting.
            Include emoji for visual engagement 📚
            Make it conversational and encouraging 🎯
            """

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            st.error(f"❌ Error generating lesson: {str(e)}")
            return "Error generating lesson content."

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

    # Sidebar navigation and reset button
    with st.sidebar:
        st.title("📚 Learning Progress")
        
        # Reset button at the top of sidebar
        if st.button("🔄 Start New Session", key="reset_button", help="Reset the application and upload a new document"):
            reset_application()
        
        st.divider()
        
        # Topic navigation
        if st.session_state.topics:
            for i, topic in enumerate(st.session_state.topics):
                progress_status = "✅" if i < st.session_state.current_topic else "📍" if i == st.session_state.current_topic else "⭕️"
                if st.button(f"{progress_status} {topic['title']}", key=f"topic_{i}"):
                    st.session_state.current_topic = i
                    st.rerun()

    # Main content area
    if not st.session_state.topics:
        st.markdown("""
        <div class="welcome-screen">
            <h1>📚 Welcome to AI Teaching Assistant</h1>
            <p style="font-size: 1.2rem; color: #666; margin: 1rem 0;">
                Transform your documents into interactive learning experiences.
            </p>
            <div class="upload-section">
                <p style="color: #4A90E2; font-size: 1.1rem; margin-bottom: 1rem;">
                    📄 Drop your document here to begin
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "",  # Empty label as we have custom HTML above
            type=['pdf', 'docx', 'md'],
            help="Supported formats: PDF, DOCX, MD"
        )
        
        if uploaded_file:
            with st.spinner("🔄 Processing your document..."):
                try:
                    content = process_uploaded_file(uploaded_file)
                    
                    if content["text"]:
                        topics = teacher.analyze_document(content)
                        if topics:
                            st.session_state.topics = topics
                            st.rerun()
                        else:
                            st.error("❌ Could not extract topics from the document. Please try a different document.")
                    else:
                        st.error("❌ Could not extract text from the document. Please check if the file is readable.")
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")

    else:
        # Display current topic
        current_topic = st.session_state.topics[st.session_state.current_topic]
        
        # Topic header with improved styling
        st.markdown(f"""
        <div class="topic-header">
            <h2>{current_topic['title']}</h2>
            <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                Topic {st.session_state.current_topic + 1} of {len(st.session_state.topics)}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate teaching content
        with st.spinner("🎓 Preparing your lesson..."):
            teaching_content = teacher.teach_topic(
                current_topic, 
                st.session_state.user_progress
            )
        
        # Display teaching content in a clean container
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.markdown(teaching_content)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Navigation buttons with improved layout
        st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        cols = st.columns([1, 2, 1])  # Create three columns for better button spacing
        
        with cols[0]:  # Previous button
            if st.session_state.current_topic > 0:
                if st.button("⬅️ Previous Topic", key="prev_topic", help="Go to previous topic"):
                    st.session_state.current_topic -= 1
                    st.rerun()
        
        with cols[2]:  # Next button
            if st.session_state.current_topic < len(st.session_state.topics) - 1:
                if st.button("Next Topic ➡️", key="next_topic", help="Go to next topic"):
                    st.session_state.current_topic += 1
                    st.rerun()
            elif st.session_state.current_topic == len(st.session_state.topics) - 1:
                st.success("🎉 Congratulations! You've completed all topics!")

if __name__ == "__main__":
    main()
