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
    page_title="DocuMentor",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Original CSS remains exactly the same
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
        
        genai.configure(api_key=st.session_state.gemini_api_key)
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
        # Check file size (15MB limit)
        MAX_FILE_SIZE = 15 * 1024 * 1024  # 15MB in bytes
        
        # Get file size
        file_content = uploaded_file.read()
        file_size = len(file_content)
        
        if file_size > MAX_FILE_SIZE:
            raise Exception(f"File size ({file_size / 1024 / 1024:.1f}MB) exceeds the maximum limit of 15MB")
            
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
        self._content_cache = {}

    def analyze_document(self, content: Dict[str, Any]) -> List[Dict]:
        """Initial document analysis with comprehensive topic breakdown"""
        try:
            text_content = content['text']
            
            # Process a larger chunk of text, broken into sections
            text_chunks = [text_content[i:i+8000] for i in range(0, len(text_content), 8000)]
            first_chunk = text_chunks[0] if text_chunks else ""
            
            prompt = {
                "role": "user",
                "parts": [f"""You are an expert curriculum designer. Analyze this educational content and create a detailed learning structure with comprehensive topic breakdown.
                
                Content to analyze:
                {first_chunk}

                Additional instructions:
                1. Break down the content into 8-12 distinct topics
                2. Ensure topics flow logically from foundational to advanced concepts
                3. Each topic should be specific and focused
                4. Include both theoretical and practical aspects where relevant

                Create a learning structure following this exact JSON format:
                {{
                    "topics": [
                        {{
                            "title": "Clear and specific topic title",
                            "key_points": [
                                "Detailed key point 1",
                                "Detailed key point 2",
                                "Detailed key point 3",
                                "Detailed key point 4",
                                "Detailed key point 5"
                            ],
                            "content": "Relevant content section",
                            "teaching_style": "conceptual",
                            "difficulty": "beginner|intermediate|advanced"
                        }}
                    ]
                }}

                Requirements:
                - Create at least 8 topics
                - Each topic must have exactly 5 key points
                - Make topics specific rather than general
                - Ensure progressive difficulty across topics
                - Include practical applications where relevant"""]
            }

            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.4,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 4096
                }
            )

            # Safely extract and parse JSON
            try:
                response_text = response.text.strip()
                json_match = response_text[response_text.find('{'):response_text.rfind('}')+1]
                structure = json.loads(json_match)
                
                if not isinstance(structure, dict) or 'topics' not in structure:
                    raise ValueError("Invalid JSON structure")
                
                topics = structure['topics']
                if not isinstance(topics, list):
                    raise ValueError("Topics must be a list")
                
                # Process additional chunks if available
                if len(text_chunks) > 1:
                    for topic in topics:
                        topic_content = topic['content']
                        for chunk in text_chunks[1:]:
                            if any(kp.lower() in chunk.lower() for kp in topic['key_points']):
                                topic_content += "\n\n" + chunk[:1000]
                        topic['content'] = topic_content
                
                # Ensure each topic has the required fields
                for topic in topics:
                    if not isinstance(topic, dict):
                        continue
                    while len(topic.get('key_points', [])) < 5:
                        topic.setdefault('key_points', []).append(f"Additional key point for {topic['title']}")
                    topic['content'] = topic.get('content', '')
                    topic['teaching_style'] = topic.get('teaching_style', 'conceptual')
                    topic['difficulty'] = topic.get('difficulty', 'beginner')
                
                return topics
                
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Error parsing model response: {str(e)}")
                return [{
                    'title': 'Document Overview',
                    'key_points': [
                        'Key concepts from the document',
                        'Main ideas and themes',
                        'Important principles covered',
                        'Practical applications',
                        'Learning objectives'
                    ],
                    'content': text_content[:8000],
                    'teaching_style': 'conceptual',
                    'difficulty': 'beginner'
                }]

        except Exception as e:
            st.error(f"Error in document analysis: {str(e)}")
            return [{
                'title': 'Document Overview',
                'key_points': [
                    'Document analysis needs review',
                    'Content structure pending',
                    'Topics to be organized',
                    'Key points to be extracted',
                    'Learning path to be defined'
                ],
                'content': text_content[:8000] if text_content else 'Content unavailable',
                'teaching_style': 'conceptual',
                'difficulty': 'beginner'
            }]

    @lru_cache(maxsize=32)
    def _generate_lesson_content(self, topic_str: str) -> str:
        """Cached method to generate lesson content"""
        try:
            topic = json.loads(topic_str)
            
            prompt = {
                "role": "user",
                "parts": [{
                    "text": f"""
                    Create an engaging lesson for: {topic['title']}

                    Key points to cover:
                    {json.dumps(topic['key_points'], indent=2)}

                    Content to teach from:
                    {topic['content']}

                    Context:
                    - Style: {topic['teaching_style']}
                    - Level: {topic['difficulty']}

                    Create a comprehensive lesson that:
                    1. Introduces the topic clearly
                    2. Explains each concept thoroughly with examples
                    3. Uses helpful analogies where appropriate
                    4. Provides detailed explanations
                    5. Includes practical applications
                    6. Ends with key takeaways

                    Use markdown formatting and emojis for engagement.
                    """
                }]
            }

            response = self.model.generate_content(
                prompt,
                generation_config={
                    'temperature': 0.4,
                    'top_p': 0.8,
                    'top_k': 40,
                    'max_output_tokens': 4096
                }
            )
            
            # Properly handle the response format
            if response.candidates:
                # Get all parts of the response and join them
                parts = response.candidates[0].content.parts
                return "".join(part.text for part in parts if hasattr(part, 'text'))
            else:
                return "Could not generate lesson content."
                
        except Exception as e:
            st.error(f"Error in _generate_lesson_content: {str(e)}")
            return f"Error generating lesson content: {str(e)}"

    def teach_topic(self, topic: Dict, user_progress: Dict) -> str:
        """Generate teaching content using cache"""
        try:
            # Convert topic to string for caching
            topic_str = json.dumps(topic)
            return self._generate_lesson_content(topic_str)
        except Exception as e:
            st.error(f"Error in teach_topic: {str(e)}")
            return f"Error generating lesson: {str(e)}"

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
        return

    # Initialize teacher
    teacher = DynamicTeacher(model)

    # Sidebar navigation
    with st.sidebar:
        st.title("Restart")
        
        if st.button("🔄 Start New Session", key="reset_button", help="Reset the application and upload a new document"):
            reset_application()
        
        st.divider()
        
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
            <h1>📚 Welcome to DocuMentor</h1>
            <p style="font-size: 1.2rem; color: #666; margin: 1rem 0;">
                Transform your documents into interactive learning experiences.
            </p>
            <div class="upload-section">
                <p style="color: #4A90E2; font-size: 1.1rem; margin-bottom: 1rem;">
                    📄 Upload you document to begin
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "",
            type=['pdf', 'docx', 'md'],
            help="Supported formats: PDF, DOCX, MD"
        )
        
        if uploaded_file:
            with st.spinner("🔄 Processing your document..."):
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

    else:
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
        
        # Generate teaching content with caching
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
        st.markdown("<br>", unsafe_allow_html=True)
        cols = st.columns([1, 2, 1])
        
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
