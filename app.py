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
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #f8f9fa;
    }
    
    .main > div {
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem;
    }
    
    .chapter-navigation {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: #fff;
        border-radius: 8px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }

    .learning-objectives {
        background-color: #f0f7ff;
        padding: 1rem;
        border-left: 4px solid #4A90E2;
        margin: 1rem 0;
    }

    .knowledge-check {
        background-color: #fff3e0;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .practical-exercise {
        background-color: #e8f5e9;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .progress-indicator {
        height: 8px;
        background-color: #e9ecef;
        border-radius: 4px;
        margin: 1rem 0;
    }

    .progress-bar {
        height: 100%;
        background-color: #4A90E2;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)

def initialize_model():
    """Initialize or get the Google Gemini model with enhanced error handling"""
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

def process_text_from_file(file_content, file_type) -> str:
    """Enhanced file processing with better error handling and support for more formats"""
    try:
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = " ".join(page.extract_text() for page in pdf_reader.pages)
            
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(file_content))
            text = " ".join(paragraph.text for paragraph in doc.paragraphs)
            
        elif file_type == "text/markdown":
            text = file_content.decode()
        
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        # Clean and preprocess text
        text = clean_text(text)
        return text
    except Exception as e:
        raise Exception(f"Error processing file content: {str(e)}")

def clean_text(text: str) -> str:
    """Clean and preprocess text content"""
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Basic normalization
    text = text.replace('\n\n', '\n').strip()
    return text

class EnhancedTeacher:
    def __init__(self, model):
        self.model = model

    def analyze_document(self, content: Dict[str, Any]) -> List[Dict]:
        """Enhanced document analysis with comprehensive learning structure"""
        try:
            text_content = content['text']
            
            prompt = f"""
            You are an expert curriculum designer and instructor. Analyze this educational content
            and create a comprehensive learning structure:

            Content to analyze:
            {text_content[:2000]}...

            Create a structured curriculum that includes:
            1. Clear learning objectives for each section
            2. Key concepts and principles
            3. Practical exercises and examples
            4. Knowledge check questions
            5. Real-world applications
            6. Additional resources and tips

            Format as JSON:
            {{
                "topics": [
                    {{
                        "title": "Topic title",
                        "learning_objectives": ["objective 1", "objective 2"],
                        "key_points": ["point 1", "point 2"],
                        "practical_exercises": ["exercise 1", "exercise 2"],
                        "knowledge_check": {{
                            "questions": [
                                {{
                                    "question": "Question text",
                                    "options": ["option 1", "option 2", "option 3", "option 4"],
                                    "correct_answer": "Correct option",
                                    "explanation": "Why this is correct"
                                }}
                            ]
                        }},
                        "real_world_applications": ["application 1", "application 2"],
                        "additional_resources": ["resource 1", "resource 2"],
                        "content": "Main content text",
                        "estimated_time": "30 minutes",
                        "difficulty": "beginner"
                    }}
                ]
            }}
            """

            response = self.model.generate_content(prompt)
            
            try:
                json_str = self._extract_json(response.text)
                structure = json.loads(json_str)
                
                # Validate and process the structure
                if 'topics' not in structure:
                    structure = {'topics': []}
                
                return structure['topics']
                
            except json.JSONDecodeError as e:
                st.error(f"❌ Error parsing curriculum structure: {str(e)}")
                return self._create_fallback_structure(text_content)

        except Exception as e:
            st.error(f"❌ Error analyzing document: {str(e)}")
            return self._create_fallback_structure(text_content)

    def _extract_json(self, text: str) -> str:
        """Extract JSON content from text"""
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return text[start:end]
        raise ValueError("No JSON object found in response")

    def _create_fallback_structure(self, content: str) -> List[Dict]:
        """Create a basic structure when full analysis fails"""
        return [{
            'title': 'Document Overview',
            'learning_objectives': ['Understand the main concepts'],
            'key_points': ['Key concepts from the document'],
            'content': content[:1000],
            'difficulty': 'beginner',
            'estimated_time': '15 minutes'
        }]

    def teach_topic(self, topic: Dict, user_progress: Dict) -> str:
        """Generate engaging, interactive lesson content"""
        try:
            prompt = f"""
            Create an engaging lesson for topic: {topic['title']}

            Learning objectives:
            {json.dumps(topic.get('learning_objectives', []), indent=2)}

            Key points:
            {json.dumps(topic.get('key_points', []), indent=2)}

            Student context:
            - Current level: {user_progress.get('understanding_level', 'beginner')}
            - Previous topics completed: {len(user_progress.get('completed_topics', []))}

            Create an interactive lesson that:
            1. Starts with a hook or engaging question
            2. Explains concepts clearly with examples
            3. Includes practical exercises
            4. Provides knowledge check points
            5. Connects to real-world applications
            6. Ends with a summary and next steps

            Format your response in markdown with clear sections and engaging emoji.
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
        st.session_state.user_progress = {
            'understanding_level': 'beginner',
            'completed_topics': [],
            'quiz_scores': {}
        }
    
    # Initialize model
    model = initialize_model()
    if model is None:
        st.stop()
        return

    # Initialize teacher
    teacher = EnhancedTeacher(model)

    # Sidebar navigation and progress
    with st.sidebar:
        st.title("📚 Learning Progress")
        
        if st.button("🔄 Start New Session", help="Reset and start fresh"):
            for key in list(st.session_state.keys()):
                if key != 'gemini_api_key':
                    del st.session_state[key]
            st.rerun()
        
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
            <h1>📚 Welcome to AI Teaching Assistant</h1>
            <p>Upload your learning materials and let's create an interactive learning experience.</p>
            <div class="upload-section">
                <p>📄 Drop your document here</p>
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
                try:
                    content = {
                        "text": process_text_from_file(uploaded_file.read(), uploaded_file.type),
                        "structure": {"sections": []}
                    }
                    
                    if content["text"]:
                        topics = teacher.analyze_document(content)
                        if topics:
                            st.session_state.topics = topics
                            st.rerun()
                        else:
                            st.error("❌ Could not extract topics from the document.")
                    else:
                        st.error("❌ Could not extract text from the document.")
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")

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
        
        # Learning objectives
        with st.expander("🎯 Learning Objectives", expanded=True):
            for objective in current_topic.get('learning_objectives', []):
                st.markdown(f"- {objective}")
        
        # Generate and display lesson content
        with st.spinner("🎓 Preparing your lesson..."):
            lesson_content = teacher.teach_topic(
                current_topic, 
                st.session_state.user_progress
            )
            st.markdown(lesson_content)
        
        # Knowledge check
        if 'knowledge_check' in current_topic:
            with st.expander("📝 Knowledge Check"):
                for q in current_topic['knowledge_check']['questions']:
                    answer = st.radio(q['question'], q['options'])
                    if st.button("Check Answer", key=f"check_{q['question']}"):
                        if answer == q['correct_answer']:
                            st.success(f"Correct! {q['explanation']}")
                        else:
                            st.error(f"Not quite. {q['explanation']}")
        
        # Navigation
        cols = st.columns([1, 2, 1])
        
        with cols[0]:
            if st.session_state.current_topic > 0:
                if st.button("⬅️ Previous Topic"):
                    st.session_state.current_topic -= 1
                    st.rerun()
        
        with cols[2]:
            if st.session_state.current_topic < len(st.session_state.topics) - 1:
                if st.button("Next Topic ➡️"):
                    st.session_state.current_topic += 1
                    st.rerun()
            elif st.session_state.current_topic == len(st.session_state.topics) - 1:
                st.success("🎉 Congratulations! You've completed all topics!")

if __name__ == "__main__":
    main()
