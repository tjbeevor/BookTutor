import streamlit as st
from typing import Dict, List, Any
import json
import google.generativeai as genai
import PyPDF2
import docx
import io
from functools import lru_cache

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

class EnhancedTeacher:
    def __init__(self, model):
        self.model = model

    def analyze_document(self, content: Dict[str, Any]) -> List[Dict]:
        """Enhanced document analysis with comprehensive content processing"""
        try:
            text_content = content['text']
            sections = self._split_into_chunks(text_content)
            all_topics = []
            
            for section in sections:
                prompt = f"""
                You are an expert curriculum designer. Analyze this content and create a detailed learning module:

                Content:
                {section[:4000]}

                Create a structured learning module that includes:
                1. Clear title and learning objectives
                2. Key concepts and principles
                3. Practical examples and exercises
                4. Knowledge check questions
                5. Additional resources

                Response format (JSON):
                {{
                    "title": "Clear section title",
                    "learning_objectives": ["List of specific learning objectives"],
                    "key_points": ["Key concepts to master"],
                    "content": "Main educational content",
                    "practical_exercises": ["Relevant exercises"],
                    "knowledge_check": {{
                        "questions": [
                            {{
                                "question": "Question text",
                                "options": ["A", "B", "C", "D"],
                                "correct_answer": "Correct option",
                                "explanation": "Why this is correct"
                            }}
                        ]
                    }},
                    "additional_resources": ["Helpful resources"],
                    "difficulty": "beginner/intermediate/advanced",
                    "estimated_time": "Time estimate"
                }}
                """

                try:
                    response = self.model.generate_content(prompt)
                    topic = json.loads(self._extract_json(response.text))
                    if self._validate_topic(topic):
                        all_topics.append(topic)
                except Exception as e:
                    st.warning(f"Warning: Skipped section due to processing error: {str(e)}")
                    continue

            return all_topics if all_topics else [self._create_fallback_structure(text_content)]

        except Exception as e:
            st.error(f"Error analyzing document: {str(e)}")
            return [self._create_fallback_structure(text_content)]

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split content into logical sections"""
        lines = text.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if self._is_section_header(line) and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # If no natural sections found, create chunks of reasonable size
        if len(sections) <= 1:
            return [text[i:i+4000] for i in range(0, len(text), 4000)]
            
        return sections

    def _is_section_header(self, line: str) -> bool:
        """Detect if a line is likely a section header"""
        line = line.strip()
        if not line:
            return False
            
        # Common section header patterns
        patterns = [
            lambda x: x.isupper() and len(x) > 10,
            lambda x: x.startswith(('#', 'Chapter', 'Section')),
            lambda x: any(char.isdigit() for char in x[:5]) and len(x) < 100,
            lambda x: x.endswith(':') and len(x) < 100
        ]
        
        return any(pattern(line) for pattern in patterns)

    def _validate_topic(self, topic: Dict) -> bool:
        """Validate topic structure"""
        required_fields = ['title', 'learning_objectives', 'key_points', 'content']
        return all(field in topic for field in required_fields)

    def _extract_json(self, text: str) -> str:
        """Safely extract JSON from text"""
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return text[start:end]
            raise ValueError("No JSON found")
        except Exception:
            raise ValueError("Invalid JSON structure")

    def _create_fallback_structure(self, content: str) -> Dict:
        """Create basic structure when parsing fails"""
        return {
            'title': 'Content Overview',
            'learning_objectives': ['Understand the main concepts presented'],
            'key_points': ['Key concepts from the material'],
            'content': content[:1000],
            'practical_exercises': ['Review and summarize the main points'],
            'knowledge_check': {
                'questions': [{
                    'question': 'What is the main topic covered?',
                    'options': ['Review the content to answer'],
                    'correct_answer': 'Review the content to answer',
                    'explanation': 'Please review the material carefully'
                }]
            },
            'difficulty': 'beginner',
            'estimated_time': '15 minutes'
        }

    def teach_topic(self, topic: Dict, user_progress: Dict) -> str:
        """Generate engaging lesson content"""
        try:
            prompt = f"""
            Create an engaging lesson for: {topic['title']}
            
            Learning objectives:
            {json.dumps(topic.get('learning_objectives', []), indent=2)}
            
            Key points:
            {json.dumps(topic.get('key_points', []), indent=2)}
            
            Main content:
            {topic.get('content', '')}

            Create an engaging lesson with:
            1. Clear introduction
            2. Detailed explanations
            3. Practical examples
            4. Interactive elements
            5. Summary and key takeaways

            Use markdown formatting and engage with emoji where appropriate.
            """

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            st.error(f"Error generating lesson: {str(e)}")
            return "Error generating lesson content."

def process_text_from_file(file_content, file_type) -> str:
    """Process uploaded file content"""
    try:
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = "\n\n".join(page.extract_text() for page in pdf_reader.pages)
            
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(file_content))
            text = "\n\n".join(paragraph.text for paragraph in doc.paragraphs)
            
        elif file_type == "text/markdown":
            text = file_content.decode()
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return text.strip()
        
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

def reset_application():
    """Reset the application state"""
    for key in list(st.session_state.keys()):
        if key != 'gemini_api_key':  # Preserve API key
            del st.session_state[key]
    st.rerun()

def main():
    # Configure page settings
    st.set_page_config(
        page_title="AI Teaching Assistant",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Add CSS
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
    </style>
    """, unsafe_allow_html=True)

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
        
        if st.button("🔄 Start New Session", key="reset_button", help="Reset and start fresh"):
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
        current_topic = st.session_state.topics[st.session_state.current_topic]
        
        st.markdown(f"""
        <div class="topic-header">
            <h2>{current_topic['title']}</h2>
            <p>Topic {st.session_state.current_topic + 1} of {len(st.session_state.topics)}</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.spinner("🎓 Preparing your lesson..."):
            lesson_content = teacher.teach_topic(
                current_topic, 
                st.session_state.user_progress
            )
            st.markdown(lesson_content)
        
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
