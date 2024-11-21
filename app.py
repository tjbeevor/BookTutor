import streamlit as st
from typing import Dict, List, Any
import json
import google.generativeai as genai
import PyPDF2
import docx
import io
import concurrent.futures
from functools import lru_cache
import time

# Custom CSS for better UI
st.set_page_config(page_title="AI Teaching Assistant", layout="wide")
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #4A90E2;
        --secondary-color: #2ECC71;
        --background-color: #F7F9FC;
        --text-color: #2C3E50;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: var(--text-color);
        font-family: 'Inter', sans-serif;
        font-weight: 600;
    }
    
    /* Card-like containers */
    .stTextInput, .stTextArea {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 20px;
        padding: 10px 25px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Progress indicators */
    .topic-progress {
        padding: 10px;
        margin: 5px 0;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
    .topic-progress:hover {
        background-color: #f0f2f5;
    }
    
    /* Loading animations */
    .stSpinner {
        border-width: 3px;
    }
    
    /* File uploader */
    .uploadedFile {
        border-radius: 10px;
        border: 2px dashed #4A90E2;
        padding: 20px;
    }
    
    /* Feedback sections */
    .feedback-box {
        background-color: #f8f9fa;
        border-left: 4px solid var(--primary-color);
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

@lru_cache(maxsize=32)
def clean_json_string(json_str: str) -> str:
    """Cache-optimized JSON string cleaner"""
    start = json_str.find('{')
    end = json_str.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError("No valid JSON found in string")
    return json_str[start:end]

def process_uploaded_file(uploaded_file) -> Dict:
    """Optimized file processing with chunking for large files"""
    content = {"text": "", "structure": {"sections": []}}
    CHUNK_SIZE = 1024 * 1024  # 1MB chunks
    
    try:
        if uploaded_file.type == "application/pdf":
            with st.spinner("Processing PDF... This might take a moment for large files"):
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
                
                # Process pages in parallel
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    text_chunks = list(executor.map(
                        lambda page: page.extract_text(),
                        pdf_reader.pages
                    ))
                content["text"] = " ".join(text_chunks)
                
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            # Process paragraphs in chunks
            content["text"] = " ".join([p.text for p in doc.paragraphs])
            
        elif uploaded_file.type == "text/markdown":
            content["text"] = uploaded_file.read().decode()
        
        # Improved section detection with caching
        return process_sections(content)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return {"text": "", "structure": {"sections": []}}

@lru_cache(maxsize=16)
def process_sections(content: Dict) -> Dict:
    """Cache-optimized section processing"""
    paragraphs = content["text"].split("\n\n")
    sections = []
    current_section = {"title": "Introduction", "content": []}
    
    for para in paragraphs:
        if para.strip().isupper() or para.strip().endswith(":"):
            if current_section["content"]:
                sections.append(current_section.copy())
            current_section = {"title": para.strip(), "content": []}
        else:
            current_section["content"].append(para)
    
    if current_section["content"]:
        sections.append(current_section.copy())
    
    content["structure"]["sections"] = sections
    return content

class DynamicTeacher:
    def __init__(self, model):
        self.model = model
        self._cache = {}
    
    @lru_cache(maxsize=32)
    def _cached_generate_content(self, prompt: str) -> str:
        """Cache-optimized content generation"""
        return self.model.generate_content(prompt).text
    
    def analyze_document(self, content: Dict[str, Any]) -> List[Dict]:
        """Optimized document analysis with progress tracking"""
        try:
            text_content = ' '.join(content['text']) if isinstance(content['text'], list) else content['text']
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Break analysis into chunks for progress tracking
            chunks = [text_content[i:i+2000] for i in range(0, len(text_content), 2000)]
            total_chunks = len(chunks)
            
            all_topics = []
            for i, chunk in enumerate(chunks):
                progress = (i + 1) / total_chunks
                progress_bar.progress(progress)
                status_text.text(f"Analyzing chunk {i+1} of {total_chunks}...")
                
                prompt = self._create_analysis_prompt(chunk)
                response = self._cached_generate_content(prompt)
                
                try:
                    chunk_topics = json.loads(clean_json_string(response))['topics']
                    all_topics.extend(chunk_topics)
                except json.JSONDecodeError:
                    continue
            
            # Clean up progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Deduplicate and merge similar topics
            return self._merge_similar_topics(all_topics)
            
        except Exception as e:
            st.error(f"Error analyzing document: {str(e)}")
            return []
    
    def _merge_similar_topics(self, topics: List[Dict]) -> List[Dict]:
        """Merge similar topics to avoid duplication"""
        merged = []
        for topic in topics:
            similar_found = False
            for existing in merged:
                if self._similarity_score(topic['title'], existing['title']) > 0.8:
                    # Merge key points and content
                    existing['key_points'] = list(set(existing['key_points'] + topic['key_points']))
                    existing['content'] += "\n" + topic['content']
                    similar_found = True
                    break
            if not similar_found:
                merged.append(topic)
        return merged
    
    def _similarity_score(self, str1: str, str2: str) -> float:
        """Simple string similarity score"""
        s1 = set(str1.lower().split())
        s2 = set(str2.lower().split())
        return len(s1 & s2) / len(s1 | s2) if s1 or s2 else 0
    
    def _create_analysis_prompt(self, text: str) -> str:
        return f"""
        Analyze this educational content and create a learning structure.
        Content: {text}...

        Create a structured learning path that:
        1. Identifies main topics to be taught
        2. Sequences them logically
        3. Identifies key points for each topic

        Return the structure as JSON with this format:
        {{
            "topics": [
                {{
                    "title": "Topic title",
                    "key_points": ["point 1", "point 2"],
                    "content": "Relevant content from the document",
                    "teaching_style": "conceptual|technical|practical",
                    "difficulty": "beginner|intermediate|advanced"
                }}
            ]
        }}

        Respond ONLY with the JSON, no other text.
        """

    def teach_topic(self, topic: Dict, user_progress: Dict) -> str:
        """Enhanced teaching content generation with progress tracking"""
        cache_key = f"{topic['title']}_{user_progress.get('understanding_level', 'beginner')}"
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        try:
            with st.spinner("Generating personalized lesson content..."):
                prompt = self._create_teaching_prompt(topic, user_progress)
                response = self._cached_generate_content(prompt)
                self._cache[cache_key] = response
                return response
        except Exception as e:
            st.error(f"Error generating lesson: {str(e)}")
            return "Error generating lesson content."
    
    def _create_teaching_prompt(self, topic: Dict, user_progress: Dict) -> str:
        return f"""
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

    def evaluate_response(self, topic: Dict, user_response: str) -> Dict:
        """Enhanced response evaluation with caching"""
        try:
            cache_key = f"{topic['title']}_{hash(user_response)}"
            if cache_key in self._cache:
                return self._cache[cache_key]
                
            prompt = self._create_evaluation_prompt(topic, user_response)
            response = self._cached_generate_content(prompt)
            evaluation = json.loads(clean_json_string(response))
            
            self._cache[cache_key] = evaluation
            return evaluation
            
        except Exception as e:
            st.error(f"Error evaluating response: {str(e)}")
            return {
                "understanding_level": 50,
                "feedback": "Unable to evaluate response.",
                "areas_to_review": [],
                "next_steps": "Please try again."
            }
    
    def _create_evaluation_prompt(self, topic: Dict, user_response: str) -> str:
        return f"""
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

        Respond ONLY with the JSON, no other text.
        """

def initialize_model():
    """Enhanced model initialization with better error handling"""
    try:
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = st.secrets.get('GOOGLE_API_KEY', '')
        
        if not st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = st.text_input(
                'Enter Google API Key:', 
                type='password',
                help="Your API key is stored securely in the session state"
            )
            if not st.session_state.gemini_api_key:
                st.warning('⚠️ Please enter your Google API key to continue.')
                st.stop()
        
        with st.spinner("Initializing AI model..."):
            genai.configure(api_key=st.session_state.gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')
            
            # Test the model with a simple prompt
            test_response = model.generate_content("Hello")
            if not test_response:
                raise Exception("Model initialization test failed")
                
            return model
    
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        st.error("Please check your API key and internet connection.")
        st.stop()

def main():
    # Initialize session state
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'current_topic' not in st.session_state:
        st.session_state.current_topic = 0
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = {'understanding_level': 'beginner'}
    if 'model' not in st.session_state:
        st.session_state.model = initialize_model()

    # Initialize teacher
    teacher = DynamicTeacher(st.session_state.model)

    # Enhanced sidebar with better navigation
    with st.sidebar:
        st.title("📚 Learning Journey")
        st.markdown("---")
        if st.session_state.topics:
            st.markdown("### Progress Tracker")
            for i, topic in enumerate(st.session_state.topics):
                status = "✅" if i < st.session_state.current_topic else "📍" if i == st.session_state.current_topic else "⭕️"
                button_color = "#2ECC71" if status == "✅" else "#4A90E2" if status == "📍" else "#95A5A6"
                
                st.markdown(
                    f"""
                    <div class="topic-progress" style="border-left: 4px solid {button_color};">
                        <span style="color: {button_color};">{status}</span> {topic['title']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if st.button(f"Jump to Topic {i+1}", key=f"topic_{i}", help=f"Switch to {topic['title']}"):
                    st.session_state.current_topic = i
                    st.rerun()

# Main content area with enhanced UI
    if not st.session_state.topics:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1>📚 Welcome to AI Teaching Assistant</h1>
            <p style="font-size: 1.2rem; color: #666; margin: 1rem 0;">
                Upload your learning material and let AI help you master the content.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Enhanced file upload interface
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown("""
            <div class="uploadedFile">
                <h3 style="text-align: center;">📄 Upload Your Document</h3>
            """, unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Supported formats: PDF, DOCX, MD",
                type=['pdf', 'docx', 'md'],
                help="Your document will be processed securely"
            )
            
            if uploaded_file:
                with st.spinner("🔍 Analyzing your document..."):
                    # Show processing status
                    status = st.empty()
                    progress_bar = st.progress(0)
                    
                    # Process document with progress updates
                    status.text("Processing document...")
                    progress_bar.progress(25)
                    content = process_uploaded_file(uploaded_file)
                    
                    status.text("Analyzing content...")
                    progress_bar.progress(50)
                    topics = teacher.analyze_document(content)
                    
                    status.text("Preparing learning materials...")
                    progress_bar.progress(75)
                    st.session_state.topics = topics
                    
                    status.text("Ready!")
                    progress_bar.progress(100)
                    time.sleep(0.5)  # Brief pause to show completion
                    st.rerun()

    else:
        # Display current topic with enhanced UI
        current_topic = st.session_state.topics[st.session_state.current_topic]
        
        # Topic header with progress indicator
        progress_percent = (st.session_state.current_topic + 1) / len(st.session_state.topics) * 100
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
            <h2 style="margin:0">{current_topic['title']}</h2>
            <div style="margin-top: 0.5rem;">
                <div style="background-color: #e9ecef; height: 8px; border-radius: 4px;">
                    <div style="background-color: #4A90E2; width: {progress_percent}%; height: 100%; border-radius: 4px;"></div>
                </div>
                <p style="text-align: right; color: #666; margin-top: 0.25rem;">
                    Topic {st.session_state.current_topic + 1} of {len(st.session_state.topics)}
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate and display teaching content with loading animation
        with st.spinner("📚 Preparing your personalized lesson..."):
            teaching_content = teacher.teach_topic(
                current_topic, 
                st.session_state.user_progress
            )
        
        # Display teaching content in a card-like container
        st.markdown("""
        <div style="background-color: white; padding: 2rem; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
        """, unsafe_allow_html=True)
        st.markdown(teaching_content)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Input for user response with enhanced UI
        st.markdown("""
        <div style="margin-top: 2rem;">
            <h3>✍️ Practice Time</h3>
            <p style="color: #666;">Answer the practice questions above to check your understanding.</p>
        </div>
        """, unsafe_allow_html=True)
        
        user_response = st.text_area(
            "Your response:",
            height=150,
            help="Write your answers here. Be thorough to get the best feedback!"
        )
        
        if user_response:
            with st.spinner("🤔 Evaluating your response..."):
                evaluation = teacher.evaluate_response(current_topic, user_response)
            
            # Show feedback with enhanced UI
            st.markdown(f"""
            <div class="feedback-box">
                <h3>📝 Feedback</h3>
                <p>{evaluation['feedback']}</p>
                <div style="margin-top: 1rem;">
                    <div style="background-color: #e9ecef; height: 8px; border-radius: 4px;">
                        <div style="background-color: {'#2ECC71' if evaluation['understanding_level'] >= 70 else '#4A90E2'}; width: {evaluation['understanding_level']}%; height: 100%; border-radius: 4px;"></div>
                    </div>
                    <p style="text-align: right; color: #666; margin-top: 0.25rem;">Understanding Level: {evaluation['understanding_level']}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            if evaluation['areas_to_review']:
                st.markdown("""
                <div style="margin-top: 1rem;">
                    <h4>🎯 Areas to Review</h4>
                </div>
                """, unsafe_allow_html=True)
                for area in evaluation['areas_to_review']:
                    st.markdown(f"- {area}")
            
            # Update progress if understanding is good
            if evaluation['understanding_level'] >= 70:
                st.success("✨ Great understanding! Ready to move forward!")
                if st.session_state.current_topic < len(st.session_state.topics) - 1:
                    if st.button("Next Topic ➡️", help="Move to the next topic"):
                        st.session_state.current_topic += 1
                        st.rerun()
                else:
                    st.balloons()
                    st.success("🎉 Congratulations! You've completed all topics!")
                    st.markdown("""
                    <div style="text-align: center; margin-top: 2rem;">
                        <h2>🌟 Learning Journey Complete!</h2>
                        <p>Great job mastering all the topics!</p>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
