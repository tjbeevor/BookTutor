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

[Previous EnhancedTeacher class and all other functions remain exactly the same...]

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
