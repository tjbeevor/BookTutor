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
    try:
        paragraphs = content["text"].split("\n\n")
        sections = []
        current_section = {"title": "Introduction", "content": []}
        
        for para in paragraphs:
            if para.strip().isupper() or para.strip().endswith(":"):
                if current_section["content"]:
                    sections.append(dict(current_section))
                current_section = {"title": para.strip(), "content": []}
            else:
                current_section["content"].append(para)
        
        if current_section["content"]:
            sections.append(dict(current_section))
        
        return {"text": content["text"], "structure": {"sections": sections}}
    except Exception as e:
        st.error(f"Error in section processing: {str(e)}")
        return {"text": content["text"], "structure": {"sections": []}}

def process_uploaded_file(uploaded_file) -> Dict:
    """Process uploaded file and extract content"""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text_content = " ".join([page.extract_text() for page in pdf_reader.pages])
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            text_content = " ".join([p.text for p in doc.paragraphs])
        elif uploaded_file.type == "text/markdown":
            text_content = uploaded_file.read().decode()
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.type}")
        
        content = {"text": text_content}
        return process_sections(content)
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return {"text": "", "structure": {"sections": []}}

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

    # Main content area
    if not st.session_state.topics:
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h1 style="color: #1F4287;">📚 Welcome to AI Teaching Assistant</h1>
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
            try:
                with st.spinner("Processing your document..."):
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
        <div style="background-color: #f8f9fa; padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
            <h2 style="color: #1F4287; margin:0">{current_topic['title']}</h2>
            <p style="color: #666; margin-top: 0.5rem;">
                Topic {st.session_state.current_topic + 1} of {len(st.session_state.topics)}
            </p>
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
            st.markdown("### Feedback")
            st.write(evaluation['feedback'])
            
            if evaluation['areas_to_review']:
                st.markdown("### Areas to Review")
                for area in evaluation['areas_to_review']:
                    st.write(f"- {area}")
            
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
