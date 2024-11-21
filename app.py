import streamlit as st
from typing import Dict, List, Any
import json
import google.generativeai as genai
import PyPDF2
import docx
import io

def clean_json_string(json_str: str) -> str:
    """Clean and validate JSON string"""
    # Find the first { and last } to extract valid JSON
    start = json_str.find('{')
    end = json_str.rfind('}') + 1
    if start == -1 or end == 0:
        raise ValueError("No valid JSON found in string")
    return json_str[start:end]

def process_uploaded_file(uploaded_file) -> Dict:
    """Process uploaded file and extract content"""
    content = {"text": "", "structure": {"sections": []}}
    
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            content["text"] = " ".join([page.extract_text() for page in pdf_reader.pages])
            
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            content["text"] = " ".join([paragraph.text for paragraph in doc.paragraphs])
            
        elif uploaded_file.type == "text/markdown":
            content["text"] = uploaded_file.read().decode()
            
        else:
            raise ValueError(f"Unsupported file type: {uploaded_file.type}")
            
        # Basic section detection
        paragraphs = content["text"].split("\n\n")
        current_section = {"title": "Introduction", "content": []}
        
        for para in paragraphs:
            if para.strip().isupper() or para.strip().endswith(":"):
                if current_section["content"]:
                    content["structure"]["sections"].append(current_section)
                current_section = {"title": para.strip(), "content": []}
            else:
                current_section["content"].append(para)
                
        if current_section["content"]:
            content["structure"]["sections"].append(current_section)
            
        return content
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return {"text": "", "structure": {"sections": []}}

class DynamicTeacher:
    def __init__(self, model):
        self.model = model

    def analyze_document(self, content: Dict[str, Any]) -> List[Dict]:
        """Initial document analysis to create learning structure"""
        try:
            # Extract text and structure
            text_content = ' '.join(content['text']) if isinstance(content['text'], list) else content['text']
            sections = content.get('structure', {}).get('sections', [])

            # Create initial structure prompt
            prompt = f"""
            Analyze this educational content and create a learning structure.
            Content: {text_content[:2000]}...

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

            # Get structure from Gemini
            response = self.model.generate_content(prompt)
            structure = json.loads(clean_json_string(response.text))
            return structure['topics']

        except Exception as e:
            st.error(f"Error analyzing document: {str(e)}")
            return []

    def teach_topic(self, topic: Dict, user_progress: Dict) -> str:
        """Dynamically generate teaching content for a specific topic"""
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

            # Generate teaching content
            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            st.error(f"Error generating lesson: {str(e)}")
            return "Error generating lesson content."

    def evaluate_response(self, topic: Dict, user_response: str) -> Dict:
        """Evaluate user's understanding based on their response"""
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

            Respond ONLY with the JSON, no other text.
            """

            # Get evaluation
            response = self.model.generate_content(prompt)
            return json.loads(clean_json_string(response.text))

        except Exception as e:
            st.error(f"Error evaluating response: {str(e)}")
            return {
                "understanding_level": 50,
                "feedback": "Unable to evaluate response.",
                "areas_to_review": [],
                "next_steps": "Please try again."
            }

def initialize_model():
    """Initialize or get the Gemini model"""
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
                st.stop()
        
        # Configure the Gemini API
        genai.configure(api_key=st.session_state.gemini_api_key)
        
        # Initialize the model - using the most capable model
        model = genai.GenerativeModel('gemini-pro')
        return model
    
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        st.stop()

def main():
    st.set_page_config(page_title="AI Teaching Assistant", layout="wide")

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

    # Sidebar for navigation
    with st.sidebar:
        st.title("📚 Learning Progress")
        if st.session_state.topics:
            for i, topic in enumerate(st.session_state.topics):
                if st.button(
                    f"{'✅' if i < st.session_state.current_topic else '📍' if i == st.session_state.current_topic else '⭕️'} {topic['title']}",
                    key=f"topic_{i}"
                ):
                    st.session_state.current_topic = i

    # Main content area
    if not st.session_state.topics:
        # File upload interface
        st.title("📚 Upload Learning Material")
        uploaded_file = st.file_uploader("Upload your document", type=['pdf', 'docx', 'md'])
        
        if uploaded_file:
            with st.spinner("Analyzing document..."):
                # Process document
                content = process_uploaded_file(uploaded_file)
                
                # Analyze content
                topics = teacher.analyze_document(content)
                st.session_state.topics = topics
                st.rerun()

    else:
        # Display current topic
        current_topic = st.session_state.topics[st.session_state.current_topic]
        
        # Generate teaching content
        teaching_content = teacher.teach_topic(
            current_topic, 
            st.session_state.user_progress
        )
        
        # Display teaching content
        st.markdown(teaching_content)
        
        # Input for user response
        user_response = st.text_area("Your response to the practice questions:")
        
        if user_response:
            # Evaluate response
            evaluation = teacher.evaluate_response(current_topic, user_response)
            
            # Show feedback
            st.write("### Feedback")
            st.write(evaluation['feedback'])
            
            if evaluation['areas_to_review']:
                st.write("### Areas to Review")
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
                    st.success("🎉 Congratulations! You've completed all topics!")

if __name__ == "__main__":
    main()
