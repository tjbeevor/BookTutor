import streamlit as st
import google.generativeai as genai
import PyPDF2
import io
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import time

@dataclass
class Topic:
    title: str
    content: str
    subtopics: List['Topic']
    completed: bool = False
    parent: Optional['Topic'] = None

class TutorialState:
    def __init__(self):
        self.topics: List[Topic] = []
        self.current_topic_index: int = 0
        self.current_subtopic_index: int = -1
        self.conversation_history: List[Dict] = []
        self.current_teaching_phase: str = "introduction"
        self.understanding_level: int = 0
        
    def reset(self):
        self.__init__()
        
    def get_current_topic(self) -> Optional[Topic]:
        if not self.topics:
            return None
        if self.current_subtopic_index == -1:
            return self.topics[self.current_topic_index] if self.current_topic_index < len(self.topics) else None
        return (self.topics[self.current_topic_index].subtopics[self.current_subtopic_index] 
                if self.current_topic_index < len(self.topics) and 
                self.current_subtopic_index < len(self.topics[self.current_topic_index].subtopics) 
                else None)

    def advance_phase(self) -> bool:
        phases = ["introduction", "explanation", "examples", "practice"]
        current_index = phases.index(self.current_teaching_phase)
        
        if current_index < len(phases) - 1:
            self.current_teaching_phase = phases[current_index + 1]
            return True
        else:
            self.current_teaching_phase = "introduction"
            return self.advance_topic()
    
    def advance_topic(self) -> bool:
        if not self.topics:
            return False
            
        current_topic = self.topics[self.current_topic_index]
        
        if self.current_subtopic_index == -1:
            if current_topic.subtopics:
                self.current_subtopic_index = 0
                return True
            elif self.current_topic_index < len(self.topics) - 1:
                self.current_topic_index += 1
                return True
            return False
        else:
            if self.current_subtopic_index < len(current_topic.subtopics) - 1:
                self.current_subtopic_index += 1
                return True
            elif self.current_topic_index < len(self.topics) - 1:
                self.current_topic_index += 1
                self.current_subtopic_index = -1
                return True
            return False

def init_gemini(api_key: str = None):
    try:
        if api_key:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-pro')
        return None
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        return None

def process_pdf(pdf_file) -> str:
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            except Exception as e:
                st.warning(f"Warning: Could not process a page in the PDF. Error: {str(e)}")
                continue
                
        text = text.replace('\x00', '')
        text = ' '.join(text.split())
        text = text.strip()
        
        if not text:
            raise ValueError("No readable text found in the PDF")
            
        return text
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        raise

def clean_json_string(json_str: str) -> str:
    try:
        json_str = json_str.replace("```json", "").replace("```", "")
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No valid JSON structure found")
            
        json_str = json_str[start_idx:end_idx]
        json_str = ' '.join(json_str.split())
        
        # Test if it's valid JSON
        json.loads(json_str)
        return json_str
        
    except Exception as e:
        st.error(f"Error cleaning JSON: {str(e)}")
        raise

def generate_tutorial_structure(content: str, model) -> List[Topic]:
    try:
        analysis_prompt = f"""
        Create a learning structure for this content. Return a JSON object with exactly this structure:
        {{
            "title": "Main Topic Title",
            "lessons": [
                {{
                    "title": "Specific Lesson Title",
                    "content": "Clear explanation",
                    "key_points": ["Point 1", "Point 2"],
                    "practice": ["Practice item 1"],
                    "difficulty": "beginner"
                }}
            ]
        }}
        Content: {content[:3000]}
        """
        
        response = model.generate_content(analysis_prompt)
        structure = json.loads(clean_json_string(response.text))
        
        topics = []
        for lesson in structure.get('lessons', []):
            topic_content = f"""
            Learning Outcome: Master {lesson['title']}
            
            Key Points:
            {chr(10).join('- ' + point for point in lesson.get('key_points', []))}
            
            Content:
            {lesson.get('content', '')}
            
            Practice:
            {chr(10).join('- ' + practice for practice in lesson.get('practice', []))}
            """
            
            topics.append(Topic(
                title=lesson.get('title', 'Untitled Topic'),
                content=topic_content,
                subtopics=[],
                completed=False
            ))
        
        return topics if topics else [Topic(
            title="Getting Started",
            content="Introduction to the subject matter.",
            subtopics=[],
            completed=False
        )]
        
    except Exception as e:
        st.error(f"Error generating tutorial structure: {str(e)}")
        return [Topic(
            title="Getting Started",
            content="Introduction to the subject matter.",
            subtopics=[],
            completed=False
        )]

def generate_teaching_message(topic: Topic, phase: str, conversation_history: List[Dict], model) -> dict:
    try:
        prompt = f"""
        Create a micro-lesson about: {topic.title}
        Return a JSON object with exactly this structure:
        {{
            "explanation": "Clear explanation of the concept",
            "examples": "Practical example or demonstration",
            "question": "Understanding check question",
            "key_points": ["Expected point 1", "Expected point 2"]
        }}
        """
        
        response = model.generate_content(prompt)
        content = json.loads(clean_json_string(response.text))
        
        return {
            "explanation": content.get('explanation', ''),
            "examples": content.get('examples', ''),
            "question": content.get('question', ''),
            "key_points": content.get('key_points', [])
        }
        
    except Exception as e:
        st.error(f"Error generating teaching message: {str(e)}")
        return {
            "explanation": topic.content,
            "examples": "Let's practice applying this concept.",
            "question": f"Can you explain the key points of {topic.title}?",
            "key_points": ["Understanding of basic concept"]
        }

def evaluate_response(user_response: str, expected_points: List[str], current_topic: Topic, model) -> dict:
    try:
        prompt = f"""
        Evaluate this response about {current_topic.title}.
        User response: {user_response}
        Expected points: {', '.join(expected_points)}
        
        Return a JSON object with exactly this structure:
        {{
            "feedback": "Specific, constructive feedback",
            "understanding_level": 75
        }}
        """
        
        response = model.generate_content(prompt)
        evaluation = json.loads(clean_json_string(response.text))
        
        return {
            "feedback": evaluation.get('feedback', ''),
            "understanding_level": int(evaluation.get('understanding_level', 50))
        }
        
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return {
            "feedback": "Thank you for your response. Keep practicing!",
            "understanding_level": 50
        }

def main():
    st.set_page_config(
        page_title="AI Learning Assistant",
        page_icon="🎓",
        layout="wide"
    )

    # Initialize session state
    if 'tutorial_state' not in st.session_state:
        st.session_state.tutorial_state = TutorialState()
    
    if 'expected_points' not in st.session_state:
        st.session_state.expected_points = []

    # Header
    st.title("🎓 AI Learning Assistant")
    st.markdown("Transform your learning experience with personalized AI guidance")

    # API Key Management
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        api_key = st.text_input("Enter your Gemini API Key:", type="password")

    if not api_key:
        st.error("Please provide your API key to continue")
        st.stop()

    # Initialize model
    if 'model' not in st.session_state:
        st.session_state.model = init_gemini(api_key)

    if not st.session_state.model:
        st.error("Failed to initialize AI model")
        st.stop()

    # Main UI Layout
    col1, col2 = st.columns([7, 3])

    with col1:
        if not st.session_state.tutorial_state.topics:
            st.markdown("### 📚 Upload Learning Material")
            pdf_file = st.file_uploader("Upload Educational PDF", type="pdf")
            
            if pdf_file:
                with st.spinner("Processing content..."):
                    try:
                        content = process_pdf(pdf_file)
                        st.session_state.tutorial_state.topics = generate_tutorial_structure(
                            content, st.session_state.model
                        )
                        st.success("Tutorial created! Let's begin.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            # Chat Interface
            state = st.session_state.tutorial_state
            current_topic = state.get_current_topic()

            if current_topic and not current_topic.completed:
                # Display conversation history
                for message in state.conversation_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Generate new teaching content if needed
                if len(state.conversation_history) == 0 or (
                    len(state.conversation_history) > 0 and 
                    state.conversation_history[-1]["role"] == "assistant" and 
                    "next topic" in state.conversation_history[-1]["content"].lower()
                ):
                    teaching_content = generate_teaching_message(
                        current_topic,
                        state.current_teaching_phase,
                        state.conversation_history,
                        st.session_state.model
                    )
                    
                    content = f"""
                    ## {current_topic.title}

                    {teaching_content['explanation']}

                    {teaching_content['examples']}

                    💡 **Check Your Understanding:**
                    {teaching_content['question']}
                    """
                    
                    with st.chat_message("assistant"):
                        st.markdown(content)
                    
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": content
                    })
                    
                    st.session_state.expected_points = teaching_content["key_points"]

                # Handle user input
                user_input = st.chat_input("Your response...")
                if user_input:
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    
                    state.conversation_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    evaluation = evaluate_response(
                        user_input,
                        st.session_state.expected_points,
                        current_topic,
                        st.session_state.model
                    )
                    
                    feedback = f"""
                    ### Feedback
                    {evaluation['feedback']}
                    
                    Understanding Level: {evaluation['understanding_level']}%
                    
                    ---
                    *Moving to next topic...*
                    """
                    
                    with st.chat_message("assistant"):
                        st.markdown(feedback)
                    
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": feedback
                    })
                    
                    if evaluation['understanding_level'] >= 70:
                        current_topic.completed = True
                        if state.advance_topic():
                            st.rerun()
                        else:
                            st.balloons()
                            st.success("🎉 Congratulations! You've completed the tutorial!")
    
    with col2:
        if st.session_state.tutorial_state.topics:
            st.markdown("### 📊 Progress")
            completed = sum(1 for t in state.topics if t.completed)
            progress = int((completed / len(state.topics)) * 100)
            st.progress(progress)
            
            st.markdown("### 📑 Topics")
            for i, topic in enumerate(state.topics, 1):
                status = "✅" if topic.completed else "📍" if topic == current_topic else "⭕️"
                st.markdown(f"{status} {i}. {topic.title}")
            
            if st.button("🔄 Reset Tutorial"):
                st.session_state.tutorial_state.reset()
                st.rerun()

if __name__ == "__main__":
    main()
