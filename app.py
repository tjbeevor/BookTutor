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
    if api_key:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-pro')
    return None

def process_pdf(pdf_file) -> str:
    """Process PDF file and extract text with better error handling."""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = " ".join(
            page.extract_text().strip()
            for page in pdf_reader.pages
            if page.extract_text()
        )
        
        if not text:
            raise ValueError("No readable text found in the PDF")
            
        return text
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        raise

def clean_json_string(json_str: str) -> str:
    """Clean and validate JSON string."""
    try:
        # Remove markdown and find valid JSON
        json_str = json_str.replace("```json", "").replace("```", "")
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No valid JSON structure found")
            
        json_str = json_str[start_idx:end_idx]
        
        # Clean up formatting
        json_str = ' '.join(json_str.split())
        
        # Validate JSON
        json.loads(json_str)
        return json_str
        
    except Exception as e:
        st.error(f"Error cleaning JSON: {str(e)}")
        raise

def generate_tutorial_structure(content: str, model) -> List[Topic]:
    """Generate tutorial structure from content."""
    def chunk_content(text: str, max_size: int = 4000) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_size:
                current_chunk.append(word)
                current_length += len(word) + 1
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
                
        if current_chunk:
            chunks.append(' '.join(current_chunk))
            
        return chunks

    try:
        chunks = chunk_content(content)
        all_topics = []
        
        for chunk_idx, chunk in enumerate(chunks):
            prompt = f"""
            Create a structured tutorial outline for the following content.
            Focus on main topics and logical subtopics.
            Return valid JSON in this format:
            {{
                "topics": [
                    {{
                        "title": "Topic Title",
                        "content": "Topic explanation",
                        "subtopics": [
                            {{
                                "title": "Subtopic Title",
                                "content": "Subtopic explanation",
                                "subtopics": []
                            }}
                        ]
                    }}
                ]
            }}

            Content chunk {chunk_idx + 1}:
            {chunk}
            """
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(prompt)
                    structure = json.loads(clean_json_string(response.text))
                    
                    if "topics" not in structure or not structure["topics"]:
                        raise ValueError("Invalid tutorial structure generated")
                    
                    all_topics.extend(structure["topics"])
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)

        def create_topic(topic_data: dict, parent: Optional[Topic] = None) -> Topic:
            topic = Topic(
                title=topic_data["title"],
                content=topic_data["content"],
                subtopics=[],
                parent=parent
            )
            
            for subtopic_data in topic_data.get("subtopics", []):
                subtopic = create_topic(subtopic_data, topic)
                topic.subtopics.append(subtopic)
                
            return topic

        topics = [create_topic(t) for t in all_topics]
        
        if not topics:
            raise ValueError("No valid topics generated")
            
        return topics
        
    except Exception as e:
        st.error(f"Error generating tutorial structure: {str(e)}")
        raise

def generate_teaching_message(topic: Topic, phase: str, conversation_history: List[Dict], model) -> dict:
    """
    Generate teaching content while maintaining proper structure and phase progression.
    """
    previous_topics = [
        msg["content"].split("<h2>")[1].split("</h2>")[0]
        for msg in conversation_history 
        if msg["role"] == "assistant" and "<h2>" in msg["content"]
    ]

    prompt = f"""
    Create a lesson about {topic.title} for the {phase} phase.
    
    Context:
    - Topic content: {topic.content}
    - Teaching phase: {phase}
    - Previous topics: {', '.join(previous_topics) if previous_topics else 'First topic'}
    
    Return the lesson in this exact JSON format:
    {{
        "explanation": "Main content with clear sections",
        "examples": [
            {{
                "title": "Example title",
                "description": "Scenario description",
                "steps": ["Step 1", "Step 2", "Step 3"]
            }}
        ],
        "question": "Understanding check question",
        "key_points": ["Point 1", "Point 2", "Point 3"]
    }}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            lesson_json = clean_json_string(response.text)
            lesson = json.loads(lesson_json)
            
            # Validate required fields
            required_fields = ["explanation", "examples", "question", "key_points"]
            if not all(field in lesson for field in required_fields):
                raise ValueError("Missing required fields in lesson content")
            
            # Format content with proper HTML structure while maintaining phase logic
            formatted_content = f"""
            <div class="lesson-container">
                <h2>{topic.title}</h2>
                
                <section class="phase-content">
                    <h3>📚 {phase.title()} Phase</h3>
                    {lesson['explanation']}
                </section>
                
                <section class="examples">
                    <h3>🔍 Examples</h3>
                    {''.join(f"""
                        <div class="example">
                            <h4>{example['title']}</h4>
                            <p>{example['description']}</p>
                            <ul>
                                {''.join(f'<li>{step}</li>' for step in example['steps'])}
                            </ul>
                        </div>
                    """ for example in lesson['examples'])}
                </section>
                
                <section class="understanding-check">
                    <h3>💡 Understanding Check</h3>
                    <p>{lesson['question']}</p>
                    <div class="key-points">
                        <h4>Key Points:</h4>
                        <ul>
                            {''.join(f'<li>{point}</li>' for point in lesson['key_points'])}
                        </ul>
                    </div>
                </section>
            </div>
            """
            
            return {
                "content": formatted_content,
                "examples": lesson["examples"],
                "question": lesson["question"],
                "key_points": lesson["key_points"]
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Error generating lesson content: {str(e)}")
                return generate_fallback_content(topic, phase)
            time.sleep(1)

def evaluate_response(user_response: str, expected_points: List[str], current_topic: Topic, model) -> dict:
    """
    Evaluate user response while maintaining tutorial flow.
    """
    prompt = f"""
    Evaluate this response about {current_topic.title}.
    
    Topic Content:
    {current_topic.content}
    
    Expected Key Points:
    {json.dumps(expected_points, indent=2)}
    
    User's Response:
    {user_response}
    
    Provide evaluation in this exact JSON format:
    {{
        "feedback": "Specific feedback on response",
        "complete_answer": "Model answer incorporating key points",
        "understanding_level": "Number between 0-100"
    }}
    """
    
    try:
        response = model.generate_content(prompt)
        evaluation = json.loads(clean_json_string(response.text))
        
        formatted_feedback = f"""
        <div class="evaluation-container">
            <section class="feedback">
                <h3>💭 Feedback</h3>
                <p>{evaluation['feedback']}</p>
            </section>
            
            <section class="complete-answer">
                <h3>📝 Complete Answer</h3>
                <p>{evaluation['complete_answer']}</p>
            </section>
            
            <section class="progress">
                <div class="progress-indicator">
                    Understanding Level: {evaluation['understanding_level']}%
                </div>
                <hr>
                <p><em>🎯 Ready to continue learning...</em></p>
            </section>
        </div>
        """
        
        return {
            "feedback": formatted_feedback,
            "understanding_level": int(evaluation['understanding_level'])
        }
        
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return generate_fallback_evaluation(current_topic)

def generate_fallback_content(topic: Topic, phase: str) -> dict:
    """
    Provide fallback content if generation fails.
    """
    return {
        "content": f"""
        <div class="lesson-container">
            <h2>{topic.title}</h2>
            <section class="phase-content">
                <h3>📚 {phase.title()} Phase</h3>
                <p>{topic.content}</p>
            </section>
            <section class="understanding-check">
                <h3>💡 Understanding Check</h3>
                <p>Please explain your understanding of {topic.title}.</p>
            </section>
        </div>
        """,
        "examples": [{"title": "Basic Example", "steps": ["Review content", "Practice concepts"]}],
        "question": f"Explain your understanding of {topic.title}.",
        "key_points": ["Understanding core concepts", "Practical application"]
    }

def generate_fallback_evaluation(topic: Topic) -> dict:
    """
    Provide fallback evaluation if evaluation fails.
    """
    return {
        "feedback": f"""
        <div class="evaluation-container">
            <section class="feedback">
                <h3>💭 Feedback</h3>
                <p>Thank you for your response. Let's review the key concepts of {topic.title}.</p>
            </section>
            <section class="complete-answer">
                <h3>📝 Complete Answer</h3>
                <p>{topic.content}</p>
            </section>
            <hr>
            <p><em>🎯 Ready to continue learning...</em></p>
        </div>
        """,
        "understanding_level": 50
    }

# In the main() function, update the chat interface section:
if current_topic and not current_topic.completed:
    # Generate teaching content when needed
    if len(state.conversation_history) == 0 or (
        state.conversation_history[-1]["role"] == "assistant" and 
        "Ready to continue learning" in state.conversation_history[-1]["content"]
    ):
        teaching_content = generate_teaching_message(
            current_topic,
            state.current_teaching_phase,
            state.conversation_history,
            st.session_state.model
        )
        
        with st.chat_message("assistant"):
            st.markdown(teaching_content["content"], unsafe_allow_html=True)
        
        state.conversation_history.append({
            "role": "assistant",
            "content": teaching_content["content"]
        })
        st.session_state.expected_points = teaching_content["key_points"]
        
    # Handle user input and evaluation
    user_input = st.chat_input("Share your thoughts...")
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
        
        with st.chat_message("assistant"):
            st.markdown(evaluation["feedback"], unsafe_allow_html=True)
            
        state.conversation_history.append({
            "role": "assistant",
            "content": evaluation["feedback"]
        })
        
        current_topic.completed = True
        
        if state.advance_topic():
            st.rerun()
        else:
            st.balloons()
            st.success("🎉 Congratulations! You've completed the tutorial!")

def main():
    st.set_page_config(
        page_title="AI Learning Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if 'tutorial_state' not in st.session_state:
        st.session_state.tutorial_state = TutorialState()

    # Page header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://via.placeholder.com/80", width=80)
    with col2:
        st.title("AI Learning Assistant")
        st.markdown("*Transform your learning experience with personalized AI guidance*")

    # Main layout
    main_content, sidebar = st.columns([7, 3])

    # Main content area
    with main_content:
        # Handle API key
        api_key = st.secrets.get("GEMINI_API_KEY") or st.text_input(
            "Enter your Gemini API Key:",
            type="password",
            help="Get your API key from Google AI Studio"
        )

        if not api_key:
            st.error("❌ Please provide your API key to continue")
            st.stop()

        # Initialize model
        if 'model' not in st.session_state:
            st.session_state.model = init_gemini(api_key)

        # File upload and processing
        if not st.session_state.tutorial_state.topics:
            st.markdown("""
                <div class="upload-section">
                    <h3>📚 Upload Learning Material</h3>
                    <p>Upload your educational PDF to begin.</p>
                </div>
            """, unsafe_allow_html=True)
            
            pdf_file = st.file_uploader("Upload Educational PDF", type="pdf")
            
            if pdf_file:
                if pdf_file.size > 15 * 1024 * 1024:
                    st.error("📤 File size exceeds 15MB limit")
                    st.stop()

                with st.spinner("🔄 Processing your material..."):
                    try:
                        content = process_pdf(pdf_file)
                        progress_bar = st.progress(0)
                        
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        st.session_state.tutorial_state.topics = generate_tutorial_structure(
                            content, st.session_state.model
                        )
                        st.success("✨ Tutorial structure created! Let's begin learning.")
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        st.stop()

        # Chat interface
        if st.session_state.tutorial_state.topics:
            chat_container = st.container()
            state = st.session_state.tutorial_state
            current_topic = state.get_current_topic()

            with chat_container:
                for message in state.conversation_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)

            if current_topic and not current_topic.completed:
                [Previous chat interface code goes here]

    # Sidebar
    with sidebar:
        if st.session_state.tutorial_state.topics:
            [Previous sidebar code goes here]

if __name__ == "__main__":
    main()
