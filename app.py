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
        self.current_teaching_phase: str = "introduction"  # Phases: introduction, explanation, examples, practice
        self.understanding_level: int = 0  # 0-100 scale
        
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
    """
    Process PDF file and extract text with better error handling and text cleaning.
    """
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            except Exception as e:
                st.warning(f"Warning: Could not process a page in the PDF. Continuing with rest of document. Error: {str(e)}")
                continue
                
        # Clean and normalize the text
        text = text.replace('\x00', '')  # Remove null bytes
        text = ' '.join(text.split())  # Normalize whitespace
        text = text.strip()
        
        if not text:
            raise ValueError("No readable text found in the PDF")
            
        return text
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        raise

def clean_json_string(json_str: str) -> str:
    """
    More robust JSON string cleaning with better error handling.
    """
    try:
        # Remove any markdown formatting
        json_str = json_str.replace("```json", "").replace("```", "")
        
        # Find the first { and last } to extract valid JSON
        start_idx = json_str.find('{')
        end_idx = json_str.rfind('}') + 1
        
        if start_idx == -1 or end_idx == 0:
            raise ValueError("No valid JSON structure found")
            
        json_str = json_str[start_idx:end_idx]
        
        # Clean up common formatting issues
        json_str = json_str.replace('\n', ' ')
        json_str = json_str.replace('\r', ' ')
        json_str = json_str.replace('\t', ' ')
        json_str = ' '.join(json_str.split())
        
        # Fix common JSON syntax issues
        json_str = json_str.replace('} {', '},{')
        json_str = json_str.replace(']"', ']')
        json_str = json_str.replace('""', '"')
        
        # Test if it's valid JSON
        json.loads(json_str)
        return json_str
        
    except Exception as e:
        st.error(f"Error cleaning JSON: {str(e)}")
        raise

def generate_tutorial_structure(content: str, model) -> List[Topic]:
    """
    Generate a more robust tutorial structure with improved content chunking and error handling.
    """
    # Split content into manageable chunks
    def chunk_content(text: str, max_chunk_size: int = 4000) -> List[str]:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 <= max_chunk_size:
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
        # Process content in chunks
        chunks = chunk_content(content)
        all_topics = []
        
        for chunk_idx, chunk in enumerate(chunks):
            prompt = f"""
            Create a structured tutorial outline for the following content.
            Focus on identifying main topics and breaking them down into logical subtopics.
            Keep the structure focused and coherent.

            Rules:
            1. Each topic should be clear and self-contained
            2. Topics should flow logically from basic to advanced concepts
            3. Include 2-3 sentences of explanatory content for each topic/subtopic
            4. Ensure subtopics are directly related to their parent topic

            Return the structure as valid JSON in this format:
            {{
                "topics": [
                    {{
                        "title": "Main Topic Title",
                        "content": "Clear explanation of the topic",
                        "subtopics": [
                            {{
                                "title": "Subtopic Title",
                                "content": "Detailed subtopic explanation",
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
                    response_text = clean_json_string(response.text)
                    structure = json.loads(response_text)
                    
                    if "topics" not in structure or not structure["topics"]:
                        raise ValueError("Invalid tutorial structure generated")
                    
                    # Add topics from this chunk
                    all_topics.extend(structure["topics"])
                    break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        st.error(f"Failed to process chunk {chunk_idx + 1} after {max_retries} attempts")
                        raise
                    time.sleep(1)  # Brief pause before retry

        # Merge and organize topics
        def create_topics(topic_data: dict, parent: Optional[Topic] = None) -> Topic:
            topic = Topic(
                title=topic_data["title"],
                content=topic_data.get("content", ""),
                subtopics=[],
                completed=False,
                parent=parent
            )
            
            # Create subtopics
            for subtopic_data in topic_data.get("subtopics", []):
                subtopic = create_topics(subtopic_data, topic)
                topic.subtopics.append(subtopic)
                
            return topic

        # Convert all topics to Topic objects
        final_topics = [create_topics(t) for t in all_topics]
        
        # Remove duplicates and merge similar topics
        def merge_similar_topics(topics: List[Topic]) -> List[Topic]:
            merged = []
            for topic in topics:
                # Check if similar topic exists
                similar_found = False
                for existing in merged:
                    if similar_titles(existing.title, topic.title):
                        # Merge content and subtopics
                        existing.content = combine_content(existing.content, topic.content)
                        existing.subtopics.extend(topic.subtopics)
                        similar_found = True
                        break
                        
                if not similar_found:
                    merged.append(topic)
                    
            return merged

        def similar_titles(title1: str, title2: str) -> bool:
            # Simple similarity check - can be improved
            title1 = title1.lower().strip()
            title2 = title2.lower().strip()
            return (title1 in title2 or title2 in title1 or
                   len(set(title1.split()) & set(title2.split())) >= 2)

        def combine_content(content1: str, content2: str) -> str:
            # Combine unique content
            sentences1 = set(content1.split('. '))
            sentences2 = set(content2.split('. '))
            return '. '.join(sentences1 | sentences2)

        final_topics = merge_similar_topics(final_topics)
        
        if not final_topics:
            raise ValueError("No valid topics could be generated from the content")
            
        return final_topics
        
    except Exception as e:
        st.error(f"Error generating tutorial structure: {str(e)}")
        raise

def generate_teaching_message(topic: Topic, phase: str, conversation_history: List[Dict], model) -> dict:
    """
    Generate more contextual and engaging teaching content.
    """
    try:
        # Create a more detailed context from conversation history
        previous_topics = []
        for msg in conversation_history:
            if msg["role"] == "assistant" and "<h2>" in msg["content"]:
                topic_title = msg["content"].split("<h2>")[1].split("</h2>")[0]
                previous_topics.append(topic_title)

        prompt = f"""
        Create an engaging, well-structured lesson about: {topic.title}
        
        Context:
        - Main topic content: {topic.content}
        - Teaching phase: {phase}
        - Previously covered topics: {', '.join(previous_topics) if previous_topics else 'This is the first topic'}
        
        Create a lesson that:
        1. Starts with a clear introduction of the concept
        2. Builds on any previously covered topics
        3. Uses concrete, relatable examples
        4. Includes interactive elements
        5. Leads to practical applications
        
        The explanation should:
        - Break down complex ideas into digestible parts
        - Use clear, concise language
        - Include relevant analogies or comparisons
        - Connect to real-world applications
        
        The examples should:
        - Be specific and detailed
        - Range from simple to more complex
        - Connect to practical applications
        - Include step-by-step explanations where appropriate
        
        The question should:
        - Directly relate to the content just covered
        - Test understanding of key concepts
        - Encourage critical thinking
        - Allow for demonstration of practical application
        
        Return the lesson as JSON in this format:
        {{
            "explanation": "Multi-paragraph explanation with clear structure and progression",
            "examples": "2-3 specific, detailed examples with explanations",
            "question": "A thought-provoking question that tests understanding of the specific content covered",
            "key_points": ["3-4 specific key points from this lesson"]
        }}
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                response_text = clean_json_string(response.text)
                lesson_content = json.loads(response_text)
                
                # Validate content
                required_keys = ["explanation", "examples", "question", "key_points"]
                if not all(key in lesson_content for key in required_keys):
                    raise ValueError("Missing required content sections")
                    
                # Ensure content is substantial
                if len(lesson_content["explanation"]) < 100:
                    raise ValueError("Explanation too short")
                    
                return lesson_content
                
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to generate teaching content after {max_retries} attempts")
                    raise
                time.sleep(1)
                
    except Exception as e:
        st.error(f"Error generating teaching content: {str(e)}")
        return {
            "explanation": f"Let's explore {topic.title}:\n\n{topic.content}",
            "examples": "Let's look at some practical examples...",
            "question": f"Based on what we've covered about {topic.title}, explain how you would apply these concepts in a practical situation.",
            "key_points": ["Understanding core concepts", "Practical applications", "Key takeaways"]
        }
def evaluate_response(answer: str, expected_points: List[str], topic: Topic, model) -> dict:
    """
    Provide more targeted and contextual evaluation of user responses.
    """
    try:
        prompt = f"""
        Evaluate this student response about: {topic.title}
        
        Context:
        Topic Content: {topic.content}
        Expected Key Points: {', '.join(expected_points)}
        
        Student's Response: {answer}
        
        Provide an evaluation that:
        1. Specifically addresses the student's response in relation to the topic
        2. References specific parts of their answer
        3. Connects feedback to the key points and content
        4. Offers concrete suggestions for improvement
        
        The feedback should:
        - Start with positive aspects of their response
        - Point out specific connections they made correctly
        - Identify specific areas where understanding could be deepened
        - Suggest specific ways to improve understanding
        
        The complete answer should:
        - Build on correct parts of their response
        - Fill in any gaps in understanding
        - Connect back to the main topic concepts
        - Provide additional context where needed
        
        Return your evaluation as JSON in this format:
        {{
            "feedback": "Specific, constructive feedback that references their actual response",
            "complete_answer": "Comprehensive explanation that builds on their understanding",
            "mastered": boolean indicating if they demonstrated good understanding
        }}
        """
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = model.generate_content(prompt)
                response_text = clean_json_string(response.text)
                evaluation = json.loads(response_text)
                
                # Validate evaluation content
                required_keys = ["feedback", "complete_answer", "mastered"]
                if not all(key in evaluation for key in required_keys):
                    raise ValueError("Missing required evaluation sections")
                
                # Ensure feedback is substantial and specific
                if len(evaluation["feedback"]) < 100:
                    raise ValueError("Feedback too brief")
                    
                return evaluation
                
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"Failed to generate evaluation after {max_retries} attempts")
                    raise
                time.sleep(1)
                
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return {
            "feedback": f"Thank you for your thoughts on {topic.title}. Let's review the key concepts.",
            "complete_answer": f"Here's a comprehensive overview of {topic.title}:\n\n{topic.content}",
            "mastered": False
        }

def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="AI Learning Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Custom CSS
    st.markdown("""
        <style>
        .stApp {
            background-color: #f8f9fa;
        }
        .main > div {
            padding: 2rem;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            height: 3em;
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border: none;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #45a049;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        .css-1d391kg {
            padding: 2rem 1rem;
        }
        .stAlert {
            border-radius: 10px;
        }
        h1 {
            color: #2E4053;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 2rem;
        }
        h2 {
            color: #34495E;
            font-size: 1.8rem;
            font-weight: 600;
        }
        h3 {
            color: #2C3E50;
            font-size: 1.4rem;
            font-weight: 500;
        }
        .lesson-content, .evaluation-content {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: #f8f9fa;
            margin: 1rem 0;
        }
        .concept-section, .examples-section, .question-section {
            margin: 1.5rem 0;
            padding: 1rem;
            border-left: 4px solid #4CAF50;
            background-color: white;
        }
        .feedback-section, .complete-answer-section {
            margin: 1rem 0;
            padding: 1rem;
            border-left: 4px solid #2196F3;
            background-color: white;
        }
        .next-topic-prompt {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #dee2e6;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    # 3. Initialize session state
    if 'tutorial_state' not in st.session_state:
        st.session_state.tutorial_state = TutorialState()

    # 4. Header
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://via.placeholder.com/80", width=80)
    with col2:
        st.title("AI Learning Assistant")
        st.markdown("*Transform your learning experience with personalized AI guidance*")

    # 5. Main Layout
    main_content, sidebar = st.columns([7, 3])

    # 6. Main Content Area
    with main_content:
        # API Key Management
        api_key = None
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            st.markdown("""
                <div style='background-color: #F8F9FA; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                    <h3>🔑 API Configuration</h3>
                </div>
            """, unsafe_allow_html=True)
            api_key_input = st.text_input(
                "Enter your Gemini API Key:",
                type="password",
                help="Get your API key from Google AI Studio",
                placeholder="Enter your API key here..."
            )
            if api_key_input:
                api_key = api_key_input

        if not api_key:
            st.error("❌ Please provide your API key to continue")
            st.stop()

        # Initialize model
        if 'model' not in st.session_state:
            st.session_state.model = init_gemini(api_key)

        # File Upload Section
        if not st.session_state.tutorial_state.topics:
            st.markdown("""
                <div style='background-color: #F8F9FA; padding: 1rem; border-radius: 10px; margin: 2rem 0;'>
                    <h3>📚 Upload Learning Material</h3>
                    <p>Upload your educational PDF to begin the interactive learning experience.</p>
                </div>
            """, unsafe_allow_html=True)
            
            pdf_file = st.file_uploader("Upload Educational PDF", type="pdf")
            
            if pdf_file:
                if pdf_file.size > 15 * 1024 * 1024:
                    st.error("📤 File size exceeds 15MB limit")
                    st.stop()

                with st.spinner("🔄 Processing your learning material..."):
                    try:
                        content = process_pdf(pdf_file)
                        if not content.strip():
                            st.error("📄 No text could be extracted from the PDF")
                            st.stop()

                        progress_bar = st.progress(0)
                        st.info("🔍 Analyzing content and creating tutorial structure...")
                        
                        for i in range(100):
                            time.sleep(0.01)
                            progress_bar.progress(i + 1)
                        
                        st.session_state.tutorial_state.topics = generate_tutorial_structure(
                            content, st.session_state.model
                        )
                        st.success("✨ Tutorial structure created successfully! Let's begin learning.")
                        time.sleep(1)
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error processing content: {str(e)}")
                        st.stop()

        # Chat Interface
        if st.session_state.tutorial_state.topics:
            chat_container = st.container()
            state = st.session_state.tutorial_state
            current_topic = state.get_current_topic()

            with chat_container:
                for message in state.conversation_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"], unsafe_allow_html=True)

            if current_topic and not current_topic.completed:
                teaching_content = None
                
                if len(state.conversation_history) == 0 or (
                    len(state.conversation_history) > 0 and 
                    state.conversation_history[-1]["role"] == "assistant" and 
                    "Moving on to the next topic" in state.conversation_history[-1]["content"]
                ):
                    teaching_content = generate_teaching_message(
                        current_topic,
                        state.current_teaching_phase,
                        state.conversation_history,
                        st.session_state.model
                    )
                    
                    lesson_content = f"""
                    <div class="lesson-content">
                        <h2>{current_topic.title}</h2>
                        <div class="concept-section">
                            <h3>📚 Understanding the Concepts</h3>
                            {teaching_content["explanation"]}
                        </div>
                        <div class="examples-section">
                            <h3>🔍 Practical Examples</h3>
                            {teaching_content["examples"]}
                        </div>
                        <div class="question-section">
                            <h3>💡 Understanding Check</h3>
                            {teaching_content["question"]}
                        </div>
                    </div>
                    """
                    
                    with st.chat_message("assistant"):
                        st.markdown(lesson_content, unsafe_allow_html=True)
                    
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": lesson_content
                    })
                    
                    st.session_state.expected_points = teaching_content["key_points"]
                
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
                    
                    evaluation_response = f"""
                    <div class="evaluation-content">
                        <div class="feedback-section">
                            <h3>💭 Feedback on Your Response</h3>
                            {evaluation['feedback']}
                        </div>
                        <div class="complete-answer-section">
                            <h3>📝 Complete Explanation</h3>
                            {evaluation['complete_answer']}
                        </div>
                        <div class="next-topic-prompt">
                            <hr>
                            <p><em>🎯 Moving on to the next topic...</em></p>
                        </div>
                    </div>
                    """
                    
                    with st.chat_message("assistant"):
                        st.markdown(evaluation_response, unsafe_allow_html=True)
                    
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": evaluation_response
                    })
                    
                    current_topic.completed = True
                    
                    if state.advance_topic():
                        st.rerun()
                    else:
                        st.balloons()
                        st.markdown("""
                        <div class="completion-message">
                            <h2>🎉 Congratulations!</h2>
                            <p>You've successfully completed the tutorial!</p>
                        </div>
                        """, unsafe_allow_html=True)

    # 7. Sidebar
    with sidebar:
        if st.session_state.tutorial_state.topics:
            st.markdown("""
                <div style='background-color: #F8F9FA; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
                    <h3>📊 Learning Progress</h3>
                </div>
            """, unsafe_allow_html=True)

            if current_topic:
                completed_topics = sum(1 for t in st.session_state.tutorial_state.topics if t.completed)
                total_topics = len(st.session_state.tutorial_state.topics)
                progress = int((completed_topics / total_topics) * 100)
                
                st.progress(progress)
                st.info(f"📍 Current Topic: {current_topic.title}")
                st.write(f"🎯 Phase: {state.current_teaching_phase.title()}")

            st.markdown("""
                <div style='background-color: #F8F9FA; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                    <h3>📑 Topic Overview</h3>
                </div>
            """, unsafe_allow_html=True)
            
            for i, topic in enumerate(state.topics, 1):
                status = "✅" if topic.completed else "📍" if topic == current_topic else "⭕️"
                st.markdown(f"<div style='margin: 0.5rem 0;'>{status} **{i}. {topic.title}**</div>", unsafe_allow_html=True)
                for j, subtopic in enumerate(topic.subtopics, 1):
                    status = "✅" if subtopic.completed else "📍" if subtopic == current_topic else "⭕️"
                    st.markdown(f"<div style='margin: 0.3rem 0 0.3rem 2rem;'>{status} {i}.{j} {subtopic.title}</div>", unsafe_allow_html=True)

            st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
            if st.button("🔄 Reset Tutorial"):
                st.session_state.tutorial_state.reset()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
