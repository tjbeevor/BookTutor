import streamlit as st
import google.generativeai as genai
import PyPDF2
import io
from typing import Dict, List, Optional
from dataclasses import dataclass
import json

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
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        raise

def clean_json_string(json_str: str) -> str:
    if "```json" in json_str:
        json_str = json_str.replace("```json", "")
    if "```" in json_str:
        json_str = json_str.replace("```", "")
    return json_str.strip()

def generate_tutorial_structure(content: str, model) -> List[Topic]:
    prompt = f"""
    Analyze this educational content and create a detailed tutorial outline.
    Focus on creating a hierarchical structure with main topics and relevant subtopics.
    For each topic and subtopic, provide 2-3 sentences of content that summarize the key points.
    
    Return the structure as valid JSON in this exact format:
    {{
        "topics": [
            {{
                "title": "Main Topic Title",
                "content": "Detailed content explanation",
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

    Content to analyze: {content[:5000]}
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = clean_json_string(response.text)
        
        structure = json.loads(response_text)
        
        if "topics" not in structure or not structure["topics"]:
            raise ValueError("Invalid tutorial structure generated")
        
        def create_topics(topic_data, parent=None) -> Topic:
            topic = Topic(
                title=topic_data["title"],
                content=topic_data["content"],
                subtopics=[],
                completed=False,
                parent=parent
            )
            topic.subtopics = [create_topics(st, topic) for st in topic_data.get("subtopics", [])]
            return topic
        
        return [create_topics(t) for t in structure["topics"]]
            
    except Exception as e:
        st.error(f"Error generating tutorial structure: {str(e)}")
        raise

def generate_teaching_message(topic: Topic, phase: str, conversation_history: List[Dict], model) -> dict:
    prompt = f"""
    You are an expert tutor teaching: {topic.title}
    Content to cover: {topic.content}
    
    Create a comprehensive, engaging lesson that deeply explains the topic.
    Include rich examples and real-world applications.
    End with ONE thought-provoking question to check understanding.
    
    Return your response in this exact JSON format:
    {{
        "explanation": "Provide a detailed, well-structured explanation (4-5 paragraphs) that:
            - Starts with a clear introduction of the concept
            - Breaks down complex ideas into digestible parts
            - Explains relationships and interconnections
            - Uses clear, precise language
            - Builds understanding progressively",
        
        "examples": "Provide 2-3 detailed, real-world examples that:
            - Start with a simple, relatable example
            - Progress to more complex, nuanced examples
            - Explain why each example matters
            - Connect examples to the main concepts
            - Show practical applications",
            
        "question": "ONE thought-provoking question that:
            - Tests deeper understanding
            - Requires application of concepts
            - Encourages critical thinking",
            
        "key_points": ["3-4 main points you expect to see in a good answer"]
    }}

    Important:
    - Be thorough and detailed in explanations
    - Use clear, engaging language
    - Make real-world connections
    - Focus on depth of understanding
    - Avoid surface-level or obvious content
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.parts[0].text
        
        # Clean up the response
        response_text = clean_json_string(response_text)
        
        # Extract the first complete JSON object
        def extract_first_json(text):
            # Find the first opening brace
            start = text.find('{')
            if start == -1:
                return None
            
            # Track nested braces
            count = 0
            for i in range(start, len(text)):
                if text[i] == '{':
                    count += 1
                elif text[i] == '}':
                    count -= 1
                    if count == 0:
                        # Found complete JSON object
                        return text[start:i+1]
            return None
        
        # Get the first complete JSON object
        json_str = extract_first_json(response_text)
        if not json_str:
            raise ValueError("No valid JSON object found in response")
            
        # Clean up the extracted JSON
        json_str = json_str.replace('\n', ' ')
        json_str = json_str.replace('\r', ' ')
        json_str = json_str.replace('\t', ' ')
        json_str = json_str.replace('**', '')  # Remove markdown formatting
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse JSON response: {str(e)}")
            st.code(json_str)  # Display the problematic response
            
            # Fallback response
            return {
                "explanation": "There was an error generating the lesson content. Please try again.",
                "examples": "Examples could not be generated.",
                "question": "Please refresh the page to try again.",
                "key_points": ["Understanding of core concepts", "Application of knowledge"]
            }
            
    except Exception as e:
        st.error(f"Error generating teaching content: {str(e)}")
        raise
def evaluate_response(answer: str, expected_points: List[str], topic: Topic, model) -> dict:
    prompt = f"""
    Topic: {topic.title}
    Student's answer: {answer}
    Key points expected: {', '.join(expected_points)}
    
    Provide a helpful response in this exact format:
    {{
        "feedback": "Brief feedback on what was good/missing in the answer",
        "complete_answer": "Provide a thorough explanation of the correct answer that:
            - Covers all key points
            - Explains concepts clearly
            - Makes connections between ideas
            - Uses specific examples where helpful",
        "mastered": false
    }}
    
    Make the complete_answer thorough and educational - this is a teaching opportunity.
    Focus on explaining the correct answer rather than critiquing the student's response.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.parts[0].text
        
        # Clean up and parse response...
        evaluation = json.loads(json_str)  # (keeping existing JSON parsing code)
        
        return evaluation
            
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return {
            "feedback": "There was an error evaluating your response.",
            "complete_answer": "Let's continue with the next topic.",
            "mastered": False
        }

def main():
    st.set_page_config(page_title="Interactive AI Tutor", layout="wide")
    
    # Initialize session state
    if 'tutorial_state' not in st.session_state:
        st.session_state.tutorial_state = TutorialState()
    
    st.title("Interactive AI Tutor")
    
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # API Key Management
        api_key = None
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            api_key_input = st.text_input(
                "Please enter your Gemini API Key:",
                type="password",
                help="Get your API key from Google AI Studio"
            )
            if api_key_input:
                api_key = api_key_input
                
        if not api_key:
            st.error("No API key provided. Please set up your API key to continue.")
            st.stop()
        
        # Initialize model
        if 'model' not in st.session_state:
            st.session_state.model = init_gemini(api_key)
        
        # File upload section
        if not st.session_state.tutorial_state.topics:
            st.subheader("Upload Learning Material")
            pdf_file = st.file_uploader("Upload Educational PDF", type="pdf")
            
            if pdf_file:
                if pdf_file.size > 15 * 1024 * 1024:
                    st.error("File size exceeds 15MB limit.")
                    st.stop()
                    
                with st.spinner("Processing your learning material..."):
                    try:
                        content = process_pdf(pdf_file)
                        if not content.strip():
                            st.error("No text could be extracted from the PDF.")
                            st.stop()
                        
                        st.info("Analyzing content and creating tutorial structure...")
                        st.session_state.tutorial_state.topics = generate_tutorial_structure(
                            content, st.session_state.model
                        )
                        st.success("Tutorial structure created! Let's begin learning.")
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Error processing content: {str(e)}")
                        st.stop()
        
        # Chat interface (only shown after content is processed)
        if st.session_state.tutorial_state.topics:
            chat_container = st.container()
            state = st.session_state.tutorial_state
            current_topic = state.get_current_topic()
            
            # Display conversation history
            with chat_container:
                for message in state.conversation_history:
                    with st.chat_message(message["role"]):
                        st.write(message["content"])
            
            # Current teaching content
            if current_topic and not current_topic.completed:
                teaching_content = None
                
                # Check if we need to generate new content
                if len(state.conversation_history) == 0 or (
                    len(state.conversation_history) > 0 and
                    state.conversation_history[-1]["role"] == "user"
                ):
                    teaching_content = generate_teaching_message(
                        current_topic,
                        state.current_teaching_phase,
                        state.conversation_history,
                        st.session_state.model
                    )
                    
                    with st.chat_message("assistant"):
                        # Display lesson content
                        st.markdown("## " + current_topic.title)
                        
                        # Explanation section
                        st.markdown("### Understanding the Concepts")
                        st.markdown(teaching_content["explanation"])
                        
                        # Examples section
                        st.markdown("### Examples & Applications")
                        st.markdown(teaching_content["examples"])
                        
                        # Question section
                        st.markdown("### Knowledge Check")
                        st.markdown(teaching_content["question"])
                    
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": teaching_content["question"]
                    })
                    
                    # Store key points for evaluation
                    st.session_state.expected_points = teaching_content["key_points"]
                
                # Handle user response
                user_input = st.chat_input("Your answer...")
                if user_input:
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    
                    state.conversation_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Evaluate understanding
                    evaluation = evaluate_response(
                        user_input,
                        st.session_state.expected_points,
                        current_topic,
                        st.session_state.model
                    )
                    
                    with st.chat_message("assistant"):
                        if evaluation["feedback"]:
                            st.markdown("**Feedback on Your Response:**")
                            st.markdown(evaluation["feedback"])
                            st.markdown("---")
                        
                        st.markdown("**Complete Explanation:**")
                        st.markdown(evaluation["complete_answer"])
                        
                        st.markdown("---")
                        st.markdown("🎯 *Moving on to the next topic...*")
                    
                    # Always advance after providing the complete answer
                    current_topic.completed = True
                    if state.advance_topic():
                        st.rerun()
                    else:
                        st.balloons()
                        st.success("🎉 Congratulations! You've completed all topics!")
    
    with col2:
        if st.session_state.tutorial_state.topics:
            # Progress and topic overview
            st.subheader("Learning Progress")
            
            # Current topic info
            if current_topic:
                st.info(f"Current Topic: {current_topic.title}")
                st.write(f"Phase: {state.current_teaching_phase.title()}")
            
            # Topic tree with status indicators
            st.subheader("Topic Overview")
            for i, topic in enumerate(state.topics, 1):
                status = "✅" if topic.completed else "📍" if topic == current_topic else "⭕️"
                st.write(f"{status} {i}. {topic.title}")
                for j, subtopic in enumerate(topic.subtopics, 1):
                    status = "✅" if subtopic.completed else "📍" if subtopic == current_topic else "⭕️"
                    st.write(f"   {status} {i}.{j} {subtopic.title}")
            
            # Reset button
            if st.button("Reset Tutorial"):
                st.session_state.tutorial_state.reset()
                st.rerun()

if __name__ == "__main__":
    main()
