import streamlit as st
import google.generativeai as genai
import PyPDF2
import io
from typing import Dict, List, Optional
import os
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
        self.current_subtopic_index: int = -1  # -1 means we're on the main topic
        self.questions_asked: int = 0
        self.max_questions_per_topic: int = 3
        self.current_path: List[int] = []  # Tracks the current topic/subtopic path
        self.conversation_history: List[Dict] = []

    def reset(self):
        self.current_topic_index = 0
        self.current_subtopic_index = -1
        self.current_path = []
        self.questions_asked = 0
        self.conversation_history = []
        for topic in self.topics:
            topic.completed = False
            for subtopic in topic.subtopics:
                subtopic.completed = False

    def get_current_topic(self) -> Optional[Topic]:
        if not self.topics:
            return None
        if self.current_subtopic_index == -1:
            return self.topics[self.current_topic_index]
        return self.topics[self.current_topic_index].subtopics[self.current_subtopic_index]

    def advance(self):
        current_topic = self.topics[self.current_topic_index]
        
        # If we're on the main topic and haven't started subtopics
        if self.current_subtopic_index == -1:
            if current_topic.subtopics:
                # Move to first subtopic
                self.current_subtopic_index = 0
                return True
            else:
                # No subtopics, move to next main topic
                if self.current_topic_index < len(self.topics) - 1:
                    self.current_topic_index += 1
                    return True
                return False
        
        # If we're on subtopics
        if self.current_subtopic_index < len(current_topic.subtopics) - 1:
            # Move to next subtopic
            self.current_subtopic_index += 1
            return True
        else:
            # Finished all subtopics, move to next main topic
            self.current_topic_index += 1
            self.current_subtopic_index = -1
            return self.current_topic_index < len(self.topics)

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
        structure = json.loads(response.text.strip())
        
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
        st.code(response.text)
        raise

def generate_teaching_content(topic: Topic, conversation_history: List[Dict], model) -> str:
    # Prepare conversation context
    context = "\n".join([
        f"{'AI' if msg['role'] == 'assistant' else 'Student'}: {msg['content']}"
        for msg in conversation_history[-3:]  # Include last 3 messages for context
    ])
    
    prompt = f"""
    You are an expert tutor teaching this topic: {topic.title}
    
    Previous conversation context:
    {context}
    
    Content to teach: {topic.content}
    
    Create an engaging lesson that includes:
    1. A brief introduction to catch interest
    2. Clear explanation of the main concepts
    3. Real-world examples or analogies
    4. A thought-provoking question to check understanding
    
    Keep the tone conversational and encouraging. End with exactly one clear question.
    """
    
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        st.error(f"Error generating teaching content: {str(e)}")
        raise

def evaluate_answer(answer: str, topic: Topic, conversation_history: List[Dict], model) -> tuple[bool, str]:
    context = "\n".join([
        f"{'AI' if msg['role'] == 'assistant' else 'Student'}: {msg['content']}"
        for msg in conversation_history[-3:]
    ])
    
    prompt = f"""
    Topic being taught: {topic.title}
    Content: {topic.content}
    
    Previous conversation context:
    {context}
    
    Student's answer: {answer}
    
    Evaluate the student's understanding. Be encouraging but thorough.
    Respond in exactly one of these formats:
    UNDERSTOOD: [Explanation of what they got right and any minor points to reinforce]
    or
    NOT_UNDERSTOOD: [Supportive explanation of misconceptions and a helpful hint or explanation]
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        if response_text.startswith("UNDERSTOOD:"):
            return True, response_text[11:].strip()
        elif response_text.startswith("NOT_UNDERSTOOD:"):
            return False, response_text[14:].strip()
        else:
            return False, "Let's try to understand this topic better."
    except Exception as e:
        st.error(f"Error evaluating answer: {str(e)}")
        raise

def main():
    st.set_page_config(page_title="AI Educational Tutor", layout="wide")
    
    st.title("AI Educational Tutor")
    
    # Initialize session state
    if 'tutorial_state' not in st.session_state:
        st.session_state.tutorial_state = TutorialState()
    if 'show_topic_tree' not in st.session_state:
        st.session_state.show_topic_tree = False
    
    # Create two columns
    col1, col2 = st.columns([7, 3])
    
    with col1:
        # API Key Management
        api_key = None
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except KeyError:
            st.warning("⚠️ Gemini API Key not found in secrets.")
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
        
        # File upload
        pdf_file = st.file_uploader("Upload Educational PDF", type="pdf")
        
        if pdf_file:
            if pdf_file.size > 15 * 1024 * 1024:
                st.error("File size exceeds 15MB limit.")
                return
                
            # Process PDF if not already processed
            if not st.session_state.tutorial_state.topics:
                with st.spinner("Processing PDF content..."):
                    try:
                        content = process_pdf(pdf_file)
                        if not content.strip():
                            st.error("No text could be extracted from the PDF.")
                            return
                        
                        st.info("Generating tutorial structure...")
                        st.session_state.tutorial_state.topics = generate_tutorial_structure(
                            content, st.session_state.model
                        )
                        st.success("Tutorial structure generated!")
                    except Exception as e:
                        st.error(f"Error processing content: {str(e)}")
                        return
            
            # Display current topic
            state = st.session_state.tutorial_state
            current_topic = state.get_current_topic()
            
            if current_topic:
                # Progress tracking
                total_topics = sum(1 + len(t.subtopics) for t in state.topics)
                completed_topics = sum(
                    (1 if t.completed else 0) + 
                    sum(1 if st.completed else 0 for st in t.subtopics)
                    for t in state.topics
                )
                progress = (completed_topics / total_topics) * 100
                
                st.progress(progress)
                st.write(f"Progress: {progress:.1f}% ({completed_topics}/{total_topics} topics)")
                
                # Topic navigation
                st.subheader(f"Current Topic: {current_topic.title}")
                
                if not current_topic.completed:
                    # Generate and display teaching content
                    teaching_content = generate_teaching_content(
                        current_topic,
                        state.conversation_history,
                        st.session_state.model
                    )
                    st.markdown(teaching_content)
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": teaching_content
                    })
                    
                    # Get and evaluate student's answer
                    student_answer = st.text_area("Your Answer:", key=f"answer_{len(state.conversation_history)}")
                    if st.button("Submit", key=f"submit_{len(state.conversation_history)}"):
                        state.conversation_history.append({
                            "role": "user",
                            "content": student_answer
                        })
                        
                        understood, explanation = evaluate_answer(
                            student_answer,
                            current_topic,
                            state.conversation_history,
                            st.session_state.model
                        )
                        
                        if understood:
                            st.success(explanation)
                            current_topic.completed = True
                            if not state.advance():
                                st.balloons()
                                st.success("🎉 Congratulations! You've completed the tutorial!")
                            st.experimental_rerun()
                        else:
                            st.warning(explanation)
                            state.questions_asked += 1
                            if state.questions_asked >= state.max_questions_per_topic:
                                st.info("Let's move on to ensure we cover all topics.")
                                current_topic.completed = True
                                if not state.advance():
                                    st.success("Tutorial completed!")
                                st.experimental_rerun()
    
    with col2:
        # Topic Tree and Controls
        st.subheader("Tutorial Overview")
        
        # Reset button
        if st.button("Reset Tutorial"):
            st.session_state.tutorial_state.reset()
            st.experimental_rerun()
        
        # Toggle topic tree
        st.session_state.show_topic_tree = st.toggle("Show Topic Tree", st.session_state.show_topic_tree)
        
        if st.session_state.show_topic_tree and st.session_state.tutorial_state.topics:
            st.write("Topic Structure:")
            for i, topic in enumerate(st.session_state.tutorial_state.topics, 1):
                status = "✅" if topic.completed else "📍" if i-1 == state.current_topic_index else "⭕️"
                st.markdown(f"{status} **{i}. {topic.title}**")
                for j, subtopic in enumerate(topic.subtopics, 1):
                    status = "✅" if subtopic.completed else "📍" if (i-1 == state.current_topic_index and j-1 == state.current_subtopic_index) else "⭕️"
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;{status} {i}.{j} {subtopic.title}")

if __name__ == "__main__":
    main()
