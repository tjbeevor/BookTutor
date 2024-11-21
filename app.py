import streamlit as st
import google.generativeai as genai
import PyPDF2
import io
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import time
import random

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
        # [Previous prompt code remains the same]
        
        response = model.generate_content(prompt)
        content = json.loads(clean_json_string(response.text))
        
        # Improved formatting with better error handling
        try:
            overview_section = f"""
# {topic.title} 📚

## Overview 📋
{content.get('overview', {}).get('basic_definition', 'No basic definition provided.')}

{content.get('overview', {}).get('context', 'No context provided.')}

First aid skills are particularly valuable because:
{"".join(f"- {benefit}\n" for benefit in content.get('overview', {}).get('key_benefits', ['No benefits provided.']))}
"""

            statistics_section = """
Statistics show that immediate first aid can significantly improve outcomes in emergencies:
""" + "".join(f"- {stat.get('stat', '')}: {stat.get('impact', '')}\n" 
              for stat in content.get('overview', {}).get('statistics', []))

            key_points_section = """
## Key Points 🎯
""" + "".join(f"• **{point.get('point', '')}**: {point.get('explanation', '')}\n" 
              for point in content.get('key_points', []))

            detailed_section = """
## Detailed Explanation 📝
"""
            for concept in content.get('detailed_explanation', {}).get('main_concepts', []):
                detailed_section += f"""
### {concept.get('title', 'Untitled Concept')}
{concept.get('description', 'No description provided.')}

Key steps:
{"".join(f"- {step}\n" for step in concept.get('steps', ['No steps provided.']))}

Important notes:
{"".join(f"- {note}\n" for note in concept.get('important_notes', ['No notes provided.']))}
"""

            example_section = f"""
## Real-World Example 💡
> **Scenario**  
> {content.get('example_scenario', {}).get('situation', 'No scenario provided.')}

> **Application**  
{"".join(f"> {i+1}. {step}\n" for i, step in enumerate(content.get('example_scenario', {}).get('application_steps', ['No steps provided.'])))}

> **Critical Decision Points**  
{"".join(f"> - {decision}\n" for decision in content.get('example_scenario', {}).get('critical_decisions', ['No decision points provided.']))}

> **Outcome**  
> Immediate Results: {content.get('example_scenario', {}).get('outcome', {}).get('immediate_results', 'No immediate results provided.')}
> 
> Long-term Impact: {content.get('example_scenario', {}).get('outcome', {}).get('long_term_impact', 'No long-term impact provided.')}
> 
> Lessons Learned:
{"".join(f"> - {lesson}\n" for lesson in content.get('example_scenario', {}).get('outcome', {}).get('lessons_learned', ['No lessons provided.']))}
"""

            practice_section = f"""
## Practice Question ✍️
{content.get('practice_question', {}).get('main_question', 'No main question provided.')}

Additional considerations:
{"".join(f"{i+1}. {question}\n" for i, question in enumerate(content.get('practice_question', {}).get('sub_questions', ['No sub-questions provided.'])))}

Key points to address:
{"".join(f"- {point}\n" for point in content.get('practice_question', {}).get('key_considerations', ['No key considerations provided.']))}

---
**Remember**: {random.choice([
    "Knowledge and preparation are key to effective response.",
    "Stay calm and follow the procedures you've learned.",
    "Every second counts - act decisively but carefully.",
    "Your actions can make a life-changing difference."
])}
"""

            # Combine all sections
            formatted_content = (
                overview_section + 
                statistics_section + 
                key_points_section + 
                detailed_section + 
                example_section + 
                practice_section
            )

            return {
                "content": formatted_content,
                "examples": content.get('example_scenario', {}),
                "question": content.get('practice_question', {}).get('main_question', 'No question provided.'),
                "key_points": content.get('expected_points', ['No points provided.'])
            }

        except KeyError as ke:
            st.warning(f"Missing expected data in content structure: {str(ke)}")
            # Provide simplified but complete content when data is missing
            return create_fallback_content(topic)
            
        except Exception as e:
            st.warning(f"Error formatting content: {str(e)}")
            return create_fallback_content(topic)
            
    except Exception as e:
        st.error(f"Error generating teaching message: {str(e)}")
        return create_fallback_content(topic)

def create_fallback_content(topic: Topic) -> dict:
    """Create a basic content structure when the main generation fails."""
    return {
        "content": f"""
# {topic.title} 📚

## Overview 📋
{topic.content}

## Key Points 🎯
• Basic understanding of the topic is essential
• Practice and repetition help master the concepts
• Seek additional resources for deeper understanding

## Practice Question ✍️
Can you explain the main concepts of {topic.title}?
""",
        "examples": {},
        "question": f"Can you explain the key points of {topic.title}?",
        "key_points": ["Basic understanding of concepts"]
    }
def main():
    # Page configuration
    st.set_page_config(
        page_title="AI Learning Assistant 📚",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for improved styling
    st.markdown("""
        <style>
        /* Global Styles */
        .main {
            padding: 1rem;
        }
        
        /* Typography */
        .main h1 {
            color: #1E3A8A;
            padding: 1.5rem 0;
            border-bottom: 2px solid #E5E7EB;
            margin-bottom: 2rem;
        }
        .main h2 {
            color: #2563EB;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            font-size: 1.5rem;
        }
        .main h3 {
            color: #3B82F6;
            margin-top: 1.25rem;
            font-size: 1.25rem;
        }
        
        /* Content Blocks */
        .content-block {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #E5E7EB;
            margin-bottom: 1rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        
        /* Chat Messages */
        .chat-message {
            padding: 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border: 1px solid #E5E7EB;
        }
        .chat-message.assistant {
            background: #F8FAFC;
        }
        .chat-message.user {
            background: #F0F9FF;
        }
        
        /* Progress Bar */
        .stProgress .st-bo {
            background-color: #3B82F6;
            height: 8px;
            border-radius: 4px;
        }
        
        /* Topic List */
        .topic-item {
            padding: 0.75rem;
            margin: 0.25rem 0;
            border-radius: 4px;
            transition: all 0.2s;
        }
        .topic-item:hover {
            background: #F3F4F6;
        }
        .topic-item.active {
            background: #EFF6FF;
            border-left: 3px solid #3B82F6;
        }
        
        /* Blockquotes for examples */
        blockquote {
            background: #F3F4F6;
            border-left: 4px solid #3B82F6;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }
        
        /* Custom file uploader */
        .stFileUploader {
            padding: 1rem;
            border: 2px dashed #E5E7EB;
            border-radius: 8px;
            text-align: center;
        }
        
        /* Custom buttons */
        .stButton button {
            background: #3B82F6;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            border: none;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }
        .stButton button:hover {
            background: #2563EB;
        }
        </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'tutorial_state' not in st.session_state:
        st.session_state.tutorial_state = TutorialState()
    
    if 'expected_points' not in st.session_state:
        st.session_state.expected_points = []

    # Header Section
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>🎓 AI Learning Assistant</h1>
            <p style='font-size: 1.2rem; color: #4B5563;'>Transform your learning experience with personalized AI guidance</p>
        </div>
    """, unsafe_allow_html=True)

    # API Key Management
    api_key = None
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
    except:
        with st.expander("🔑 API Key Configuration"):
            api_key = st.text_input("Enter your Gemini API Key:", type="password")

    if not api_key:
        st.error("⚠️ Please provide your API key to continue")
        st.stop()

    # Initialize model
    if 'model' not in st.session_state:
        st.session_state.model = init_gemini(api_key)

    if not st.session_state.model:
        st.error("❌ Failed to initialize AI model")
        st.stop()

    # Main Layout
    col1, col2 = st.columns([7, 3])

    with col1:
        if not st.session_state.tutorial_state.topics:
            st.markdown("""
                <div class='content-block'>
                    <h2>📚 Upload Learning Material</h2>
                    <p>Upload your educational content to begin the learning journey.</p>
                </div>
            """, unsafe_allow_html=True)
            
            pdf_file = st.file_uploader("Upload Educational PDF", type="pdf")
            
            if pdf_file:
                with st.spinner("🔄 Processing content..."):
                    try:
                        content = process_pdf(pdf_file)
                        st.session_state.tutorial_state.topics = generate_tutorial_structure(
                            content, st.session_state.model
                        )
                        st.success("✅ Tutorial created successfully! Let's begin.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
        else:
            # Chat Interface
            state = st.session_state.tutorial_state
            current_topic = state.get_current_topic()

            if current_topic and not current_topic.completed:
                # Display conversation history
                for message in state.conversation_history:
                    message_class = "assistant" if message["role"] == "assistant" else "user"
                    with st.chat_message(message["role"]):
                        st.markdown(f"<div class='chat-message {message_class}'>{message['content']}</div>", 
                                  unsafe_allow_html=True)

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
                    
                    with st.chat_message("assistant"):
                        st.markdown(teaching_content["content"], unsafe_allow_html=True)
                    
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": teaching_content["content"]
                    })
                    
                    st.session_state.expected_points = teaching_content["key_points"]

                # Handle user input
                user_input = st.chat_input("Your response...")
                if user_input:
                    with st.chat_message("user"):
                        st.markdown(f"<div class='chat-message user'>{user_input}</div>", 
                                  unsafe_allow_html=True)
                    
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
                    
                    if evaluation['understanding_level'] >= 70:
                        current_topic.completed = True
                        if state.advance_topic():
                            st.rerun()
                        else:
                            st.balloons()
                            st.success("🎉 Congratulations! You've completed the tutorial!")
            else:
                st.success("🎉 All topics completed! Start a new tutorial or reset the current one.")

    with col2:
        if st.session_state.tutorial_state.topics:
            # Progress Section
            st.markdown("""
                <div class='content-block'>
                    <h2>📊 Learning Progress</h2>
                </div>
            """, unsafe_allow_html=True)
            
            completed = sum(1 for t in state.topics if t.completed)
            progress = int((completed / len(state.topics)) * 100)
            st.progress(progress)
            st.markdown(f"**{progress}%** completed ({completed}/{len(state.topics)} topics)")
            
            # Topics Section
            st.markdown("""
                <div class='content-block'>
                    <h2>📑 Topics Overview</h2>
                </div>
            """, unsafe_allow_html=True)
            
            for i, topic in enumerate(state.topics, 1):
                status = "✅" if topic.completed else "📍" if topic == current_topic else "⭕️"
                topic_class = "active" if topic == current_topic else ""
                st.markdown(
                    f"<div class='topic-item {topic_class}'>{status} {i}. {topic.title}</div>", 
                    unsafe_allow_html=True
                )
            
            # Reset Button
            if st.button("🔄 Reset Tutorial", key="reset_button"):
                st.session_state.tutorial_state.reset()
                st.rerun()

if __name__ == "__main__":
    main()
