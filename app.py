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
    """Clean and format JSON string to ensure valid parsing."""
    # Remove markdown code block indicators
    json_str = json_str.replace("```json", "").replace("```", "")
    
    # Replace numbered list items with actual strings
    json_str = json_str.replace("[1.", '["').replace("2.", '","').replace("3.", '","').replace("4.", '","')
    if json_str.count('[') > json_str.count(']'):
        json_str = json_str + '"]'
    
    # Fix common formatting issues
    json_str = json_str.replace('\n', ' ')
    json_str = json_str.replace('\r', ' ')
    json_str = json_str.replace('\t', ' ')
    json_str = json_str.replace('**', '')
    
    # Fix list formatting
    json_str = json_str.replace('* ', '')
    
    # Remove multiple spaces
    json_str = ' '.join(json_str.split())
    
    # Fix common delimiter issues
    json_str = json_str.replace('} {', '},{')
    
    # Clean up array delimiters
    json_str = json_str.replace('[ ', '[').replace(' ]', ']')
    
    return json_str

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
    Create a structured lesson about: {topic.title}
    Main content: {topic.content}

    Provide your response in this EXACT format without any deviations:
    {{
        "explanation": "Write a detailed, multi-paragraph explanation here that introduces and explains the key concepts",
        "examples": "Provide 2-3 specific, real-world examples that illustrate these concepts",
        "question": "Write a specific question that tests understanding of both the concepts and examples provided",
        "key_points": ["key point 1", "key point 2", "key point 3"]
    }}

    Your response should meet these criteria:

    1. The explanation must:
       - Introduce the topic clearly
       - Break down key concepts
       - Build understanding progressively
       - Connect ideas logically
       - Use clear, precise language

    2. The examples must:
       - Directly relate to the explanation
       - Show real-world applications
       - Move from simple to complex
       - Demonstrate practical relevance

    3. The question must:
       - Test understanding of both concepts and examples covered
       - Be specific to what was taught
       - Require analytical thinking

    CRITICAL: Return ONLY the JSON object. No additional text or formatting.
    """

    try:
        response = model.generate_content(prompt)
        response_text = response.parts[0].text.strip()
        
        # Remove any markdown formatting or extra content
        if '```json' in response_text:
            response_text = response_text.split('```json')[1].split('```')[0]
        
        # Ensure we have a clean JSON string
        response_text = response_text.strip()
        if not response_text.startswith('{'):
            response_text = response_text[response_text.find('{'):]
        if not response_text.endswith('}'):
            response_text = response_text[:response_text.rfind('}')+1]

        try:
            lesson_content = json.loads(response_text)
            
            # Validate the content structure
            required_keys = ["explanation", "examples", "question", "key_points"]
            if not all(key in lesson_content for key in required_keys):
                raise ValueError("Missing required content sections")
                
            return lesson_content

        except json.JSONDecodeError:
            # If JSON parsing fails, create a structured response from the topic content
            return {
                "explanation": f"Let's understand {topic.title}:\n\n{topic.content}",
                "examples": "We'll examine this through practical examples:\n\n" + 
                           "1. [First relevant example based on the content]\n" +
                           "2. [Second example showing practical application]",
                "question": f"Based on our discussion of {topic.title}, explain how [key concept] impacts [relevant aspect].",
                "key_points": [
                    "Understanding the core concepts",
                    "Practical applications",
                    "Real-world implications"
                ]
            }
            
    except Exception as e:
        st.error(f"Error generating content: {str(e)}")
        return {
            "explanation": topic.content,
            "examples": "Let's examine some practical applications.",
            "question": "What are the key concepts you've learned?",
            "key_points": ["Core concepts", "Applications", "Implications"]
        }
def evaluate_response(answer: str, expected_points: List[str], topic: Topic, model) -> dict:
    prompt = f"""
    Topic: {topic.title}
    Student's answer: {answer}
    Expected points: {', '.join(expected_points)}
    
    Return your evaluation as a single JSON object:
    {{
        "feedback": "Brief feedback on the answer",
        "complete_answer": "Thorough explanation of the correct answer",
        "mastered": false
    }}
    
    Keep your response simple and focused. Do not use special formatting or numbered lists.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = response.parts[0].text
        cleaned_json = clean_json_string(response_text)
        
        try:
            evaluation = json.loads(cleaned_json)
            return evaluation
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse evaluation JSON: {str(e)}")
            return {
                "feedback": "Thank you for your response.",
                "complete_answer": f"Here's what you should understand about this topic:\n\n{topic.content}",
                "mastered": False
            }
            
    except Exception as e:
        st.error(f"Error in evaluation: {str(e)}")
        return {
            "feedback": "Thank you for your response.",
            "complete_answer": f"Let's review the key points:\n\n{topic.content}",
            "mastered": False
        }

def main():
    st.set_page_config(page_title="Interactive AI Tutor", layout="wide")
    
    # Custom styling
    st.markdown("""
        <style>
        .main {
            padding: 1.5rem;
        }
        .stMarkdown h2 {
            color: #1E88E5;
            padding: 1rem 0 0.5rem 0;
            border-bottom: 1px solid #e0e0e0;
        }
        .stMarkdown h3 {
            color: #0D47A1;
            padding: 1rem 0 0.5rem 0;
        }
        .stMarkdown p {
            line-height: 1.6;
            margin: 0.8rem 0;
        }
        .stMarkdown ul {
            margin: 0.8rem 0;
            padding-left: 1.5rem;
        }
        pre {
            background-color: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            padding: 1rem;
            margin: 1rem 0;
        }
        code {
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.9rem;
            color: #1a1a1a;
        }
        .python-code {
            background-color: #f8f9fa;
            border-left: 3px solid #1E88E5;
        }
        .knowledge-check {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        .feedback {
            padding: 1rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)
    
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
        
        # Initialize tutorial state
        if 'tutorial_state' not in st.session_state:
            st.session_state.tutorial_state = TutorialState()
        
        # File upload section
        if not st.session_state.tutorial_state.topics:
            st.markdown("<div class='main'>", unsafe_allow_html=True)
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
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Chat interface
        if st.session_state.tutorial_state.topics:
            st.markdown("<div class='main'>", unsafe_allow_html=True)
            chat_container = st.container()
            state = st.session_state.tutorial_state
            current_topic = state.get_current_topic()
            
            # Display conversation history
            with chat_container:
                for message in state.conversation_history:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Current teaching content
            if current_topic and not current_topic.completed:
                teaching_content = None
                
                # Generate new lesson content if needed
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
                    
                    lesson_content = f"""## {current_topic.title}

### 📚 Understanding the Concepts
{teaching_content["explanation"]}

---

### 🔍 Real-World Examples & Applications
<div class="python-code">
{teaching_content["examples"]}
</div>

---

<div class="knowledge-check">
### 💡 Knowledge Check
{teaching_content["question"]}
</div>"""
                    
                    with st.chat_message("assistant"):
                        st.markdown(lesson_content, unsafe_allow_html=True)
                    
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": lesson_content
                    })
                    
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
                    
                    evaluation_response = f"""<div class="feedback">
### 🎯 Feedback
{evaluation['feedback']}

### ✨ Complete Explanation
{evaluation['complete_answer']}

---
### 📝 Next Steps
Moving on to the next topic in our learning journey...
</div>"""
                    
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
                        st.success("🎉 Congratulations! You've completed the tutorial!")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.tutorial_state.topics:
            st.markdown("<div class='sidebar'>", unsafe_allow_html=True)
            st.subheader("Learning Progress")
            
            if current_topic:
                st.info(f"Current Topic: {current_topic.title}")
                st.write(f"Phase: {state.current_teaching_phase.title()}")
            
            st.subheader("Topic Overview")
            for i, topic in enumerate(state.topics, 1):
                status = "✅" if topic.completed else "📍" if topic == current_topic else "⭕️"
                st.write(f"{status} {i}. {topic.title}")
                for j, subtopic in enumerate(topic.subtopics, 1):
                    status = "✅" if subtopic.completed else "📍" if subtopic == current_topic else "⭕️"
                    st.write(f"   {status} {i}.{j} {subtopic.title}")
            
            if st.button("Reset Tutorial"):
                st.session_state.tutorial_state.reset()
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
