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
    Generate cohesive, well-structured teaching content with enhanced formatting.
    """
    previous_topics = []
    for msg in conversation_history:
        if msg["role"] == "assistant" and "<h2>" in msg["content"]:
            topic_title = msg["content"].split("<h2>")[1].split("</h2>")[0]
            previous_topics.append(topic_title)

    prompt = f"""
    Create a comprehensive lesson about: {topic.title}
    
    Context:
    - Main topic content: {topic.content}
    - Teaching phase: {phase}
    - Previously covered topics: {', '.join(previous_topics) if previous_topics else 'This is the first topic'}
    
    Format the response using this exact structure:
    {{
        "introduction": "A clear 2-3 sentence introduction to the topic",
        "key_concepts": [
            {{
                "title": "Concept Title",
                "explanation": "Clear explanation",
                "bullet_points": [
                    "Key point 1",
                    "Key point 2"
                ]
            }}
        ],
        "examples": [
            {{
                "title": "Example Title",
                "scenario": "Detailed example scenario",
                "steps": [
                    "Step 1",
                    "Step 2"
                ]
            }}
        ],
        "practice_question": "A thought-provoking question about the content",
        "key_takeaways": [
            "Key takeaway 1",
            "Key takeaway 2"
        ]
    }}
    
    Make the content practical and easy to understand.
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            response_text = clean_json_string(response.text)
            content = json.loads(response_text)
            
            # Format the content with clean markdown
            formatted_content = f"""
## {topic.title}

### 📚 Understanding the Concepts

{content['introduction']}

"""
            # Add key concepts
            for concept in content['key_concepts']:
                formatted_content += f"""
#### {concept['title']}
{concept['explanation']}

"""
                if concept['bullet_points']:
                    for point in concept['bullet_points']:
                        formatted_content += f"* {point}\n"
                formatted_content += "\n"

            # Add practical applications
            formatted_content += "\n### 🔍 Practical Applications\n"
            for i, example in enumerate(content['examples'], 1):
                formatted_content += f"""
#### Example {i}: {example['title']}
{example['scenario']}

Steps:
"""
                for step in example['steps']:
                    formatted_content += f"* {step}\n"
                formatted_content += "\n"

            # Add understanding check
            formatted_content += f"""
### 💡 Understanding Check

{content['practice_question']}

Key Takeaways:
"""
            for takeaway in content['key_takeaways']:
                formatted_content += f"* {takeaway}\n"
            
            return {
                "explanation": formatted_content,
                "examples": content['examples'],
                "question": content['practice_question'],
                "key_points": content['key_takeaways']
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to generate teaching content after {max_retries} attempts: {str(e)}")
                # Return a basic formatted version if all attempts fail
                return {
                    "explanation": f"""
## {topic.title}

### 📚 Understanding the Concepts

{topic.content}

### 🔍 Practical Applications

#### Example 1: Basic Application
* Start with the fundamentals
* Apply the concept step by step
* Practice regularly

### 💡 Understanding Check

Please explain your understanding of {topic.title}.

Key Takeaways:
* Understand the basic principles
* Practice regularly
* Apply concepts systematically
""",
                    "examples": [{"title": "Basic Application", "steps": ["Step 1", "Step 2", "Step 3"]}],
                    "question": f"Please explain your understanding of {topic.title}.",
                    "key_points": ["Understanding core concepts", "Practical applications", "Key takeaways"]
                }
            time.sleep(1)

def evaluate_response(user_response: str, expected_points: List[str], current_topic: Topic, model) -> dict:
    """
    Evaluate user's response against expected learning points.
    
    Args:
        user_response: The user's answer text
        expected_points: List of key points that should be covered
        current_topic: Current topic being discussed
        model: The AI model instance
    
    Returns:
        dict: Contains feedback and complete answer
    """
    prompt = f"""
    Evaluate this response about {current_topic.title}.
    
    User's response:
    {user_response}
    
    Expected key points:
    {" - " + "\n - ".join(expected_points)}
    
    Provide an evaluation in this JSON format:
    {{
        "feedback": "Detailed, constructive feedback on the response, highlighting strengths and areas for improvement",
        "complete_answer": "A comprehensive explanation of the topic, incorporating all key points",
        "understanding_level": "A number from 0-100 indicating comprehension level"
    }}
    
    Make the feedback encouraging but thorough. Include specific examples from their response.
    """
    
    try:
        response = model.generate_content(prompt)
        response_text = clean_json_string(response.text)
        evaluation = json.loads(response_text)
        
        # Format the feedback and complete answer with markdown
        formatted_feedback = f"""
#### Feedback on Your Understanding
{evaluation['feedback']}

#### Complete Explanation
{evaluation['complete_answer']}

Understanding Level: {evaluation['understanding_level']}%
"""
        
        return {
            "feedback": formatted_feedback,
            "complete_answer": evaluation['complete_answer'],
            "understanding_level": int(evaluation['understanding_level'])
        }
        
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return {
            "feedback": """
#### Feedback on Your Understanding
Thank you for your response. Let's review the key points to ensure complete understanding.

#### Complete Explanation
""" + current_topic.content,
            "complete_answer": current_topic.content,
            "understanding_level": 50
        }

def main():
    # 1. Page Configuration
    st.set_page_config(
        page_title="AI Learning Assistant",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # 2. Custom CSS kept from original - assuming it's defined above

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
                # Check if we need to generate new teaching content
                if len(state.conversation_history) == 0 or (
                    len(state.conversation_history) > 0 and 
                    state.conversation_history[-1]["role"] == "assistant" and 
                    "Moving on to the next topic" in state.conversation_history[-1]["content"]
                ):
                    # Generate teaching content
                    teaching_content = generate_teaching_message(
                        current_topic,
                        state.current_teaching_phase,
                        state.conversation_history,
                        st.session_state.model
                    )
                    
                    # Format and display the content
                    formatted_content = f"""
                    ## {current_topic.title}
            
                    ### 📚 Understanding the Concepts
                    {teaching_content["explanation"]}
            
                    ### 🔍 Practical Applications
                    {teaching_content["examples"]}
            
                    ### 💡 Understanding Check
                    {teaching_content["question"]}
                    """
                    
                    with st.chat_message("assistant"):
                        st.markdown(formatted_content, unsafe_allow_html=True)
                    
                    # Update conversation history
                    state.conversation_history.append({
                        "role": "assistant",
                        "content": formatted_content
                    })
                    
                    # Store expected points for evaluation
                    st.session_state.expected_points = teaching_content["key_points"]
                
                # Handle user input
                user_input = st.chat_input("Share your thoughts...")
                if user_input:
                    with st.chat_message("user"):
                        st.markdown(user_input)
                    
                    state.conversation_history.append({
                        "role": "user",
                        "content": user_input
                    })
                    
                    # Generate and display evaluation
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
                    
                    # Mark current topic as completed
                    current_topic.completed = True
                    
                    # Advance to next topic or show completion
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
