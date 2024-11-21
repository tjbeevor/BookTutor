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
    Generate a structured tutorial breakdown with full logic and robust JSON handling.
    """
    def chunk_content(text: str, max_chunk_size: int = 3000) -> List[str]:
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for paragraph in paragraphs:
            if current_length + len(paragraph) + 2 <= max_chunk_size:
                current_chunk.append(paragraph)
                current_length += len(paragraph) + 2
            else:
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
                
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
            
        return chunks

    def safe_json_loads(json_str: str, context: str = "") -> dict:
        """Safely parse JSON with enhanced error handling"""
        try:
            # Remove any markdown formatting
            json_str = json_str.replace("```json", "").replace("```", "")
            
            # Find the first { and last }
            start_idx = json_str.find('{')
            end_idx = json_str.rfind('}') + 1
            
            if start_idx == -1 or end_idx == 0:
                st.warning(f"No valid JSON structure found in {context}")
                return {"micro_lessons": []}
                
            json_str = json_str[start_idx:end_idx]
            
            # Clean up common formatting issues
            json_str = json_str.replace('\n', ' ')
            json_str = json_str.replace('\r', ' ')
            json_str = json_str.replace('\t', ' ')
            json_str = ' '.join(json_str.split())
            json_str = json_str.replace('} {', '},{')
            json_str = json_str.replace(']"', ']')
            json_str = json_str.replace('""', '"')
            json_str = json_str.replace("'", '"')
            json_str = json_str.replace(",]", "]")
            json_str = json_str.replace(",}", "}")
            
            return json.loads(json_str)
        except Exception as e:
            st.warning(f"JSON parsing error in {context}: {str(e)}")
            return {"micro_lessons": []}

    try:
        # Initial analysis to understand document structure and learning flow
        analysis_prompt = """
        Analyze this educational content and create a detailed learning structure that:
        1. Identifies the main subject and key topics
        2. Breaks complex topics into focused micro-lessons
        3. Establishes clear prerequisites and learning flow
        4. Ensures logical progression of concepts
        
        Return ONLY this exact JSON format (no additional text):
        {
          "main_subject": "subject name",
          "concept_breakdown": [
            {
              "topic": "major topic name",
              "micro_lessons": [
                {
                  "title": "specific micro-lesson title",
                  "prerequisites": ["required knowledge"],
                  "key_concepts": ["main points"],
                  "learning_outcome": "specific outcome",
                  "difficulty": "beginner|intermediate|advanced"
                }
              ],
              "sequence": ["lesson 1", "lesson 2"]
            }
          ]
        }
        """
        
        # Get initial analysis
        response = model.generate_content(analysis_prompt + "\n\nContent:\n" + content[:3000])
        analysis = safe_json_loads(response.text, "initial analysis")
        
        # Process content in chunks with full logical structure
        chunks = chunk_content(content)
        all_topics = []
        
        # Extract learning sequence and prerequisites
        learning_sequence = []
        prerequisites_map = {}
        
        for concept in analysis.get('concept_breakdown', []):
            for lesson in concept.get('micro_lessons', []):
                learning_sequence.extend(concept.get('sequence', []))
                prerequisites_map[lesson.get('title')] = lesson.get('prerequisites', [])

        for chunk_idx, chunk in enumerate(chunks):
            chunk_prompt = f"""
            Create focused micro-lessons for this content about {analysis.get('main_subject', 'the subject')}.
            
            Requirements:
            1. Each micro-lesson covers ONE specific concept
            2. Clear prerequisites and learning outcomes
            3. Specific practice activities
            4. Build on previous knowledge systematically
            
            Previous lessons covered: {[t.get('title', '') for t in all_topics]}
            
            Return ONLY this exact JSON format (no additional text):
            {{
              "micro_lessons": [
                {{
                  "title": "specific lesson title",
                  "content": "clear explanation",
                  "prerequisites": ["required knowledge"],
                  "key_concepts": ["main points"],
                  "practice_activities": ["specific practice tasks"],
                  "learning_outcome": "specific outcome",
                  "difficulty": "beginner|intermediate|advanced"
                }}
              ]
            }}

            Content chunk:
            {chunk}
            """
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = model.generate_content(chunk_prompt)
                    structure = safe_json_loads(response.text, f"chunk {chunk_idx + 1}")
                    
                    if "micro_lessons" in structure and structure["micro_lessons"]:
                        # Process each micro-lesson
                        for lesson in structure["micro_lessons"]:
                            if not any(similar_titles(existing.get('title', ''), lesson.get('title', '')) 
                                     for existing in all_topics):
                                # Enhance lesson with context and prerequisites
                                lesson['prerequisites'] = (
                                    prerequisites_map.get(lesson.get('title', ''), []) +
                                    lesson.get('prerequisites', [])
                                )
                                all_topics.append(lesson)
                        break
                    
                except Exception as e:
                    if attempt == max_retries - 1:
                        st.warning(f"Failed to process chunk {chunk_idx + 1}: {str(e)}")
                    time.sleep(1)

        def similar_titles(title1: str, title2: str) -> bool:
            """Enhanced similarity check for topics"""
            if not title1 or not title2:
                return False
            title1_words = set(title1.lower().strip().split())
            title2_words = set(title2.lower().strip().split())
            common_words = title1_words & title2_words
            return (len(common_words) >= min(len(title1_words), len(title2_words)) * 0.7 or
                   title1.lower() in title2.lower() or title2.lower() in title1.lower())

        # Sort topics based on learning sequence and prerequisites
        def get_topic_score(topic: dict) -> tuple:
            # Get sequence position
            sequence_pos = next((i for i, seq in enumerate(learning_sequence) 
                               if similar_titles(topic.get('title', ''), seq)), len(learning_sequence))
            
            # Calculate prerequisite depth
            prereq_count = len(topic.get('prerequisites', []))
            difficulty_score = {'beginner': 0, 'intermediate': 1, 'advanced': 2}
            diff_score = difficulty_score.get(topic.get('difficulty', 'beginner'), 0)
            
            return (sequence_pos, prereq_count, diff_score)

        all_topics.sort(key=get_topic_score)

        # Convert to Topic objects with enhanced content structure
        def create_topic(topic_data: dict) -> Topic:
            content = f"""Learning Outcome:
{topic_data.get('learning_outcome', 'Not specified')}

Key Concepts:
{chr(10).join('- ' + concept for concept in topic_data.get('key_concepts', []))}

Content:
{topic_data.get('content', '')}

Practice Activities:
{chr(10).join('- ' + activity for activity in topic_data.get('practice_activities', []))}

Prerequisites:
{chr(10).join('- ' + prereq for prereq in topic_data.get('prerequisites', []))}
"""
            return Topic(
                title=topic_data.get('title', 'Untitled Topic'),
                content=content,
                subtopics=[],
                completed=False,
                parent=None
            )

        final_topics = [create_topic(t) for t in all_topics if isinstance(t, dict)]
        
        if not final_topics:
            st.warning("No topics generated, creating default structure")
            final_topics = [
                Topic(
                    title="Introduction to " + analysis.get('main_subject', 'the Subject'),
                    content="Basic introduction to the subject matter.",
                    subtopics=[],
                    completed=False
                )
            ]
        
        return final_topics
        
    except Exception as e:
        st.error(f"Error in tutorial structure generation: {str(e)}")
        return [
            Topic(
                title="Getting Started",
                content="Introduction to the subject matter.",
                subtopics=[],
                completed=False
            )
        ]

def generate_teaching_message(topic: Topic, phase: str, conversation_history: List[Dict], model) -> dict:
    """
    Generate focused micro-learning content with clear progression and validation points.
    """
    # Create context from conversation history
    previous_topics = []
    for msg in conversation_history:
        if msg["role"] == "assistant" and "<h2>" in msg["content"]:
            topic_title = msg["content"].split("<h2>")[1].split("</h2>")[0]
            previous_topics.append(topic_title)

    prompt = f"""
    Create a focused micro-lesson about: {topic.title}
    Previous topics covered: {', '.join(previous_topics) if previous_topics else 'This is the first topic'}
    
    Guidelines:
    1. Focus on ONE specific concept or skill
    2. Content should be concise but thorough
    3. Use clear, simple language
    4. Include only the most relevant examples
    5. Ask focused questions that test understanding of the specific concept
    
    Return a JSON object with this exact structure:
    {{
        "micro_lesson": {{
            "concept": {{
                "title": "Clear, specific concept title",
                "key_point": "One main point to remember",
                "explanation": "2-3 clear sentences explaining the concept"
            }},
            "practical_example": {{
                "scenario": "A single, clear real-world example",
                "steps": ["Step 1", "Step 2", "Step 3"],
                "key_consideration": "One important thing to remember"
            }},
            "understanding_check": {{
                "question": "A specific question testing understanding of this concept",
                "expected_elements": ["Point 1", "Point 2", "Point 3"]
            }}
        }}
    }}
    """
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            response_text = clean_json_string(response.text)
            content = json.loads(response_text)
            
            # Format the content with clear visual separation
            formatted_content = f"""## {content['micro_lesson']['concept']['title']}

### 📚 Key Concept
{content['micro_lesson']['concept']['explanation']}

**Remember:** {content['micro_lesson']['concept']['key_point']}

### 🔍 Real-World Example
{content['micro_lesson']['practical_example']['scenario']}

Steps:
"""
            for step in content['micro_lesson']['practical_example']['steps']:
                formatted_content += f"* {step}\n"
                
            formatted_content += f"\n**Important:** {content['micro_lesson']['practical_example']['key_consideration']}\n\n"
            
            formatted_content += f"""### 💡 Check Your Understanding
{content['micro_lesson']['understanding_check']['question']}"""
            
            return {
                "explanation": formatted_content,
                "expected_points": content['micro_lesson']['understanding_check']['expected_elements'],
                "question": content['micro_lesson']['understanding_check']['question']
            }
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"Failed to generate teaching content after {max_retries} attempts: {str(e)}")
                return {
                    "explanation": f"""## {topic.title}

### 📚 Key Concept
{topic.content}

### 🔍 Real-World Example
Practice applying this concept in a simple scenario.

### 💡 Check Your Understanding
Can you explain the key points of {topic.title}?
""",
                    "expected_points": ["Understanding of basic concept", "Application of concept", "Key considerations"],
                    "question": f"Can you explain the key points of {topic.title}?"
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
