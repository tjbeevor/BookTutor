import streamlit as st
from typing import Dict, List, Any
import json
import google.generativeai as genai
import PyPDF2
import docx
import io

def initialize_model():
    """Initialize or get the Google PaLM model"""
    try:
        if 'api_key' not in st.session_state:
            st.session_state.api_key = st.secrets.get('GOOGLE_API_KEY', '')
        
        if not st.session_state.api_key:
            st.session_state.api_key = st.text_input(
                'Enter Google API Key:', 
                type='password',
                help="Enter your Google API key to access the PaLM model"
            )
            if not st.session_state.api_key:
                st.warning('⚠️ Please enter your Google API key to continue.')
                return None
        
        genai.configure(api_key=st.session_state.api_key)
        
        # List available models
        models = genai.list_models()
        available_models = [model.name for model in models]  # Corrected line
        if not available_models:
            st.error("No models available. Please check your API key and permissions.")
            return None

        # Allow the user to select a model
        st.session_state.model_name = st.selectbox(
            "Select a model:",
            available_models,
            index=0
        )
        return st.session_state.model_name
        
    except Exception as e:
        st.error(f"❌ Error initializing PaLM model: {str(e)}")
        return None

class EnhancedTeacher:
    def __init__(self, model):
        self.model = model

    def analyze_document(self, content: Dict[str, Any]) -> List[Dict]:
        try:
            text_content = content['text']
            if text_content is None:
                text_content = ''  # Ensure text_content is never None
                
            sections = self._split_into_chunks(text_content)
            all_topics = []
            
            for section in sections:
                prompt = """You are an expert curriculum designer. Create a detailed learning module from the provided content.
Your response must be a valid JSON object with no trailing commas, following this EXACT structure:

{
  "title": "Specific topic title",
  "learning_objectives": [
    "First learning objective",
    "Second learning objective"
  ],
  "key_points": [
    "First key point",
    "Second key point"
  ],
  "content": "Main educational content",
  "practical_exercises": [
    "First exercise",
    "Second exercise"
  ],
  "knowledge_check": {
    "questions": [
      {
        "question": "Sample question?",
        "options": [
          "First option",
          "Second option",
          "Third option",
          "Fourth option"
        ],
        "correct_answer": "First option",
        "explanation": "Explanation of the correct answer"
      }
    ]
  },
  "difficulty": "beginner",
  "estimated_time": "30 minutes"
}

Content to analyze:
"""
                # Add the section content with clear delimiter
                prompt += f"\n\n{section[:4000]}\n\nRespond ONLY with the JSON structure, no additional text or explanations."

                try:
                    response = genai.generate_text(
                        model=self.model,
                        prompt=prompt,
                        temperature=0.1,
                        top_p=0.95,
                        top_k=40
                    )
                    
                    response_text = response.result
                    # Clean the response text more aggressively
                    response_text = self._clean_json_text(response_text)
                    
                    try:
                        topic = json.loads(response_text)
                        if self._validate_topic(topic):
                            topic = self._enhance_topic(topic, section)
                            all_topics.append(topic)
                            continue
                    except json.JSONDecodeError as e:
                        st.warning(f"JSON parsing error: {str(e)}")
                    
                    # If we get here, either the response was empty or parsing failed
                    st.warning("Using fallback structure for this section")
                    fallback = self._create_fallback_structure(section[:1000])
                    all_topics.append(fallback)

                except Exception as e:
                    st.warning(f"Processing error for section: {str(e)}")
                    fallback = self._create_fallback_structure(section[:1000])
                    all_topics.append(fallback)
                    continue

            # Post-process all topics to ensure coherent flow
            processed_topics = self._post_process_topics(all_topics) if all_topics else [self._create_fallback_structure(text_content)]
            return processed_topics

        except Exception as e:
            st.error(f"Error in document analysis: {str(e)}")
            return [self._create_fallback_structure(text_content)]

    def _clean_json_text(self, text: str) -> str:
        """More aggressive JSON text cleaning"""
        try:
            # Remove any markdown code block markers and extra whitespace
            text = text.replace('```json', '').replace('```', '')
            text = text.strip()
            
            # Find the first '{' and last '}'
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1:
                text = text[start:end+1]
            
            # More aggressive cleaning
            text = text.replace('\\"', '"')  # Fix escaped quotes
            text = text.replace('\n', ' ')   # Remove newlines
            text = text.replace('\\n', ' ')  # Remove escaped newlines
            text = text.replace('\\', '')    # Remove remaining backslashes
            text = ' '.join(text.split())    # Normalize spacing
            
            # Fix common JSON formatting issues
            text = text.replace('"{', '{')   # Fix escaped objects
            text = text.replace('}"', '}')   # Fix escaped objects
            text = text.replace(',}', '}')   # Remove trailing commas
            text = text.replace(',]', ']')   # Remove trailing commas
            
            # Replace single quotes with double quotes
            text = text.replace("'", '"')
            
            # Fix missing quotes around property names
            import re
            text = re.sub(r'(\s*{?\s*)(\w+)(\s*:)', r'\1"\2"\3', text)
            
            # Validate JSON structure before returning
            json.loads(text)  # Test if it's valid JSON
            return text
            
        except json.JSONDecodeError:
            # Create a minimal valid JSON structure
            return json.dumps(self._create_fallback_structure("Content parsing error"))
        
        except Exception as e:
            st.error(f"Error cleaning JSON text: {str(e)}")
            return json.dumps(self._create_fallback_structure("Content parsing error"))

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split content into logical sections with size limits"""
        max_chunk_size = 4000  # Adjust as needed
        lines = text.split('\n')
        sections = []
        current_section = []

        for line in lines:
            if self._is_section_header(line) and current_section:
                section_text = '\n'.join(current_section).strip()
                if section_text:
                    sections.append(section_text)
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            section_text = '\n'.join(current_section).strip()
            if section_text:
                sections.append(section_text)
        
        # If no natural sections found, create chunks of reasonable size
        if len(sections) <= 1:
            chunks = [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]
            # Ensure chunks break at sentence boundaries where possible
            return self._refine_chunks(chunks)
            
        return sections

    def _refine_chunks(self, chunks: List[str]) -> List[str]:
        """Refine chunk boundaries to break at natural points"""
        refined_chunks = []
        for chunk in chunks:
            if len(chunk) < 4000:
                refined_chunks.append(chunk)
            else:
                # Break at last full stop within chunk
                last_period = chunk.rfind('.')
                if last_period != -1:
                    refined_chunks.append(chunk[:last_period+1])
                else:
                    refined_chunks.append(chunk)
        return refined_chunks

    def _is_section_header(self, line: str) -> bool:
        """Detect if a line is likely a section header"""
        line = line.strip()
        if not line:
            return False
            
        patterns = [
            lambda x: x.isupper() and len(x) > 10,
            lambda x: x.startswith(('#', 'Chapter', 'Section')),
            lambda x: any(char.isdigit() for char in x[:5]) and len(x) < 100,
            lambda x: x.endswith(':') and len(x) < 100
        ]
        
        return any(pattern(line) for pattern in patterns)

    def _validate_topic(self, topic: Dict) -> bool:
        """Validate topic structure with detailed feedback"""
        # First check if content exists and is None
        if 'content' in topic and topic['content'] is None:
            topic['content'] = ''  # Set empty string instead of None
        
        required_fields = [
            'title', 
            'learning_objectives', 
            'key_points', 
            'content'
        ]
        
        missing_fields = [field for field in required_fields if field not in topic]
        if missing_fields:
            st.warning(f"Missing required fields: {', '.join(missing_fields)}")
            return False
    
        # Validate field types
        type_checks = {
            'title': str,
            'learning_objectives': list,
            'key_points': list,
            'content': str
        }
        
        for field, expected_type in type_checks.items():
            if field in topic and not isinstance(topic[field], expected_type):
                # Convert None to empty string for content field
                if field == 'content' and topic[field] is None:
                    topic[field] = ''
                else:
                    st.warning(f"Field '{field}' has incorrect type. Expected {expected_type}, got {type(topic[field])}")
                    return False
                    
        return True

    def _enhance_topic(self, topic: Dict, original_section: str) -> Dict:
        """Enhance topic with additional context and structure"""
        # Ensure all fields are present
        topic.setdefault('practical_exercises', [])
        topic.setdefault('knowledge_check', {'questions': []})
        topic.setdefault('real_world_applications', [])
        topic.setdefault('additional_resources', [])
        topic.setdefault('difficulty', 'beginner')
        topic.setdefault('estimated_time', '30 minutes')

        # Preserve original content structure
        if 'content' in topic:
            topic['content'] = self._preserve_formatting(topic['content'], original_section)

        return topic

    def _preserve_formatting(self, content: str, original: str) -> str:
        """Preserve original formatting and structure where possible"""
        # Preserve lists
        lines = content.split('\n')
        formatted_lines = []
        for line in lines:
            if line.strip().startswith(('•', '-', '*', '1.', '2.')):
                formatted_lines.append(f"\n{line}")
            else:
                formatted_lines.append(line)
        return '\n'.join(formatted_lines)

    def _post_process_topics(self, topics: List[Dict]) -> List[Dict]:
        """Ensure topics flow logically and maintain document structure"""
        processed_topics = []
        current_difficulty = "beginner"
        
        for i, topic in enumerate(topics):
            # Ensure progression of difficulty
            if i > len(topics) // 2:
                current_difficulty = "intermediate"
            if i > len(topics) * 0.8:
                current_difficulty = "advanced"
            
            # Add cross-references and connections
            if i > 0:
                topic['prerequisites'] = [topics[i-1]['title']]
            if i < len(topics) - 1:
                topic['next_steps'] = [topics[i+1]['title']]
            
            topic['difficulty'] = current_difficulty
            processed_topics.append(topic)
        
        return processed_topics

    def _create_fallback_structure(self, content: str) -> Dict:
        """Create more informative fallback structure when parsing fails"""
        return {
            'title': 'Content Section',
            'learning_objectives': ['Understand the key concepts in this section'],
            'key_points': ['Review the main points presented in this material'],
            'content': content[:1000] + ("..." if len(content) > 1000 else ""),
            'practical_exercises': ['Summarize the main concepts presented'],
            'knowledge_check': {
                'questions': [{
                    'question': 'What are the main topics covered in this section?',
                    'options': [
                        'Review the content to identify key topics',
                        'Analyze the main concepts presented',
                        'Summarize the key points',
                        'All of the above'
                    ],
                    'correct_answer': 'All of the above',
                    'explanation': 'A thorough review of the content will help identify and understand the key topics.'
                }]
            },
            'difficulty': 'beginner',
            'estimated_time': '15 minutes'
        }

    def teach_topic(self, topic: Dict, user_progress: Dict) -> str:
        """Generate engaging lesson content with proper response handling"""
        try:
            prompt = f"""
Create an engaging lesson for: {topic['title']}

Learning objectives:
{json.dumps(topic.get('learning_objectives', []), indent=2)}

Key points:
{json.dumps(topic.get('key_points', []), indent=2)}

Main content:
{topic.get('content', '')}

Create an engaging lesson with:
1. Clear introduction
2. Detailed explanations
3. Practical examples
4. Interactive elements
5. Summary and key takeaways

Student context:
- Level: {user_progress.get('understanding_level', 'beginner')}
- Topics completed: {len(user_progress.get('completed_topics', []))}

Use markdown formatting for clear structure and engagement.
"""

            response = genai.generate_text(
                model=self.model,
                prompt=prompt,
                temperature=0.7,
                max_output_tokens=1024  # Adjust as needed
            )
            
            # Properly handle the response
            lesson_content = response.result
            return lesson_content

        except Exception as e:
            st.error(f"Error generating lesson: {str(e)}")
            return "Error generating lesson content."

def process_text_from_file(file_content, file_type) -> str:
    """Process uploaded file content"""
    try:
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = "\n\n".join(page.extract_text() or '' for page in pdf_reader.pages)
            
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(file_content))
            text = "\n\n".join(paragraph.text for paragraph in doc.paragraphs)
            
        elif file_type == "text/markdown":
            text = file_content.decode()
            
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return text.strip()
        
    except Exception as e:
        raise Exception(f"Error processing file: {str(e)}")

def reset_application():
    """Reset the application state"""
    for key in list(st.session_state.keys()):
        if key not in ['api_key', 'model_name']:  # Preserve API key and model name
            del st.session_state[key]
    st.experimental_rerun()

def main():
    """Main application function"""
    st.set_page_config(
        page_title="Enhanced Learning Assistant",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Enhanced Learning Assistant")
    st.write("Upload your educational content and let AI help transform it into an interactive learning experience.")

    # Initialize model
    model = initialize_model()
    if not model:
        return

    # Initialize teacher
    teacher = EnhancedTeacher(model)

    # Session state initialization
    if 'current_topics' not in st.session_state:
        st.session_state.current_topics = []
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = {
            'understanding_level': 'beginner',
            'completed_topics': []
        }

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        
        # Reset button
        if st.button("Reset Application", type="secondary"):
            reset_application()
        
        # Understanding level selector
        st.session_state.user_progress['understanding_level'] = st.selectbox(
            "Select your understanding level:",
            ["beginner", "intermediate", "advanced"],
            index=["beginner", "intermediate", "advanced"].index(
                st.session_state.user_progress.get('understanding_level', 'beginner')
            )
        )

    # Main content area
    uploaded_file = st.file_uploader(
        "Upload your educational content (PDF, DOCX, or Markdown)",
        type=['pdf', 'docx', 'md'],
        help="Upload your educational material to begin"
    )

    if uploaded_file:
        try:
            # Process the uploaded file
            file_content = uploaded_file.read()
            text_content = process_text_from_file(file_content, uploaded_file.type)

            # Analyze content
            with st.spinner("Analyzing content and creating learning modules..."):
                topics = teacher.analyze_document({'text': text_content})
                st.session_state.current_topics = topics

            # Display topics
            if st.session_state.current_topics:
                st.subheader("📑 Learning Modules")
                
                for i, topic in enumerate(st.session_state.current_topics):
                    with st.expander(f"Module {i+1}: {topic['title']}", expanded=i==0):
                        # Topic metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"Difficulty: {topic['difficulty']}")
                        with col2:
                            st.caption(f"Estimated time: {topic['estimated_time']}")
                        with col3:
                            status = 'Completed' if topic['title'] in st.session_state.user_progress['completed_topics'] else 'Not Started'
                            st.caption(f"Status: {status}")

                        # Display lesson content
                        if st.button(f"Start Module {i+1}", key=f"start_module_{i}"):
                            lesson_content = teacher.teach_topic(topic, st.session_state.user_progress)
                            st.markdown(lesson_content)

                            # Knowledge check section
                            if topic.get('knowledge_check', {}).get('questions'):
                                st.subheader("✍️ Knowledge Check")
                                for q_idx, question in enumerate(topic['knowledge_check']['questions']):
                                    st.write(f"Q{q_idx + 1}: {question['question']}")
                                    answer = st.radio(
                                        "Select your answer:",
                                        question['options'],
                                        key=f"q_{i}_{q_idx}"
                                    )
                                    if st.button("Check Answer", key=f"check_{i}_{q_idx}"):
                                        if answer == question['correct_answer']:
                                            st.success("Correct! " + question['explanation'])
                                        else:
                                            st.error(f"Not quite. The correct answer is: {question['correct_answer']}")

                            # Mark as complete button
                            if st.button("Mark as Complete", key=f"complete_{i}"):
                                if topic['title'] not in st.session_state.user_progress['completed_topics']:
                                    st.session_state.user_progress['completed_topics'].append(topic['title'])
                                st.success("Module marked as complete!")
                                st.experimental_rerun()

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.write("Please try uploading a different file or contact support if the issue persists.")

if __name__ == "__main__":
    main()
