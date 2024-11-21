# 1. KEEP THESE IMPORTS AT THE TOP
import streamlit as st
from typing import Dict, List, Any
import json
import google.generativeai as genai
import PyPDF2
import docx
import io
from functools import lru_cache

# 2. KEEP THIS FUNCTION
def initialize_model():
    """Initialize or get the Google Gemini model"""
    try:
        if 'gemini_api_key' not in st.session_state:
            st.session_state.gemini_api_key = st.secrets.get('GOOGLE_API_KEY', '')
        
        if not st.session_state.gemini_api_key:
            st.session_state.gemini_api_key = st.text_input(
                'Enter Google API Key:', 
                type='password',
                help="Enter your Google API key to access the Gemini model"
            )
            if not st.session_state.gemini_api_key:
                st.warning('⚠️ Please enter your Google API key to continue.')
                return None
        
        genai.configure(api_key=st.session_state.gemini_api_key)
        return genai.GenerativeModel('gemini-pro')
    
    except Exception as e:
        st.error(f"❌ Error initializing Gemini model: {str(e)}")
        return None

class EnhancedTeacher:
    def __init__(self, model):
        self.model = model

    def analyze_document(self, content: Dict[str, Any]) -> List[Dict]:
        """Enhanced document analysis with comprehensive content processing and fixed JSON handling"""
        try:
            text_content = content['text']
            sections = self._split_into_chunks(text_content)
            all_topics = []
            
            for section in sections:
                prompt = f"""
                You are an expert curriculum designer. Create a detailed learning module from this content. 
                Focus on extracting meaningful educational content while preserving the original material's structure.
                Output ONLY valid JSON with no additional text.

                Content to analyze:
                {section[:4000]}

                JSON Response Format:
                {{
                    "title": "Clear and specific section title",
                    "learning_objectives": [
                        "Specific learning objective 1",
                        "Specific learning objective 2"
                    ],
                    "key_points": [
                        "Key concept 1",
                        "Key concept 2"
                    ],
                    "content": "Detailed educational content from the source material",
                    "practical_exercises": [
                        "Specific exercise 1",
                        "Specific exercise 2"
                    ],
                    "knowledge_check": {{
                        "questions": [
                            {{
                                "question": "Specific question about the content?",
                                "options": [
                                    "Option A",
                                    "Option B",
                                    "Option C",
                                    "Option D"
                                ],
                                "correct_answer": "Correct option",
                                "explanation": "Why this answer is correct"
                            }}
                        ]
                    }},
                    "real_world_applications": [
                        "Application 1",
                        "Application 2"
                    ],
                    "additional_resources": [
                        "Resource 1",
                        "Resource 2"
                    ],
                    "difficulty": "beginner/intermediate/advanced",
                    "estimated_time": "Time estimate"
                }}"""

                try:
                    response = self.model.generate_content(
                        prompt,
                        generation_config={
                            'temperature': 0.3,
                            'top_p': 0.8,
                            'top_k': 40
                        }
                    )

                    # Clean and parse response
                    response_text = self._clean_json_text(response.text)
                    topic = json.loads(response_text)

                    if self._validate_topic(topic):
                        # Post-process topic to enhance content
                        topic = self._enhance_topic(topic, section)
                        all_topics.append(topic)
                    else:
                        st.warning(f"Skipped invalid topic structure")

                except Exception as e:
                    st.warning(f"Processing error for section: {str(e)}")
                    continue

            # Post-process all topics to ensure coherent flow
            processed_topics = self._post_process_topics(all_topics) if all_topics else [self._create_fallback_structure(text_content)]
            return processed_topics

        except Exception as e:
            st.error(f"Error in document analysis: {str(e)}")
            return [self._create_fallback_structure(text_content)]

    def _clean_json_text(self, text: str) -> str:
        """Clean and fix common JSON formatting issues"""
        try:
            # Remove any markdown code block markers
            text = text.replace('```json', '').replace('```', '')
            
            # Find the first '{' and last '}'
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1:
                text = text[start:end+1]
            
            # Fix common JSON formatting issues
            text = text.replace('\n', ' ')  # Remove newlines
            text = ' '.join(text.split())   # Normalize spacing
            
            return text
        except Exception as e:
            st.error(f"Error cleaning JSON text: {str(e)}")
            raise

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split content into logical sections"""
        lines = text.split('\n')
        sections = []
        current_section = []
        
        for line in lines:
            if self._is_section_header(line) and current_section:
                sections.append('\n'.join(current_section))
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        # If no natural sections found, create chunks of reasonable size
        if len(sections) <= 1:
            chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
            # Ensure chunks break at sentence boundaries where possible
            return self._refine_chunks(chunks)
            
        return sections

    def _refine_chunks(self, chunks: List[str]) -> List[str]:
        """Refine chunk boundaries to break at natural points"""
        refined_chunks = []
        for chunk in chunks:
            sentences = chunk.split('.')
            if len(sentences) > 1:
                # Remove last incomplete sentence if any
                complete_chunk = '.'.join(sentences[:-1]) + '.'
                refined_chunks.append(complete_chunk)
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

    def teach_topic(self, topic: Dict, user_progress: Dict) -> str:
        """Generate engaging lesson content"""
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

            response = self.model.generate_content(prompt)
            return response.text

        except Exception as e:
            st.error(f"Error generating lesson: {str(e)}")
            return "Error generating lesson content."

    def _create_fallback_structure(self, content: str) -> Dict:
        """Create basic structure when parsing fails"""
        return {
            'title': 'Content Overview',
            'learning_objectives': ['Understand the main concepts presented'],
            'key_points': ['Key concepts from the material'],
            'content': content[:1000],
            'practical_exercises': ['Review and summarize the main points'],
            'knowledge_check': {
                'questions': [{
                    'question': 'What is the main topic covered?',
                    'options': ['Review the content to answer'],
                    'correct_answer': 'Review the content to answer',
                    'explanation': 'Please review the material carefully'
                }]
            },
            'difficulty': 'beginner',
            'estimated_time': '15 minutes'
        }

# 4. KEEP THIS FUNCTION
def process_text_from_file(file_content, file_type) -> str:
    """Process uploaded file content"""
    try:
        if file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text = "\n\n".join(page.extract_text() for page in pdf_reader.pages)
            
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

# 5. KEEP THIS FUNCTION
def reset_application():
    """Reset the application state"""
    for key in list(st.session_state.keys()):
        if key != 'gemini_api_key':  # Preserve API key
            del st.session_state[key]
    st.rerun()

# 6. KEEP THE MAIN FUNCTION AND EVERYTHING ELSE
def main():
    [Keep your existing main() function exactly as is]

if __name__ == "__main__":
    main()
