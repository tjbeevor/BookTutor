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
        available_models = [model['name'] for model in models]
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

    # ... (rest of the class methods remain the same)

    # Ensure that in teach_topic, you're also using self.model correctly
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

# ... (rest of the code remains the same)

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
