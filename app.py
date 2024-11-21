import streamlit as st
import google.generativeai as genai
import PyPDF2
import docx
import markdown
import bs4
import io
import json
import time
import random
from typing import Dict, List, Optional, Any, BinaryIO
from dataclasses import dataclass
from abc import ABC, abstractmethod
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Custom Exceptions
class DocumentProcessingError(Exception):
    """Custom exception for document processing errors"""
    pass

class UnsupportedFormatError(Exception):
    """Custom exception for unsupported file formats"""
    pass

# Base Classes
@dataclass
class Topic:
    title: str
    content: str
    subtopics: List['Topic']
    completed: bool = False
    parent: Optional['Topic'] = None
    metadata: Dict[str, Any] = None
    content_type: str = "general"
    difficulty_level: str = "intermediate"

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class UserProgress:
    understanding_level: float = 50.0
    completed_topics: List[str] = None
    current_approach: str = "practical"
    difficulty_level: str = "beginner"
    
    def __post_init__(self):
        if self.completed_topics is None:
            self.completed_topics = []

class BaseDocumentProcessor(ABC):
    @abstractmethod
    def extract_content(self, file_obj: BinaryIO) -> Dict[str, Any]:
        """Extract content from the document"""
        pass

    @abstractmethod
    def validate_content(self, content: Dict[str, Any]) -> bool:
        """Validate extracted content"""
        pass

    def standardize_content(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Convert content to standard internal format"""
        return {
            'text': content.get('text', ''),
            'metadata': content.get('metadata', {}),
            'structure': content.get('structure', {}),
            'type': content.get('type', 'general')
        }
# Document Processors
class PDFProcessor(BaseDocumentProcessor):
    def extract_content(self, file_obj: BinaryIO) -> Dict[str, Any]:
        """Extract content from PDF files"""
        try:
            content = {
                'text': [],
                'metadata': {},
                'structure': {'sections': []},
                'type': 'pdf'
            }
            
            pdf_reader = PyPDF2.PdfReader(file_obj)
            
            # Extract metadata
            content['metadata'] = {
                'pages': len(pdf_reader.pages),
                'title': pdf_reader.metadata.get('/Title', 'Untitled'),
                'author': pdf_reader.metadata.get('/Author', 'Unknown'),
                'subject': pdf_reader.metadata.get('/Subject', ''),
                'creation_date': pdf_reader.metadata.get('/CreationDate', '')
            }
            
            # Process each page
            current_section = {'title': None, 'content': []}
            
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    if not page_text:
                        continue
                        
                    # Split into lines for analysis
                    lines = page_text.split('\n')
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                            
                        # Detect if line is likely a header
                        if self._is_header(line):
                            if current_section['title'] and current_section['content']:
                                content['structure']['sections'].append(current_section.copy())
                            current_section = {'title': line, 'content': []}
                        else:
                            current_section['content'].append(line)
                            
                    content['text'].append(page_text)
                    
                except Exception as e:
                    st.warning(f"Warning: Error processing page {page_num}: {str(e)}")
                    continue
            
            # Add the last section if it exists
            if current_section['title'] and current_section['content']:
                content['structure']['sections'].append(current_section)
                
            return content
            
        except Exception as e:
            raise DocumentProcessingError(f"Error processing PDF: {str(e)}")

    def validate_content(self, content: Dict[str, Any]) -> bool:
        """Validate PDF content"""
        if not content.get('text'):
            return False
        if not any(text.strip() for text in content['text']):
            return False
        return True

    def _is_header(self, line: str) -> bool:
        """Detect if a line is likely a header"""
        # Simple heuristics for header detection
        line = line.strip()
        if not line:
            return False
            
        # Check for common header patterns
        header_indicators = [
            line.isupper(),
            line.startswith('#'),
            len(line.split()) <= 5 and line[0].isupper(),
            any(line.startswith(str(i) + '.') for i in range(1, 100))
        ]
        
        return any(header_indicators)

class DocxProcessor(BaseDocumentProcessor):
    def extract_content(self, file_obj: BinaryIO) -> Dict[str, Any]:
        """Extract content from DOCX files"""
        try:
            doc = docx.Document(file_obj)
            content = {
                'text': [],
                'metadata': {},
                'structure': {'sections': []},
                'type': 'docx'
            }
            
            # Extract document properties
            content['metadata'] = {
                'title': doc.core_properties.title or 'Untitled',
                'author': doc.core_properties.author or 'Unknown',
                'created': str(doc.core_properties.created or ''),
                'modified': str(doc.core_properties.modified or '')
            }
            
            current_section = {'title': None, 'content': []}
            
            for paragraph in doc.paragraphs:
                text = paragraph.text.strip()
                if not text:
                    continue
                    
                # Check if paragraph is a heading
                if paragraph.style.name.startswith('Heading'):
                    if current_section['title'] and current_section['content']:
                        content['structure']['sections'].append(current_section.copy())
                    current_section = {'title': text, 'content': []}
                else:
                    current_section['content'].append(text)
                    content['text'].append(text)
                    
            # Add the last section if it exists
            if current_section['title'] and current_section['content']:
                content['structure']['sections'].append(current_section)
                
            # Extract tables if they exist
            content['tables'] = []
            for table in doc.tables:
                table_data = []
                for row in table.rows:
                    row_data = [cell.text.strip() for cell in row.cells]
                    table_data.append(row_data)
                content['tables'].append(table_data)
                
            return content
            
        except Exception as e:
            raise DocumentProcessingError(f"Error processing DOCX: {str(e)}")

    def validate_content(self, content: Dict[str, Any]) -> bool:
        """Validate DOCX content"""
        return bool(content.get('text') or content.get('tables'))

class MarkdownProcessor(BaseDocumentProcessor):
    def extract_content(self, file_obj: BinaryIO) -> Dict[str, Any]:
        """Extract content from Markdown files"""
        try:
            text = file_obj.read().decode('utf-8')
            html = markdown.markdown(text, extensions=['tables', 'fenced_code'])
            soup = bs4.BeautifulSoup(html, 'html.parser')
            
            content = {
                'text': [],
                'metadata': {},
                'structure': {'sections': []},
                'type': 'markdown'
            }
            
            current_section = {'title': None, 'content': []}
            
            # Process headers and content
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'code', 'pre']):
                text = element.get_text().strip()
                if not text:
                    continue
                    
                if element.name.startswith('h'):
                    if current_section['title'] and current_section['content']:
                        content['structure']['sections'].append(current_section.copy())
                    current_section = {'title': text, 'content': []}
                else:
                    current_section['content'].append(text)
                    content['text'].append(text)
                    
            # Add the last section if it exists
            if current_section['title'] and current_section['content']:
                content['structure']['sections'].append(current_section)
                
            # Extract code blocks
            content['code_blocks'] = []
            for code_block in soup.find_all('pre'):
                content['code_blocks'].append({
                    'language': code_block.find('code').get('class', [''])[0] if code_block.find('code') else '',
                    'code': code_block.get_text().strip()
                })
                
            return content
            
        except Exception as e:
            raise DocumentProcessingError(f"Error processing Markdown: {str(e)}")

    def validate_content(self, content: Dict[str, Any]) -> bool:
        """Validate Markdown content"""
        return bool(content.get('text'))

class DocumentProcessor:
    def __init__(self):
        self.processors = {
            'pdf': PDFProcessor(),
            'docx': DocxProcessor(),
            'md': MarkdownProcessor()
        }
        
    def process_document(self, file_obj: BinaryIO, file_type: str) -> Dict[str, Any]:
        """Process document and return standardized content"""
        if file_type not in self.processors:
            raise UnsupportedFormatError(f"Format {file_type} not supported")
            
        processor = self.processors[file_type]
        content = processor.extract_content(file_obj)
        
        if not processor.validate_content(content):
            raise DocumentProcessingError("Invalid or empty document content")
            
        return processor.standardize_content(content)

    def detect_file_type(self, file_obj) -> str:
        """Detect file type from file object"""
        filename = getattr(file_obj, 'name', '').lower()
        if filename.endswith('.pdf'):
            return 'pdf'
        elif filename.endswith('.docx'):
            return 'docx'
        elif filename.endswith('.md'):
            return 'md'
        else:
            raise UnsupportedFormatError("Unsupported file type")

def process_uploaded_file(file_obj) -> Dict[str, Any]:
    """Process an uploaded file and return standardized content"""
    try:
        processor = DocumentProcessor()
        file_type = processor.detect_file_type(file_obj)
        return processor.process_document(file_obj, file_type)
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        raise

# Content Analysis and Tutorial Generation
class ContentAnalyzer:
    def __init__(self, model):
        self.model = model
        
    def analyze_content(self, content: Dict[str, Any]) -> List[Topic]:
        """Analyze content and create topic structure"""
        try:
            # Detect content characteristics
            content_type = self._detect_content_type(content)
            complexity = self._assess_complexity(content)
            structure = self._analyze_structure(content)
            
            # Generate initial topics using AI
            topics = self._generate_topics(content, content_type, complexity)
            
            # Enhance topics with metadata and structure
            enhanced_topics = self._enhance_topics(topics, structure)
            
            return enhanced_topics
            
        except Exception as e:
            st.error(f"Error analyzing content: {str(e)}")
            return self._create_fallback_structure()

    def _detect_content_type(self, content: Dict[str, Any]) -> str:
        """Detect the type of content based on text analysis"""
        text = ' '.join(content['text']) if isinstance(content['text'], list) else content['text']
        
        # Keyword sets for different content types
        keywords = {
            'technical': {
                'code', 'implementation', 'function', 'class', 'method',
                'algorithm', 'programming', 'database', 'API', 'system',
                'technical', 'software', 'development', 'framework'
            },
            'theoretical': {
                'theory', 'concept', 'principle', 'hypothesis', 'analysis',
                'research', 'study', 'methodology', 'framework', 'model',
                'approach', 'perspective', 'paradigm'
            },
            'practical': {
                'guide', 'tutorial', 'step', 'practice', 'example',
                'application', 'use case', 'implementation', 'hands-on',
                'exercise', 'workshop', 'demonstration'
            }
        }
        
        # Calculate scores for each content type
        scores = {}
        text_lower = text.lower()
        for content_type, keyword_set in keywords.items():
            score = sum(1 for keyword in keyword_set if keyword in text_lower)
            scores[content_type] = score / len(keyword_set)  # Normalize score
            
        # Return the content type with highest score
        return max(scores.items(), key=lambda x: x[1])[0]

    def _assess_complexity(self, content: Dict[str, Any]) -> str:
        """Assess content complexity using various metrics"""
        text = ' '.join(content['text']) if isinstance(content['text'], list) else content['text']
        
        # Calculate complexity metrics
        metrics = {
            'avg_word_length': self._calculate_avg_word_length(text),
            'avg_sentence_length': self._calculate_avg_sentence_length(text),
            'technical_term_density': self._calculate_technical_term_density(text),
            'structure_complexity': self._calculate_structure_complexity(content)
        }
        
        # Calculate overall complexity score
        complexity_score = (
            metrics['avg_word_length'] * 0.3 +
            metrics['avg_sentence_length'] * 0.3 +
            metrics['technical_term_density'] * 0.2 +
            metrics['structure_complexity'] * 0.2
        )
        
        # Map score to difficulty level
        if complexity_score > 0.7:
            return "advanced"
        elif complexity_score > 0.4:
            return "intermediate"
        else:
            return "beginner"

    def _calculate_avg_word_length(self, text: str) -> float:
        words = text.split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)

    def _calculate_avg_sentence_length(self, text: str) -> float:
        sentences = text.split('.')
        if not sentences:
            return 0
        return sum(len(sentence.split()) for sentence in sentences) / len(sentences)

    def _calculate_technical_term_density(self, text: str) -> float:
        # List of common technical terms
        technical_terms = {
            'algorithm', 'function', 'method', 'class', 'object',
            'system', 'process', 'database', 'interface', 'api',
            'framework', 'architecture', 'protocol', 'module', 'component'
        }
        
        words = text.lower().split()
        if not words:
            return 0
        return sum(1 for word in words if word in technical_terms) / len(words)

    def _calculate_structure_complexity(self, content: Dict[str, Any]) -> float:
        # Analyze content structure complexity
        structure = content.get('structure', {})
        sections = structure.get('sections', [])
        
        if not sections:
            return 0
            
        # Calculate structural metrics
        depth = max(self._calculate_section_depth(section) for section in sections)
        breadth = len(sections)
        
        # Normalize metrics
        normalized_depth = min(depth / 5, 1)  # Cap at depth of 5
        normalized_breadth = min(breadth / 10, 1)  # Cap at 10 sections
        
        return (normalized_depth + normalized_breadth) / 2

    def _calculate_section_depth(self, section: Dict[str, Any], current_depth: int = 1) -> int:
        """Calculate the depth of nested sections"""
        if not isinstance(section, dict):
            return current_depth
            
        subsections = section.get('subsections', [])
        if not subsections:
            return current_depth
            
        return max(self._calculate_section_depth(subsec, current_depth + 1) 
                  for subsec in subsections)

    def _analyze_structure(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document structure for topic organization"""
        structure = content.get('structure', {})
        
        # Extract sections and their relationships
        sections = structure.get('sections', [])
        analyzed_structure = {
            'main_topics': [],
            'relationships': [],
            'hierarchy': {}
        }
        
        # Process sections
        for i, section in enumerate(sections):
            analyzed_structure['main_topics'].append({
                'title': section.get('title', f'Section {i+1}'),
                'content': section.get('content', []),
                'importance': self._assess_section_importance(section)
            })
            
            # Find relationships between sections
            for j in range(i + 1, len(sections)):
                relationship = self._find_section_relationship(section, sections[j])
                if relationship:
                    analyzed_structure['relationships'].append(relationship)
                    
        # Build topic hierarchy
        analyzed_structure['hierarchy'] = self._build_topic_hierarchy(sections)
        
        return analyzed_structure

    def _assess_section_importance(self, section: Dict[str, Any]) -> float:
        """Assess the importance of a section based on various factors"""
        factors = {
            'content_length': len(' '.join(section.get('content', []))),
            'has_subsections': bool(section.get('subsections')),
            'referenced_by_others': False  # Could be enhanced with cross-reference analysis
        }
        
        # Calculate importance score (0-1)
        importance = (
            min(factors['content_length'] / 1000, 1) * 0.6 +
            (0.3 if factors['has_subsections'] else 0) +
            (0.1 if factors['referenced_by_others'] else 0)
        )
        
        return importance

    def _generate_topics(self, content: Dict[str, Any], content_type: str, 
                        complexity: str) -> List[Topic]:
        """Generate topics using AI model"""
        try:
            # Create prompt for AI
            prompt = self._create_analysis_prompt(content, content_type, complexity)
            
            # Get AI response
            response = self.model.generate_content(prompt)
            structure = json.loads(clean_json_string(response.text))
            
            # Convert AI response to topics
            topics = []
            for lesson in structure.get('lessons', []):
                topic = Topic(
                    title=lesson.get('title', 'Untitled Topic'),
                    content=lesson.get('content', ''),
                    subtopics=[],
                    metadata={
                        'key_points': lesson.get('key_points', []),
                        'practice': lesson.get('practice', []),
                        'difficulty': lesson.get('difficulty', complexity),
                        'content_type': content_type
                    }
                )
                topics.append(topic)
                
            return topics
            
        except Exception as e:
            st.warning(f"Error generating topics: {str(e)}")
            return self._create_fallback_structure()

    def _create_analysis_prompt(self, content: Dict[str, Any], content_type: str, 
                              complexity: str) -> str:
        """Create prompt for AI analysis"""
        return f"""
        Analyze this {content_type} content with {complexity} complexity level.
        Create a learning structure with clear topics and subtopics.
        
        Return a JSON object with this structure:
        {{
            "lessons": [
                {{
                    "title": "Topic Title",
                    "content": "Main content and explanation",
                    "key_points": ["Key Point 1", "Key Point 2"],
                    "practice": ["Practice Item 1", "Practice Item 2"],
                    "difficulty": "{complexity}"
                }}
            ]
        }}
        
        Content to analyze:
        {' '.join(content['text']) if isinstance(content['text'], list) else content['text']}
        """

    
    def _create_fallback_structure(self) -> List[Topic]:
        """Create a basic topic structure when analysis fails"""
        return [Topic(
            title="Introduction to the Subject",
            content="Basic introduction to the subject matter.",
            subtopics=[],
            metadata={
                'key_points': ['Understand basic concepts'],
                'practice': ['Review the material'],
                'difficulty': 'beginner',
                'content_type': 'general'
            }
        )]

    def _enhance_topics(self, topics: List[Topic], structure: Dict[str, Any]) -> List[Topic]:
        """Enhance topics with structural information and relationships"""
        # Map topics to structural elements
        topic_map = {topic.title: topic for topic in topics}
        
        # Add relationships and hierarchy information
        for relationship in structure.get('relationships', []):
            source = relationship.get('source')
            target = relationship.get('target')
            if source in topic_map and target in topic_map:
                if 'relationships' not in topic_map[source].metadata:
                    topic_map[source].metadata['relationships'] = []
                topic_map[source].metadata['relationships'].append(relationship)
                
        # Add importance scores
        for topic in topic_map.values():
            for main_topic in structure.get('main_topics', []):
                if main_topic['title'] == topic.title:
                    topic.metadata['importance'] = main_topic['importance']
                    break
                    
        return list(topic_map.values())

# Tutorial Generation Templates
class TutorialTemplate(ABC):
    @abstractmethod
    def create_tutorial_content(self, topic: Topic, user_performance: float) -> Dict[str, Any]:
        pass

class TechnicalTemplate(TutorialTemplate):
    def create_tutorial_content(self, topic: Topic, user_performance: float) -> Dict[str, Any]:
        difficulty = self._adjust_difficulty(user_performance)
        return {
            "overview": {
                "title": topic.title,
                "description": self._generate_description(topic),
                "prerequisites": self._get_prerequisites(topic),
                "objectives": self._generate_objectives(topic, difficulty)
            },
            "content": {
                "explanation": self._generate_explanation(topic, difficulty),
                "code_examples": self._generate_code_examples(topic, difficulty),
                "practice_exercises": self._generate_exercises(topic, difficulty)
            },
            "difficulty": difficulty
        }

    def _adjust_difficulty(self, user_performance: float) -> str:
        if user_performance >= 85:
            return "advanced"
        elif user_performance >= 65:
            return "intermediate"
        return "beginner"

    def _generate_description(self, topic: Topic) -> str:
        return f"Technical deep-dive into {topic.title}"

    def _get_prerequisites(self, topic: Topic) -> List[str]:
        return ["Basic programming knowledge", "Understanding of data structures"]

    def _generate_objectives(self, topic: Topic, difficulty: str) -> List[str]:
        return [f"Master {point}" for point in topic.metadata.get('key_points', [])]

    def _generate_explanation(self, topic: Topic, difficulty: str) -> str:
        return topic.content

    def _generate_code_examples(self, topic: Topic, difficulty: str) -> List[Dict[str, Any]]:
        return [{"title": "Example", "code": "# Code example"}]

    def _generate_exercises(self, topic: Topic, difficulty: str) -> List[Dict[str, Any]]:
        return [{"title": "Exercise", "description": "Practice exercise"}]

class TheoreticalTemplate(TutorialTemplate):
    def create_tutorial_content(self, topic: Topic, user_performance: float) -> Dict[str, Any]:
        difficulty = self._adjust_difficulty(user_performance)
        return {
            "overview": {
                "title": topic.title,
                "theoretical_framework": self._generate_framework(topic),
                "learning_objectives": self._generate_objectives(topic, difficulty)
            },
            "content": {
                "concepts": self._generate_concepts(topic),
                "examples": self._generate_examples(topic, difficulty),
                "discussion_points": self._generate_discussion_points(topic)
            },
            "difficulty": difficulty
        }

class PracticalTemplate(TutorialTemplate):
    def create_tutorial_content(self, topic: Topic, user_performance: float) -> Dict[str, Any]:
        difficulty = self._adjust_difficulty(user_performance)
        return {
            "overview": {
                "title": topic.title,
                "practical_context": self._generate_context(topic),
                "learning_goals": self._generate_goals(topic, difficulty)
            },
            "content": {
                "steps": self._generate_steps(topic),
                "examples": self._generate_examples(topic, difficulty),
                "exercises": self._generate_exercises(topic, difficulty)
            },
            "difficulty": difficulty
        }

class AdaptiveTutorialGenerator:
    def __init__(self):
        self.templates = {
            'technical': TechnicalTemplate(),
            'theoretical': TheoreticalTemplate(),
            'practical': PracticalTemplate()
        }
        
    def generate_tutorial(self, topic: Topic, user_performance: float) -> Dict[str, Any]:
        """Generate tutorial content based on topic type and user performance"""
        template = self.templates.get(topic.content_type, TechnicalTemplate())
        return template.create_tutorial_content(topic, user_performance)

    def adapt_difficulty(self, user_performance: float, current_difficulty: str) -> str:
        """Adapt difficulty based on user performance"""
        if user_performance >= 85 and current_difficulty != "advanced":
            return "advanced"
        elif user_performance >= 65 and current_difficulty == "beginner":
            return "intermediate"
        elif user_performance < 65 and current_difficulty != "beginner":
            return "beginner"
        return current_difficulty

    def generate_alternative_explanation(self, topic: Topic, 
                                      previous_approach: str) -> Dict[str, Any]:
        """Generate alternative explanation when user struggles"""
        # Define possible approaches
        approaches = ["practical", "theoretical", "technical"]
        
        # Choose next approach (different from previous)
        next_approach = next(
            (a for a in approaches if a != previous_approach), 
            "practical"
        )
        
        # Use corresponding template for new approach
        template = self.templates[next_approach]
        
        # Generate content with modified difficulty
        alternative_content = template.create_tutorial_content(topic, 50.0)  # Start at middle difficulty
        
        # Add meta information about the alternative approach
        alternative_content["meta"] = {
            "approach": next_approach,
            "reason": "Alternative explanation provided due to learning difficulty",
            "focus": self._get_approach_focus(next_approach)
        }
        
        return alternative_content

    def _get_approach_focus(self, approach: str) -> str:
        """Get the focus area for each approach type"""
        focus_map = {
            "practical": "hands-on examples and real-world applications",
            "theoretical": "underlying concepts and principles",
            "technical": "implementation details and specific techniques"
        }
        return focus_map.get(approach, "general understanding")

class TutorialManager:
    """Manages the overall tutorial experience"""
    
    def __init__(self, model):
        self.content_analyzer = ContentAnalyzer(model)
        self.tutorial_generator = AdaptiveTutorialGenerator()
        self.current_performance: float = 50.0
        self.approach_history: List[str] = []
        
    def create_tutorial(self, content: Dict[str, Any]) -> List[Topic]:
        """Create a new tutorial from content"""
        return self.content_analyzer.analyze_content(content)
        
    def generate_next_content(self, topic: Topic) -> Dict[str, Any]:
        """Generate next piece of tutorial content"""
        return self.tutorial_generator.generate_tutorial(
            topic, 
            self.current_performance
        )
        
    def update_performance(self, evaluation_result: Dict[str, Any]) -> None:
        """Update user performance metrics"""
        self.current_performance = evaluation_result.get('understanding_level', self.current_performance)
        
    def get_alternative_content(self, topic: Topic) -> Optional[Dict[str, Any]]:
        """Get alternative explanation if needed"""
        if self.current_performance < 65 and self.approach_history:
            return self.tutorial_generator.generate_alternative_explanation(
                topic,
                self.approach_history[-1]
            )
        return None

class EvaluationEngine:
    """Handles evaluation of user responses and progress"""
    
    def __init__(self, model):
        self.model = model
        
    def evaluate_response(self, user_response: str, 
                         expected_points: List[str], 
                         topic: Topic) -> Dict[str, Any]:
        """Evaluate user response and provide detailed feedback"""
        try:
            # Generate evaluation prompt
            prompt = self._create_evaluation_prompt(
                user_response,
                expected_points,
                topic
            )
            
            # Get AI evaluation
            response = self.model.generate_content(prompt)
            evaluation = json.loads(clean_json_string(response.text))
            
            # Calculate understanding level
            understanding_level = self._calculate_understanding_level(evaluation)
            
            # Format feedback
            feedback = self._format_feedback(evaluation)
            
            return {
                "understanding_level": understanding_level,
                "feedback": feedback,
                "evaluation_data": evaluation,
                "recommendations": self._generate_recommendations(evaluation)
            }
            
        except Exception as e:
            st.error(f"Error in evaluation: {str(e)}")
            return self._create_fallback_evaluation()
            
    def _create_evaluation_prompt(self, user_response: str, 
                                expected_points: List[str], 
                                topic: Topic) -> str:
        """Create prompt for evaluation"""
        return f"""
        Evaluate this response about {topic.title}.
        
        User response: {user_response}
        
        Expected key points: {', '.join(expected_points)}
        
        Return a JSON object with this structure:
        {{
            "points_covered": [
                "Point 1",
                "Point 2"
            ],
            "missing_points": [
                "Point 3"
            ],
            "misconceptions": [
                "Misconception 1"
            ],
            "understanding_level": 75,
            "strengths": [
                "Strength 1"
            ],
            "areas_for_improvement": [
                "Area 1"
            ],
            "suggestions": [
                "Suggestion 1"
            ]
        }}
        
        The understanding_level should be between 0 and 100.
        """
        
    def _calculate_understanding_level(self, evaluation: Dict[str, Any]) -> float:
        """Calculate overall understanding level"""
        base_level = evaluation.get('understanding_level', 50)
        
        # Adjust based on points covered and missing
        points_covered = len(evaluation.get('points_covered', []))
        points_missing = len(evaluation.get('missing_points', []))
        misconceptions = len(evaluation.get('misconceptions', []))
        
        # Apply adjustments
        adjusted_level = base_level
        if points_covered > 0:
            adjusted_level += min(points_covered * 5, 20)
        if points_missing > 0:
            adjusted_level -= min(points_missing * 5, 20)
        if misconceptions > 0:
            adjusted_level -= min(misconceptions * 10, 30)
            
        # Ensure result is between 0 and 100
        return max(0, min(100, adjusted_level))
        
    def _format_feedback(self, evaluation: Dict[str, Any]) -> str:
        """Format evaluation feedback for display"""
        return f"""
## Feedback Analysis 📊

### ✅ Strengths
{"".join(f"- {strength}\n" for strength in evaluation.get('strengths', []))}

### 📝 Areas for Improvement
{"".join(f"- {area}\n" for area in evaluation.get('areas_for_improvement', []))}

### 🎯 Key Points Covered
{"".join(f"- {point}\n" for point in evaluation.get('points_covered', []))}

### ⚠️ Missing Points
{"".join(f"- {point}\n" for point in evaluation.get('missing_points', []))}

### 💡 Suggestions
{"".join(f"- {suggestion}\n" for suggestion in evaluation.get('suggestions', []))}

### Understanding Level
{'▰' * int(evaluation.get('understanding_level', 50)/10)}{'▱' * (10-int(evaluation.get('understanding_level', 50)/10))} {evaluation.get('understanding_level', 50)}%
"""

    def _generate_recommendations(self, evaluation: Dict[str, Any]) -> List[str]:
        """Generate learning recommendations based on evaluation"""
        recommendations = []
        
        # Add recommendations based on missing points
        if evaluation.get('missing_points'):
            recommendations.append("Review the following topics: " + 
                                ", ".join(evaluation['missing_points']))
            
        # Add recommendations based on misconceptions
        if evaluation.get('misconceptions'):
            recommendations.append("Clarify understanding of: " + 
                                ", ".join(evaluation['misconceptions']))
            
        # Add general improvement suggestions
        if evaluation.get('suggestions'):
            recommendations.extend(evaluation['suggestions'])
            
        return recommendations
        
    def _create_fallback_evaluation(self) -> Dict[str, Any]:
        """Create basic evaluation when AI evaluation fails"""
        return {
            "understanding_level": 50,
            "feedback": """
## Feedback Analysis 📊

Thank you for your response. Due to processing limitations, 
detailed feedback couldn't be generated.

### Understanding Level
▰▰▰▰▰▱▱▱▱▱ 50%

Please continue with the next topic.
""",
            "evaluation_data": {},
            "recommendations": ["Continue to next topic"]
        }
# Main Application Logic and Streamlit Interface

def init_gemini(api_key: str = None) -> Optional[Any]:
    """Initialize the Gemini AI model"""
    try:
        if api_key:
            genai.configure(api_key=api_key)
            return genai.GenerativeModel('gemini-pro')
        return None
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        return None

def init_session_state():
    """Initialize or reset session state"""
    if 'tutorial_manager' not in st.session_state:
        st.session_state.tutorial_manager = None
    if 'current_topic_index' not in st.session_state:
        st.session_state.current_topic_index = 0
    if 'topics' not in st.session_state:
        st.session_state.topics = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = {
            'understanding_level': 50.0,
            'completed_topics': [],
            'current_approach': 'practical'
        }

def render_sidebar():
    """Render the application sidebar"""
    with st.sidebar:
        st.markdown("## 📚 Learning Progress")
        
        if st.session_state.topics:
            # Display progress bar
            completed = len(st.session_state.user_progress['completed_topics'])
            total = len(st.session_state.topics)
            progress = int((completed / total) * 100)
            st.progress(progress)
            st.markdown(f"**{progress}%** completed ({completed}/{total} topics)")
            
            # Display topic list
            st.markdown("### Topics Overview")
            for i, topic in enumerate(st.session_state.topics):
                status = "✅" if topic.title in st.session_state.user_progress['completed_topics'] else \
                         "📍" if i == st.session_state.current_topic_index else "⭕️"
                st.markdown(
                    f"<div class='topic-item {status}'>{i+1}. {topic.title}</div>",
                    unsafe_allow_html=True
                )
            
            # Display current stats
            st.markdown("### Current Stats")
            st.markdown(f"Understanding Level: {st.session_state.user_progress['understanding_level']:.1f}%")
            st.markdown(f"Learning Approach: {st.session_state.user_progress['current_approach'].title()}")
            
            # Reset button
            if st.button("🔄 Reset Tutorial"):
                reset_session()
                st.rerun()

def render_header():
    """Render the application header"""
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>🎓 AI Learning Assistant</h1>
            <p style='font-size: 1.2rem; color: #4B5563;'>
                Your personalized learning journey with AI guidance
            </p>
        </div>
    """, unsafe_allow_html=True)

def render_file_upload():
    """Render the file upload section"""
    st.markdown("""
        <div class='content-block'>
            <h2>📚 Upload Learning Material</h2>
            <p>Upload your educational content to begin the learning journey.</p>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Upload Educational Content",
        type=["pdf", "docx", "md"],
        help="Supported formats: PDF, Word documents, and Markdown files"
    )
    
    return uploaded_file

def render_chat_interface():
    """Render the chat interface"""
    if not st.session_state.topics:
        return
        
    # Get current topic
    current_topic = st.session_state.topics[st.session_state.current_topic_index]
    
    # Display conversation history
    for message in st.session_state.conversation_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    
    # Generate new content if needed
    if not st.session_state.conversation_history or \
       st.session_state.conversation_history[-1]["role"] == "user":
        tutorial_content = st.session_state.tutorial_manager.generate_next_content(current_topic)
        
        with st.chat_message("assistant"):
            st.markdown(tutorial_content["content"], unsafe_allow_html=True)
        
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": tutorial_content["content"]
        })
    
    # Handle user input
    user_input = st.chat_input("Your response...")
    if user_input:
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input, unsafe_allow_html=True)
        
        st.session_state.conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Evaluate response
        evaluation = st.session_state.tutorial_manager.evaluate_response(
            user_input,
            current_topic
        )
        
        # Display feedback
        with st.chat_message("assistant"):
            st.markdown(evaluation["feedback"], unsafe_allow_html=True)
        
        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": evaluation["feedback"]
        })
        
        # Update progress
        st.session_state.user_progress['understanding_level'] = evaluation["understanding_level"]
        
        # Check if topic is completed
        if evaluation["understanding_level"] >= 70:
            st.session_state.user_progress['completed_topics'].append(current_topic.title)
            advance_topic()
            st.rerun()

def handle_file_upload(uploaded_file):
    """Process uploaded file and initialize tutorial"""
    try:
        with st.spinner("🔄 Processing content..."):
            # Process the file
            content = process_uploaded_file(uploaded_file)
            
            # Initialize tutorial manager if not exists
            if not st.session_state.tutorial_manager:
                st.session_state.tutorial_manager = TutorialManager(st.session_state.model)
            
            # Create tutorial
            st.session_state.topics = st.session_state.tutorial_manager.create_tutorial(content)
            
            st.success("✅ Tutorial created successfully! Let's begin.")
            st.rerun()
            
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")

def advance_topic():
    """Advance to the next topic"""
    if st.session_state.current_topic_index < len(st.session_state.topics) - 1:
        st.session_state.current_topic_index += 1
        st.session_state.conversation_history = []
    else:
        show_completion_message()

def show_completion_message():
    """Show tutorial completion message"""
    st.balloons()
    st.success("""
        🎉 Congratulations! You've completed the tutorial!
        
        Here's your learning summary:
        - Topics completed: {} out of {}
        - Final understanding level: {:.1f}%
        - Learning approach: {}
        
        Would you like to:
        1. Start a new tutorial with different content
        2. Review specific topics
        3. Get a detailed progress report
    """.format(
        len(st.session_state.user_progress['completed_topics']),
        len(st.session_state.topics),
        st.session_state.user_progress['understanding_level'],
        st.session_state.user_progress['current_approach']
    ))

def reset_session():
    """Reset the session state"""
    st.session_state.tutorial_manager = None
    st.session_state.current_topic_index = 0
    st.session_state.topics = []
    st.session_state.conversation_history = []
    st.session_state.user_progress = {
        'understanding_level': 50.0,
        'completed_topics': [],
        'current_approach': 'practical'
    }

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        /* Global Styles */
        .main {
            padding: 1rem;
            max-width: 1200px;
            margin: 0 auto;
        }
        
        /* Typography */
        h1 {
            color: #1E3A8A;
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }
        h2 {
            color: #2563EB;
            font-size: 1.8rem;
            margin-bottom: 0.8rem;
        }
        h3 {
            color: #3B82F6;
            font-size: 1.5rem;
            margin-bottom: 0.6rem;
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
        
        /* Chat Interface */
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
        
        /* Topic List */
        .topic-item {
            padding: 0.75rem;
            margin: 0.25rem 0;
            border-radius: 4px;
            transition: background-color 0.2s;
        }
        .topic-item:hover {
            background: #F3F4F6;
        }
        .topic-item.✅ {
            color: #059669;
        }
        .topic-item.📍 {
            color: #2563EB;
            font-weight: bold;
        }
        
        /* Progress Bar */
        .stProgress .st-bo {
            height: 8px;
            border-radius: 4px;
        }
        
        /* Buttons */
        .stButton button {
            background: #2563EB;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-weight: 500;
            transition: background-color 0.2s;
        }
        .stButton button:hover {
            background: #1D4ED8;
        }
        
        /* File Uploader */
        .stFileUploader {
            padding: 2rem;
            border: 2px dashed #E5E7EB;
            border-radius: 8px;
            text-align: center;
        }
        
        /* Feedback Sections */
        .feedback-section {
            margin: 1rem 0;
            padding: 1rem;
            border-left: 4px solid #2563EB;
            background: #F8FAFC;
        }
        </style>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    # Page configuration
    st.set_page_config(
        page_title="AI Learning Assistant 📚",
        page_icon="🎓",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize session state
    init_session_state()
    
    # Get API key
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
    
    # Render header
    render_header()
    
    # Create two-column layout
    col1, col2 = st.columns([7, 3])
    
    with col1:
        if not st.session_state.topics:
            uploaded_file = render_file_upload()
            if uploaded_file:
                handle_file_upload(uploaded_file)
        else:
            render_chat_interface()
    
    with col2:
        render_sidebar()

if __name__ == "__main__":
    main()
