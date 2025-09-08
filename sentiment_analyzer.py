"""
SentilensAI - Advanced Sentiment Analysis for AI Chatbot Messages

This module provides comprehensive sentiment analysis capabilities specifically designed
for analyzing AI chatbot conversations using LangChain integration and multiple ML models.

Features:
- Multi-model sentiment analysis (VADER, TextBlob, spaCy, Transformers)
- LangChain integration for intelligent conversation analysis
- Real-time sentiment tracking for chatbot interactions
- Advanced emotion detection and classification
- Context-aware sentiment analysis for conversational AI

Author: Pravin Selvamuthu
Repository: https://github.com/kernelseed/sentilens-ai
"""

import re
import json
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# LangChain Integration
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.output_parsers import BaseOutputParser

# Transformers for advanced sentiment analysis
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Multilingual support
try:
    from multilingual_sentiment import MultilingualSentimentAnalyzer, MultilingualSentimentResult
    MULTILINGUAL_AVAILABLE = True
except ImportError:
    MULTILINGUAL_AVAILABLE = False

# spaCy for advanced NLP
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SentimentResult:
    """Data class for sentiment analysis results"""
    text: str
    sentiment: str  # positive, negative, neutral
    confidence: float
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    emotions: Dict[str, float]
    timestamp: datetime
    model_used: str
    metadata: Dict[str, Any]


@dataclass
class ChatbotMessage:
    """Data class for chatbot message analysis"""
    message_id: str
    user_message: str
    bot_response: str
    timestamp: datetime
    conversation_id: str
    user_sentiment: SentimentResult
    bot_sentiment: SentimentResult
    conversation_sentiment: str
    satisfaction_score: float


class SentimentOutputParser(BaseOutputParser):
    """Custom output parser for LangChain sentiment analysis"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse sentiment analysis output from LLM"""
        try:
            # Try to parse as JSON first
            if text.strip().startswith('{'):
                return json.loads(text)
            
            # Extract sentiment information using regex
            sentiment_match = re.search(r'sentiment["\']?\s*:\s*["\']?(\w+)', text, re.IGNORECASE)
            confidence_match = re.search(r'confidence["\']?\s*:\s*([0-9.]+)', text, re.IGNORECASE)
            polarity_match = re.search(r'polarity["\']?\s*:\s*([-0-9.]+)', text, re.IGNORECASE)
            
            result = {
                'sentiment': sentiment_match.group(1).lower() if sentiment_match else 'neutral',
                'confidence': float(confidence_match.group(1)) if confidence_match else 0.5,
                'polarity': float(polarity_match.group(1)) if polarity_match else 0.0,
                'raw_output': text
            }
            
            return result
        except Exception as e:
            logger.warning(f"Failed to parse sentiment output: {e}")
            return {
                'sentiment': 'neutral',
                'confidence': 0.5,
                'polarity': 0.0,
                'raw_output': text
            }


class SentilensAIAnalyzer:
    """
    Advanced sentiment analysis for AI chatbot messages using multiple models and LangChain
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, model_cache_dir: str = "./model_cache", 
                 enable_multilingual: bool = True):
        """
        Initialize the SentimentsAI analyzer
        
        Args:
            openai_api_key: OpenAI API key for LangChain integration
            model_cache_dir: Directory to cache downloaded models
            enable_multilingual: Enable multilingual support for English, Spanish, and Chinese
        """
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Multilingual support
        self.enable_multilingual = enable_multilingual and MULTILINGUAL_AVAILABLE
        if self.enable_multilingual:
            try:
                self.multilingual_analyzer = MultilingualSentimentAnalyzer()
                logger.info("✅ Multilingual support enabled (English, Spanish, Chinese)")
            except Exception as e:
                logger.warning(f"Failed to initialize multilingual analyzer: {e}")
                self.enable_multilingual = False
        else:
            self.multilingual_analyzer = None
        
        # Initialize sentiment analyzers
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
        
        # Initialize spaCy model
        self.spacy_model = None
        if SPACY_AVAILABLE:
            try:
                self.spacy_model = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model 'en_core_web_sm' not found. Install with: python -m spacy download en_core_web_sm")
        
        # Initialize transformers pipeline
        self.transformers_pipeline = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.transformers_pipeline = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    cache_dir=self.model_cache_dir
                )
            except Exception as e:
                logger.warning(f"Failed to load transformers pipeline: {e}")
        
        # Initialize LangChain components
        self.llm = None
        self.sentiment_chain = None
        if openai_api_key:
            try:
                self.llm = OpenAI(api_key=openai_api_key, temperature=0.1)
                self._setup_langchain_components()
            except Exception as e:
                logger.warning(f"Failed to initialize OpenAI LLM: {e}")
        
        # Emotion detection patterns
        self.emotion_patterns = {
            'joy': [r'\b(happy|joy|excited|great|wonderful|amazing|fantastic|love|adore)\b'],
            'sadness': [r'\b(sad|depressed|upset|disappointed|hurt|grief|sorrow)\b'],
            'anger': [r'\b(angry|mad|furious|rage|annoyed|irritated|frustrated)\b'],
            'fear': [r'\b(scared|afraid|worried|anxious|nervous|terrified|panic)\b'],
            'surprise': [r'\b(surprised|shocked|amazed|wow|incredible|unbelievable)\b'],
            'disgust': [r'\b(disgusted|revolted|sick|gross|nasty|awful|terrible)\b']
        }
    
    def _setup_langchain_components(self):
        """Setup LangChain components for sentiment analysis"""
        if not self.llm:
            return
        
        # Create sentiment analysis prompt template
        sentiment_prompt = PromptTemplate(
            input_variables=["text", "context"],
            template="""
            Analyze the sentiment of the following text from an AI chatbot conversation.
            Consider the context of the conversation and provide a detailed sentiment analysis.
            
            Text: "{text}"
            Context: "{context}"
            
            Please provide your analysis in the following JSON format:
            {{
                "sentiment": "positive|negative|neutral",
                "confidence": 0.0-1.0,
                "polarity": -1.0 to 1.0,
                "reasoning": "Brief explanation of your analysis",
                "emotions": {{
                    "joy": 0.0-1.0,
                    "sadness": 0.0-1.0,
                    "anger": 0.0-1.0,
                    "fear": 0.0-1.0,
                    "surprise": 0.0-1.0,
                    "disgust": 0.0-1.0
                }}
            }}
            """
        )
        
        # Create the sentiment analysis chain
        self.sentiment_chain = LLMChain(
            llm=self.llm,
            prompt=sentiment_prompt,
            output_parser=SentimentOutputParser()
        )
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for sentiment analysis
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs, mentions, and hashtags
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\!\?\,\;\:]', '', text)
        
        return text
    
    def extract_emotions(self, text: str) -> Dict[str, float]:
        """
        Extract emotion scores from text using pattern matching
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of emotion scores
        """
        emotions = {emotion: 0.0 for emotion in self.emotion_patterns.keys()}
        
        for emotion, patterns in self.emotion_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                emotions[emotion] += len(matches) * 0.1  # Simple scoring
        
        # Normalize scores
        total_score = sum(emotions.values())
        if total_score > 0:
            emotions = {k: min(v / total_score, 1.0) for k, v in emotions.items()}
        
        return emotions
    
    def analyze_with_vader(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using VADER"""
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(scores['compound']),
            'polarity': scores['compound'],
            'subjectivity': 0.5,  # VADER doesn't provide subjectivity
            'scores': scores
        }
    
    def analyze_with_textblob(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using TextBlob"""
        blob = TextBlob(text)
        
        # Determine sentiment
        if blob.sentiment.polarity > 0.1:
            sentiment = 'positive'
        elif blob.sentiment.polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(blob.sentiment.polarity),
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity
        }
    
    def analyze_with_spacy(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using spaCy (if available)"""
        if not self.spacy_model:
            return self.analyze_with_textblob(text)  # Fallback
        
        doc = self.spacy_model(text)
        
        # Simple sentiment analysis using spaCy's token attributes
        positive_words = 0
        negative_words = 0
        total_words = 0
        
        for token in doc:
            if not token.is_stop and not token.is_punct and token.is_alpha:
                total_words += 1
                # Simple heuristic based on word sentiment
                if token.lemma_.lower() in ['good', 'great', 'excellent', 'amazing', 'wonderful']:
                    positive_words += 1
                elif token.lemma_.lower() in ['bad', 'terrible', 'awful', 'horrible', 'worst']:
                    negative_words += 1
        
        if total_words == 0:
            polarity = 0.0
        else:
            polarity = (positive_words - negative_words) / total_words
        
        # Determine sentiment
        if polarity > 0.1:
            sentiment = 'positive'
        elif polarity < -0.1:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        return {
            'sentiment': sentiment,
            'confidence': abs(polarity),
            'polarity': polarity,
            'subjectivity': 0.5  # spaCy doesn't provide subjectivity
        }
    
    def analyze_with_transformers(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using Transformers (if available)"""
        if not self.transformers_pipeline:
            return self.analyze_with_textblob(text)  # Fallback
        
        try:
            result = self.transformers_pipeline(text)[0]
            
            # Map transformer labels to our format
            label_mapping = {
                'LABEL_0': 'negative',
                'LABEL_1': 'neutral', 
                'LABEL_2': 'positive'
            }
            
            sentiment = label_mapping.get(result['label'], 'neutral')
            confidence = result['score']
            
            # Estimate polarity from confidence and sentiment
            if sentiment == 'positive':
                polarity = confidence
            elif sentiment == 'negative':
                polarity = -confidence
            else:
                polarity = 0.0
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': 0.5  # Transformers don't provide subjectivity
            }
        except Exception as e:
            logger.warning(f"Transformers analysis failed: {e}")
            return self.analyze_with_textblob(text)  # Fallback
    
    def analyze_with_langchain(self, text: str, context: str = "") -> Dict[str, Any]:
        """Analyze sentiment using LangChain and LLM"""
        if not self.sentiment_chain:
            return self.analyze_with_textblob(text)  # Fallback
        
        try:
            result = self.sentiment_chain.run(text=text, context=context)
            
            # Ensure we have the required fields
            if not isinstance(result, dict):
                result = {'sentiment': 'neutral', 'confidence': 0.5, 'polarity': 0.0}
            
            # Validate and normalize the result
            sentiment = result.get('sentiment', 'neutral')
            if sentiment not in ['positive', 'negative', 'neutral']:
                sentiment = 'neutral'
            
            confidence = max(0.0, min(1.0, float(result.get('confidence', 0.5))))
            polarity = max(-1.0, min(1.0, float(result.get('polarity', 0.0))))
            
            # Extract emotions if available
            emotions = result.get('emotions', {})
            if not isinstance(emotions, dict):
                emotions = self.extract_emotions(text)
            
            return {
                'sentiment': sentiment,
                'confidence': confidence,
                'polarity': polarity,
                'subjectivity': 0.5,  # LLM doesn't provide subjectivity
                'emotions': emotions,
                'reasoning': result.get('reasoning', '')
            }
        except Exception as e:
            logger.warning(f"LangChain analysis failed: {e}")
            return self.analyze_with_textblob(text)  # Fallback
    
    def analyze_sentiment(self, text: str, method: str = 'ensemble', context: str = "") -> SentimentResult:
        """
        Analyze sentiment using specified method
        
        Args:
            text: Text to analyze
            method: Analysis method ('vader', 'textblob', 'spacy', 'transformers', 'langchain', 'ensemble')
            context: Additional context for analysis
            
        Returns:
            SentimentResult object
        """
        if not text or not text.strip():
            return SentimentResult(
                text=text,
                sentiment='neutral',
                confidence=0.0,
                polarity=0.0,
                subjectivity=0.0,
                emotions={},
                timestamp=datetime.now(),
                model_used=method,
                metadata={}
            )
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        if method == 'ensemble':
            # Use ensemble of all available methods
            results = []
            
            # VADER
            vader_result = self.analyze_with_vader(processed_text)
            results.append(vader_result)
            
            # TextBlob
            textblob_result = self.analyze_with_textblob(processed_text)
            results.append(textblob_result)
            
            # spaCy
            spacy_result = self.analyze_with_spacy(processed_text)
            results.append(spacy_result)
            
            # Transformers
            if self.transformers_pipeline:
                transformers_result = self.analyze_with_transformers(processed_text)
                results.append(transformers_result)
            
            # LangChain
            if self.sentiment_chain:
                langchain_result = self.analyze_with_langchain(processed_text, context)
                results.append(langchain_result)
            
            # Ensemble voting
            sentiment_votes = [r['sentiment'] for r in results]
            sentiment_counts = {s: sentiment_votes.count(s) for s in set(sentiment_votes)}
            final_sentiment = max(sentiment_counts, key=sentiment_counts.get)
            
            # Average confidence and polarity
            avg_confidence = np.mean([r['confidence'] for r in results])
            avg_polarity = np.mean([r['polarity'] for r in results])
            avg_subjectivity = np.mean([r.get('subjectivity', 0.5) for r in results])
            
            # Combine emotions
            all_emotions = {}
            for result in results:
                if 'emotions' in result:
                    for emotion, score in result['emotions'].items():
                        all_emotions[emotion] = all_emotions.get(emotion, 0) + score
            emotions = {k: v / len(results) for k, v in all_emotions.items()}
            
            if not emotions:
                emotions = self.extract_emotions(processed_text)
            
            final_result = {
                'sentiment': final_sentiment,
                'confidence': avg_confidence,
                'polarity': avg_polarity,
                'subjectivity': avg_subjectivity,
                'emotions': emotions
            }
            
        else:
            # Use specific method
            if method == 'vader':
                final_result = self.analyze_with_vader(processed_text)
            elif method == 'textblob':
                final_result = self.analyze_with_textblob(processed_text)
            elif method == 'spacy':
                final_result = self.analyze_with_spacy(processed_text)
            elif method == 'transformers':
                final_result = self.analyze_with_transformers(processed_text)
            elif method == 'langchain':
                final_result = self.analyze_with_langchain(processed_text, context)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Extract emotions if not provided
            if 'emotions' not in final_result:
                final_result['emotions'] = self.extract_emotions(processed_text)
        
        return SentimentResult(
            text=text,
            sentiment=final_result['sentiment'],
            confidence=final_result['confidence'],
            polarity=final_result['polarity'],
            subjectivity=final_result.get('subjectivity', 0.5),
            emotions=final_result['emotions'],
            timestamp=datetime.now(),
            model_used=method,
            metadata=final_result
        )
    
    def analyze_sentiment_multilingual(self, text: str, target_language: Optional[str] = None,
                                     enable_cross_language: bool = False) -> MultilingualSentimentResult:
        """
        Analyze sentiment with multilingual support (English, Spanish, Chinese)
        
        Args:
            text: Text to analyze
            target_language: Specific language to use ('en', 'es', 'zh') or None for auto-detection
            enable_cross_language: Enable cross-language consensus analysis
            
        Returns:
            MultilingualSentimentResult object
        """
        if not self.enable_multilingual or not self.multilingual_analyzer:
            # Fallback to regular analysis
            regular_result = self.analyze_sentiment(text, method='ensemble')
            return MultilingualSentimentResult(
                text=text,
                detected_language='en',
                language_confidence=0.5,
                sentiment=regular_result.sentiment,
                confidence=regular_result.confidence,
                emotions=regular_result.emotions,
                methods_used=[regular_result.model_used],
                language_specific_analysis={'fallback': True}
            )
        
        return self.multilingual_analyzer.analyze_sentiment_multilingual(
            text, target_language, enable_cross_language
        )
    
    def analyze_conversation_multilingual(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a conversation with multilingual support
        
        Args:
            conversation: Conversation dictionary with messages
            
        Returns:
            Dictionary with multilingual analysis results
        """
        if not self.enable_multilingual or not self.multilingual_analyzer:
            # Fallback to regular analysis
            messages = conversation.get('messages', [])
            regular_results = []
            for msg in messages:
                user_text = msg.get('user', '')
                bot_text = msg.get('bot', '')
                if user_text:
                    regular_results.append(self.analyze_sentiment(user_text))
                if bot_text:
                    regular_results.append(self.analyze_sentiment(bot_text))
            return {'fallback': True, 'results': regular_results}
        
        return self.multilingual_analyzer.analyze_conversation_multilingual(conversation)
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages for multilingual analysis"""
        if self.enable_multilingual and self.multilingual_analyzer:
            return self.multilingual_analyzer.get_supported_languages()
        return ['en']  # Default to English only
    
    def get_language_name(self, language_code: str) -> str:
        """Get human-readable language name"""
        if self.enable_multilingual and self.multilingual_analyzer:
            return self.multilingual_analyzer.get_language_name(language_code)
        return {'en': 'English'}.get(language_code, language_code)
    
    def analyze_chatbot_conversation(self, messages: List[Dict[str, Any]]) -> List[ChatbotMessage]:
        """
        Analyze a complete chatbot conversation
        
        Args:
            messages: List of message dictionaries with 'user', 'bot', 'timestamp', 'conversation_id'
            
        Returns:
            List of ChatbotMessage objects
        """
        results = []
        
        for i, msg in enumerate(messages):
            user_text = msg.get('user', '')
            bot_text = msg.get('bot', '')
            timestamp = msg.get('timestamp', datetime.now())
            conversation_id = msg.get('conversation_id', f'conv_{i}')
            message_id = msg.get('message_id', f'{conversation_id}_{i}')
            
            # Analyze user message
            user_sentiment = self.analyze_sentiment(user_text, method='ensemble')
            
            # Analyze bot response
            bot_sentiment = self.analyze_sentiment(bot_text, method='ensemble', context=user_text)
            
            # Determine overall conversation sentiment
            if user_sentiment.sentiment == bot_sentiment.sentiment:
                conversation_sentiment = user_sentiment.sentiment
            else:
                # Use weighted average based on confidence
                user_weight = user_sentiment.confidence
                bot_weight = bot_sentiment.confidence
                total_weight = user_weight + bot_weight
                
                if total_weight > 0:
                    user_polarity_weighted = user_sentiment.polarity * (user_weight / total_weight)
                    bot_polarity_weighted = bot_sentiment.polarity * (bot_weight / total_weight)
                    combined_polarity = user_polarity_weighted + bot_polarity_weighted
                    
                    if combined_polarity > 0.1:
                        conversation_sentiment = 'positive'
                    elif combined_polarity < -0.1:
                        conversation_sentiment = 'negative'
                    else:
                        conversation_sentiment = 'neutral'
                else:
                    conversation_sentiment = 'neutral'
            
            # Calculate satisfaction score (0-1)
            satisfaction_score = self._calculate_satisfaction_score(user_sentiment, bot_sentiment)
            
            chatbot_message = ChatbotMessage(
                message_id=message_id,
                user_message=user_text,
                bot_response=bot_text,
                timestamp=timestamp,
                conversation_id=conversation_id,
                user_sentiment=user_sentiment,
                bot_sentiment=bot_sentiment,
                conversation_sentiment=conversation_sentiment,
                satisfaction_score=satisfaction_score
            )
            
            results.append(chatbot_message)
        
        return results
    
    def _calculate_satisfaction_score(self, user_sentiment: SentimentResult, bot_sentiment: SentimentResult) -> float:
        """Calculate satisfaction score based on sentiment alignment"""
        # Base score from user sentiment
        base_score = (user_sentiment.polarity + 1) / 2  # Convert -1,1 to 0,1
        
        # Adjust based on bot response sentiment
        if user_sentiment.sentiment == 'positive' and bot_sentiment.sentiment == 'positive':
            adjustment = 0.2
        elif user_sentiment.sentiment == 'negative' and bot_sentiment.sentiment == 'positive':
            adjustment = 0.3  # Bot being positive to negative user is good
        elif user_sentiment.sentiment == 'neutral' and bot_sentiment.sentiment == 'positive':
            adjustment = 0.1
        else:
            adjustment = -0.1
        
        # Factor in confidence
        confidence_factor = (user_sentiment.confidence + bot_sentiment.confidence) / 2
        
        final_score = base_score + adjustment
        final_score = max(0.0, min(1.0, final_score))  # Clamp to 0-1
        
        return final_score * confidence_factor
    
    def get_sentiment_summary(self, results: List[SentimentResult]) -> Dict[str, Any]:
        """Get summary statistics for sentiment analysis results"""
        if not results:
            return {}
        
        sentiments = [r.sentiment for r in results]
        confidences = [r.confidence for r in results]
        polarities = [r.polarity for r in results]
        
        sentiment_counts = {s: sentiments.count(s) for s in set(sentiments)}
        total = len(sentiments)
        
        return {
            'total_messages': total,
            'sentiment_distribution': {k: v/total for k, v in sentiment_counts.items()},
            'average_confidence': np.mean(confidences),
            'average_polarity': np.mean(polarities),
            'sentiment_trend': sentiments,
            'confidence_trend': confidences,
            'polarity_trend': polarities
        }
    
    def export_results(self, results: List[Union[SentimentResult, ChatbotMessage]], 
                      filename: str, format: str = 'json') -> str:
        """
        Export analysis results to file
        
        Args:
            results: List of analysis results
            filename: Output filename
            format: Export format ('json', 'csv', 'excel')
            
        Returns:
            Path to exported file
        """
        output_path = Path(filename)
        
        if format == 'json':
            # Convert results to dictionaries
            data = []
            for result in results:
                if isinstance(result, SentimentResult):
                    data.append({
                        'text': result.text,
                        'sentiment': result.sentiment,
                        'confidence': result.confidence,
                        'polarity': result.polarity,
                        'subjectivity': result.subjectivity,
                        'emotions': result.emotions,
                        'timestamp': result.timestamp.isoformat(),
                        'model_used': result.model_used
                    })
                elif isinstance(result, ChatbotMessage):
                    data.append({
                        'message_id': result.message_id,
                        'user_message': result.user_message,
                        'bot_response': result.bot_response,
                        'timestamp': result.timestamp.isoformat(),
                        'conversation_id': result.conversation_id,
                        'user_sentiment': result.user_sentiment.sentiment,
                        'user_confidence': result.user_sentiment.confidence,
                        'user_polarity': result.user_sentiment.polarity,
                        'bot_sentiment': result.bot_sentiment.sentiment,
                        'bot_confidence': result.bot_sentiment.confidence,
                        'bot_polarity': result.bot_sentiment.polarity,
                        'conversation_sentiment': result.conversation_sentiment,
                        'satisfaction_score': result.satisfaction_score
                    })
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            # Convert to DataFrame and save as CSV
            data = []
            for result in results:
                if isinstance(result, SentimentResult):
                    data.append({
                        'text': result.text,
                        'sentiment': result.sentiment,
                        'confidence': result.confidence,
                        'polarity': result.polarity,
                        'subjectivity': result.subjectivity,
                        'timestamp': result.timestamp.isoformat(),
                        'model_used': result.model_used
                    })
                elif isinstance(result, ChatbotMessage):
                    data.append({
                        'message_id': result.message_id,
                        'user_message': result.user_message,
                        'bot_response': result.bot_response,
                        'timestamp': result.timestamp.isoformat(),
                        'conversation_id': result.conversation_id,
                        'user_sentiment': result.user_sentiment.sentiment,
                        'user_confidence': result.user_sentiment.confidence,
                        'bot_sentiment': result.bot_sentiment.sentiment,
                        'bot_confidence': result.bot_sentiment.confidence,
                        'conversation_sentiment': result.conversation_sentiment,
                        'satisfaction_score': result.satisfaction_score
                    })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        elif format == 'excel':
            # Convert to DataFrame and save as Excel
            data = []
            for result in results:
                if isinstance(result, SentimentResult):
                    data.append({
                        'text': result.text,
                        'sentiment': result.sentiment,
                        'confidence': result.confidence,
                        'polarity': result.polarity,
                        'subjectivity': result.subjectivity,
                        'timestamp': result.timestamp.isoformat(),
                        'model_used': result.model_used
                    })
                elif isinstance(result, ChatbotMessage):
                    data.append({
                        'message_id': result.message_id,
                        'user_message': result.user_message,
                        'bot_response': result.bot_response,
                        'timestamp': result.timestamp.isoformat(),
                        'conversation_id': result.conversation_id,
                        'user_sentiment': result.user_sentiment.sentiment,
                        'user_confidence': result.user_sentiment.confidence,
                        'bot_sentiment': result.bot_sentiment.sentiment,
                        'bot_confidence': result.bot_sentiment.confidence,
                        'conversation_sentiment': result.conversation_sentiment,
                        'satisfaction_score': result.satisfaction_score
                    })
            
            df = pd.DataFrame(data)
            df.to_excel(output_path, index=False, engine='openpyxl')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_path)


def main():
    """Demo function to showcase SentimentsAI capabilities"""
    print("🤖 SentilensAI - Advanced Sentiment Analysis for AI Chatbot Messages")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = SentilensAIAnalyzer()
    
    # Sample chatbot messages
    sample_messages = [
        {
            'user': 'I love this chatbot! It\'s so helpful and friendly.',
            'bot': 'Thank you so much! I\'m thrilled to hear that you\'re having a great experience. Is there anything else I can help you with today?',
            'timestamp': datetime.now(),
            'conversation_id': 'demo_001'
        },
        {
            'user': 'This is terrible. The bot keeps giving me wrong answers.',
            'bot': 'I apologize for the confusion. Let me help you get the correct information. Could you please provide more details about what you\'re looking for?',
            'timestamp': datetime.now(),
            'conversation_id': 'demo_002'
        },
        {
            'user': 'Can you help me with my account balance?',
            'bot': 'Of course! I\'d be happy to help you check your account balance. Please provide your account number or login credentials.',
            'timestamp': datetime.now(),
            'conversation_id': 'demo_003'
        }
    ]
    
    print("\n📊 Analyzing sample chatbot conversations...")
    
    # Analyze conversations
    results = analyzer.analyze_chatbot_conversation(sample_messages)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n--- Conversation {i} ---")
        print(f"User: {result.user_message}")
        print(f"Bot: {result.bot_response}")
        print(f"User Sentiment: {result.user_sentiment.sentiment} (confidence: {result.user_sentiment.confidence:.2f})")
        print(f"Bot Sentiment: {result.bot_sentiment.sentiment} (confidence: {result.bot_sentiment.confidence:.2f})")
        print(f"Conversation Sentiment: {result.conversation_sentiment}")
        print(f"Satisfaction Score: {result.satisfaction_score:.2f}")
    
    # Get summary
    sentiment_results = [r.user_sentiment for r in results] + [r.bot_sentiment for r in results]
    summary = analyzer.get_sentiment_summary(sentiment_results)
    
    print(f"\n📈 Summary Statistics:")
    print(f"Total Messages: {summary['total_messages']}")
    print(f"Sentiment Distribution: {summary['sentiment_distribution']}")
    print(f"Average Confidence: {summary['average_confidence']:.2f}")
    print(f"Average Polarity: {summary['average_polarity']:.2f}")
    
    # Export results
    output_file = analyzer.export_results(results, 'sentiment_analysis_results.json', 'json')
    print(f"\n💾 Results exported to: {output_file}")
    
    print("\n✅ SentilensAI demo completed successfully!")
    print("🚀 Ready for production use with LangChain and ML models!")


if __name__ == "__main__":
    main()
