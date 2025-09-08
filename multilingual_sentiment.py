#!/usr/bin/env python3
"""
SentilensAI - Multilingual Sentiment Analysis Module

Advanced multilingual sentiment analysis supporting:
- English (en)
- Spanish (es) 
- Chinese (zh)
- Automatic language detection
- Language-specific sentiment models
- Cross-language sentiment comparison

Author: Pravin Selvamuthu
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime

# Multilingual NLP libraries
try:
    import langdetect
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultilingualSentimentResult:
    """Result of multilingual sentiment analysis"""
    text: str
    detected_language: str
    language_confidence: float
    sentiment: str
    confidence: float
    emotions: Dict[str, float]
    methods_used: List[str]
    language_specific_analysis: Dict[str, Any]
    cross_language_consensus: Optional[Dict[str, Any]] = None

class MultilingualSentimentAnalyzer:
    """Advanced multilingual sentiment analyzer for English, Spanish, and Chinese"""
    
    def __init__(self, model_cache_dir: str = "./multilingual_models"):
        self.model_cache_dir = model_cache_dir
        self.supported_languages = ['en', 'es', 'zh']
        self.language_names = {
            'en': 'English',
            'es': 'Spanish', 
            'zh': 'Chinese'
        }
        
        # Language detection patterns
        self.language_patterns = {
            'en': r'[a-zA-Z]',
            'es': r'[ñáéíóúüÑÁÉÍÓÚÜ]',
            'zh': r'[\u4e00-\u9fff]'
        }
        
        # Language-specific sentiment models
        self.sentiment_models = {
            'en': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'es': 'pysentimiento/robertuito-sentiment-analysis',
            'zh': 'uer/roberta-base-finetuned-dianping-chinese'
        }
        
        # Initialize language-specific models
        self.models = {}
        self.tokenizers = {}
        self._load_language_models()
    
    def _load_language_models(self):
        """Load language-specific models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers not available. Multilingual features limited.")
            return
        
        for lang_code in self.supported_languages:
            try:
                model_name = self.sentiment_models[lang_code]
                logger.info(f"Loading {self.language_names[lang_code]} model: {model_name}")
                
                self.tokenizers[lang_code] = AutoTokenizer.from_pretrained(model_name)
                self.models[lang_code] = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=3,  # positive, negative, neutral
                    ignore_mismatched_sizes=True
                )
                logger.info(f"✅ {self.language_names[lang_code]} model loaded successfully")
                
            except Exception as e:
                logger.warning(f"Failed to load {self.language_names[lang_code]} model: {e}")
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect the language of the input text"""
        
        # Clean and preprocess text
        cleaned_text = self._clean_text(text)
        
        if not cleaned_text.strip():
            return 'en', 0.0
        
        # Method 1: Use langdetect if available
        if LANGDETECT_AVAILABLE:
            try:
                detected_langs = detect_langs(cleaned_text)
                if detected_langs:
                    best_lang = detected_langs[0]
                    if best_lang.lang in self.supported_languages:
                        return best_lang.lang, best_lang.prob
            except Exception as e:
                logger.warning(f"Language detection failed: {e}")
        
        # Method 2: Pattern-based detection
        pattern_scores = {}
        for lang_code, pattern in self.language_patterns.items():
            matches = len(re.findall(pattern, cleaned_text))
            pattern_scores[lang_code] = matches / len(cleaned_text) if cleaned_text else 0
        
        # Method 3: Character-based detection for Chinese
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', cleaned_text))
        if chinese_chars > 0:
            pattern_scores['zh'] = chinese_chars / len(cleaned_text)
        
        # Select best language
        if pattern_scores:
            best_lang = max(pattern_scores.items(), key=lambda x: x[1])
            confidence = min(best_lang[1] * 2, 1.0)  # Scale confidence
            return best_lang[0], confidence
        
        # Default to English
        return 'en', 0.5
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for language detection"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize unicode
        text = unicodedata.normalize('NFKD', text)
        
        # Remove URLs, mentions, hashtags
        text = re.sub(r'http\S+|www\S+|@\w+|#\w+', '', text)
        
        return text
    
    def analyze_sentiment_multilingual(self, text: str, 
                                     target_language: Optional[str] = None,
                                     enable_cross_language: bool = False) -> MultilingualSentimentResult:
        """Analyze sentiment in multiple languages"""
        
        # Detect language if not specified
        if target_language is None:
            detected_lang, lang_confidence = self.detect_language(text)
        else:
            detected_lang = target_language
            lang_confidence = 1.0
        
        # Ensure language is supported
        if detected_lang not in self.supported_languages:
            detected_lang = 'en'
            lang_confidence = 0.5
        
        # Analyze sentiment in detected language
        sentiment_result = self._analyze_sentiment_language_specific(text, detected_lang)
        
        # Cross-language analysis if enabled
        cross_language_consensus = None
        if enable_cross_language and len(self.supported_languages) > 1:
            cross_language_consensus = self._analyze_cross_language_consensus(text)
        
        return MultilingualSentimentResult(
            text=text,
            detected_language=detected_lang,
            language_confidence=lang_confidence,
            sentiment=sentiment_result['sentiment'],
            confidence=sentiment_result['confidence'],
            emotions=sentiment_result['emotions'],
            methods_used=sentiment_result['methods_used'],
            language_specific_analysis=sentiment_result['language_analysis'],
            cross_language_consensus=cross_language_consensus
        )
    
    def _analyze_sentiment_language_specific(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze sentiment using language-specific models"""
        
        result = {
            'sentiment': 'neutral',
            'confidence': 0.5,
            'emotions': {},
            'methods_used': [],
            'language_analysis': {}
        }
        
        # Method 1: Transformer model for specific language
        if language in self.models and self.models[language] is not None:
            try:
                transformer_result = self._analyze_with_transformer(text, language)
                result['sentiment'] = transformer_result['sentiment']
                result['confidence'] = transformer_result['confidence']
                result['methods_used'].append(f'transformer_{language}')
                result['language_analysis']['transformer'] = transformer_result
            except Exception as e:
                logger.warning(f"Transformer analysis failed for {language}: {e}")
        
        # Method 2: Language-specific rules and patterns
        rule_based_result = self._analyze_with_language_rules(text, language)
        if rule_based_result['confidence'] > result['confidence']:
            result['sentiment'] = rule_based_result['sentiment']
            result['confidence'] = rule_based_result['confidence']
            result['methods_used'].append(f'rules_{language}')
        
        result['language_analysis']['rule_based'] = rule_based_result
        
        # Method 3: Emotion analysis
        emotions = self._analyze_emotions_language_specific(text, language)
        result['emotions'] = emotions
        result['methods_used'].append(f'emotions_{language}')
        
        return result
    
    def _analyze_with_transformer(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze sentiment using transformer model"""
        
        if language not in self.models or self.models[language] is None:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        try:
            tokenizer = self.tokenizers[language]
            model = self.models[language]
            
            # Tokenize input
            inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
            
            # Get predictions
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.softmax(outputs.logits, dim=-1)
                prediction = torch.argmax(probabilities, dim=-1).item()
                confidence = torch.max(probabilities).item()
            
            # Map to sentiment labels
            sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            sentiment = sentiment_map.get(prediction, 'neutral')
            
            return {
                'sentiment': sentiment,
                'confidence': float(confidence),
                'probabilities': {
                    'negative': float(probabilities[0][0]),
                    'neutral': float(probabilities[0][1]),
                    'positive': float(probabilities[0][2])
                }
            }
            
        except Exception as e:
            logger.warning(f"Transformer analysis failed: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.5}
    
    def _analyze_with_language_rules(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze sentiment using language-specific rules"""
        
        # Language-specific sentiment words
        sentiment_words = {
            'en': {
                'positive': ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like', 'happy', 'pleased'],
                'negative': ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'frustrated', 'disappointed', 'sad']
            },
            'es': {
                'positive': ['bueno', 'excelente', 'maravilloso', 'fantástico', 'genial', 'amor', 'me gusta', 'feliz', 'contento', 'satisfecho'],
                'negative': ['malo', 'terrible', 'horrible', 'odio', 'no me gusta', 'enojado', 'frustrado', 'decepcionado', 'triste', 'molesto']
            },
            'zh': {
                'positive': ['好', '很好', '优秀', '棒', '喜欢', '爱', '高兴', '满意', '开心', '不错'],
                'negative': ['坏', '糟糕', '讨厌', '不喜欢', '生气', '失望', '难过', '愤怒', '烦恼', '不好']
            }
        }
        
        if language not in sentiment_words:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        text_lower = text.lower()
        positive_count = sum(1 for word in sentiment_words[language]['positive'] if word in text_lower)
        negative_count = sum(1 for word in sentiment_words[language]['negative'] if word in text_lower)
        
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return {'sentiment': 'neutral', 'confidence': 0.5}
        
        if positive_count > negative_count:
            sentiment = 'positive'
            confidence = positive_count / total_sentiment_words
        elif negative_count > positive_count:
            sentiment = 'negative'
            confidence = negative_count / total_sentiment_words
        else:
            sentiment = 'neutral'
            confidence = 0.5
        
        return {
            'sentiment': sentiment,
            'confidence': min(confidence, 1.0),
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def _analyze_emotions_language_specific(self, text: str, language: str) -> Dict[str, float]:
        """Analyze emotions using language-specific patterns"""
        
        # Language-specific emotion patterns
        emotion_patterns = {
            'en': {
                'joy': ['happy', 'joy', 'excited', 'delighted', 'cheerful', 'elated'],
                'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'rage'],
                'sadness': ['sad', 'depressed', 'melancholy', 'gloomy', 'sorrowful'],
                'fear': ['afraid', 'scared', 'terrified', 'worried', 'anxious', 'nervous'],
                'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned']
            },
            'es': {
                'joy': ['alegre', 'feliz', 'contento', 'emocionado', 'dichoso', 'gozoso'],
                'anger': ['enojado', 'furioso', 'irritado', 'molesto', 'rabioso', 'colérico'],
                'sadness': ['triste', 'deprimido', 'melancólico', 'afligido', 'apenado'],
                'fear': ['asustado', 'temeroso', 'preocupado', 'ansioso', 'nervioso'],
                'surprise': ['sorprendido', 'asombrado', 'atónito', 'desconcertado']
            },
            'zh': {
                'joy': ['高兴', '快乐', '开心', '兴奋', '愉快', '欣喜'],
                'anger': ['生气', '愤怒', '恼火', '愤怒', '气愤', '暴怒'],
                'sadness': ['悲伤', '难过', '沮丧', '忧郁', '哀伤', '痛苦'],
                'fear': ['害怕', '恐惧', '担心', '焦虑', '紧张', '不安'],
                'surprise': ['惊讶', '震惊', '吃惊', '意外', '诧异', '惊愕']
            }
        }
        
        if language not in emotion_patterns:
            return {}
        
        text_lower = text.lower()
        emotions = {}
        
        for emotion, words in emotion_patterns[language].items():
            count = sum(1 for word in words if word in text_lower)
            emotions[emotion] = min(count / len(words), 1.0) if words else 0.0
        
        return emotions
    
    def _analyze_cross_language_consensus(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment across multiple languages for consensus"""
        
        consensus_results = {}
        
        for language in self.supported_languages:
            if language in self.models and self.models[language] is not None:
                try:
                    result = self._analyze_sentiment_language_specific(text, language)
                    consensus_results[language] = {
                        'sentiment': result['sentiment'],
                        'confidence': result['confidence'],
                        'language': self.language_names[language]
                    }
                except Exception as e:
                    logger.warning(f"Cross-language analysis failed for {language}: {e}")
        
        if not consensus_results:
            return None
        
        # Calculate consensus
        sentiments = [result['sentiment'] for result in consensus_results.values()]
        confidences = [result['confidence'] for result in consensus_results.values()]
        
        # Most common sentiment
        from collections import Counter
        sentiment_counts = Counter(sentiments)
        consensus_sentiment = sentiment_counts.most_common(1)[0][0]
        
        # Average confidence
        avg_confidence = sum(confidences) / len(confidences)
        
        # Agreement rate
        agreement_rate = sentiment_counts[consensus_sentiment] / len(sentiments)
        
        return {
            'consensus_sentiment': consensus_sentiment,
            'average_confidence': avg_confidence,
            'agreement_rate': agreement_rate,
            'language_results': consensus_results,
            'total_languages': len(consensus_results)
        }
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return self.supported_languages
    
    def get_language_name(self, language_code: str) -> str:
        """Get human-readable language name"""
        return self.language_names.get(language_code, language_code)
    
    def analyze_conversation_multilingual(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a conversation with multilingual support"""
        
        results = {
            'conversation_id': conversation.get('conversation_id', 'unknown'),
            'timestamp': conversation.get('timestamp'),
            'language_analysis': {},
            'sentiment_analysis': {},
            'cross_language_insights': {},
            'multilingual_metrics': {}
        }
        
        messages = conversation.get('messages', [])
        language_distribution = {}
        sentiment_by_language = {}
        
        for i, message in enumerate(messages):
            user_text = message.get('user', '')
            bot_text = message.get('bot', '')
            
            message_analysis = {
                'message_index': i + 1,
                'timestamp': message.get('timestamp'),
                'user_analysis': None,
                'bot_analysis': None
            }
            
            # Analyze user message
            if user_text:
                user_result = self.analyze_sentiment_multilingual(user_text, enable_cross_language=True)
                message_analysis['user_analysis'] = user_result
                
                # Track language distribution
                lang = user_result.detected_language
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
                
                # Track sentiment by language
                if lang not in sentiment_by_language:
                    sentiment_by_language[lang] = []
                sentiment_by_language[lang].append(user_result.sentiment)
            
            # Analyze bot message
            if bot_text:
                bot_result = self.analyze_sentiment_multilingual(bot_text, enable_cross_language=True)
                message_analysis['bot_analysis'] = bot_result
                
                # Track language distribution
                lang = bot_result.detected_language
                language_distribution[lang] = language_distribution.get(lang, 0) + 1
                
                # Track sentiment by language
                if lang not in sentiment_by_language:
                    sentiment_by_language[lang] = []
                sentiment_by_language[lang].append(bot_result.sentiment)
            
            results['sentiment_analysis'][f'message_{i+1}'] = message_analysis
        
        # Calculate multilingual metrics
        results['multilingual_metrics'] = {
            'language_distribution': language_distribution,
            'sentiment_by_language': sentiment_by_language,
            'total_languages_detected': len(language_distribution),
            'primary_language': max(language_distribution.items(), key=lambda x: x[1])[0] if language_distribution else 'en',
            'language_diversity': len(language_distribution) / len(self.supported_languages)
        }
        
        return results

def main():
    """Demo function for multilingual sentiment analysis"""
    print("🌍 SentilensAI - Multilingual Sentiment Analysis Demo")
    print("=" * 60)
    
    # Initialize multilingual analyzer
    analyzer = MultilingualSentimentAnalyzer()
    
    # Sample texts in different languages
    sample_texts = [
        {
            'text': "I love this product! It's amazing and works perfectly.",
            'expected_lang': 'en',
            'description': 'English positive sentiment'
        },
        {
            'text': "¡Me encanta este producto! Es increíble y funciona perfectamente.",
            'expected_lang': 'es',
            'description': 'Spanish positive sentiment'
        },
        {
            'text': "这个产品太棒了！我非常喜欢它，效果很好。",
            'expected_lang': 'zh',
            'description': 'Chinese positive sentiment'
        },
        {
            'text': "This is terrible. I hate it and want a refund immediately.",
            'expected_lang': 'en',
            'description': 'English negative sentiment'
        },
        {
            'text': "Esto es terrible. Lo odio y quiero un reembolso inmediatamente.",
            'expected_lang': 'es',
            'description': 'Spanish negative sentiment'
        },
        {
            'text': "这太糟糕了。我讨厌它，想要立即退款。",
            'expected_lang': 'zh',
            'description': 'Chinese negative sentiment'
        }
    ]
    
    print(f"🔍 Analyzing {len(sample_texts)} texts in multiple languages...")
    print(f"Supported languages: {', '.join([analyzer.get_language_name(lang) for lang in analyzer.get_supported_languages()])}")
    print()
    
    for i, sample in enumerate(sample_texts, 1):
        print(f"📝 Sample {i}: {sample['description']}")
        print(f"Text: {sample['text']}")
        
        # Analyze with multilingual support
        result = analyzer.analyze_sentiment_multilingual(
            sample['text'], 
            enable_cross_language=True
        )
        
        print(f"Detected Language: {analyzer.get_language_name(result.detected_language)} (confidence: {result.language_confidence:.2f})")
        print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
        print(f"Methods Used: {', '.join(result.methods_used)}")
        
        if result.emotions:
            print(f"Emotions: {', '.join([f'{k}: {v:.2f}' for k, v in result.emotions.items() if v > 0])}")
        
        if result.cross_language_consensus:
            consensus = result.cross_language_consensus
            print(f"Cross-language Consensus: {consensus['consensus_sentiment']} (agreement: {consensus['agreement_rate']:.2f})")
        
        print("-" * 50)
    
    # Test conversation analysis
    print("\n🗣️ Multilingual Conversation Analysis:")
    print("=" * 40)
    
    multilingual_conversation = {
        'conversation_id': 'multilingual_demo_001',
        'timestamp': '2024-01-15T10:30:00Z',
        'messages': [
            {
                'user': 'Hello, I need help with my account',
                'bot': 'Hola, puedo ayudarte con tu cuenta',
                'timestamp': '2024-01-15T10:30:15Z'
            },
            {
                'user': '谢谢你的帮助！',
                'bot': 'You\'re welcome! I\'m happy to help.',
                'timestamp': '2024-01-15T10:30:30Z'
            }
        ]
    }
    
    conversation_result = analyzer.analyze_conversation_multilingual(multilingual_conversation)
    
    print(f"Conversation ID: {conversation_result['conversation_id']}")
    print(f"Languages Detected: {conversation_result['multilingual_metrics']['total_languages_detected']}")
    print(f"Primary Language: {analyzer.get_language_name(conversation_result['multilingual_metrics']['primary_language'])}")
    print(f"Language Distribution: {conversation_result['multilingual_metrics']['language_distribution']}")
    print(f"Language Diversity: {conversation_result['multilingual_metrics']['language_diversity']:.2f}")
    
    print(f"\n✅ Multilingual sentiment analysis demo completed!")
    print(f"🌍 SentilensAI now supports {len(analyzer.get_supported_languages())} languages!")
    print(f"🚀 Ready for global AI chatbot conversations!")

if __name__ == "__main__":
    main()
