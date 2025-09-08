#!/usr/bin/env python3
"""
SentilensAI - Enhanced Analysis with Deep Learning and Learning Recommendations

This module provides comprehensive analysis combining traditional ML, deep learning,
and generates specific learning recommendations for agent training improvement.

Author: Pravin Selvamuthu
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import our modules
from sentiment_analyzer import SentilensAIAnalyzer
from deep_learning_sentiment import DeepLearningSentimentAnalyzer
from ml_training_pipeline import SentilensAITrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedSentimentAnalyzer:
    """Enhanced sentiment analyzer combining traditional ML and deep learning"""
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.traditional_analyzer = SentilensAIAnalyzer(openai_api_key=openai_api_key)
        self.deep_learning_analyzer = DeepLearningSentimentAnalyzer()
        self.ml_trainer = SentilensAITrainer()
        
        self.analysis_results = {}
        self.learning_recommendations = {}
    
    def analyze_conversation_enhanced(self, conversation: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation using both traditional and deep learning methods"""
        
        conv_id = conversation.get('conversation_id', 'unknown')
        messages = conversation.get('messages', [])
        
        logger.info(f"Analyzing conversation {conv_id} with enhanced methods...")
        
        enhanced_results = {
            'conversation_id': conv_id,
            'timestamp': conversation.get('timestamp'),
            'message_analysis': [],
            'deep_learning_insights': {},
            'learning_recommendations': {},
            'quality_metrics': {},
            'improvement_opportunities': []
        }
        
        # Analyze each message with multiple methods
        for i, message in enumerate(messages):
            user_text = message.get('user', '')
            bot_text = message.get('bot', '')
            
            message_analysis = {
                'message_index': i + 1,
                'timestamp': message.get('timestamp'),
                'user_message': user_text,
                'bot_message': bot_text,
                'traditional_analysis': {},
                'deep_learning_analysis': {},
                'consensus_analysis': {}
            }
            
            # Traditional analysis
            if user_text:
                traditional_user = self.traditional_analyzer.analyze_sentiment(user_text)
                message_analysis['traditional_analysis']['user'] = {
                    'sentiment': traditional_user.sentiment,
                    'confidence': traditional_user.confidence,
                    'emotions': traditional_user.emotions,
                    'methods_used': getattr(traditional_user, 'methods_used', ['traditional_ml'])
                }
            
            if bot_text:
                traditional_bot = self.traditional_analyzer.analyze_sentiment(bot_text)
                message_analysis['traditional_analysis']['bot'] = {
                    'sentiment': traditional_bot.sentiment,
                    'confidence': traditional_bot.confidence,
                    'emotions': traditional_bot.emotions,
                    'methods_used': getattr(traditional_bot, 'methods_used', ['traditional_ml'])
                }
            
            # Deep learning analysis
            if user_text:
                dl_user = self.deep_learning_analyzer.analyze_sentiment_deep_learning(user_text)
                message_analysis['deep_learning_analysis']['user'] = dl_user
            
            if bot_text:
                dl_bot = self.deep_learning_analyzer.analyze_sentiment_deep_learning(bot_text)
                message_analysis['deep_learning_analysis']['bot'] = dl_bot
            
            # Consensus analysis
            message_analysis['consensus_analysis'] = self._generate_consensus_analysis(
                message_analysis['traditional_analysis'],
                message_analysis['deep_learning_analysis']
            )
            
            enhanced_results['message_analysis'].append(message_analysis)
        
        # Generate conversation-level insights
        enhanced_results['deep_learning_insights'] = self._analyze_conversation_patterns(enhanced_results)
        enhanced_results['learning_recommendations'] = self._generate_learning_recommendations(enhanced_results)
        enhanced_results['quality_metrics'] = self._calculate_enhanced_quality_metrics(enhanced_results)
        enhanced_results['improvement_opportunities'] = self._identify_improvement_opportunities(enhanced_results)
        
        return enhanced_results
    
    def _generate_consensus_analysis(self, traditional: Dict, deep_learning: Dict) -> Dict[str, Any]:
        """Generate consensus analysis between traditional and deep learning methods"""
        
        consensus = {
            'user_sentiment_consensus': None,
            'bot_sentiment_consensus': None,
            'confidence_consensus': {},
            'method_agreement': {},
            'reliability_score': 0.0
        }
        
        # Analyze user sentiment consensus
        if 'user' in traditional and 'user' in deep_learning:
            trad_sentiment = traditional['user']['sentiment']
            dl_sentiment = deep_learning['user']['ensemble_prediction']
            
            if trad_sentiment == dl_sentiment:
                consensus['user_sentiment_consensus'] = trad_sentiment
                consensus['method_agreement']['user'] = 1.0
            else:
                # Use confidence-weighted consensus
                trad_conf = traditional['user']['confidence']
                dl_conf = deep_learning['user']['average_confidence']
                
                if trad_conf > dl_conf:
                    consensus['user_sentiment_consensus'] = trad_sentiment
                else:
                    consensus['user_sentiment_consensus'] = dl_sentiment
                
                consensus['method_agreement']['user'] = 0.5
        
        # Analyze bot sentiment consensus
        if 'bot' in traditional and 'bot' in deep_learning:
            trad_sentiment = traditional['bot']['sentiment']
            dl_sentiment = deep_learning['bot']['ensemble_prediction']
            
            if trad_sentiment == dl_sentiment:
                consensus['bot_sentiment_consensus'] = trad_sentiment
                consensus['method_agreement']['bot'] = 1.0
            else:
                trad_conf = traditional['bot']['confidence']
                dl_conf = deep_learning['bot']['average_confidence']
                
                if trad_conf > dl_conf:
                    consensus['bot_sentiment_consensus'] = trad_sentiment
                else:
                    consensus['bot_sentiment_consensus'] = dl_sentiment
                
                consensus['method_agreement']['bot'] = 0.5
        
        # Calculate overall reliability score
        agreement_scores = list(consensus['method_agreement'].values())
        consensus['reliability_score'] = np.mean(agreement_scores) if agreement_scores else 0.0
        
        return consensus
    
    def _analyze_conversation_patterns(self, enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze conversation patterns using deep learning insights"""
        
        patterns = {
            'sentiment_evolution': {},
            'confidence_trends': {},
            'model_agreement_analysis': {},
            'deep_learning_insights': {}
        }
        
        messages = enhanced_results['message_analysis']
        
        # Analyze sentiment evolution
        user_sentiments = []
        bot_sentiments = []
        confidence_scores = []
        agreement_scores = []
        
        for msg in messages:
            consensus = msg['consensus_analysis']
            
            if consensus['user_sentiment_consensus']:
                user_sentiments.append(consensus['user_sentiment_consensus'])
            if consensus['bot_sentiment_consensus']:
                bot_sentiments.append(consensus['bot_sentiment_consensus'])
            
            confidence_scores.append(consensus['reliability_score'])
            agreement_scores.append(consensus['reliability_score'])
        
        patterns['sentiment_evolution'] = {
            'user_trend': user_sentiments,
            'bot_trend': bot_sentiments,
            'sentiment_stability': self._calculate_sentiment_stability(user_sentiments, bot_sentiments)
        }
        
        patterns['confidence_trends'] = {
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'confidence_consistency': 1.0 - np.std(confidence_scores) if confidence_scores else 0.0,
            'low_confidence_points': [i for i, conf in enumerate(confidence_scores) if conf < 0.5]
        }
        
        patterns['model_agreement_analysis'] = {
            'average_agreement': np.mean(agreement_scores) if agreement_scores else 0.0,
            'agreement_consistency': 1.0 - np.std(agreement_scores) if agreement_scores else 0.0,
            'disagreement_points': [i for i, agree in enumerate(agreement_scores) if agree < 0.7]
        }
        
        # Deep learning specific insights
        patterns['deep_learning_insights'] = {
            'ensemble_benefit': self._calculate_ensemble_benefit(messages),
            'model_diversity': self._calculate_model_diversity(messages),
            'prediction_confidence': self._calculate_prediction_confidence(messages)
        }
        
        return patterns
    
    def _calculate_sentiment_stability(self, user_sentiments: List[str], bot_sentiments: List[str]) -> float:
        """Calculate sentiment stability score"""
        
        if not user_sentiments or not bot_sentiments:
            return 0.0
        
        # Calculate sentiment changes
        user_changes = sum(1 for i in range(1, len(user_sentiments)) 
                          if user_sentiments[i] != user_sentiments[i-1])
        bot_changes = sum(1 for i in range(1, len(bot_sentiments)) 
                         if bot_sentiments[i] != bot_sentiments[i-1])
        
        total_possible_changes = len(user_sentiments) + len(bot_sentiments) - 2
        stability = 1.0 - (user_changes + bot_changes) / total_possible_changes if total_possible_changes > 0 else 1.0
        
        return max(0.0, stability)
    
    def _calculate_ensemble_benefit(self, messages: List[Dict]) -> float:
        """Calculate the benefit of using ensemble methods"""
        
        model_agreements = []
        for msg in messages:
            dl_analysis = msg.get('deep_learning_analysis', {})
            for role in ['user', 'bot']:
                if role in dl_analysis:
                    agreement = dl_analysis[role].get('model_agreement', 0.0)
                    model_agreements.append(agreement)
        
        return np.mean(model_agreements) if model_agreements else 0.0
    
    def _calculate_model_diversity(self, messages: List[Dict]) -> float:
        """Calculate model diversity in predictions"""
        
        all_models = set()
        for msg in messages:
            dl_analysis = msg.get('deep_learning_analysis', {})
            for role in ['user', 'bot']:
                if role in dl_analysis:
                    models_used = dl_analysis[role].get('models_used', [])
                    all_models.update(models_used)
        
        return len(all_models) / 5.0  # Normalize by expected number of models
    
    def _calculate_prediction_confidence(self, messages: List[Dict]) -> float:
        """Calculate average prediction confidence"""
        
        confidences = []
        for msg in messages:
            dl_analysis = msg.get('deep_learning_analysis', {})
            for role in ['user', 'bot']:
                if role in dl_analysis:
                    avg_conf = dl_analysis[role].get('average_confidence', 0.0)
                    confidences.append(avg_conf)
        
        return np.mean(confidences) if confidences else 0.0
    
    def _generate_learning_recommendations(self, enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive learning recommendations"""
        
        recommendations = {
            'immediate_actions': [],
            'short_term_improvements': [],
            'long_term_strategy': [],
            'technical_recommendations': [],
            'training_priorities': {},
            'model_optimization': {}
        }
        
        patterns = enhanced_results['deep_learning_insights']
        quality_metrics = enhanced_results['quality_metrics']
        
        # Immediate actions based on critical issues
        if quality_metrics.get('negative_response_rate', 0) > 0.1:
            recommendations['immediate_actions'].append({
                'priority': 'CRITICAL',
                'action': 'Eliminate negative bot responses',
                'description': 'Implement real-time tone checking to prevent negative responses',
                'expected_impact': 'High',
                'implementation_time': '1-2 days'
            })
        
        if patterns['confidence_trends']['average_confidence'] < 0.6:
            recommendations['immediate_actions'].append({
                'priority': 'HIGH',
                'action': 'Improve response confidence',
                'description': 'Enhance training data and model architecture for better confidence',
                'expected_impact': 'High',
                'implementation_time': '1 week'
            })
        
        # Short-term improvements
        if patterns['model_agreement_analysis']['average_agreement'] < 0.8:
            recommendations['short_term_improvements'].append({
                'area': 'Model Agreement',
                'action': 'Implement ensemble methods',
                'description': 'Use multiple models and consensus mechanisms for better agreement',
                'benefit': 'Improved prediction reliability',
                'effort': 'Medium'
            })
        
        if patterns['deep_learning_insights']['ensemble_benefit'] < 0.7:
            recommendations['short_term_improvements'].append({
                'area': 'Ensemble Learning',
                'action': 'Optimize ensemble configuration',
                'description': 'Fine-tune ensemble weights and model selection',
                'benefit': 'Better prediction accuracy',
                'effort': 'Medium'
            })
        
        # Long-term strategy
        recommendations['long_term_strategy'] = [
            {
                'area': 'Continuous Learning',
                'action': 'Implement active learning pipeline',
                'description': 'Continuously improve models with new data and feedback',
                'timeline': '3-6 months',
                'investment': 'High'
            },
            {
                'area': 'Model Architecture',
                'action': 'Develop custom domain-specific models',
                'description': 'Train specialized models for chatbot conversations',
                'timeline': '6-12 months',
                'investment': 'Very High'
            }
        ]
        
        # Technical recommendations
        recommendations['technical_recommendations'] = [
            {
                'category': 'Infrastructure',
                'recommendation': 'Deploy model serving infrastructure',
                'priority': 'High',
                'description': 'Set up scalable model serving for real-time inference'
            },
            {
                'category': 'Monitoring',
                'recommendation': 'Implement model performance monitoring',
                'priority': 'High',
                'description': 'Track model performance and drift over time'
            },
            {
                'category': 'Data',
                'recommendation': 'Create data quality pipeline',
                'priority': 'Medium',
                'description': 'Ensure high-quality training and validation data'
            }
        ]
        
        # Training priorities
        recommendations['training_priorities'] = {
            'immediate': ['Tone checking', 'Confidence improvement'],
            'short_term': ['Ensemble optimization', 'Model agreement'],
            'long_term': ['Active learning', 'Custom models']
        }
        
        # Model optimization
        recommendations['model_optimization'] = {
            'current_performance': {
                'ensemble_benefit': patterns['deep_learning_insights']['ensemble_benefit'],
                'model_diversity': patterns['deep_learning_insights']['model_diversity'],
                'prediction_confidence': patterns['deep_learning_insights']['prediction_confidence']
            },
            'optimization_targets': {
                'ensemble_benefit': 0.8,
                'model_diversity': 0.9,
                'prediction_confidence': 0.8
            },
            'optimization_strategies': [
                'Fine-tune transformer models on domain data',
                'Implement model selection based on confidence',
                'Add data augmentation techniques',
                'Use transfer learning from larger datasets'
            ]
        }
        
        return recommendations
    
    def _calculate_enhanced_quality_metrics(self, enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate enhanced quality metrics using both traditional and deep learning insights"""
        
        messages = enhanced_results['message_analysis']
        
        # Traditional quality metrics
        positive_responses = 0
        negative_responses = 0
        total_responses = 0
        confidence_scores = []
        agreement_scores = []
        
        for msg in messages:
            consensus = msg['consensus_analysis']
            
            if consensus['bot_sentiment_consensus']:
                total_responses += 1
                if consensus['bot_sentiment_consensus'] == 'positive':
                    positive_responses += 1
                elif consensus['bot_sentiment_consensus'] == 'negative':
                    negative_responses += 1
            
            confidence_scores.append(consensus['reliability_score'])
            agreement_scores.append(consensus['reliability_score'])
        
        # Enhanced metrics
        quality_metrics = {
            'response_quality_score': positive_responses / total_responses if total_responses > 0 else 0.0,
            'negative_response_rate': negative_responses / total_responses if total_responses > 0 else 0.0,
            'average_confidence': np.mean(confidence_scores) if confidence_scores else 0.0,
            'confidence_consistency': 1.0 - np.std(confidence_scores) if confidence_scores else 0.0,
            'model_agreement_rate': np.mean(agreement_scores) if agreement_scores else 0.0,
            'reliability_score': np.mean(agreement_scores) if agreement_scores else 0.0
        }
        
        # Overall quality score (weighted combination)
        quality_metrics['overall_quality_score'] = (
            quality_metrics['response_quality_score'] * 0.3 +
            (1 - quality_metrics['negative_response_rate']) * 0.3 +
            quality_metrics['average_confidence'] * 0.2 +
            quality_metrics['model_agreement_rate'] * 0.2
        )
        
        return quality_metrics
    
    def _identify_improvement_opportunities(self, enhanced_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify specific improvement opportunities"""
        
        opportunities = []
        messages = enhanced_results['message_analysis']
        patterns = enhanced_results['deep_learning_insights']
        
        # Low confidence opportunities
        for i, msg in enumerate(messages):
            consensus = msg['consensus_analysis']
            if consensus['reliability_score'] < 0.6:
                opportunities.append({
                    'type': 'Low Confidence',
                    'message_index': i + 1,
                    'description': f'Message {i+1} has low confidence ({consensus["reliability_score"]:.2f})',
                    'suggestion': 'Improve training data or model architecture',
                    'priority': 'High'
                })
        
        # Model disagreement opportunities
        for i, msg in enumerate(messages):
            consensus = msg['consensus_analysis']
            if consensus['reliability_score'] < 0.7:
                opportunities.append({
                    'type': 'Model Disagreement',
                    'message_index': i + 1,
                    'description': f'Models disagree on message {i+1} sentiment',
                    'suggestion': 'Implement ensemble consensus mechanism',
                    'priority': 'Medium'
                })
        
        # Sentiment deterioration opportunities
        user_sentiments = [msg['consensus_analysis']['user_sentiment_consensus'] 
                          for msg in messages if msg['consensus_analysis']['user_sentiment_consensus']]
        
        if len(user_sentiments) >= 2:
            if user_sentiments[0] in ['positive', 'neutral'] and user_sentiments[-1] == 'negative':
                opportunities.append({
                    'type': 'Sentiment Deterioration',
                    'message_index': 'Multiple',
                    'description': 'User sentiment deteriorated during conversation',
                    'suggestion': 'Implement proactive sentiment management',
                    'priority': 'High'
                })
        
        return opportunities

def main():
    """Demo function for enhanced analysis"""
    print("🤖 SentilensAI - Enhanced Analysis with Deep Learning Demo")
    print("=" * 70)
    
    # Initialize enhanced analyzer
    analyzer = EnhancedSentimentAnalyzer()
    
    # Load sample conversation
    sample_conversation = {
        "conversation_id": "enhanced_demo_001",
        "timestamp": "2024-01-15T10:30:00Z",
        "messages": [
            {
                "user": "Hi, I need help with my account",
                "bot": "Hello! I'd be happy to help you with your account. What specific issue are you experiencing?",
                "timestamp": "2024-01-15T10:30:15Z"
            },
            {
                "user": "I can't log in and I'm getting frustrated",
                "bot": "I understand your frustration. Let me help you troubleshoot this login issue.",
                "timestamp": "2024-01-15T10:30:45Z"
            },
            {
                "user": "Thank you so much! You're amazing and very helpful",
                "bot": "You're very welcome! I'm glad I could help resolve this for you.",
                "timestamp": "2024-01-15T10:31:20Z"
            }
        ]
    }
    
    print("🔍 Analyzing conversation with enhanced methods...")
    
    # Perform enhanced analysis
    enhanced_results = analyzer.analyze_conversation_enhanced(sample_conversation)
    
    # Display results
    print(f"\n📊 Enhanced Analysis Results:")
    print(f"   Conversation ID: {enhanced_results['conversation_id']}")
    print(f"   Overall Quality Score: {enhanced_results['quality_metrics']['overall_quality_score']:.2f}")
    print(f"   Model Agreement Rate: {enhanced_results['quality_metrics']['model_agreement_rate']:.2f}")
    print(f"   Reliability Score: {enhanced_results['quality_metrics']['reliability_score']:.2f}")
    
    # Display learning recommendations
    recommendations = enhanced_results['learning_recommendations']
    print(f"\n🎓 Learning Recommendations:")
    print(f"   Immediate Actions: {len(recommendations['immediate_actions'])}")
    print(f"   Short-term Improvements: {len(recommendations['short_term_improvements'])}")
    print(f"   Long-term Strategy: {len(recommendations['long_term_strategy'])}")
    
    # Display improvement opportunities
    opportunities = enhanced_results['improvement_opportunities']
    print(f"\n💡 Improvement Opportunities: {len(opportunities)}")
    for opp in opportunities[:3]:  # Show first 3
        print(f"   • {opp['type']}: {opp['description']}")
    
    print(f"\n✅ Enhanced analysis completed successfully!")
    print(f"🚀 Deep learning insights and recommendations ready!")

if __name__ == "__main__":
    main()
