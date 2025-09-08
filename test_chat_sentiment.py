#!/usr/bin/env python3
"""
SentilensAI - Chat Message Sentiment Analysis Test

This script loads chat messages from a JSON file and analyzes their sentiment
using SentilensAI's advanced sentiment analysis capabilities.

Author: Pravin Selvamuthu
"""

import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from sentiment_analyzer import SentilensAIAnalyzer, SentimentResult
from chatbot_integration import SentilensAIChatbotIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_chat_messages(json_file: str) -> List[Dict[str, Any]]:
    """
    Load chat messages from JSON file
    
    Args:
        json_file: Path to JSON file containing chat messages
        
    Returns:
        List of conversation data
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('conversations', [])
    except Exception as e:
        logger.error(f"Error loading chat messages: {e}")
        return []

def analyze_conversation_sentiment(analyzer: SentilensAIAnalyzer, 
                                 conversation: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze sentiment for a single conversation
    
    Args:
        analyzer: SentilensAI analyzer instance
        conversation: Conversation data with messages
        
    Returns:
        Analysis results for the conversation
    """
    conv_id = conversation.get('conversation_id', 'unknown')
    messages = conversation.get('messages', [])
    
    print(f"\n🔍 Analyzing Conversation: {conv_id}")
    print("=" * 50)
    
    conversation_results = {
        'conversation_id': conv_id,
        'timestamp': conversation.get('timestamp'),
        'message_analysis': [],
        'overall_sentiment': {
            'user_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0},
            'bot_sentiment': {'positive': 0, 'negative': 0, 'neutral': 0}
        },
        'sentiment_trend': [],
        'conversation_quality': 0.0
    }
    
    user_sentiments = []
    bot_sentiments = []
    
    for i, message in enumerate(messages):
        user_text = message.get('user', '')
        bot_text = message.get('bot', '')
        timestamp = message.get('timestamp', '')
        
        print(f"\n📝 Message {i+1} ({timestamp}):")
        print(f"👤 User: {user_text}")
        print(f"🤖 Bot: {bot_text}")
        
        # Analyze user message
        if user_text:
            user_result = analyzer.analyze_sentiment(user_text)
            user_sentiments.append(user_result.sentiment)
            conversation_results['overall_sentiment']['user_sentiment'][user_result.sentiment] += 1
            
            print(f"   👤 User Sentiment: {user_result.sentiment} (Confidence: {user_result.confidence:.2f})")
            if user_result.emotions:
                print(f"   😊 Emotions: {', '.join([f'{k}: {v:.2f}' for k, v in user_result.emotions.items()])}")
        
        # Analyze bot message
        if bot_text:
            bot_result = analyzer.analyze_sentiment(bot_text)
            bot_sentiments.append(bot_result.sentiment)
            conversation_results['overall_sentiment']['bot_sentiment'][bot_result.sentiment] += 1
            
            print(f"   🤖 Bot Sentiment: {bot_result.sentiment} (Confidence: {bot_result.confidence:.2f})")
            if bot_result.emotions:
                print(f"   😊 Emotions: {', '.join([f'{k}: {v:.2f}' for k, v in bot_result.emotions.items()])}")
        
        # Store message analysis
        message_analysis = {
            'message_index': i + 1,
            'timestamp': timestamp,
            'user_message': user_text,
            'bot_message': bot_text,
            'user_sentiment': user_result.sentiment if user_text else None,
            'user_confidence': user_result.confidence if user_text else None,
            'bot_sentiment': bot_result.sentiment if bot_text else None,
            'bot_confidence': bot_result.confidence if bot_text else None,
            'user_emotions': user_result.emotions if user_text else None,
            'bot_emotions': bot_result.emotions if bot_text else None
        }
        conversation_results['message_analysis'].append(message_analysis)
        
        # Track sentiment trend
        conversation_results['sentiment_trend'].append({
            'message_index': i + 1,
            'user_sentiment': user_result.sentiment if user_text else None,
            'bot_sentiment': bot_result.sentiment if bot_text else None
        })
    
    # Calculate conversation quality score
    if user_sentiments and bot_sentiments:
        positive_user = sum(1 for s in user_sentiments if s == 'positive')
        positive_bot = sum(1 for s in bot_sentiments if s == 'positive')
        total_messages = len(user_sentiments)
        
        conversation_results['conversation_quality'] = (positive_user + positive_bot) / (total_messages * 2)
    
    # Print conversation summary
    print(f"\n📊 Conversation Summary:")
    print(f"   User Sentiment Distribution: {conversation_results['overall_sentiment']['user_sentiment']}")
    print(f"   Bot Sentiment Distribution: {conversation_results['overall_sentiment']['bot_sentiment']}")
    print(f"   Conversation Quality Score: {conversation_results['conversation_quality']:.2f}")
    
    return conversation_results

def main():
    """Main function to test chat message sentiment analysis"""
    print("🤖 SentilensAI - Chat Message Sentiment Analysis Test")
    print("=" * 60)
    
    # Initialize analyzer
    print("🔧 Initializing SentilensAI analyzer...")
    analyzer = SentilensAIAnalyzer()
    print("✅ Analyzer initialized successfully!")
    
    # Load chat messages
    json_file = "sample_chat_messages.json"
    print(f"\n📁 Loading chat messages from: {json_file}")
    conversations = load_chat_messages(json_file)
    
    if not conversations:
        print("❌ No conversations found in the JSON file!")
        return
    
    print(f"✅ Loaded {len(conversations)} conversations")
    
    # Analyze each conversation
    all_results = []
    total_quality = 0.0
    
    for conversation in conversations:
        result = analyze_conversation_sentiment(analyzer, conversation)
        all_results.append(result)
        total_quality += result['conversation_quality']
    
    # Calculate overall statistics
    avg_quality = total_quality / len(conversations) if conversations else 0
    
    print(f"\n🎯 Overall Analysis Summary:")
    print("=" * 40)
    print(f"Total Conversations Analyzed: {len(conversations)}")
    print(f"Average Conversation Quality: {avg_quality:.2f}")
    
    # Count sentiment distribution across all conversations
    total_user_sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
    total_bot_sentiment = {'positive': 0, 'negative': 0, 'neutral': 0}
    
    for result in all_results:
        for sentiment, count in result['overall_sentiment']['user_sentiment'].items():
            total_user_sentiment[sentiment] += count
        for sentiment, count in result['overall_sentiment']['bot_sentiment'].items():
            total_bot_sentiment[sentiment] += count
    
    print(f"\n📈 Overall Sentiment Distribution:")
    print(f"User Messages: {total_user_sentiment}")
    print(f"Bot Messages: {total_bot_sentiment}")
    
    # Save detailed results
    output_file = f"sentiment_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_timestamp': datetime.now().isoformat(),
            'total_conversations': len(conversations),
            'average_quality_score': avg_quality,
            'overall_sentiment_distribution': {
                'user': total_user_sentiment,
                'bot': total_bot_sentiment
            },
            'conversation_results': all_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed results saved to: {output_file}")
    print("\n✅ SentilensAI chat sentiment analysis completed successfully!")
    print("🚀 Ready for production sentiment monitoring!")

if __name__ == "__main__":
    main()
