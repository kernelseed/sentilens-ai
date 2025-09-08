#!/usr/bin/env python3
"""
SentilensAI - Multilingual Demo

This script demonstrates the multilingual capabilities of SentilensAI
supporting English, Spanish, and Chinese sentiment analysis.

Author: Pravin Selvamuthu
"""

import json
from datetime import datetime
from sentiment_analyzer import SentilensAIAnalyzer

def create_multilingual_conversations():
    """Create sample conversations in multiple languages"""
    
    conversations = [
        {
            "conversation_id": "multilingual_001",
            "timestamp": "2024-01-15T10:30:00Z",
            "description": "English conversation - positive sentiment",
            "messages": [
                {
                    "user": "Hi, I need help with my account",
                    "bot": "Hello! I'd be happy to help you with your account. What specific issue are you experiencing?",
                    "timestamp": "2024-01-15T10:30:15Z"
                },
                {
                    "user": "I love this service! It's amazing and very helpful",
                    "bot": "Thank you so much! I'm delighted to hear that you're happy with our service.",
                    "timestamp": "2024-01-15T10:30:30Z"
                }
            ]
        },
        {
            "conversation_id": "multilingual_002", 
            "timestamp": "2024-01-15T11:15:00Z",
            "description": "Spanish conversation - mixed sentiment",
            "messages": [
                {
                    "user": "Hola, necesito ayuda con mi cuenta",
                    "bot": "¡Hola! Estaré encantado de ayudarte con tu cuenta. ¿Qué problema específico tienes?",
                    "timestamp": "2024-01-15T11:15:15Z"
                },
                {
                    "user": "Estoy muy frustrado con este servicio",
                    "bot": "Lamento escuchar que estás frustrado. Déjame ayudarte a resolver este problema.",
                    "timestamp": "2024-01-15T11:15:30Z"
                },
                {
                    "user": "¡Gracias! Ahora estoy mucho mejor",
                    "bot": "¡De nada! Me alegra saber que te sientes mejor. ¿Hay algo más en lo que pueda ayudarte?",
                    "timestamp": "2024-01-15T11:15:45Z"
                }
            ]
        },
        {
            "conversation_id": "multilingual_003",
            "timestamp": "2024-01-15T14:20:00Z", 
            "description": "Chinese conversation - positive sentiment",
            "messages": [
                {
                    "user": "你好，我需要帮助",
                    "bot": "你好！我很乐意帮助你。你遇到什么具体问题了吗？",
                    "timestamp": "2024-01-15T14:20:15Z"
                },
                {
                    "user": "这个产品太棒了！我非常喜欢",
                    "bot": "谢谢你的赞美！我很高兴听到你喜欢我们的产品。",
                    "timestamp": "2024-01-15T14:20:30Z"
                }
            ]
        },
        {
            "conversation_id": "multilingual_004",
            "timestamp": "2024-01-15T16:45:00Z",
            "description": "Mixed language conversation",
            "messages": [
                {
                    "user": "Hello, I'm interested in your services",
                    "bot": "Hola! Me complace saber que estás interesado en nuestros servicios.",
                    "timestamp": "2024-01-15T16:45:15Z"
                },
                {
                    "user": "这个服务怎么样？",
                    "bot": "Our service is excellent! We provide comprehensive support in multiple languages.",
                    "timestamp": "2024-01-15T16:45:30Z"
                }
            ]
        }
    ]
    
    return conversations

def run_multilingual_demo():
    """Run the multilingual sentiment analysis demo"""
    
    print("🌍 SentilensAI - Multilingual Sentiment Analysis Demo")
    print("=" * 70)
    
    # Initialize analyzer with multilingual support
    print("🔧 Initializing SentilensAI with multilingual support...")
    analyzer = SentilensAIAnalyzer(enable_multilingual=True)
    
    # Check supported languages
    supported_languages = analyzer.get_supported_languages()
    print(f"✅ Supported languages: {', '.join([analyzer.get_language_name(lang) for lang in supported_languages])}")
    
    # Create sample conversations
    print("\n📝 Creating multilingual sample conversations...")
    conversations = create_multilingual_conversations()
    print(f"✅ Created {len(conversations)} multilingual conversations")
    
    # Analyze each conversation
    print("\n🔍 Analyzing conversations with multilingual capabilities...")
    print("=" * 70)
    
    all_results = []
    
    for i, conversation in enumerate(conversations, 1):
        print(f"\n📋 CONVERSATION {i}: {conversation['conversation_id']}")
        print(f"Description: {conversation['description']}")
        print("-" * 50)
        
        # Analyze conversation with multilingual support
        multilingual_result = analyzer.analyze_conversation_multilingual(conversation)
        
        # Display results
        print(f"Languages Detected: {multilingual_result['multilingual_metrics']['total_languages_detected']}")
        print(f"Primary Language: {analyzer.get_language_name(multilingual_result['multilingual_metrics']['primary_language'])}")
        print(f"Language Distribution: {multilingual_result['multilingual_metrics']['language_distribution']}")
        print(f"Language Diversity: {multilingual_result['multilingual_metrics']['language_diversity']:.2f}")
        
        # Analyze individual messages
        for msg_key, msg_analysis in multilingual_result['sentiment_analysis'].items():
            if msg_key.startswith('message_'):
                print(f"\n  {msg_key.upper()}:")
                
                # User analysis
                if msg_analysis['user_analysis']:
                    user_result = msg_analysis['user_analysis']
                    print(f"    👤 User ({analyzer.get_language_name(user_result.detected_language)}):")
                    print(f"      Text: {user_result.text}")
                    print(f"      Sentiment: {user_result.sentiment} (confidence: {user_result.confidence:.2f})")
                    print(f"      Language Confidence: {user_result.language_confidence:.2f}")
                    if user_result.emotions:
                        emotions_str = ', '.join([f'{k}: {v:.2f}' for k, v in user_result.emotions.items() if v > 0])
                        print(f"      Emotions: {emotions_str}")
                
                # Bot analysis
                if msg_analysis['bot_analysis']:
                    bot_result = msg_analysis['bot_analysis']
                    print(f"    🤖 Bot ({analyzer.get_language_name(bot_result.detected_language)}):")
                    print(f"      Text: {bot_result.text}")
                    print(f"      Sentiment: {bot_result.sentiment} (confidence: {bot_result.confidence:.2f})")
                    print(f"      Language Confidence: {bot_result.language_confidence:.2f}")
                    if bot_result.emotions:
                        emotions_str = ', '.join([f'{k}: {v:.2f}' for k, v in bot_result.emotions.items() if v > 0])
                        print(f"      Emotions: {emotions_str}")
                
                # Cross-language consensus if available
                if msg_analysis['user_analysis'] and msg_analysis['user_analysis'].cross_language_consensus:
                    consensus = msg_analysis['user_analysis'].cross_language_consensus
                    print(f"    🌍 Cross-language Consensus:")
                    print(f"      Consensus Sentiment: {consensus['consensus_sentiment']}")
                    print(f"      Agreement Rate: {consensus['agreement_rate']:.2f}")
                    print(f"      Languages Analyzed: {consensus['total_languages']}")
        
        all_results.append(multilingual_result)
    
    # Generate summary statistics
    print(f"\n📊 MULTILINGUAL ANALYSIS SUMMARY")
    print("=" * 50)
    
    total_conversations = len(conversations)
    total_languages_detected = sum(result['multilingual_metrics']['total_languages_detected'] for result in all_results)
    avg_language_diversity = sum(result['multilingual_metrics']['language_diversity'] for result in all_results) / total_conversations
    
    print(f"Total Conversations Analyzed: {total_conversations}")
    print(f"Total Languages Detected: {total_languages_detected}")
    print(f"Average Language Diversity: {avg_language_diversity:.2f}")
    
    # Language distribution across all conversations
    all_language_dist = {}
    for result in all_results:
        for lang, count in result['multilingual_metrics']['language_distribution'].items():
            all_language_dist[lang] = all_language_dist.get(lang, 0) + count
    
    print(f"\nOverall Language Distribution:")
    for lang, count in all_language_dist.items():
        print(f"  {analyzer.get_language_name(lang)}: {count} messages")
    
    # Sentiment analysis by language
    print(f"\nSentiment Analysis by Language:")
    for result in all_results:
        for lang, sentiments in result['multilingual_metrics']['sentiment_by_language'].items():
            from collections import Counter
            sentiment_counts = Counter(sentiments)
            print(f"  {analyzer.get_language_name(lang)}: {dict(sentiment_counts)}")
    
    # Test individual multilingual analysis
    print(f"\n🧪 INDIVIDUAL MULTILINGUAL ANALYSIS TEST")
    print("=" * 50)
    
    test_texts = [
        ("I love this product!", "en", "English positive"),
        ("¡Me encanta este producto!", "es", "Spanish positive"),
        ("这个产品太棒了！", "zh", "Chinese positive"),
        ("This is terrible!", "en", "English negative"),
        ("¡Esto es terrible!", "es", "Spanish negative"),
        ("这太糟糕了！", "zh", "Chinese negative")
    ]
    
    for text, expected_lang, description in test_texts:
        print(f"\nTesting: {description}")
        print(f"Text: {text}")
        
        # Analyze with multilingual support
        result = analyzer.analyze_sentiment_multilingual(text, enable_cross_language=True)
        
        print(f"Detected Language: {analyzer.get_language_name(result.detected_language)} (confidence: {result.language_confidence:.2f})")
        print(f"Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
        print(f"Methods Used: {', '.join(result.methods_used)}")
        
        if result.emotions:
            emotions_str = ', '.join([f'{k}: {v:.2f}' for k, v in result.emotions.items() if v > 0])
            print(f"Emotions: {emotions_str}")
        
        if result.cross_language_consensus:
            consensus = result.cross_language_consensus
            print(f"Cross-language Consensus: {consensus['consensus_sentiment']} (agreement: {consensus['agreement_rate']:.2f})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"multilingual_analysis_results_{timestamp}.json"
    
    # Convert MultilingualSentimentResult objects to dictionaries for JSON serialization
    serializable_results = []
    for result in all_results:
        serializable_result = dict(result)
        # Convert any MultilingualSentimentResult objects in sentiment_analysis
        for msg_key, msg_analysis in result['sentiment_analysis'].items():
            if 'user_analysis' in msg_analysis and msg_analysis['user_analysis']:
                msg_analysis['user_analysis'] = {
                    'text': msg_analysis['user_analysis'].text,
                    'detected_language': msg_analysis['user_analysis'].detected_language,
                    'language_confidence': msg_analysis['user_analysis'].language_confidence,
                    'sentiment': msg_analysis['user_analysis'].sentiment,
                    'confidence': msg_analysis['user_analysis'].confidence,
                    'emotions': msg_analysis['user_analysis'].emotions,
                    'methods_used': msg_analysis['user_analysis'].methods_used,
                    'language_specific_analysis': msg_analysis['user_analysis'].language_specific_analysis,
                    'cross_language_consensus': msg_analysis['user_analysis'].cross_language_consensus
                }
            if 'bot_analysis' in msg_analysis and msg_analysis['bot_analysis']:
                msg_analysis['bot_analysis'] = {
                    'text': msg_analysis['bot_analysis'].text,
                    'detected_language': msg_analysis['bot_analysis'].detected_language,
                    'language_confidence': msg_analysis['bot_analysis'].language_confidence,
                    'sentiment': msg_analysis['bot_analysis'].sentiment,
                    'confidence': msg_analysis['bot_analysis'].confidence,
                    'emotions': msg_analysis['bot_analysis'].emotions,
                    'methods_used': msg_analysis['bot_analysis'].methods_used,
                    'language_specific_analysis': msg_analysis['bot_analysis'].language_specific_analysis,
                    'cross_language_consensus': msg_analysis['bot_analysis'].cross_language_consensus
                }
        serializable_results.append(serializable_result)
    
    with open(results_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': timestamp,
            'total_conversations': total_conversations,
            'total_languages_detected': total_languages_detected,
            'average_language_diversity': avg_language_diversity,
            'language_distribution': all_language_dist,
            'conversation_results': serializable_results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Results saved to: {results_filename}")
    
    print(f"\n✅ Multilingual sentiment analysis demo completed!")
    print(f"🌍 SentilensAI now supports {len(supported_languages)} languages!")
    print(f"🚀 Ready for global AI chatbot conversations!")

def main():
    """Main function to run multilingual demo"""
    try:
        run_multilingual_demo()
    except Exception as e:
        print(f"❌ Error in multilingual demo: {e}")
        print("Please ensure all dependencies are installed:")
        print("pip install langdetect transformers torch")

if __name__ == "__main__":
    main()
