"""
SentilensAI - Comprehensive Usage Examples

This module provides comprehensive examples and demonstrations of SentilensAI
capabilities for sentiment analysis in AI chatbot conversations.

Author: Pravin Selvamuthu
Repository: https://github.com/kernelseed/sentilens-ai
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# Import SentilensAI modules
from sentiment_analyzer import SentilensAIAnalyzer, SentimentResult
from chatbot_integration import SentilensAIChatbotIntegration, AlertConfig
from ml_training_pipeline import SentilensAITrainer
from visualization import SentilensAIVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def demo_basic_sentiment_analysis():
    """Demonstrate basic sentiment analysis capabilities"""
    print("🤖 SentilensAI - Basic Sentiment Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = SentilensAIAnalyzer()
    
    # Sample texts for analysis
    sample_texts = [
        "I love this chatbot! It's amazing and so helpful!",
        "This is terrible. The bot doesn't understand anything.",
        "Can you help me with my account balance?",
        "I'm so frustrated with this service. It's the worst!",
        "Thank you so much! You've been incredibly helpful.",
        "I need to speak to a manager immediately!",
        "This is okay, nothing special but it works.",
        "Wow! This is exactly what I was looking for!"
    ]
    
    print("\n📊 Analyzing sample texts with different methods...")
    
    # Test different analysis methods
    methods = ['vader', 'textblob', 'ensemble']
    
    for method in methods:
        print(f"\n--- {method.upper()} Method ---")
        for i, text in enumerate(sample_texts, 1):
            try:
                result = analyzer.analyze_sentiment(text, method=method)
                print(f"{i}. '{text[:50]}...'")
                print(f"   Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
                print(f"   Polarity: {result.polarity:.2f}")
                if result.emotions:
                    top_emotion = max(result.emotions.items(), key=lambda x: x[1])
                    print(f"   Top Emotion: {top_emotion[0]} ({top_emotion[1]:.2f})")
            except Exception as e:
                print(f"   Error: {e}")
            print()


def demo_chatbot_conversation_analysis():
    """Demonstrate chatbot conversation analysis"""
    print("🤖 SentilensAI - Chatbot Conversation Analysis Demo")
    print("=" * 60)
    
    # Configure alerts
    alert_config = AlertConfig(
        negative_sentiment_threshold=0.3,
        satisfaction_threshold=0.4,
        escalation_keywords=['manager', 'supervisor', 'complaint', 'terrible']
    )
    
    # Initialize integration
    integration = SentilensAIChatbotIntegration(alert_config=alert_config)
    
    # Sample conversation
    conversation = [
        {
            'user': 'Hi, I need help with my account',
            'bot': 'Hello! I\'d be happy to help you with your account. What specific issue are you experiencing?',
            'conversation_id': 'demo_conv_001',
            'timestamp': datetime.now() - timedelta(minutes=10)
        },
        {
            'user': 'I can\'t log in. This is so frustrating!',
            'bot': 'I understand your frustration. Let me help you resolve this login issue. Can you tell me what error message you\'re seeing?',
            'conversation_id': 'demo_conv_001',
            'timestamp': datetime.now() - timedelta(minutes=9)
        },
        {
            'user': 'It says my password is wrong but I know it\'s correct',
            'bot': 'That can be very frustrating. Let\'s try resetting your password. I\'ll guide you through the process step by step.',
            'conversation_id': 'demo_conv_001',
            'timestamp': datetime.now() - timedelta(minutes=8)
        },
        {
            'user': 'Thank you! That worked perfectly. You\'re amazing!',
            'bot': 'You\'re very welcome! I\'m so glad I could help you get back into your account. Is there anything else I can assist you with today?',
            'conversation_id': 'demo_conv_001',
            'timestamp': datetime.now() - timedelta(minutes=7)
        },
        {
            'user': 'Actually, I want to speak to a manager about this terrible experience',
            'bot': 'I understand you\'d like to speak with a manager. I\'m sorry that your experience wasn\'t what you expected. Let me connect you with our supervisor right away.',
            'conversation_id': 'demo_conv_001',
            'timestamp': datetime.now() - timedelta(minutes=6)
        }
    ]
    
    print("\n📊 Processing conversation...")
    
    # Process conversation
    results = integration.process_conversation_batch(conversation)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n--- Message {i} ---")
        print(f"User: {result.user_message}")
        print(f"Bot: {result.bot_response}")
        print(f"User Sentiment: {result.user_sentiment.sentiment} (confidence: {result.user_sentiment.confidence:.2f})")
        print(f"Bot Sentiment: {result.bot_sentiment.sentiment} (confidence: {result.bot_sentiment.confidence:.2f})")
        print(f"Conversation Sentiment: {result.conversation_sentiment}")
        print(f"Satisfaction Score: {result.satisfaction_score:.2f}")
        
        # Show emotions
        if result.user_sentiment.emotions:
            top_emotions = sorted(result.user_sentiment.emotions.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"Top Emotions: {', '.join([f'{e}({s:.2f})' for e, s in top_emotions])}")
    
    # Get conversation analysis
    print(f"\n📈 Conversation Analysis:")
    metrics = integration.get_conversation_analysis('demo_conv_001')
    if metrics:
        print(f"Total Messages: {metrics.total_messages}")
        print(f"Average User Sentiment: {metrics.average_user_sentiment:.2f}")
        print(f"Average Bot Sentiment: {metrics.average_bot_sentiment:.2f}")
        print(f"Satisfaction Score: {metrics.satisfaction_score:.2f}")
        print(f"Conversation Quality: {metrics.conversation_quality}")
        print(f"Key Emotions: {metrics.key_emotions}")
        print(f"Escalation Events: {len(metrics.escalation_events)}")
    
    # Get dashboard data
    print(f"\n📊 Dashboard Data:")
    dashboard = integration.get_sentiment_dashboard_data()
    print(f"Total Conversations: {dashboard['total_conversations']}")
    print(f"Total Messages: {dashboard['total_messages']}")
    print(f"Average Satisfaction: {dashboard['average_satisfaction']:.2f}")
    print(f"Active Alerts: {dashboard['active_alerts']}")
    
    # Show recent alerts
    if dashboard['recent_alerts']:
        print(f"\n🚨 Recent Alerts:")
        for alert in dashboard['recent_alerts'][:3]:
            print(f"- [{alert['severity'].upper()}] {alert['type']}: {alert['message']}")


def demo_ml_training_pipeline():
    """Demonstrate machine learning training pipeline"""
    print("🤖 SentilensAI - ML Training Pipeline Demo")
    print("=" * 50)
    
    # Initialize trainer
    trainer = SentilensAITrainer()
    
    # Get training summary
    summary = trainer.get_training_summary()
    print(f"\n📊 Training Configuration:")
    print(f"Available Models: {len(summary['available_models'])}")
    print(f"XGBoost Available: {summary['xgboost_available']}")
    print(f"LightGBM Available: {summary['lightgbm_available']}")
    print(f"CatBoost Available: {summary['catboost_available']}")
    print(f"Plotting Available: {summary['plotting_available']}")
    
    # Create synthetic training data
    print(f"\n🔄 Creating synthetic training data...")
    training_data = trainer.create_synthetic_training_data(num_samples=200)
    print(f"Created {len(training_data)} training samples")
    print(f"Sentiment distribution: {training_data['sentiment'].value_counts().to_dict()}")
    
    # Train all models
    print(f"\n🚀 Training all models...")
    results = trainer.train_all_models(training_data, optimize_hyperparameters=True)
    
    # Display results
    print(f"\n📈 Training Results:")
    print("-" * 60)
    for model_name, result in results.items():
        if 'error' not in result:
            print(f"{model_name:20} | F1: {result['f1_macro']:.3f} | Accuracy: {result['accuracy']:.3f} | Time: {result['training_time']:.1f}s")
        else:
            print(f"{model_name:20} | Error: {result['error']}")
    
    # Test predictions
    print(f"\n🔮 Testing predictions...")
    test_texts = [
        "I love this chatbot! It's amazing!",
        "This is terrible. I hate it.",
        "Can you help me with my account?"
    ]
    
    for text in test_texts:
        try:
            prediction = trainer.predict_sentiment(text, 'random_forest')
            print(f"Text: '{text}'")
            print(f"Prediction: {prediction['sentiment']} (confidence: {prediction['confidence']:.3f})")
            if prediction['probabilities']:
                probs = prediction['probabilities']
                print(f"Probabilities: {', '.join([f'{k}: {v:.2f}' for k, v in probs.items()])}")
        except Exception as e:
            print(f"Prediction failed: {e}")
        print()
    
    # Model comparison
    print(f"\n📊 Model Comparison:")
    try:
        # Extract features for comparison
        texts = training_data['text'].tolist()
        labels = training_data['sentiment'].tolist()
        X = trainer.extract_features(texts)
        y = trainer.label_encoder.fit_transform(labels)
        
        comparison = trainer.compare_models(X, y, models_to_compare=['random_forest', 'svm', 'neural_network'])
        for model_name, metrics in comparison.items():
            print(f"{model_name:20} | CV F1: {metrics['cv_mean']:.3f} ± {metrics['cv_std']:.3f} | Accuracy: {metrics['accuracy']:.3f}")
    except Exception as e:
        print(f"Model comparison failed: {e}")


def demo_visualization_capabilities():
    """Demonstrate visualization and reporting capabilities"""
    print("🤖 SentilensAI - Visualization Demo")
    print("=" * 40)
    
    # Initialize visualizer
    visualizer = SentilensAIVisualizer()
    
    print(f"📊 Visualization Libraries Available:")
    print(f"Matplotlib: {visualizer.matplotlib_available}")
    print(f"Plotly: {visualizer.plotly_available}")
    print(f"WordCloud: {visualizer.wordcloud_available}")
    
    if not visualizer.matplotlib_available and not visualizer.plotly_available:
        print("❌ No visualization libraries available. Install matplotlib or plotly to see visualizations.")
        return
    
    # Create sample data
    from sentiment_analyzer import SentimentResult
    
    sample_messages = []
    sentiments = ['positive', 'negative', 'neutral']
    
    for i in range(15):
        sentiment = np.random.choice(sentiments)
        polarity = np.random.uniform(-1, 1) if sentiment == 'neutral' else (0.5 if sentiment == 'positive' else -0.5)
        
        message = SentimentResult(
            text=f"Sample message {i+1} with {sentiment} sentiment about chatbot experience",
            sentiment=sentiment,
            confidence=np.random.uniform(0.6, 1.0),
            polarity=polarity,
            subjectivity=np.random.uniform(0.3, 0.8),
            emotions={
                'joy': np.random.uniform(0, 0.5),
                'sadness': np.random.uniform(0, 0.5),
                'anger': np.random.uniform(0, 0.5),
                'fear': np.random.uniform(0, 0.5),
                'surprise': np.random.uniform(0, 0.5),
                'disgust': np.random.uniform(0, 0.5)
            },
            timestamp=datetime.now() - timedelta(minutes=i*2),
            model_used='ensemble',
            metadata={}
        )
        sample_messages.append(message)
    
    print(f"\n📈 Creating visualizations with {len(sample_messages)} sample messages...")
    
    # Create visualizations
    generated_files = []
    
    try:
        # Sentiment trend
        trend_path = visualizer.plot_sentiment_trend(sample_messages)
        if trend_path:
            generated_files.append(trend_path)
            print(f"✅ Sentiment trend plot: {trend_path}")
    except Exception as e:
        print(f"❌ Failed to create sentiment trend: {e}")
    
    try:
        # Emotion distribution
        emotion_path = visualizer.plot_emotion_distribution(sample_messages)
        if emotion_path:
            generated_files.append(emotion_path)
            print(f"✅ Emotion distribution plot: {emotion_path}")
    except Exception as e:
        print(f"❌ Failed to create emotion distribution: {e}")
    
    try:
        # Word clouds for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            wc_path = visualizer.create_word_cloud(sample_messages, sentiment_filter=sentiment)
            if wc_path:
                generated_files.append(wc_path)
                print(f"✅ Word cloud ({sentiment}): {wc_path}")
    except Exception as e:
        print(f"❌ Failed to create word clouds: {e}")
    
    # Sample model results for comparison
    sample_model_results = {
        'random_forest': {'accuracy': 0.85, 'f1_macro': 0.82, 'training_time': 2.5},
        'svm': {'accuracy': 0.83, 'f1_macro': 0.80, 'training_time': 1.2},
        'neural_network': {'accuracy': 0.87, 'f1_macro': 0.84, 'training_time': 8.3},
        'xgboost': {'accuracy': 0.89, 'f1_macro': 0.86, 'training_time': 3.1}
    }
    
    try:
        # Model comparison
        model_path = visualizer.plot_model_comparison(sample_model_results)
        if model_path:
            generated_files.append(model_path)
            print(f"✅ Model comparison plot: {model_path}")
    except Exception as e:
        print(f"❌ Failed to create model comparison: {e}")
    
    print(f"\n📁 Generated {len(generated_files)} visualization files")
    print(f"📁 Output directory: {visualizer.output_dir}")


def demo_comprehensive_workflow():
    """Demonstrate a comprehensive SentilensAI workflow"""
    print("🤖 SentilensAI - Comprehensive Workflow Demo")
    print("=" * 60)
    
    # Step 1: Initialize all components
    print("\n🔧 Step 1: Initializing SentilensAI components...")
    analyzer = SentilensAIAnalyzer()
    trainer = SentilensAITrainer()
    integration = SentilensAIChatbotIntegration()
    visualizer = SentilensAIVisualizer()
    
    # Step 2: Train a custom model
    print("\n🚀 Step 2: Training custom sentiment model...")
    training_data = trainer.create_synthetic_training_data(num_samples=300)
    training_results = trainer.train_all_models(training_data, optimize_hyperparameters=True)
    
    best_model = max(training_results.keys(), 
                    key=lambda k: training_results[k].get('f1_macro', 0) 
                    if 'error' not in training_results[k] else 0)
    print(f"Best model: {best_model} (F1: {training_results[best_model]['f1_macro']:.3f})")
    
    # Step 3: Process real conversations
    print("\n💬 Step 3: Processing chatbot conversations...")
    conversations = [
        {
            'user': 'Hi, I need help with my order',
            'bot': 'Hello! I\'d be happy to help you with your order. Can you provide your order number?',
            'conversation_id': 'workflow_001'
        },
        {
            'user': 'I can\'t find my order number. This is so frustrating!',
            'bot': 'I understand your frustration. Let me help you locate your order. Can you provide your email address?',
            'conversation_id': 'workflow_001'
        },
        {
            'user': 'Thank you! You found it. I\'m so relieved!',
            'bot': 'You\'re very welcome! I\'m glad I could help you locate your order. Is there anything else I can assist you with?',
            'conversation_id': 'workflow_001'
        }
    ]
    
    conversation_results = integration.process_conversation_batch(conversations)
    
    # Step 4: Analyze with custom model
    print("\n🔮 Step 4: Testing custom model predictions...")
    for result in conversation_results:
        try:
            custom_prediction = trainer.predict_sentiment(result.user_message, best_model)
            print(f"User: '{result.user_message}'")
            print(f"Custom Model: {custom_prediction['sentiment']} (confidence: {custom_prediction['confidence']:.3f})")
            print(f"Ensemble Model: {result.user_sentiment.sentiment} (confidence: {result.user_sentiment.confidence:.3f})")
            print()
        except Exception as e:
            print(f"Custom prediction failed: {e}")
    
    # Step 5: Generate comprehensive report
    print("\n📊 Step 5: Generating comprehensive analysis report...")
    try:
        report_files = visualizer.generate_comprehensive_report(integration)
        print(f"Generated {len(report_files)} report files:")
        for report_type, file_path in report_files.items():
            print(f"- {report_type}: {file_path}")
    except Exception as e:
        print(f"Report generation failed: {e}")
    
    # Step 6: Export data
    print("\n💾 Step 6: Exporting conversation data...")
    try:
        export_file = integration.export_conversation_data('workflow_001', 'json')
        print(f"Conversation data exported to: {export_file}")
    except Exception as e:
        print(f"Export failed: {e}")
    
    print("\n✅ Comprehensive workflow completed successfully!")
    print("🚀 SentilensAI is ready for production use!")


def demo_advanced_features():
    """Demonstrate advanced SentilensAI features"""
    print("🤖 SentilensAI - Advanced Features Demo")
    print("=" * 50)
    
    # Advanced alert configuration
    print("\n🚨 Advanced Alert Configuration:")
    advanced_alert_config = AlertConfig(
        negative_sentiment_threshold=0.2,
        satisfaction_threshold=0.3,
        consecutive_negative_limit=2,
        escalation_keywords=['manager', 'supervisor', 'complaint', 'terrible', 'awful', 'worst', 'hate'],
        alert_channels=['console', 'webhook']  # Would need webhook handler
    )
    
    integration = SentilensAIChatbotIntegration(alert_config=advanced_alert_config)
    
    # Test advanced alerting
    test_messages = [
        "This is terrible! I want to speak to a manager!",
        "I hate this service. It's the worst experience ever!",
        "This is awful. I'm filing a complaint!",
        "I need a supervisor immediately!"
    ]
    
    print("\n📊 Testing advanced alerting system...")
    for i, message in enumerate(test_messages, 1):
        result = integration.process_message(
            user_message=message,
            bot_response="I understand your concern. Let me help you.",
            conversation_id=f"alert_test_{i}"
        )
        print(f"Message {i}: '{message}'")
        print(f"Sentiment: {result.user_sentiment.sentiment} (polarity: {result.user_sentiment.polarity:.2f})")
        print(f"Satisfaction: {result.satisfaction_score:.2f}")
        print()
    
    # Advanced emotion analysis
    print("\n😊 Advanced Emotion Analysis:")
    analyzer = SentilensAIAnalyzer()
    
    emotion_test_texts = [
        "I'm so excited about this new feature!",
        "I'm really worried about my data security.",
        "This makes me so angry!",
        "I'm surprised by how well this works!",
        "This is disgusting and revolting!",
        "I'm so sad about what happened."
    ]
    
    for text in emotion_test_texts:
        result = analyzer.analyze_sentiment(text, method='ensemble')
        top_emotions = sorted(result.emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"Text: '{text}'")
        print(f"Top Emotions: {', '.join([f'{e}({s:.2f})' for e, s in top_emotions])}")
        print()
    
    # Advanced model comparison
    print("\n📊 Advanced Model Comparison:")
    trainer = SentilensAITrainer()
    training_data = trainer.create_synthetic_training_data(num_samples=500)
    
    # Train specific models for comparison
    models_to_compare = ['random_forest', 'svm', 'neural_network']
    if trainer.get_training_summary()['xgboost_available']:
        models_to_compare.append('xgboost')
    
    comparison_results = {}
    for model_name in models_to_compare:
        try:
            texts = training_data['text'].tolist()
            labels = training_data['sentiment'].tolist()
            X = trainer.extract_features(texts)
            y = trainer.label_encoder.fit_transform(labels)
            
            result = trainer.train_model(model_name, X, y, optimize_hyperparameters=True)
            comparison_results[model_name] = result
        except Exception as e:
            print(f"Failed to train {model_name}: {e}")
    
    # Display detailed comparison
    print("\nDetailed Model Performance:")
    print("-" * 80)
    for model_name, result in comparison_results.items():
        if 'error' not in result:
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  F1-Macro: {result['f1_macro']:.3f}")
            print(f"  Precision: {result['precision_macro']:.3f}")
            print(f"  Recall: {result['recall_macro']:.3f}")
            print(f"  Training Time: {result['training_time']:.1f}s")
            print(f"  Matthews Correlation: {result['matthews_corrcoef']:.3f}")
            print(f"  Cohen's Kappa: {result['cohen_kappa']:.3f}")


def cleanup_demo_files():
    """Clean up demo files"""
    import os
    import glob
    
    print("\n🧹 Cleaning up demo files...")
    
    # Remove generated files
    patterns = [
        "sentiment_trend_*.png",
        "emotion_distribution_*.png", 
        "wordcloud_*.png",
        "model_comparison_*.png",
        "conversation_dashboard_*.png",
        "interactive_dashboard_*.html",
        "summary_report_*.md",
        "conversation_*.json",
        "sentiments_ai_*_model.pkl"
    ]
    
    removed_count = 0
    for pattern in patterns:
        files = glob.glob(pattern)
        for file in files:
            try:
                os.remove(file)
                removed_count += 1
            except:
                pass
    
    print(f"Removed {removed_count} demo files")


def main():
    """Main demo function showcasing all SentilensAI capabilities"""
    print("🤖 SentilensAI - Comprehensive Demo Suite")
    print("=" * 60)
    print("This demo showcases all major SentilensAI capabilities:")
    print("1. Basic sentiment analysis")
    print("2. Chatbot conversation analysis")
    print("3. Machine learning training pipeline")
    print("4. Visualization and reporting")
    print("5. Comprehensive workflow")
    print("6. Advanced features")
    print("=" * 60)
    
    try:
        # Run all demos
        demo_basic_sentiment_analysis()
        print("\n" + "="*60 + "\n")
        
        demo_chatbot_conversation_analysis()
        print("\n" + "="*60 + "\n")
        
        demo_ml_training_pipeline()
        print("\n" + "="*60 + "\n")
        
        demo_visualization_capabilities()
        print("\n" + "="*60 + "\n")
        
        demo_comprehensive_workflow()
        print("\n" + "="*60 + "\n")
        
        demo_advanced_features()
        
        # Multilingual demos
        demo_multilingual_analysis()
        demo_multilingual_conversation()
        
        print("\n" + "="*60)
        print("🎉 All SentilensAI demos completed successfully!")
        print("🚀 SentilensAI is ready for production use!")
        print("🌍 Multilingual support: English, Spanish, Chinese")
        print("📚 Check the README.md for detailed documentation")
        print("🔗 Repository: https://github.com/kernelseed/sentiments-ai")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed with error: {e}")
        logger.exception("Demo failed")
    finally:
        # Ask user if they want to clean up
        try:
            cleanup_choice = input("\n🧹 Clean up generated demo files? (y/n): ").lower().strip()
            if cleanup_choice in ['y', 'yes']:
                cleanup_demo_files()
        except:
            pass


def demo_multilingual_analysis():
    """Demonstrate multilingual sentiment analysis capabilities"""
    print("\n🌍 Multilingual Sentiment Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer with multilingual support
    analyzer = SentilensAIAnalyzer(enable_multilingual=True)
    
    # Sample texts in different languages
    multilingual_texts = [
        {
            'text': "I love this product! It's amazing!",
            'language': 'English',
            'expected_sentiment': 'positive'
        },
        {
            'text': "¡Me encanta este producto! ¡Es increíble!",
            'language': 'Spanish', 
            'expected_sentiment': 'positive'
        },
        {
            'text': "这个产品太棒了！我非常喜欢！",
            'language': 'Chinese',
            'expected_sentiment': 'positive'
        },
        {
            'text': "This is terrible! I hate it!",
            'language': 'English',
            'expected_sentiment': 'negative'
        },
        {
            'text': "¡Esto es terrible! ¡Lo odio!",
            'language': 'Spanish',
            'expected_sentiment': 'negative'
        },
        {
            'text': "这太糟糕了！我讨厌它！",
            'language': 'Chinese',
            'expected_sentiment': 'negative'
        }
    ]
    
    print(f"Supported Languages: {', '.join([analyzer.get_language_name(lang) for lang in analyzer.get_supported_languages()])}")
    print()
    
    for i, sample in enumerate(multilingual_texts, 1):
        print(f"📝 Sample {i} ({sample['language']}): {sample['text']}")
        
        # Analyze with multilingual support
        result = analyzer.analyze_sentiment_multilingual(
            sample['text'], 
            enable_cross_language=True
        )
        
        print(f"  Detected Language: {analyzer.get_language_name(result.detected_language)} (confidence: {result.language_confidence:.2f})")
        print(f"  Sentiment: {result.sentiment} (confidence: {result.confidence:.2f})")
        print(f"  Methods Used: {', '.join(result.methods_used)}")
        
        if result.emotions:
            emotions_str = ', '.join([f'{k}: {v:.2f}' for k, v in result.emotions.items() if v > 0])
            print(f"  Emotions: {emotions_str}")
        
        if result.cross_language_consensus:
            consensus = result.cross_language_consensus
            print(f"  Cross-Language Consensus: {consensus['consensus_sentiment']} (agreement: {consensus['agreement_rate']:.2f})")
        
        print("-" * 30)
    
    print("\n✅ Multilingual analysis demo completed!")
    print("🌍 SentilensAI supports English, Spanish, and Chinese!")
    print("🚀 Ready for global AI chatbot conversations!")


def demo_multilingual_conversation():
    """Demonstrate multilingual conversation analysis"""
    print("\n🗣️ Multilingual Conversation Analysis Demo")
    print("=" * 50)
    
    # Initialize analyzer with multilingual support
    analyzer = SentilensAIAnalyzer(enable_multilingual=True)
    
    # Sample multilingual conversation
    conversation = {
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
            },
            {
                'user': '¡Muchas gracias!',
                'bot': '不客气！很高兴能帮助您。',
                'timestamp': '2024-01-15T10:30:45Z'
            }
        ]
    }
    
    print("📋 Analyzing multilingual conversation...")
    print(f"Conversation ID: {conversation['conversation_id']}")
    print()
    
    # Analyze conversation with multilingual support
    result = analyzer.analyze_conversation_multilingual(conversation)
    
    print(f"Languages Detected: {result['multilingual_metrics']['total_languages_detected']}")
    print(f"Primary Language: {analyzer.get_language_name(result['multilingual_metrics']['primary_language'])}")
    print(f"Language Distribution: {result['multilingual_metrics']['language_distribution']}")
    print(f"Language Diversity: {result['multilingual_metrics']['language_diversity']:.2f}")
    print()
    
    # Analyze individual messages
    for msg_key, msg_analysis in result['sentiment_analysis'].items():
        if msg_key.startswith('message_'):
            print(f"  {msg_key.upper()}:")
            
            # User analysis
            if msg_analysis['user_analysis']:
                user_result = msg_analysis['user_analysis']
                print(f"    👤 User ({analyzer.get_language_name(user_result.detected_language)}):")
                print(f"      Text: {user_result.text}")
                print(f"      Sentiment: {user_result.sentiment} (confidence: {user_result.confidence:.2f})")
                print(f"      Language Confidence: {user_result.language_confidence:.2f}")
            
            # Bot analysis
            if msg_analysis['bot_analysis']:
                bot_result = msg_analysis['bot_analysis']
                print(f"    🤖 Bot ({analyzer.get_language_name(bot_result.detected_language)}):")
                print(f"      Text: {bot_result.text}")
                print(f"      Sentiment: {bot_result.sentiment} (confidence: {bot_result.confidence:.2f})")
                print(f"      Language Confidence: {bot_result.language_confidence:.2f}")
            
            print()
    
    print("✅ Multilingual conversation analysis completed!")
    print("🌍 Successfully analyzed mixed-language conversation!")
    print("🚀 SentilensAI handles global conversations seamlessly!")


if __name__ == "__main__":
    main()
