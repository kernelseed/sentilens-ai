"""
SentilensAI - Chatbot Integration Module

This module provides comprehensive integration capabilities for analyzing
sentiment in real-time chatbot conversations and AI agent interactions.

Features:
- Real-time sentiment monitoring for chatbot conversations
- Integration with popular chatbot platforms (Discord, Slack, Telegram, etc.)
- Conversation flow analysis and sentiment tracking
- Automated response quality assessment
- Customer satisfaction scoring
- Alert system for negative sentiment detection

Author: Pravin Selvamuthu
Repository: https://github.com/kernelseed/sentilens-ai
"""

import asyncio
import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from pathlib import Path
import uuid

import pandas as pd
import numpy as np
from collections import deque, defaultdict

# Import our core modules
from sentiment_analyzer import SentilensAIAnalyzer, SentimentResult, ChatbotMessage
from ml_training_pipeline import SentilensAITrainer

# Web framework for API endpoints
try:
    from fastapi import FastAPI, WebSocket, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ConversationMetrics:
    """Metrics for a complete conversation"""
    conversation_id: str
    start_time: datetime
    end_time: datetime
    total_messages: int
    user_messages: int
    bot_messages: int
    average_user_sentiment: float
    average_bot_sentiment: float
    sentiment_trend: List[str]
    satisfaction_score: float
    conversation_quality: str  # excellent, good, fair, poor
    key_emotions: Dict[str, float]
    response_times: List[float]  # in seconds
    escalation_events: List[str]


@dataclass
class AlertConfig:
    """Configuration for sentiment alerts"""
    negative_sentiment_threshold: float = 0.3
    satisfaction_threshold: float = 0.4
    consecutive_negative_limit: int = 3
    escalation_keywords: List[str] = None
    alert_channels: List[str] = None  # email, slack, webhook, etc.
    
    def __post_init__(self):
        if self.escalation_keywords is None:
            self.escalation_keywords = [
                'manager', 'supervisor', 'complaint', 'terrible', 'awful',
                'worst', 'hate', 'angry', 'frustrated', 'disappointed'
            ]
        if self.alert_channels is None:
            self.alert_channels = ['console']


@dataclass
class SentimentAlert:
    """Alert for negative sentiment detection"""
    alert_id: str
    conversation_id: str
    alert_type: str  # negative_sentiment, low_satisfaction, escalation
    severity: str  # low, medium, high, critical
    message: str
    timestamp: datetime
    context: Dict[str, Any]
    resolved: bool = False


class ConversationTracker:
    """Tracks and analyzes ongoing conversations"""
    
    def __init__(self, max_conversation_history: int = 100):
        self.max_history = max_conversation_history
        self.conversations: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_conversation_history))
        self.conversation_metrics: Dict[str, ConversationMetrics] = {}
        self.active_conversations: Dict[str, datetime] = {}
    
    def add_message(self, conversation_id: str, message: ChatbotMessage):
        """Add a message to conversation tracking"""
        self.conversations[conversation_id].append(message)
        self.active_conversations[conversation_id] = datetime.now()
    
    def get_conversation_sentiment_trend(self, conversation_id: str) -> List[str]:
        """Get sentiment trend for a conversation"""
        if conversation_id not in self.conversations:
            return []
        
        messages = list(self.conversations[conversation_id])
        return [msg.conversation_sentiment for msg in messages]
    
    def calculate_conversation_metrics(self, conversation_id: str) -> ConversationMetrics:
        """Calculate comprehensive metrics for a conversation"""
        if conversation_id not in self.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        messages = list(self.conversations[conversation_id])
        if not messages:
            raise ValueError(f"No messages found for conversation {conversation_id}")
        
        # Basic metrics
        total_messages = len(messages)
        user_messages = sum(1 for msg in messages if msg.user_message.strip())
        bot_messages = sum(1 for msg in messages if msg.bot_response.strip())
        
        # Time metrics
        start_time = min(msg.timestamp for msg in messages)
        end_time = max(msg.timestamp for msg in messages)
        
        # Sentiment metrics
        user_sentiments = [msg.user_sentiment.polarity for msg in messages if msg.user_message.strip()]
        bot_sentiments = [msg.bot_sentiment.polarity for msg in messages if msg.bot_response.strip()]
        
        avg_user_sentiment = np.mean(user_sentiments) if user_sentiments else 0.0
        avg_bot_sentiment = np.mean(bot_sentiments) if bot_sentiments else 0.0
        
        # Sentiment trend
        sentiment_trend = self.get_conversation_sentiment_trend(conversation_id)
        
        # Satisfaction score
        satisfaction_scores = [msg.satisfaction_score for msg in messages]
        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else 0.0
        
        # Conversation quality
        if avg_satisfaction >= 0.8:
            quality = 'excellent'
        elif avg_satisfaction >= 0.6:
            quality = 'good'
        elif avg_satisfaction >= 0.4:
            quality = 'fair'
        else:
            quality = 'poor'
        
        # Key emotions
        all_emotions = defaultdict(list)
        for msg in messages:
            for emotion, score in msg.user_sentiment.emotions.items():
                all_emotions[emotion].append(score)
        
        key_emotions = {
            emotion: np.mean(scores) for emotion, scores in all_emotions.items()
            if np.mean(scores) > 0.1
        }
        
        # Response times (simplified - would need actual timing data)
        response_times = []
        for i in range(1, len(messages)):
            time_diff = (messages[i].timestamp - messages[i-1].timestamp).total_seconds()
            if time_diff < 300:  # Only count responses within 5 minutes
                response_times.append(time_diff)
        
        # Escalation events
        escalation_events = []
        for msg in messages:
            if any(keyword in msg.user_message.lower() for keyword in ['manager', 'supervisor', 'complaint']):
                escalation_events.append(f"Escalation request at {msg.timestamp}")
        
        metrics = ConversationMetrics(
            conversation_id=conversation_id,
            start_time=start_time,
            end_time=end_time,
            total_messages=total_messages,
            user_messages=user_messages,
            bot_messages=bot_messages,
            average_user_sentiment=avg_user_sentiment,
            average_bot_sentiment=avg_bot_sentiment,
            sentiment_trend=sentiment_trend,
            satisfaction_score=avg_satisfaction,
            conversation_quality=quality,
            key_emotions=key_emotions,
            response_times=response_times,
            escalation_events=escalation_events
        )
        
        self.conversation_metrics[conversation_id] = metrics
        return metrics


class SentimentAlertManager:
    """Manages sentiment-based alerts and notifications"""
    
    def __init__(self, config: AlertConfig):
        self.config = config
        self.active_alerts: Dict[str, SentimentAlert] = {}
        self.alert_history: List[SentimentAlert] = []
        self.alert_handlers: Dict[str, Callable] = {}
    
    def register_alert_handler(self, channel: str, handler: Callable):
        """Register a custom alert handler"""
        self.alert_handlers[channel] = handler
    
    def check_for_alerts(self, message: ChatbotMessage, conversation_metrics: Optional[ConversationMetrics] = None) -> List[SentimentAlert]:
        """Check if message triggers any alerts"""
        alerts = []
        
        # Negative sentiment alert
        if message.user_sentiment.polarity < -self.config.negative_sentiment_threshold:
            alert = SentimentAlert(
                alert_id=str(uuid.uuid4()),
                conversation_id=message.conversation_id,
                alert_type='negative_sentiment',
                severity='high' if message.user_sentiment.polarity < -0.7 else 'medium',
                message=f"Negative sentiment detected: {message.user_sentiment.sentiment} (polarity: {message.user_sentiment.polarity:.2f})",
                timestamp=datetime.now(),
                context={
                    'user_message': message.user_message,
                    'sentiment_confidence': message.user_sentiment.confidence,
                    'emotions': message.user_sentiment.emotions
                }
            )
            alerts.append(alert)
        
        # Low satisfaction alert
        if message.satisfaction_score < self.config.satisfaction_threshold:
            alert = SentimentAlert(
                alert_id=str(uuid.uuid4()),
                conversation_id=message.conversation_id,
                alert_type='low_satisfaction',
                severity='medium',
                message=f"Low satisfaction score: {message.satisfaction_score:.2f}",
                timestamp=datetime.now(),
                context={
                    'satisfaction_score': message.satisfaction_score,
                    'user_sentiment': message.user_sentiment.sentiment,
                    'bot_sentiment': message.bot_sentiment.sentiment
                }
            )
            alerts.append(alert)
        
        # Escalation keyword alert
        escalation_found = any(
            keyword in message.user_message.lower() 
            for keyword in self.config.escalation_keywords
        )
        if escalation_found:
            alert = SentimentAlert(
                alert_id=str(uuid.uuid4()),
                conversation_id=message.conversation_id,
                alert_type='escalation',
                severity='critical',
                message=f"Escalation keywords detected in message",
                timestamp=datetime.now(),
                context={
                    'user_message': message.user_message,
                    'detected_keywords': [
                        kw for kw in self.config.escalation_keywords 
                        if kw in message.user_message.lower()
                    ]
                }
            )
            alerts.append(alert)
        
        # Process alerts
        for alert in alerts:
            self.active_alerts[alert.alert_id] = alert
            self.alert_history.append(alert)
            self._send_alert(alert)
        
        return alerts
    
    def _send_alert(self, alert: SentimentAlert):
        """Send alert through configured channels"""
        for channel in self.config.alert_channels:
            if channel in self.alert_handlers:
                try:
                    self.alert_handlers[channel](alert)
                except Exception as e:
                    logger.error(f"Failed to send alert via {channel}: {e}")
            elif channel == 'console':
                logger.warning(f"🚨 ALERT [{alert.severity.upper()}] {alert.alert_type}: {alert.message}")
    
    def resolve_alert(self, alert_id: str):
        """Mark an alert as resolved"""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].resolved = True
            del self.active_alerts[alert_id]


class SentilensAIChatbotIntegration:
    """
    Main integration class for chatbot sentiment analysis
    """
    
    def __init__(self, openai_api_key: Optional[str] = None, alert_config: Optional[AlertConfig] = None):
        """
        Initialize chatbot integration
        
        Args:
            openai_api_key: OpenAI API key for LangChain integration
            alert_config: Configuration for sentiment alerts
        """
        self.analyzer = SentilensAIAnalyzer(openai_api_key=openai_api_key)
        self.trainer = SentilensAITrainer()
        self.conversation_tracker = ConversationTracker()
        self.alert_manager = SentimentAlertManager(alert_config or AlertConfig())
        
        # Register default console alert handler
        self.alert_manager.register_alert_handler('console', self._console_alert_handler)
        
        # Statistics
        self.total_messages_processed = 0
        self.total_conversations = 0
        self.alert_count = 0
    
    def _console_alert_handler(self, alert: SentimentAlert):
        """Default console alert handler"""
        print(f"\n🚨 SENTIMENT ALERT [{alert.severity.upper()}]")
        print(f"Type: {alert.alert_type}")
        print(f"Conversation: {alert.conversation_id}")
        print(f"Message: {alert.message}")
        print(f"Time: {alert.timestamp}")
        if alert.context:
            print(f"Context: {json.dumps(alert.context, indent=2)}")
        print("-" * 50)
    
    def process_message(self, user_message: str, bot_response: str, 
                       conversation_id: str, message_id: Optional[str] = None,
                       timestamp: Optional[datetime] = None) -> ChatbotMessage:
        """
        Process a single message exchange and return analysis
        
        Args:
            user_message: User's message
            bot_response: Bot's response
            conversation_id: Unique conversation identifier
            message_id: Unique message identifier (auto-generated if None)
            timestamp: Message timestamp (current time if None)
            
        Returns:
            ChatbotMessage with sentiment analysis
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        if message_id is None:
            message_id = f"{conversation_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Analyze sentiment
        user_sentiment = self.analyzer.analyze_sentiment(user_message, method='ensemble')
        bot_sentiment = self.analyzer.analyze_sentiment(bot_response, method='ensemble', context=user_message)
        
        # Create chatbot message
        chatbot_message = ChatbotMessage(
            message_id=message_id,
            user_message=user_message,
            bot_response=bot_response,
            timestamp=timestamp,
            conversation_id=conversation_id,
            user_sentiment=user_sentiment,
            bot_sentiment=bot_sentiment,
            conversation_sentiment='neutral',  # Will be calculated
            satisfaction_score=0.0  # Will be calculated
        )
        
        # Calculate conversation sentiment and satisfaction
        chatbot_message.conversation_sentiment = self._determine_conversation_sentiment(
            user_sentiment, bot_sentiment
        )
        chatbot_message.satisfaction_score = self.analyzer._calculate_satisfaction_score(
            user_sentiment, bot_sentiment
        )
        
        # Track conversation
        self.conversation_tracker.add_message(conversation_id, chatbot_message)
        
        # Check for alerts
        alerts = self.alert_manager.check_for_alerts(chatbot_message)
        self.alert_count += len(alerts)
        
        # Update statistics
        self.total_messages_processed += 1
        
        return chatbot_message
    
    def _determine_conversation_sentiment(self, user_sentiment: SentimentResult, 
                                        bot_sentiment: SentimentResult) -> str:
        """Determine overall conversation sentiment"""
        # Weighted combination based on confidence
        user_weight = user_sentiment.confidence
        bot_weight = bot_sentiment.confidence
        total_weight = user_weight + bot_weight
        
        if total_weight > 0:
            user_polarity_weighted = user_sentiment.polarity * (user_weight / total_weight)
            bot_polarity_weighted = bot_sentiment.polarity * (bot_weight / total_weight)
            combined_polarity = user_polarity_weighted + bot_polarity_weighted
            
            if combined_polarity > 0.1:
                return 'positive'
            elif combined_polarity < -0.1:
                return 'negative'
        
        return 'neutral'
    
    def process_conversation_batch(self, messages: List[Dict[str, Any]]) -> List[ChatbotMessage]:
        """
        Process a batch of messages from a conversation
        
        Args:
            messages: List of message dictionaries with 'user', 'bot', 'conversation_id', etc.
            
        Returns:
            List of analyzed ChatbotMessage objects
        """
        results = []
        
        for msg in messages:
            chatbot_message = self.process_message(
                user_message=msg.get('user', ''),
                bot_response=msg.get('bot', ''),
                conversation_id=msg.get('conversation_id', 'unknown'),
                message_id=msg.get('message_id'),
                timestamp=msg.get('timestamp')
            )
            results.append(chatbot_message)
        
        return results
    
    def get_conversation_analysis(self, conversation_id: str) -> Optional[ConversationMetrics]:
        """Get comprehensive analysis for a conversation"""
        try:
            return self.conversation_tracker.calculate_conversation_metrics(conversation_id)
        except ValueError:
            return None
    
    def get_sentiment_dashboard_data(self) -> Dict[str, Any]:
        """Get data for sentiment analysis dashboard"""
        # Get all conversation metrics
        all_metrics = list(self.conversation_tracker.conversation_metrics.values())
        
        if not all_metrics:
            return {
                'total_conversations': 0,
                'total_messages': 0,
                'average_satisfaction': 0.0,
                'sentiment_distribution': {},
                'quality_distribution': {},
                'active_alerts': len(self.alert_manager.active_alerts),
                'recent_alerts': []
            }
        
        # Calculate aggregate metrics
        total_conversations = len(all_metrics)
        total_messages = sum(m.total_messages for m in all_metrics)
        average_satisfaction = np.mean([m.satisfaction_score for m in all_metrics])
        
        # Sentiment distribution
        all_sentiments = []
        for metrics in all_metrics:
            all_sentiments.extend(metrics.sentiment_trend)
        
        sentiment_counts = pd.Series(all_sentiments).value_counts()
        sentiment_distribution = (sentiment_counts / sentiment_counts.sum()).to_dict()
        
        # Quality distribution
        quality_counts = pd.Series([m.conversation_quality for m in all_metrics]).value_counts()
        quality_distribution = (quality_counts / quality_counts.sum()).to_dict()
        
        # Recent alerts
        recent_alerts = [
            {
                'id': alert.alert_id,
                'type': alert.alert_type,
                'severity': alert.severity,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'conversation_id': alert.conversation_id
            }
            for alert in self.alert_manager.alert_history[-10:]  # Last 10 alerts
        ]
        
        return {
            'total_conversations': total_conversations,
            'total_messages': total_messages,
            'average_satisfaction': average_satisfaction,
            'sentiment_distribution': sentiment_distribution,
            'quality_distribution': quality_distribution,
            'active_alerts': len(self.alert_manager.active_alerts),
            'recent_alerts': recent_alerts,
            'processing_stats': {
                'total_messages_processed': self.total_messages_processed,
                'total_conversations': self.total_conversations,
                'alert_count': self.alert_count
            }
        }
    
    def export_conversation_data(self, conversation_id: str, format: str = 'json') -> str:
        """Export conversation data to file"""
        if conversation_id not in self.conversation_tracker.conversations:
            raise ValueError(f"Conversation {conversation_id} not found")
        
        messages = list(self.conversation_tracker.conversations[conversation_id])
        metrics = self.conversation_tracker.conversation_metrics.get(conversation_id)
        
        data = {
            'conversation_id': conversation_id,
            'messages': [asdict(msg) for msg in messages],
            'metrics': asdict(metrics) if metrics else None,
            'export_timestamp': datetime.now().isoformat()
        }
        
        # Convert datetime objects to strings
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj
        
        # Recursively convert datetime objects
        def recursive_convert(data):
            if isinstance(data, dict):
                return {k: recursive_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [recursive_convert(item) for item in data]
            else:
                return convert_datetime(data)
        
        data = recursive_convert(data)
        
        # Save to file
        filename = f"conversation_{conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return filename
    
    def train_custom_model(self, training_data: pd.DataFrame, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Train a custom sentiment model for chatbot data"""
        logger.info(f"Training custom model: {model_name}")
        results = self.trainer.train_all_models(training_data, optimize_hyperparameters=True)
        return results.get(model_name, {})
    
    def predict_with_custom_model(self, text: str, model_name: str = 'random_forest') -> Dict[str, Any]:
        """Predict sentiment using custom trained model"""
        return self.trainer.predict_sentiment(text, model_name)


# FastAPI Integration (if available)
if FASTAPI_AVAILABLE:
    class MessageRequest(BaseModel):
        user_message: str
        bot_response: str
        conversation_id: str
        message_id: Optional[str] = None
        timestamp: Optional[datetime] = None
    
    class ConversationRequest(BaseModel):
        messages: List[Dict[str, Any]]
    
    class SentilensAIAPI:
        """FastAPI integration for SentilensAI"""
        
        def __init__(self, integration: SentilensAIChatbotIntegration):
            self.integration = integration
            self.app = FastAPI(title="SentilensAI API", version="1.0.0")
            self._setup_routes()
        
        def _setup_routes(self):
            """Setup API routes"""
            
            # Add CORS middleware
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            @self.app.post("/analyze-message")
            async def analyze_message(request: MessageRequest):
                """Analyze a single message exchange"""
                try:
                    result = self.integration.process_message(
                        user_message=request.user_message,
                        bot_response=request.bot_response,
                        conversation_id=request.conversation_id,
                        message_id=request.message_id,
                        timestamp=request.timestamp
                    )
                    return {
                        'success': True,
                        'data': asdict(result),
                        'alerts': len(self.integration.alert_manager.active_alerts)
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.app.post("/analyze-conversation")
            async def analyze_conversation(request: ConversationRequest):
                """Analyze a complete conversation"""
                try:
                    results = self.integration.process_conversation_batch(request.messages)
                    return {
                        'success': True,
                        'data': [asdict(result) for result in results],
                        'conversation_count': len(results)
                    }
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @self.app.get("/conversation/{conversation_id}")
            async def get_conversation_analysis(conversation_id: str):
                """Get analysis for a specific conversation"""
                metrics = self.integration.get_conversation_analysis(conversation_id)
                if metrics is None:
                    raise HTTPException(status_code=404, detail="Conversation not found")
                return {'success': True, 'data': asdict(metrics)}
            
            @self.app.get("/dashboard")
            async def get_dashboard_data():
                """Get dashboard data"""
                data = self.integration.get_sentiment_dashboard_data()
                return {'success': True, 'data': data}
            
            @self.app.get("/alerts")
            async def get_alerts():
                """Get active alerts"""
                alerts = [
                    asdict(alert) for alert in self.integration.alert_manager.active_alerts.values()
                ]
                return {'success': True, 'data': alerts}
            
            @self.app.post("/alerts/{alert_id}/resolve")
            async def resolve_alert(alert_id: str):
                """Resolve an alert"""
                self.integration.alert_manager.resolve_alert(alert_id)
                return {'success': True, 'message': 'Alert resolved'}
            
            @self.websocket("/ws")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time sentiment monitoring"""
                await websocket.accept()
                try:
                    while True:
                        # Send dashboard data every 30 seconds
                        data = self.integration.get_sentiment_dashboard_data()
                        await websocket.send_json(data)
                        await asyncio.sleep(30)
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                finally:
                    await websocket.close()


def main():
    """Demo function to showcase SentimentsAI chatbot integration"""
    print("🤖 SentilensAI - Chatbot Integration Demo")
    print("=" * 50)
    
    # Initialize integration
    integration = SentilensAIChatbotIntegration()
    
    # Sample conversation
    sample_conversation = [
        {
            'user': 'Hi, I need help with my account',
            'bot': 'Hello! I\'d be happy to help you with your account. What specific issue are you experiencing?',
            'conversation_id': 'demo_001',
            'timestamp': datetime.now()
        },
        {
            'user': 'I can\'t log in. This is so frustrating!',
            'bot': 'I understand your frustration. Let me help you resolve this login issue. Can you tell me what error message you\'re seeing?',
            'conversation_id': 'demo_001',
            'timestamp': datetime.now()
        },
        {
            'user': 'It says my password is wrong but I know it\'s correct',
            'bot': 'That can be very frustrating. Let\'s try resetting your password. I\'ll guide you through the process step by step.',
            'conversation_id': 'demo_001',
            'timestamp': datetime.now()
        },
        {
            'user': 'Thank you! That worked perfectly. You\'re amazing!',
            'bot': 'You\'re very welcome! I\'m so glad I could help you get back into your account. Is there anything else I can assist you with today?',
            'conversation_id': 'demo_001',
            'timestamp': datetime.now()
        }
    ]
    
    print("📊 Processing sample conversation...")
    
    # Process conversation
    results = integration.process_conversation_batch(sample_conversation)
    
    # Display results
    for i, result in enumerate(results, 1):
        print(f"\n--- Message {i} ---")
        print(f"User: {result.user_message}")
        print(f"Bot: {result.bot_response}")
        print(f"User Sentiment: {result.user_sentiment.sentiment} (confidence: {result.user_sentiment.confidence:.2f})")
        print(f"Bot Sentiment: {result.bot_sentiment.sentiment} (confidence: {result.bot_sentiment.confidence:.2f})")
        print(f"Conversation Sentiment: {result.conversation_sentiment}")
        print(f"Satisfaction Score: {result.satisfaction_score:.2f}")
    
    # Get conversation analysis
    print(f"\n📈 Conversation Analysis:")
    metrics = integration.get_conversation_analysis('demo_001')
    if metrics:
        print(f"Total Messages: {metrics.total_messages}")
        print(f"Average User Sentiment: {metrics.average_user_sentiment:.2f}")
        print(f"Average Bot Sentiment: {metrics.average_bot_sentiment:.2f}")
        print(f"Satisfaction Score: {metrics.satisfaction_score:.2f}")
        print(f"Conversation Quality: {metrics.conversation_quality}")
        print(f"Key Emotions: {metrics.key_emotions}")
    
    # Get dashboard data
    print(f"\n📊 Dashboard Data:")
    dashboard = integration.get_sentiment_dashboard_data()
    print(f"Total Conversations: {dashboard['total_conversations']}")
    print(f"Total Messages: {dashboard['total_messages']}")
    print(f"Average Satisfaction: {dashboard['average_satisfaction']:.2f}")
    print(f"Active Alerts: {dashboard['active_alerts']}")
    
    # Export conversation data
    export_file = integration.export_conversation_data('demo_001', 'json')
    print(f"\n💾 Conversation data exported to: {export_file}")
    
    print("\n✅ SentilensAI chatbot integration demo completed!")
    print("🚀 Ready for real-time chatbot sentiment monitoring!")


if __name__ == "__main__":
    main()
