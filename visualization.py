"""
SentilensAI - Visualization and Reporting Module

This module provides comprehensive visualization and reporting capabilities
for sentiment analysis results from AI chatbot conversations.

Features:
- Interactive sentiment trend charts
- Emotion distribution visualizations
- Conversation quality dashboards
- Model performance comparisons
- Real-time sentiment monitoring
- Exportable reports and insights

Author: Pravin Selvamuthu
Repository: https://github.com/kernelseed/sentilens-ai
"""

import json
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# Import our modules
from sentiment_analyzer import SentimentResult, ChatbotMessage
from chatbot_integration import ConversationMetrics, SentilensAIChatbotIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for matplotlib
if MATPLOTLIB_AVAILABLE:
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")


class SentilensAIVisualizer:
    """
    Comprehensive visualization and reporting for sentiment analysis
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the visualizer
        
        Args:
            output_dir: Directory to save visualization files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Check available libraries
        self.matplotlib_available = MATPLOTLIB_AVAILABLE
        self.plotly_available = PLOTLY_AVAILABLE
        self.wordcloud_available = WORDCLOUD_AVAILABLE
        
        if not self.matplotlib_available and not self.plotly_available:
            logger.warning("No visualization libraries available. Install matplotlib or plotly for visualizations.")
        
        # Color schemes
        self.sentiment_colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'neutral': '#4682B4'    # Steel Blue
        }
        
        self.emotion_colors = {
            'joy': '#FFD700',       # Gold
            'sadness': '#4169E1',   # Royal Blue
            'anger': '#FF4500',     # Orange Red
            'fear': '#8B008B',      # Dark Magenta
            'surprise': '#FF69B4',  # Hot Pink
            'disgust': '#32CD32'    # Lime Green
        }
    
    def plot_sentiment_trend(self, messages: List[Union[SentimentResult, ChatbotMessage]], 
                           title: str = "Sentiment Trend Over Time", 
                           save_path: Optional[str] = None) -> Optional[str]:
        """
        Plot sentiment trend over time
        
        Args:
            messages: List of sentiment results or chatbot messages
            title: Chart title
            save_path: Path to save the plot (auto-generated if None)
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not self.matplotlib_available:
            logger.warning("Matplotlib not available for sentiment trend plot")
            return None
        
        # Extract data
        timestamps = []
        sentiments = []
        polarities = []
        confidences = []
        
        for msg in messages:
            if isinstance(msg, SentimentResult):
                timestamps.append(msg.timestamp)
                sentiments.append(msg.sentiment)
                polarities.append(msg.polarity)
                confidences.append(msg.confidence)
            elif isinstance(msg, ChatbotMessage):
                timestamps.append(msg.timestamp)
                sentiments.append(msg.user_sentiment.sentiment)
                polarities.append(msg.user_sentiment.polarity)
                confidences.append(msg.user_sentiment.confidence)
        
        if not timestamps:
            logger.warning("No data available for sentiment trend plot")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot 1: Sentiment categories over time
        sentiment_numeric = [1 if s == 'positive' else (-1 if s == 'negative' else 0) for s in sentiments]
        colors = [self.sentiment_colors.get(s, '#808080') for s in sentiments]
        
        ax1.scatter(timestamps, sentiment_numeric, c=colors, alpha=0.7, s=50)
        ax1.set_ylabel('Sentiment')
        ax1.set_title(f'{title} - Sentiment Categories')
        ax1.set_ylim(-1.5, 1.5)
        ax1.set_yticks([-1, 0, 1])
        ax1.set_yticklabels(['Negative', 'Neutral', 'Positive'])
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(timestamps) > 1:
            # Convert timestamps to numeric for trend calculation
            time_numeric = mdates.date2num(timestamps)
            z = np.polyfit(time_numeric, sentiment_numeric, 1)
            p = np.poly1d(z)
            ax1.plot(timestamps, p(time_numeric), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax1.legend()
        
        # Plot 2: Polarity and confidence over time
        ax2.plot(timestamps, polarities, 'b-', alpha=0.7, linewidth=2, label='Polarity')
        ax2.fill_between(timestamps, polarities, alpha=0.3, color='blue')
        
        # Add confidence as background
        ax2_twin = ax2.twinx()
        ax2_twin.plot(timestamps, confidences, 'g--', alpha=0.5, linewidth=1, label='Confidence')
        ax2_twin.set_ylabel('Confidence', color='green')
        ax2_twin.set_ylim(0, 1)
        
        ax2.set_ylabel('Polarity', color='blue')
        ax2.set_xlabel('Time')
        ax2.set_title('Polarity and Confidence Over Time')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=max(1, len(timestamps)//10)))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"sentiment_trend_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Sentiment trend plot saved to {save_path}")
        return str(save_path)
    
    def plot_emotion_distribution(self, messages: List[Union[SentimentResult, ChatbotMessage]], 
                                title: str = "Emotion Distribution", 
                                save_path: Optional[str] = None) -> Optional[str]:
        """
        Plot emotion distribution from messages
        
        Args:
            messages: List of sentiment results or chatbot messages
            title: Chart title
            save_path: Path to save the plot (auto-generated if None)
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not self.matplotlib_available:
            logger.warning("Matplotlib not available for emotion distribution plot")
            return None
        
        # Extract emotion data
        emotion_scores = {emotion: [] for emotion in self.emotion_colors.keys()}
        
        for msg in messages:
            if isinstance(msg, SentimentResult):
                emotions = msg.emotions
            elif isinstance(msg, ChatbotMessage):
                emotions = msg.user_sentiment.emotions
            else:
                continue
            
            for emotion in emotion_scores.keys():
                emotion_scores[emotion].append(emotions.get(emotion, 0.0))
        
        # Calculate statistics
        emotion_stats = {}
        for emotion, scores in emotion_scores.items():
            if scores:
                emotion_stats[emotion] = {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'max': np.max(scores),
                    'count': len(scores)
                }
        
        if not emotion_stats:
            logger.warning("No emotion data available for distribution plot")
            return None
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Mean emotion scores
        emotions = list(emotion_stats.keys())
        means = [emotion_stats[e]['mean'] for e in emotions]
        colors = [self.emotion_colors[e] for e in emotions]
        
        bars = ax1.bar(emotions, means, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        ax1.set_title(f'{title} - Mean Scores')
        ax1.set_ylabel('Mean Emotion Score')
        ax1.set_ylim(0, max(means) * 1.1 if means else 1)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Emotion intensity distribution
        all_scores = []
        all_emotions = []
        for emotion, scores in emotion_scores.items():
            all_scores.extend(scores)
            all_emotions.extend([emotion] * len(scores))
        
        if all_scores:
            df_emotions = pd.DataFrame({'emotion': all_emotions, 'score': all_scores})
            
            # Create box plot
            sns.boxplot(data=df_emotions, x='emotion', y='score', ax=ax2, palette=self.emotion_colors)
            ax2.set_title(f'{title} - Score Distribution')
            ax2.set_ylabel('Emotion Score')
            ax2.set_xlabel('Emotion')
            ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"emotion_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Emotion distribution plot saved to {save_path}")
        return str(save_path)
    
    def plot_conversation_quality_dashboard(self, metrics: ConversationMetrics, 
                                          save_path: Optional[str] = None) -> Optional[str]:
        """
        Create a comprehensive conversation quality dashboard
        
        Args:
            metrics: Conversation metrics object
            save_path: Path to save the plot (auto-generated if None)
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not self.matplotlib_available:
            logger.warning("Matplotlib not available for conversation quality dashboard")
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Overall Quality Score (large central plot)
        ax1 = fig.add_subplot(gs[0, :])
        quality_scores = {
            'Satisfaction': metrics.satisfaction_score,
            'User Sentiment': (metrics.average_user_sentiment + 1) / 2,  # Convert -1,1 to 0,1
            'Bot Sentiment': (metrics.average_bot_sentiment + 1) / 2,
            'Response Quality': min(1.0, len(metrics.response_times) / metrics.total_messages) if metrics.response_times else 0.5
        }
        
        colors = ['#2E8B57', '#4682B4', '#FFD700', '#FF6347']
        bars = ax1.bar(quality_scores.keys(), quality_scores.values(), color=colors, alpha=0.8)
        ax1.set_title(f'Conversation Quality Dashboard - {metrics.conversation_id}', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Score (0-1)')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bar, value in zip(bars, quality_scores.values()):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Sentiment Trend
        ax2 = fig.add_subplot(gs[1, 0])
        sentiment_counts = pd.Series(metrics.sentiment_trend).value_counts()
        colors_sentiment = [self.sentiment_colors.get(s, '#808080') for s in sentiment_counts.index]
        ax2.pie(sentiment_counts.values, labels=sentiment_counts.index, colors=colors_sentiment, autopct='%1.1f%%')
        ax2.set_title('Sentiment Distribution')
        
        # 3. Key Emotions
        ax3 = fig.add_subplot(gs[1, 1])
        if metrics.key_emotions:
            emotions = list(metrics.key_emotions.keys())
            scores = list(metrics.key_emotions.values())
            colors_emotion = [self.emotion_colors.get(e, '#808080') for e in emotions]
            ax3.barh(emotions, scores, color=colors_emotion, alpha=0.7)
            ax3.set_title('Key Emotions')
            ax3.set_xlabel('Intensity')
        else:
            ax3.text(0.5, 0.5, 'No significant emotions detected', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Key Emotions')
        
        # 4. Response Times
        ax4 = fig.add_subplot(gs[1, 2])
        if metrics.response_times:
            ax4.hist(metrics.response_times, bins=min(10, len(metrics.response_times)), alpha=0.7, color='skyblue', edgecolor='black')
            ax4.set_title('Response Time Distribution')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Frequency')
        else:
            ax4.text(0.5, 0.5, 'No response time data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Response Time Distribution')
        
        # 5. Conversation Timeline
        ax5 = fig.add_subplot(gs[2, :])
        duration = (metrics.end_time - metrics.start_time).total_seconds() / 60  # minutes
        ax5.barh(['Conversation'], [duration], color='lightcoral', alpha=0.7)
        ax5.set_title(f'Conversation Duration: {duration:.1f} minutes')
        ax5.set_xlabel('Duration (minutes)')
        
        # Add text annotations
        fig.text(0.02, 0.02, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', fontsize=8, alpha=0.7)
        fig.text(0.98, 0.02, f'Quality: {metrics.conversation_quality.upper()}', fontsize=8, alpha=0.7, ha='right')
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"conversation_dashboard_{metrics.conversation_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Conversation quality dashboard saved to {save_path}")
        return str(save_path)
    
    def plot_model_comparison(self, model_results: Dict[str, Any], 
                            title: str = "Model Performance Comparison", 
                            save_path: Optional[str] = None) -> Optional[str]:
        """
        Plot comparison of different ML models
        
        Args:
            model_results: Dictionary of model results
            title: Chart title
            save_path: Path to save the plot (auto-generated if None)
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not self.matplotlib_available:
            logger.warning("Matplotlib not available for model comparison plot")
            return None
        
        # Extract metrics
        models = []
        accuracies = []
        f1_scores = []
        training_times = []
        
        for model_name, results in model_results.items():
            if 'error' not in results:
                models.append(model_name.replace('_', ' ').title())
                accuracies.append(results.get('accuracy', 0))
                f1_scores.append(results.get('f1_macro', 0))
                training_times.append(results.get('training_time', 0))
        
        if not models:
            logger.warning("No valid model results for comparison plot")
            return None
        
        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Accuracy comparison
        bars1 = ax1.bar(models, accuracies, color='skyblue', alpha=0.7, edgecolor='black')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: F1 Score comparison
        bars2 = ax2.bar(models, f1_scores, color='lightgreen', alpha=0.7, edgecolor='black')
        ax2.set_title('Model F1 Score Comparison')
        ax2.set_ylabel('F1 Score (Macro)')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Training time comparison
        bars3 = ax3.bar(models, training_times, color='lightcoral', alpha=0.7, edgecolor='black')
        ax3.set_title('Model Training Time Comparison')
        ax3.set_ylabel('Training Time (seconds)')
        ax3.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, time in zip(bars3, training_times):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + max(training_times) * 0.01,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        if save_path is None:
            save_path = self.output_dir / f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Model comparison plot saved to {save_path}")
        return str(save_path)
    
    def create_word_cloud(self, messages: List[Union[SentimentResult, ChatbotMessage]], 
                         sentiment_filter: Optional[str] = None,
                         title: str = "Word Cloud", 
                         save_path: Optional[str] = None) -> Optional[str]:
        """
        Create word cloud from messages
        
        Args:
            messages: List of sentiment results or chatbot messages
            sentiment_filter: Filter by sentiment ('positive', 'negative', 'neutral')
            title: Chart title
            save_path: Path to save the plot (auto-generated if None)
            
        Returns:
            Path to saved plot or None if visualization not available
        """
        if not self.wordcloud_available or not self.matplotlib_available:
            logger.warning("WordCloud or Matplotlib not available for word cloud")
            return None
        
        # Extract text data
        texts = []
        for msg in messages:
            if isinstance(msg, SentimentResult):
                text = msg.text
                sentiment = msg.sentiment
            elif isinstance(msg, ChatbotMessage):
                text = msg.user_message
                sentiment = msg.user_sentiment.sentiment
            else:
                continue
            
            # Apply sentiment filter
            if sentiment_filter is None or sentiment == sentiment_filter:
                texts.append(text)
        
        if not texts:
            logger.warning("No text data available for word cloud")
            return None
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Create word cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            max_words=100,
            colormap='viridis',
            relative_scaling=0.5,
            random_state=42
        ).generate(combined_text)
        
        # Create plot
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'{title} - {sentiment_filter.title() if sentiment_filter else "All Sentiments"}', 
                 fontsize=16, fontweight='bold', pad=20)
        
        # Save plot
        if save_path is None:
            sentiment_suffix = f"_{sentiment_filter}" if sentiment_filter else ""
            save_path = self.output_dir / f"wordcloud{sentiment_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Word cloud saved to {save_path}")
        return str(save_path)
    
    def create_interactive_dashboard(self, integration: SentilensAIChatbotIntegration, 
                                   save_path: Optional[str] = None) -> Optional[str]:
        """
        Create an interactive dashboard using Plotly
        
        Args:
            integration: SentilensAI chatbot integration instance
            save_path: Path to save the HTML file (auto-generated if None)
            
        Returns:
            Path to saved HTML file or None if Plotly not available
        """
        if not self.plotly_available:
            logger.warning("Plotly not available for interactive dashboard")
            return None
        
        # Get dashboard data
        dashboard_data = integration.get_sentiment_dashboard_data()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sentiment Distribution', 'Conversation Quality', 
                          'Satisfaction Trend', 'Recent Alerts'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "table"}]]
        )
        
        # 1. Sentiment Distribution (Pie Chart)
        if dashboard_data['sentiment_distribution']:
            sentiments = list(dashboard_data['sentiment_distribution'].keys())
            values = list(dashboard_data['sentiment_distribution'].values())
            colors = [self.sentiment_colors.get(s, '#808080') for s in sentiments]
            
            fig.add_trace(
                go.Pie(labels=sentiments, values=values, marker_colors=colors, name="Sentiment"),
                row=1, col=1
            )
        
        # 2. Conversation Quality (Bar Chart)
        if dashboard_data['quality_distribution']:
            qualities = list(dashboard_data['quality_distribution'].keys())
            counts = list(dashboard_data['quality_distribution'].values())
            
            fig.add_trace(
                go.Bar(x=qualities, y=counts, name="Quality", marker_color='lightblue'),
                row=1, col=2
            )
        
        # 3. Satisfaction Trend (Scatter Plot)
        # This would need historical data - for now, show current average
        fig.add_trace(
            go.Scatter(x=[datetime.now()], y=[dashboard_data['average_satisfaction']], 
                      mode='markers+text', text=[f"{dashboard_data['average_satisfaction']:.3f}"],
                      name="Satisfaction", marker=dict(size=20, color='green')),
            row=2, col=1
        )
        
        # 4. Recent Alerts (Table)
        if dashboard_data['recent_alerts']:
            alerts = dashboard_data['recent_alerts']
            fig.add_trace(
                go.Table(
                    header=dict(values=['Type', 'Severity', 'Time', 'Message']),
                    cells=dict(values=[
                        [alert['type'] for alert in alerts],
                        [alert['severity'] for alert in alerts],
                        [alert['timestamp'] for alert in alerts],
                        [alert['message'][:50] + '...' if len(alert['message']) > 50 else alert['message'] for alert in alerts]
                    ])
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="SentimentsAI Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        # Save as HTML
        if save_path is None:
            save_path = self.output_dir / f"interactive_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        fig.write_html(str(save_path))
        
        logger.info(f"Interactive dashboard saved to {save_path}")
        return str(save_path)
    
    def generate_comprehensive_report(self, integration: SentilensAIChatbotIntegration, 
                                    output_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Generate a comprehensive sentiment analysis report with all visualizations
        
        Args:
            integration: SentilensAI chatbot integration instance
            output_dir: Output directory (uses default if None)
            
        Returns:
            Dictionary of generated file paths
        """
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(exist_ok=True)
        
        report_files = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get all conversations
        all_messages = []
        for conv_id, messages in integration.conversation_tracker.conversations.items():
            all_messages.extend(list(messages))
        
        if not all_messages:
            logger.warning("No conversation data available for report generation")
            return report_files
        
        # 1. Sentiment Trend
        try:
            trend_path = self.plot_sentiment_trend(all_messages, save_path=f"sentiment_trend_{timestamp}.png")
            if trend_path:
                report_files['sentiment_trend'] = trend_path
        except Exception as e:
            logger.error(f"Failed to create sentiment trend plot: {e}")
        
        # 2. Emotion Distribution
        try:
            emotion_path = self.plot_emotion_distribution(all_messages, save_path=f"emotion_distribution_{timestamp}.png")
            if emotion_path:
                report_files['emotion_distribution'] = emotion_path
        except Exception as e:
            logger.error(f"Failed to create emotion distribution plot: {e}")
        
        # 3. Word Clouds for each sentiment
        for sentiment in ['positive', 'negative', 'neutral']:
            try:
                wc_path = self.create_word_cloud(all_messages, sentiment_filter=sentiment, 
                                               save_path=f"wordcloud_{sentiment}_{timestamp}.png")
                if wc_path:
                    report_files[f'wordcloud_{sentiment}'] = wc_path
            except Exception as e:
                logger.error(f"Failed to create word cloud for {sentiment}: {e}")
        
        # 4. Conversation Quality Dashboards
        for conv_id, metrics in integration.conversation_tracker.conversation_metrics.items():
            try:
                dashboard_path = self.plot_conversation_quality_dashboard(metrics, 
                                                                        save_path=f"dashboard_{conv_id}_{timestamp}.png")
                if dashboard_path:
                    report_files[f'dashboard_{conv_id}'] = dashboard_path
            except Exception as e:
                logger.error(f"Failed to create dashboard for {conv_id}: {e}")
        
        # 5. Interactive Dashboard
        try:
            interactive_path = self.create_interactive_dashboard(integration, 
                                                               save_path=f"interactive_dashboard_{timestamp}.html")
            if interactive_path:
                report_files['interactive_dashboard'] = interactive_path
        except Exception as e:
            logger.error(f"Failed to create interactive dashboard: {e}")
        
        # 6. Generate summary report
        try:
            summary_path = self._generate_summary_report(integration, timestamp)
            if summary_path:
                report_files['summary_report'] = summary_path
        except Exception as e:
            logger.error(f"Failed to create summary report: {e}")
        
        logger.info(f"Comprehensive report generated with {len(report_files)} files")
        return report_files
    
    def _generate_summary_report(self, integration: SentilensAIChatbotIntegration, timestamp: str) -> str:
        """Generate a text summary report"""
        dashboard_data = integration.get_sentiment_dashboard_data()
        
        report_content = f"""
# SentilensAI Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- Total Conversations: {dashboard_data['total_conversations']}
- Total Messages: {dashboard_data['total_messages']}
- Average Satisfaction: {dashboard_data['average_satisfaction']:.3f}
- Active Alerts: {dashboard_data['active_alerts']}

## Sentiment Distribution
"""
        
        for sentiment, percentage in dashboard_data['sentiment_distribution'].items():
            report_content += f"- {sentiment.title()}: {percentage:.1%}\n"
        
        report_content += f"""
## Conversation Quality Distribution
"""
        
        for quality, percentage in dashboard_data['quality_distribution'].items():
            report_content += f"- {quality.title()}: {percentage:.1%}\n"
        
        report_content += f"""
## Recent Alerts
"""
        
        for alert in dashboard_data['recent_alerts'][:5]:  # Show last 5 alerts
            report_content += f"- [{alert['severity'].upper()}] {alert['type']}: {alert['message']}\n"
        
        report_content += f"""
## Processing Statistics
- Total Messages Processed: {dashboard_data['processing_stats']['total_messages_processed']}
- Total Conversations: {dashboard_data['processing_stats']['total_conversations']}
- Total Alerts Generated: {dashboard_data['processing_stats']['alert_count']}

## Recommendations
"""
        
        # Generate recommendations based on data
        if dashboard_data['average_satisfaction'] < 0.5:
            report_content += "- ⚠️ Low average satisfaction detected. Consider improving bot responses.\n"
        
        if dashboard_data['active_alerts'] > 0:
            report_content += "- 🚨 Active alerts require attention. Review negative sentiment patterns.\n"
        
        negative_percentage = dashboard_data['sentiment_distribution'].get('negative', 0)
        if negative_percentage > 0.3:
            report_content += "- 📉 High negative sentiment percentage. Investigate common issues.\n"
        
        report_content += "\n---\n"
        report_content += "Report generated by SentilensAI - Advanced Sentiment Analysis for AI Chatbots"
        
        # Save report
        report_path = self.output_dir / f"summary_report_{timestamp}.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Summary report saved to {report_path}")
        return str(report_path)


def main():
    """Demo function to showcase SentilensAI visualization capabilities"""
    print("🤖 SentilensAI - Visualization and Reporting Demo")
    print("=" * 60)
    
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
    
    for i in range(20):
        sentiment = np.random.choice(sentiments)
        polarity = np.random.uniform(-1, 1) if sentiment == 'neutral' else (0.5 if sentiment == 'positive' else -0.5)
        
        message = SentimentResult(
            text=f"Sample message {i+1} with {sentiment} sentiment",
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
            timestamp=datetime.now() - timedelta(minutes=i*5),
            model_used='ensemble',
            metadata={}
        )
        sample_messages.append(message)
    
    print(f"\n📈 Creating visualizations with {len(sample_messages)} sample messages...")
    
    # Create visualizations
    try:
        # Sentiment trend
        trend_path = visualizer.plot_sentiment_trend(sample_messages)
        if trend_path:
            print(f"✅ Sentiment trend plot: {trend_path}")
    except Exception as e:
        print(f"❌ Failed to create sentiment trend: {e}")
    
    try:
        # Emotion distribution
        emotion_path = visualizer.plot_emotion_distribution(sample_messages)
        if emotion_path:
            print(f"✅ Emotion distribution plot: {emotion_path}")
    except Exception as e:
        print(f"❌ Failed to create emotion distribution: {e}")
    
    try:
        # Word cloud
        wc_path = visualizer.create_word_cloud(sample_messages)
        if wc_path:
            print(f"✅ Word cloud: {wc_path}")
    except Exception as e:
        print(f"❌ Failed to create word cloud: {e}")
    
    # Sample model results
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
            print(f"✅ Model comparison plot: {model_path}")
    except Exception as e:
        print(f"❌ Failed to create model comparison: {e}")
    
    print(f"\n📁 All visualizations saved to: {visualizer.output_dir}")
    print("\n✅ SentilensAI visualization demo completed!")
    print("🚀 Ready for comprehensive sentiment analysis reporting!")


if __name__ == "__main__":
    main()
