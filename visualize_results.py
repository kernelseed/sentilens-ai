#!/usr/bin/env python3
"""
SentilensAI - Results Visualization

Simple visualization of chat sentiment analysis results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np

def load_results(json_file):
    """Load analysis results from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_sentiment_charts(results):
    """Create visualization charts for sentiment analysis results"""
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SentilensAI - Chat Sentiment Analysis Results', fontsize=16, fontweight='bold')
    
    # 1. Overall Sentiment Distribution
    user_sentiment = results['overall_sentiment_distribution']['user']
    bot_sentiment = results['overall_sentiment_distribution']['bot']
    
    categories = ['Positive', 'Negative', 'Neutral']
    user_counts = [user_sentiment['positive'], user_sentiment['negative'], user_sentiment['neutral']]
    bot_counts = [bot_sentiment['positive'], bot_sentiment['negative'], bot_sentiment['neutral']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, user_counts, width, label='User Messages', alpha=0.8)
    ax1.bar(x + width/2, bot_counts, width, label='Bot Messages', alpha=0.8)
    ax1.set_xlabel('Sentiment')
    ax1.set_ylabel('Message Count')
    ax1.set_title('Overall Sentiment Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Conversation Quality Scores
    conv_qualities = [conv['conversation_quality'] for conv in results['conversation_results']]
    conv_ids = [conv['conversation_id'] for conv in results['conversation_results']]
    
    bars = ax2.bar(conv_ids, conv_qualities, color=['green' if q > 0.7 else 'orange' if q > 0.4 else 'red' for q in conv_qualities])
    ax2.set_xlabel('Conversation ID')
    ax2.set_ylabel('Quality Score')
    ax2.set_title('Conversation Quality Scores')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, quality in zip(bars, conv_qualities):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{quality:.2f}', ha='center', va='bottom')
    
    # 3. Sentiment Trend Over Messages
    all_user_sentiments = []
    all_bot_sentiments = []
    
    for conv in results['conversation_results']:
        for msg in conv['sentiment_trend']:
            if msg['user_sentiment']:
                all_user_sentiments.append(msg['user_sentiment'])
            if msg['bot_sentiment']:
                all_bot_sentiments.append(msg['bot_sentiment'])
    
    # Count sentiments
    user_sentiment_counts = Counter(all_user_sentiments)
    bot_sentiment_counts = Counter(all_bot_sentiments)
    
    # Pie chart for user sentiments
    user_labels = list(user_sentiment_counts.keys())
    user_sizes = list(user_sentiment_counts.values())
    colors = ['lightgreen', 'lightcoral', 'lightblue']
    
    ax3.pie(user_sizes, labels=user_labels, autopct='%1.1f%%', colors=colors[:len(user_labels)])
    ax3.set_title('User Message Sentiment Distribution')
    
    # Pie chart for bot sentiments
    bot_labels = list(bot_sentiment_counts.keys())
    bot_sizes = list(bot_sentiment_counts.values())
    
    ax4.pie(bot_sizes, labels=bot_labels, autopct='%1.1f%%', colors=colors[:len(bot_labels)])
    ax4.set_title('Bot Message Sentiment Distribution')
    
    plt.tight_layout()
    return fig

def main():
    """Main function to create visualizations"""
    print("📊 SentilensAI - Creating Sentiment Analysis Visualizations")
    print("=" * 60)
    
    # Find the most recent results file
    import glob
    result_files = glob.glob("sentiment_analysis_results_*.json")
    if not result_files:
        print("❌ No results files found!")
        return
    
    latest_file = max(result_files)
    print(f"📁 Loading results from: {latest_file}")
    
    # Load results
    results = load_results(latest_file)
    
    # Create visualizations
    print("🎨 Creating sentiment analysis charts...")
    fig = create_sentiment_charts(results)
    
    # Save the plot
    output_file = f"sentiment_analysis_charts_{results['analysis_timestamp'][:10]}.png"
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"💾 Charts saved to: {output_file}")
    
    # Display summary
    print(f"\n📈 Analysis Summary:")
    print(f"   Total Conversations: {results['total_conversations']}")
    print(f"   Average Quality Score: {results['average_quality_score']:.2f}")
    print(f"   User Messages: {sum(results['overall_sentiment_distribution']['user'].values())}")
    print(f"   Bot Messages: {sum(results['overall_sentiment_distribution']['bot'].values())}")
    
    print("\n✅ Visualization completed successfully!")
    print("🚀 Check the generated chart file for detailed insights!")

if __name__ == "__main__":
    main()
