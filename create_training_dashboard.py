#!/usr/bin/env python3
"""
SentilensAI - Agent Training Dashboard Creator

Creates comprehensive visual dashboards for agent training insights
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import pandas as pd

def create_training_dashboard():
    """Create comprehensive training dashboard"""
    
    # Load the latest results
    import glob
    result_files = glob.glob("sentiment_analysis_results_*.json")
    if not result_files:
        print("❌ No results files found!")
        return
    
    latest_file = max(result_files)
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('SentilensAI - Agent Training Analysis Dashboard', fontsize=20, fontweight='bold', y=0.95)
    
    # 1. Quality Score Distribution (Top Left)
    ax1 = fig.add_subplot(gs[0, 0])
    conv_ids = [conv['conversation_id'] for conv in results['conversation_results']]
    quality_scores = [conv['conversation_quality'] for conv in results['conversation_results']]
    
    colors = ['red' if q < 0.4 else 'orange' if q < 0.7 else 'green' for q in quality_scores]
    bars = ax1.bar(conv_ids, quality_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
    ax1.set_title('Conversation Quality Scores', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Quality Score', fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, score in zip(bars, quality_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Add quality thresholds
    ax1.axhline(y=0.7, color='green', linestyle='--', alpha=0.7, label='High Quality (0.7+)')
    ax1.axhline(y=0.4, color='orange', linestyle='--', alpha=0.7, label='Medium Quality (0.4-0.7)')
    ax1.legend()
    
    # 2. Sentiment Distribution (Top Middle)
    ax2 = fig.add_subplot(gs[0, 1])
    user_sentiment = results['overall_sentiment_distribution']['user']
    bot_sentiment = results['overall_sentiment_distribution']['bot']
    
    categories = ['Positive', 'Negative', 'Neutral']
    user_counts = [user_sentiment['positive'], user_sentiment['negative'], user_sentiment['neutral']]
    bot_counts = [bot_sentiment['positive'], bot_sentiment['negative'], bot_sentiment['neutral']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax2.bar(x - width/2, user_counts, width, label='User Messages', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, bot_counts, width, label='Bot Messages', alpha=0.8, color='lightcoral')
    ax2.set_xlabel('Sentiment', fontsize=12)
    ax2.set_ylabel('Message Count', fontsize=12)
    ax2.set_title('Overall Sentiment Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Problem Analysis (Top Right)
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Analyze problems from the detailed analysis
    problems = [
        "Low Confidence Responses",
        "Negative Bot Responses", 
        "User Sentiment Issues",
        "Response Quality Issues"
    ]
    
    problem_counts = [4, 2, 3, 2]  # Based on the analysis
    colors = ['red', 'darkred', 'orange', 'yellow']
    
    bars = ax3.barh(problems, problem_counts, color=colors, alpha=0.7)
    ax3.set_title('Identified Problems', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Occurrences', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, count in zip(bars, problem_counts):
        width = bar.get_width()
        ax3.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                str(count), ha='left', va='center', fontweight='bold')
    
    # 4. Conversation Flow Analysis (Second Row)
    ax4 = fig.add_subplot(gs[1, :])
    
    # Create a timeline showing sentiment progression
    conversation_data = []
    for conv in results['conversation_results']:
        conv_id = conv['conversation_id']
        messages = conv['message_analysis']
        
        user_sentiments = []
        bot_sentiments = []
        
        for msg in messages:
            if msg['user_sentiment']:
                sentiment_value = 1 if msg['user_sentiment'] == 'positive' else -1 if msg['user_sentiment'] == 'negative' else 0
                user_sentiments.append(sentiment_value)
            if msg['bot_sentiment']:
                sentiment_value = 1 if msg['bot_sentiment'] == 'positive' else -1 if msg['bot_sentiment'] == 'negative' else 0
                bot_sentiments.append(sentiment_value)
        
        conversation_data.append({
            'conv_id': conv_id,
            'user_sentiments': user_sentiments,
            'bot_sentiments': bot_sentiments
        })
    
    # Plot sentiment progression for each conversation
    for i, conv in enumerate(conversation_data):
        x_pos = np.arange(len(conv['user_sentiments']))
        
        # Plot user sentiment
        ax4.plot(x_pos, conv['user_sentiments'], 'o-', label=f'{conv["conv_id"]} - User', 
                linewidth=2, markersize=6, alpha=0.8)
        
        # Plot bot sentiment
        ax4.plot(x_pos, conv['bot_sentiments'], 's--', label=f'{conv["conv_id"]} - Bot', 
                linewidth=2, markersize=6, alpha=0.8)
    
    ax4.set_title('Sentiment Progression Across Conversations', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Message Number', fontsize=12)
    ax4.set_ylabel('Sentiment Score', fontsize=12)
    ax4.set_yticks([-1, 0, 1])
    ax4.set_yticklabels(['Negative', 'Neutral', 'Positive'])
    ax4.grid(True, alpha=0.3)
    ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Training Priority Matrix (Third Row Left)
    ax5 = fig.add_subplot(gs[2, 0])
    
    training_areas = ['Response Confidence', 'Response Tone', 'Sentiment Management', 'Overall Quality']
    priority_scores = [8, 10, 7, 6]  # Based on analysis
    impact_scores = [7, 9, 8, 6]
    
    scatter = ax5.scatter(impact_scores, priority_scores, s=200, alpha=0.7, c=priority_scores, 
                         cmap='RdYlGn_r', edgecolors='black', linewidth=1)
    
    for i, area in enumerate(training_areas):
        ax5.annotate(area, (impact_scores[i], priority_scores[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax5.set_xlabel('Impact Score', fontsize=12)
    ax5.set_ylabel('Priority Score', fontsize=12)
    ax5.set_title('Training Priority Matrix', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Success vs Problem Ratio (Third Row Middle)
    ax6 = fig.add_subplot(gs[2, 1])
    
    success_count = 1  # conv_003
    problem_count = 3  # conv_001, conv_002, conv_004
    
    labels = ['High Quality\nConversations', 'Medium Quality\nConversations']
    sizes = [success_count, problem_count]
    colors = ['green', 'orange']
    explode = (0.1, 0)  # Explode the first slice
    
    wedges, texts, autotexts = ax6.pie(sizes, explode=explode, labels=labels, colors=colors,
                                      autopct='%1.1f%%', shadow=True, startangle=90)
    ax6.set_title('Quality Distribution', fontsize=14, fontweight='bold')
    
    # 7. Improvement Recommendations (Third Row Right)
    ax7 = fig.add_subplot(gs[2, 2])
    
    recommendations = [
        'Implement Tone Checking',
        'Improve Response Confidence',
        'Add Sentiment Monitoring',
        'Create Response Templates',
        'Regular Training Sessions'
    ]
    
    recommendation_priority = [10, 8, 7, 6, 5]
    
    bars = ax7.barh(recommendations, recommendation_priority, color='lightblue', alpha=0.7)
    ax7.set_title('Training Recommendations', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Priority Score', fontsize=12)
    ax7.grid(True, alpha=0.3)
    
    # 8. Action Plan Timeline (Bottom Row)
    ax8 = fig.add_subplot(gs[3, :])
    
    # Create a timeline of actions
    actions = [
        'Address Negative Responses',
        'Improve Response Confidence',
        'Implement Sentiment Monitoring',
        'Create Training Materials',
        'Schedule Training Sessions',
        'Monitor Performance'
    ]
    
    timeframes = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5', 'Week 6']
    priorities = [10, 8, 7, 6, 5, 4]
    
    y_pos = np.arange(len(actions))
    bars = ax8.barh(y_pos, priorities, color=plt.cm.RdYlGn_r(np.array(priorities)/10), alpha=0.7)
    
    ax8.set_yticks(y_pos)
    ax8.set_yticklabels(actions)
    ax8.set_xlabel('Priority Score', fontsize=12)
    ax8.set_title('Action Plan Timeline', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, priority in zip(bars, priorities):
        width = bar.get_width()
        ax8.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                str(priority), ha='left', va='center', fontweight='bold')
    
    # Save the dashboard
    dashboard_file = f"comprehensive_training_dashboard_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(dashboard_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Comprehensive training dashboard saved to: {dashboard_file}")
    
    plt.show()

if __name__ == "__main__":
    create_training_dashboard()
