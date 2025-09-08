#!/usr/bin/env python3
"""
SentilensAI - Detailed Chat Analysis for Agent Training Improvement

This script provides comprehensive analysis of chat interactions to identify
specific problems and areas for agent training improvement.

Author: Pravin Selvamuthu
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

def load_analysis_results(json_file):
    """Load sentiment analysis results"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_conversation_patterns(results):
    """Analyze conversation patterns to identify problems"""
    
    print("🔍 DETAILED CONVERSATION ANALYSIS")
    print("=" * 60)
    
    conversation_insights = []
    
    for conv in results['conversation_results']:
        conv_id = conv['conversation_id']
        messages = conv['message_analysis']
        quality_score = conv['conversation_quality']
        
        print(f"\n📋 CONVERSATION: {conv_id}")
        print(f"Quality Score: {quality_score:.2f}/1.0")
        print("-" * 40)
        
        # Analyze sentiment progression
        user_sentiments = [msg['user_sentiment'] for msg in messages if msg['user_sentiment']]
        bot_sentiments = [msg['bot_sentiment'] for msg in messages if msg['bot_sentiment']]
        
        # Identify problems
        problems = []
        improvements = []
        
        # Problem 1: Sentiment Deterioration
        if len(user_sentiments) >= 2:
            if user_sentiments[0] in ['positive', 'neutral'] and user_sentiments[-1] == 'negative':
                problems.append("User sentiment deteriorated during conversation")
        
        # Problem 2: Low Confidence Responses
        low_confidence_bot = [msg for msg in messages if msg['bot_confidence'] and msg['bot_confidence'] < 0.3]
        if low_confidence_bot:
            problems.append(f"Bot responses with low confidence: {len(low_confidence_bot)} messages")
        
        # Problem 3: Negative Bot Responses
        negative_bot_responses = [msg for msg in messages if msg['bot_sentiment'] == 'negative']
        if negative_bot_responses:
            problems.append(f"Bot generated negative responses: {len(negative_bot_responses)} messages")
        
        # Problem 4: Emotion Mismatch
        emotion_mismatches = []
        for msg in messages:
            if msg['user_emotions'] and msg['bot_emotions']:
                user_primary_emotion = max(msg['user_emotions'].items(), key=lambda x: x[1])[0]
                bot_primary_emotion = max(msg['bot_emotions'].items(), key=lambda x: x[1])[0]
                if user_primary_emotion == 'anger' and bot_primary_emotion != 'joy':
                    emotion_mismatches.append(f"User angry but bot not empathetic (Message {msg['message_index']})")
        
        if emotion_mismatches:
            problems.extend(emotion_mismatches)
        
        # Identify improvements
        if quality_score > 0.7:
            improvements.append("High quality conversation - good agent performance")
        elif quality_score > 0.4:
            improvements.append("Moderate quality - some areas for improvement")
        else:
            improvements.append("Low quality - significant training needed")
        
        # Analyze specific message patterns
        print(f"\n📊 SENTIMENT PROGRESSION:")
        print(f"   User: {' → '.join(user_sentiments)}")
        print(f"   Bot:  {' → '.join(bot_sentiments)}")
        
        print(f"\n⚠️  IDENTIFIED PROBLEMS:")
        if problems:
            for i, problem in enumerate(problems, 1):
                print(f"   {i}. {problem}")
        else:
            print("   No significant problems detected")
        
        print(f"\n✅ IMPROVEMENTS IDENTIFIED:")
        for i, improvement in enumerate(improvements, 1):
            print(f"   {i}. {improvement}")
        
        # Store insights
        conversation_insights.append({
            'conversation_id': conv_id,
            'quality_score': quality_score,
            'problems': problems,
            'improvements': improvements,
            'user_sentiment_trend': user_sentiments,
            'bot_sentiment_trend': bot_sentiments,
            'total_messages': len(messages)
        })
    
    return conversation_insights

def analyze_agent_performance_patterns(conversation_insights):
    """Analyze patterns in agent performance"""
    
    print(f"\n🎯 AGENT PERFORMANCE ANALYSIS")
    print("=" * 60)
    
    # Categorize conversations by quality
    high_quality = [conv for conv in conversation_insights if conv['quality_score'] > 0.7]
    medium_quality = [conv for conv in conversation_insights if 0.4 <= conv['quality_score'] <= 0.7]
    low_quality = [conv for conv in conversation_insights if conv['quality_score'] < 0.4]
    
    print(f"📈 CONVERSATION QUALITY DISTRIBUTION:")
    print(f"   High Quality (>0.7): {len(high_quality)} conversations")
    print(f"   Medium Quality (0.4-0.7): {len(medium_quality)} conversations")
    print(f"   Low Quality (<0.4): {len(low_quality)} conversations")
    
    # Analyze common problems
    all_problems = []
    for conv in conversation_insights:
        all_problems.extend(conv['problems'])
    
    problem_counts = Counter(all_problems)
    
    print(f"\n🚨 MOST COMMON PROBLEMS:")
    for problem, count in problem_counts.most_common():
        print(f"   • {problem} ({count} occurrences)")
    
    # Analyze sentiment trends
    print(f"\n📊 SENTIMENT TREND ANALYSIS:")
    
    # User sentiment patterns
    user_sentiment_changes = []
    for conv in conversation_insights:
        if len(conv['user_sentiment_trend']) >= 2:
            first_sentiment = conv['user_sentiment_trend'][0]
            last_sentiment = conv['user_sentiment_trend'][-1]
            user_sentiment_changes.append((first_sentiment, last_sentiment))
    
    sentiment_transitions = Counter(user_sentiment_changes)
    print(f"   User Sentiment Transitions:")
    for transition, count in sentiment_transitions.most_common():
        print(f"     {transition[0]} → {transition[1]}: {count} times")
    
    return {
        'high_quality': high_quality,
        'medium_quality': medium_quality,
        'low_quality': low_quality,
        'common_problems': problem_counts,
        'sentiment_transitions': sentiment_transitions
    }

def generate_training_recommendations(performance_analysis):
    """Generate specific training recommendations"""
    
    print(f"\n🎓 AGENT TRAINING RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    # Analyze common problems and create recommendations
    common_problems = performance_analysis['common_problems']
    
    if "User sentiment deteriorated during conversation" in common_problems:
        recommendations.append({
            'priority': 'HIGH',
            'area': 'Sentiment Management',
            'problem': 'User sentiment deteriorates during conversation',
            'recommendation': 'Train agents to recognize early warning signs of frustration and proactively address concerns',
            'specific_actions': [
                'Implement sentiment monitoring in real-time',
                'Train agents to use empathetic language when users show frustration',
                'Create escalation protocols for negative sentiment detection',
                'Practice de-escalation techniques'
            ]
        })
    
    if "Bot responses with low confidence" in common_problems:
        recommendations.append({
            'priority': 'MEDIUM',
            'area': 'Response Quality',
            'problem': 'Bot responses have low confidence scores',
            'recommendation': 'Improve response quality and confidence through better training data and validation',
            'specific_actions': [
                'Review and improve training datasets',
                'Implement response validation before sending',
                'Train agents to be more decisive in their responses',
                'Add confidence scoring to response evaluation'
            ]
        })
    
    if "Bot generated negative responses" in common_problems:
        recommendations.append({
            'priority': 'HIGH',
            'area': 'Response Tone',
            'problem': 'Bot generates negative responses',
            'recommendation': 'Ensure all bot responses maintain positive or neutral tone',
            'specific_actions': [
                'Implement tone checking before response delivery',
                'Train agents to always maintain professional, helpful tone',
                'Create positive response templates for common scenarios',
                'Add sentiment analysis to response validation pipeline'
            ]
        })
    
    if "User angry but bot not empathetic" in common_problems:
        recommendations.append({
            'priority': 'CRITICAL',
            'area': 'Emotional Intelligence',
            'problem': 'Bot lacks empathy when users are angry',
            'recommendation': 'Enhance emotional intelligence and empathy in agent responses',
            'specific_actions': [
                'Train agents to recognize and acknowledge user emotions',
                'Implement empathy training modules',
                'Create emotional response templates',
                'Practice active listening and validation techniques'
            ]
        })
    
    # Quality-based recommendations
    low_quality_count = len(performance_analysis['low_quality'])
    if low_quality_count > 0:
        recommendations.append({
            'priority': 'CRITICAL',
            'area': 'Overall Performance',
            'problem': f'{low_quality_count} conversations with low quality scores',
            'recommendation': 'Comprehensive agent retraining needed',
            'specific_actions': [
                'Conduct comprehensive performance review',
                'Implement intensive retraining program',
                'Add quality monitoring and feedback loops',
                'Consider additional human oversight for complex cases'
            ]
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['area'].upper()} - {rec['priority']} PRIORITY")
        print(f"   Problem: {rec['problem']}")
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"   Specific Actions:")
        for action in rec['specific_actions']:
            print(f"     • {action}")
    
    return recommendations

def create_agent_training_dashboard(conversation_insights, performance_analysis):
    """Create a visual dashboard for agent training insights"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SentilensAI - Agent Training Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Quality Score Distribution
    quality_scores = [conv['quality_score'] for conv in conversation_insights]
    conv_ids = [conv['conversation_id'] for conv in conversation_insights]
    
    colors = ['red' if q < 0.4 else 'orange' if q < 0.7 else 'green' for q in quality_scores]
    bars = ax1.bar(conv_ids, quality_scores, color=colors, alpha=0.7)
    ax1.set_title('Conversation Quality Scores')
    ax1.set_ylabel('Quality Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, quality_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.2f}', ha='center', va='bottom')
    
    # 2. Problem Frequency
    all_problems = []
    for conv in conversation_insights:
        all_problems.extend(conv['problems'])
    
    problem_counts = Counter(all_problems)
    if problem_counts:
        problems, counts = zip(*problem_counts.most_common())
        ax2.barh(range(len(problems)), counts, color='coral', alpha=0.7)
        ax2.set_yticks(range(len(problems)))
        ax2.set_yticklabels([p[:30] + '...' if len(p) > 30 else p for p in problems])
        ax2.set_title('Most Common Problems')
        ax2.set_xlabel('Frequency')
    
    # 3. Sentiment Transition Matrix
    sentiment_transitions = performance_analysis['sentiment_transitions']
    if sentiment_transitions:
        transitions = list(sentiment_transitions.keys())
        counts = list(sentiment_transitions.values())
        
        # Create transition labels
        transition_labels = [f"{t[0]}→{t[1]}" for t in transitions]
        
        ax3.pie(counts, labels=transition_labels, autopct='%1.1f%%', startangle=90)
        ax3.set_title('User Sentiment Transitions')
    
    # 4. Quality Categories
    high_quality = len(performance_analysis['high_quality'])
    medium_quality = len(performance_analysis['medium_quality'])
    low_quality = len(performance_analysis['low_quality'])
    
    categories = ['High Quality\n(>0.7)', 'Medium Quality\n(0.4-0.7)', 'Low Quality\n(<0.4)']
    counts = [high_quality, medium_quality, low_quality]
    colors = ['green', 'orange', 'red']
    
    ax4.bar(categories, counts, color=colors, alpha=0.7)
    ax4.set_title('Conversation Quality Distribution')
    ax4.set_ylabel('Number of Conversations')
    
    # Add value labels
    for bar, count in zip(ax4.patches, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def main():
    """Main analysis function"""
    print("🎯 SentilensAI - Detailed Agent Training Analysis")
    print("=" * 70)
    
    # Load results
    import glob
    result_files = glob.glob("sentiment_analysis_results_*.json")
    if not result_files:
        print("❌ No results files found!")
        return
    
    latest_file = max(result_files)
    print(f"📁 Loading analysis results from: {latest_file}")
    
    results = load_analysis_results(latest_file)
    
    # Perform detailed analysis
    conversation_insights = analyze_conversation_patterns(results)
    performance_analysis = analyze_agent_performance_patterns(conversation_insights)
    recommendations = generate_training_recommendations(performance_analysis)
    
    # Create training dashboard
    print(f"\n📊 Creating agent training dashboard...")
    fig = create_agent_training_dashboard(conversation_insights, performance_analysis)
    
    # Save dashboard
    dashboard_file = f"agent_training_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(dashboard_file, dpi=300, bbox_inches='tight')
    print(f"💾 Training dashboard saved to: {dashboard_file}")
    
    # Save detailed recommendations
    recommendations_file = f"training_recommendations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(recommendations_file, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_timestamp': datetime.now().isoformat(),
            'conversation_insights': conversation_insights,
            'performance_analysis': {
                'high_quality_count': len(performance_analysis['high_quality']),
                'medium_quality_count': len(performance_analysis['medium_quality']),
                'low_quality_count': len(performance_analysis['low_quality']),
                'common_problems': dict(performance_analysis['common_problems']),
                'sentiment_transitions': dict(performance_analysis['sentiment_transitions'])
            },
            'training_recommendations': recommendations
        }, f, indent=2, ensure_ascii=False)
    
    print(f"💾 Detailed recommendations saved to: {recommendations_file}")
    
    print(f"\n✅ Detailed agent training analysis completed!")
    print(f"🚀 Use these insights to improve agent performance and training!")

if __name__ == "__main__":
    main()
