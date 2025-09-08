#!/usr/bin/env python3
"""
SentilensAI - Agent Training Summary and Recommendations

Comprehensive analysis of chat interactions to identify specific problems
and provide actionable insights for improving agent training.

Author: Pravin Selvamuthu
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict, Counter

def load_analysis_results(json_file):
    """Load sentiment analysis results"""
    with open(json_file, 'r', encoding='utf-8') as f:
        return json.load(f)

def generate_comprehensive_summary():
    """Generate comprehensive agent training summary"""
    
    print("🎯 SENTILENSAI - COMPREHENSIVE AGENT TRAINING ANALYSIS")
    print("=" * 80)
    
    # Load results
    import glob
    result_files = glob.glob("sentiment_analysis_results_*.json")
    if not result_files:
        print("❌ No results files found!")
        return
    
    latest_file = max(result_files)
    print(f"📁 Analysis based on: {latest_file}")
    
    results = load_analysis_results(latest_file)
    
    # Overall Statistics
    print(f"\n📊 OVERALL PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Total Conversations Analyzed: {results['total_conversations']}")
    print(f"Average Quality Score: {results['average_quality_score']:.2f}/1.0")
    print(f"Total Messages: {sum(results['overall_sentiment_distribution']['user'].values()) + sum(results['overall_sentiment_distribution']['bot'].values())}")
    
    # Detailed Conversation Analysis
    print(f"\n🔍 DETAILED CONVERSATION BREAKDOWN")
    print("=" * 50)
    
    conversation_insights = []
    
    for conv in results['conversation_results']:
        conv_id = conv['conversation_id']
        quality_score = conv['conversation_quality']
        messages = conv['message_analysis']
        
        print(f"\n📋 {conv_id.upper()} - Quality Score: {quality_score:.2f}/1.0")
        print("-" * 40)
        
        # Analyze each message
        problems = []
        strengths = []
        
        for i, msg in enumerate(messages, 1):
            user_text = msg['user_message'][:50] + "..." if len(msg['user_message']) > 50 else msg['user_message']
            bot_text = msg['bot_message'][:50] + "..." if len(msg['bot_message']) > 50 else msg['bot_message']
            
            print(f"\n  Message {i}:")
            print(f"    👤 User: {user_text}")
            print(f"    🤖 Bot:  {bot_text}")
            
            # Analyze user sentiment
            if msg['user_sentiment']:
                user_sentiment = msg['user_sentiment']
                user_confidence = msg['user_confidence']
                print(f"    📊 User Sentiment: {user_sentiment} (Confidence: {user_confidence:.2f})")
                
                if user_sentiment == 'negative' and user_confidence > 0.5:
                    problems.append(f"Message {i}: User expressed strong negative sentiment")
            
            # Analyze bot performance
            if msg['bot_sentiment'] and msg['bot_confidence']:
                bot_sentiment = msg['bot_sentiment']
                bot_confidence = msg['bot_confidence']
                print(f"    📊 Bot Sentiment: {bot_sentiment} (Confidence: {bot_confidence:.2f})")
                
                if bot_confidence < 0.3:
                    problems.append(f"Message {i}: Bot response had low confidence ({bot_confidence:.2f})")
                
                if bot_sentiment == 'negative':
                    problems.append(f"Message {i}: Bot generated negative response")
                
                if bot_confidence > 0.6 and bot_sentiment == 'positive':
                    strengths.append(f"Message {i}: Strong positive bot response")
        
        # Identify conversation-level issues
        user_sentiments = [msg['user_sentiment'] for msg in messages if msg['user_sentiment']]
        bot_sentiments = [msg['bot_sentiment'] for msg in messages if msg['bot_sentiment']]
        
        if len(user_sentiments) >= 2:
            if user_sentiments[0] in ['positive', 'neutral'] and user_sentiments[-1] == 'negative':
                problems.append("User sentiment deteriorated during conversation")
            elif user_sentiments[0] == 'negative' and user_sentiments[-1] == 'positive':
                strengths.append("Successfully improved user sentiment")
        
        # Print analysis
        if problems:
            print(f"\n  ⚠️  PROBLEMS IDENTIFIED:")
            for problem in problems:
                print(f"    • {problem}")
        else:
            print(f"\n  ✅ No significant problems detected")
        
        if strengths:
            print(f"\n  💪 STRENGTHS IDENTIFIED:")
            for strength in strengths:
                print(f"    • {strength}")
        
        # Store insights
        conversation_insights.append({
            'conversation_id': conv_id,
            'quality_score': quality_score,
            'problems': problems,
            'strengths': strengths,
            'user_sentiment_trend': user_sentiments,
            'bot_sentiment_trend': bot_sentiments
        })
    
    # Overall Analysis
    print(f"\n🎯 OVERALL AGENT PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    # Quality distribution
    high_quality = [conv for conv in conversation_insights if conv['quality_score'] > 0.7]
    medium_quality = [conv for conv in conversation_insights if 0.4 <= conv['quality_score'] <= 0.7]
    low_quality = [conv for conv in conversation_insights if conv['quality_score'] < 0.4]
    
    print(f"📈 QUALITY DISTRIBUTION:")
    print(f"   🟢 High Quality (>0.7): {len(high_quality)} conversations ({len(high_quality)/len(conversation_insights)*100:.1f}%)")
    print(f"   🟡 Medium Quality (0.4-0.7): {len(medium_quality)} conversations ({len(medium_quality)/len(conversation_insights)*100:.1f}%)")
    print(f"   🔴 Low Quality (<0.4): {len(low_quality)} conversations ({len(low_quality)/len(conversation_insights)*100:.1f}%)")
    
    # Common problems
    all_problems = []
    for conv in conversation_insights:
        all_problems.extend(conv['problems'])
    
    problem_counts = Counter(all_problems)
    
    print(f"\n🚨 MOST COMMON PROBLEMS:")
    for problem, count in problem_counts.most_common():
        print(f"   • {problem} ({count} occurrences)")
    
    # Sentiment analysis
    print(f"\n📊 SENTIMENT ANALYSIS:")
    user_sentiment_dist = results['overall_sentiment_distribution']['user']
    bot_sentiment_dist = results['overall_sentiment_distribution']['bot']
    
    print(f"   User Messages: {user_sentiment_dist['positive']} positive, {user_sentiment_dist['negative']} negative, {user_sentiment_dist['neutral']} neutral")
    print(f"   Bot Messages: {bot_sentiment_dist['positive']} positive, {bot_sentiment_dist['negative']} negative, {bot_sentiment_dist['neutral']} neutral")
    
    # Training Recommendations
    print(f"\n🎓 SPECIFIC TRAINING RECOMMENDATIONS")
    print("=" * 50)
    
    recommendations = []
    
    # Problem 1: Low Confidence Responses
    low_confidence_count = sum(1 for conv in conversation_insights for problem in conv['problems'] if 'low confidence' in problem.lower())
    if low_confidence_count > 0:
        recommendations.append({
            'priority': 'HIGH',
            'area': 'Response Confidence',
            'issue': f'{low_confidence_count} low confidence responses detected',
            'recommendation': 'Improve response quality and agent confidence',
            'actions': [
                'Train agents to be more decisive in their responses',
                'Implement response validation before sending',
                'Provide better knowledge base and training materials',
                'Add confidence scoring to response evaluation'
            ]
        })
    
    # Problem 2: Negative Bot Responses
    negative_bot_count = sum(1 for conv in conversation_insights for problem in conv['problems'] if 'negative response' in problem.lower())
    if negative_bot_count > 0:
        recommendations.append({
            'priority': 'CRITICAL',
            'area': 'Response Tone',
            'issue': f'{negative_bot_count} negative bot responses detected',
            'recommendation': 'Ensure all responses maintain positive or neutral tone',
            'actions': [
                'Implement tone checking before response delivery',
                'Train agents to always maintain professional, helpful tone',
                'Create positive response templates for common scenarios',
                'Add sentiment analysis to response validation pipeline'
            ]
        })
    
    # Problem 3: Sentiment Deterioration
    sentiment_deterioration = sum(1 for conv in conversation_insights for problem in conv['problems'] if 'sentiment deteriorated' in problem.lower())
    if sentiment_deterioration > 0:
        recommendations.append({
            'priority': 'HIGH',
            'area': 'Sentiment Management',
            'issue': f'{sentiment_deterioration} conversations with deteriorating user sentiment',
            'recommendation': 'Improve sentiment management and de-escalation',
            'actions': [
                'Train agents to recognize early warning signs of frustration',
                'Implement proactive sentiment monitoring',
                'Practice de-escalation techniques',
                'Create escalation protocols for negative sentiment'
            ]
        })
    
    # Quality-based recommendations
    if len(medium_quality) > len(high_quality):
        recommendations.append({
            'priority': 'MEDIUM',
            'area': 'Overall Performance',
            'issue': f'More medium quality conversations ({len(medium_quality)}) than high quality ({len(high_quality)})',
            'recommendation': 'Focus on improving overall conversation quality',
            'actions': [
                'Conduct comprehensive performance review',
                'Implement quality monitoring and feedback loops',
                'Add regular training sessions',
                'Create quality benchmarks and targets'
            ]
        })
    
    # Print recommendations
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec['area'].upper()} - {rec['priority']} PRIORITY")
        print(f"   Issue: {rec['issue']}")
        print(f"   Recommendation: {rec['recommendation']}")
        print(f"   Specific Actions:")
        for action in rec['actions']:
            print(f"     • {action}")
    
    # Success Stories
    print(f"\n💪 SUCCESS STORIES")
    print("=" * 30)
    for conv in high_quality:
        print(f"✅ {conv['conversation_id']}: Quality Score {conv['quality_score']:.2f}")
        if conv['strengths']:
            for strength in conv['strengths']:
                print(f"   • {strength}")
    
    # Action Plan
    print(f"\n📋 IMMEDIATE ACTION PLAN")
    print("=" * 30)
    print("1. 🚨 URGENT: Address negative bot responses immediately")
    print("2. 📈 HIGH: Improve response confidence through training")
    print("3. 🎯 MEDIUM: Implement sentiment monitoring and de-escalation")
    print("4. 📊 ONGOING: Monitor conversation quality metrics")
    print("5. 🎓 TRAINING: Schedule regular agent training sessions")
    
    # Save summary
    summary_data = {
        'analysis_timestamp': datetime.now().isoformat(),
        'overall_metrics': {
            'total_conversations': results['total_conversations'],
            'average_quality_score': results['average_quality_score'],
            'high_quality_count': len(high_quality),
            'medium_quality_count': len(medium_quality),
            'low_quality_count': len(low_quality)
        },
        'common_problems': dict(problem_counts),
        'recommendations': recommendations,
        'conversation_insights': conversation_insights
    }
    
    summary_file = f"agent_training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed summary saved to: {summary_file}")
    print(f"\n✅ Comprehensive agent training analysis completed!")
    print(f"🚀 Use these insights to significantly improve agent performance!")

if __name__ == "__main__":
    generate_comprehensive_summary()
