#!/usr/bin/env python3
"""
SentilensAI - Enhanced Report Generator with Deep Learning Insights

Generates comprehensive reports including deep learning analysis,
learning recommendations, and actionable insights for agent training.

Author: Pravin Selvamuthu
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import logging

from enhanced_analysis import EnhancedSentimentAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedReportGenerator:
    """Generate comprehensive reports with deep learning insights"""
    
    def __init__(self):
        self.analyzer = EnhancedSentimentAnalyzer()
        self.report_data = {}
    
    def generate_comprehensive_report(self, conversations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive report with deep learning analysis"""
        
        print("🤖 SentilensAI - Generating Enhanced Comprehensive Report")
        print("=" * 70)
        
        # Analyze all conversations with enhanced methods
        enhanced_analyses = []
        for conv in conversations:
            print(f"🔍 Analyzing {conv['conversation_id']} with deep learning...")
            enhanced_analysis = self.analyzer.analyze_conversation_enhanced(conv)
            enhanced_analyses.append(enhanced_analysis)
        
        # Generate comprehensive report
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_conversations': len(conversations),
                'analysis_methods': ['Traditional ML', 'Deep Learning', 'Ensemble'],
                'report_version': '2.0.0'
            },
            'executive_summary': self._generate_executive_summary(enhanced_analyses),
            'deep_learning_insights': self._generate_deep_learning_insights(enhanced_analyses),
            'learning_recommendations': self._generate_learning_recommendations(enhanced_analyses),
            'quality_analysis': self._generate_quality_analysis(enhanced_analyses),
            'improvement_roadmap': self._generate_improvement_roadmap(enhanced_analyses),
            'technical_insights': self._generate_technical_insights(enhanced_analyses),
            'conversation_details': enhanced_analyses
        }
        
        return report
    
    def _generate_executive_summary(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate executive summary with key insights"""
        
        # Calculate overall metrics
        total_conversations = len(analyses)
        quality_scores = [analysis['quality_metrics']['overall_quality_score'] for analysis in analyses]
        avg_quality = np.mean(quality_scores)
        
        # Categorize conversations
        high_quality = sum(1 for score in quality_scores if score > 0.7)
        medium_quality = sum(1 for score in quality_scores if 0.4 <= score <= 0.7)
        low_quality = sum(1 for score in quality_scores if score < 0.4)
        
        # Deep learning metrics
        ensemble_benefits = []
        model_agreements = []
        prediction_confidences = []
        
        for analysis in analyses:
            dl_insights = analysis['deep_learning_insights']
            ensemble_benefits.append(dl_insights['deep_learning_insights']['ensemble_benefit'])
            model_agreements.append(dl_insights['model_agreement_analysis']['average_agreement'])
            prediction_confidences.append(dl_insights['deep_learning_insights']['prediction_confidence'])
        
        avg_ensemble_benefit = np.mean(ensemble_benefits)
        avg_model_agreement = np.mean(model_agreements)
        avg_prediction_confidence = np.mean(prediction_confidences)
        
        return {
            'overall_performance': {
                'average_quality_score': avg_quality,
                'quality_distribution': {
                    'high_quality': high_quality,
                    'medium_quality': medium_quality,
                    'low_quality': low_quality
                },
                'performance_grade': self._calculate_performance_grade(avg_quality)
            },
            'deep_learning_metrics': {
                'ensemble_benefit': avg_ensemble_benefit,
                'model_agreement': avg_model_agreement,
                'prediction_confidence': avg_prediction_confidence,
                'ai_readiness_score': self._calculate_ai_readiness_score(
                    avg_ensemble_benefit, avg_model_agreement, avg_prediction_confidence
                )
            },
            'key_insights': self._generate_key_insights(analyses),
            'critical_issues': self._identify_critical_issues(analyses),
            'success_factors': self._identify_success_factors(analyses)
        }
    
    def _calculate_performance_grade(self, avg_quality: float) -> str:
        """Calculate performance grade based on quality score"""
        if avg_quality >= 0.8:
            return 'A+ (Excellent)'
        elif avg_quality >= 0.7:
            return 'A (Very Good)'
        elif avg_quality >= 0.6:
            return 'B (Good)'
        elif avg_quality >= 0.5:
            return 'C (Average)'
        else:
            return 'D (Needs Improvement)'
    
    def _calculate_ai_readiness_score(self, ensemble_benefit: float, model_agreement: float, 
                                    prediction_confidence: float) -> float:
        """Calculate AI readiness score"""
        return (ensemble_benefit * 0.4 + model_agreement * 0.3 + prediction_confidence * 0.3)
    
    def _generate_key_insights(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Generate key insights from analysis"""
        insights = []
        
        # Calculate metrics
        quality_scores = [analysis['quality_metrics']['overall_quality_score'] for analysis in analyses]
        avg_quality = np.mean(quality_scores)
        
        ensemble_benefits = [analysis['deep_learning_insights']['deep_learning_insights']['ensemble_benefit'] 
                           for analysis in analyses]
        avg_ensemble_benefit = np.mean(ensemble_benefits)
        
        # Generate insights
        if avg_quality > 0.7:
            insights.append("High overall conversation quality indicates effective agent training")
        elif avg_quality < 0.5:
            insights.append("Low conversation quality suggests need for comprehensive training overhaul")
        
        if avg_ensemble_benefit > 0.8:
            insights.append("Strong ensemble benefit indicates effective use of multiple AI models")
        elif avg_ensemble_benefit < 0.6:
            insights.append("Limited ensemble benefit suggests need for model optimization")
        
        # Count improvement opportunities
        total_opportunities = sum(len(analysis['improvement_opportunities']) for analysis in analyses)
        if total_opportunities > len(analyses) * 2:
            insights.append("Multiple improvement opportunities identified across conversations")
        
        return insights
    
    def _identify_critical_issues(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify critical issues requiring immediate attention"""
        critical_issues = []
        
        for analysis in analyses:
            conv_id = analysis['conversation_id']
            quality_metrics = analysis['quality_metrics']
            
            # Check for critical quality issues
            if quality_metrics['negative_response_rate'] > 0.2:
                critical_issues.append({
                    'conversation_id': conv_id,
                    'issue': 'High negative response rate',
                    'severity': 'CRITICAL',
                    'description': f"Negative response rate: {quality_metrics['negative_response_rate']:.2f}",
                    'action_required': 'Immediate tone checking implementation'
                })
            
            if quality_metrics['average_confidence'] < 0.4:
                critical_issues.append({
                    'conversation_id': conv_id,
                    'issue': 'Very low confidence scores',
                    'severity': 'HIGH',
                    'description': f"Average confidence: {quality_metrics['average_confidence']:.2f}",
                    'action_required': 'Confidence improvement training'
                })
            
            if quality_metrics['model_agreement_rate'] < 0.5:
                critical_issues.append({
                    'conversation_id': conv_id,
                    'issue': 'Poor model agreement',
                    'severity': 'MEDIUM',
                    'description': f"Model agreement: {quality_metrics['model_agreement_rate']:.2f}",
                    'action_required': 'Ensemble optimization'
                })
        
        return critical_issues
    
    def _identify_success_factors(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify success factors from high-performing conversations"""
        success_factors = []
        
        # Find high-quality conversations
        high_quality_convs = [analysis for analysis in analyses 
                            if analysis['quality_metrics']['overall_quality_score'] > 0.7]
        
        if high_quality_convs:
            # Analyze common patterns
            high_confidence_count = sum(1 for conv in high_quality_convs 
                                      if conv['quality_metrics']['average_confidence'] > 0.7)
            
            if high_confidence_count > 0:
                success_factors.append({
                    'factor': 'High Confidence Responses',
                    'description': f"{high_confidence_count} high-quality conversations had high confidence",
                    'recommendation': 'Replicate confidence-building techniques'
                })
            
            # Check for positive sentiment trends
            positive_trend_count = 0
            for conv in high_quality_convs:
                user_sentiments = conv['deep_learning_insights']['sentiment_evolution']['user_trend']
                if len(user_sentiments) >= 2 and user_sentiments[-1] == 'positive':
                    positive_trend_count += 1
            
            if positive_trend_count > 0:
                success_factors.append({
                    'factor': 'Positive Sentiment Management',
                    'description': f"{positive_trend_count} conversations successfully maintained positive sentiment",
                    'recommendation': 'Document and train on positive sentiment techniques'
                })
        
        return success_factors
    
    def _generate_deep_learning_insights(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate deep learning specific insights"""
        
        # Aggregate deep learning metrics
        all_ensemble_benefits = []
        all_model_agreements = []
        all_prediction_confidences = []
        all_model_diversities = []
        
        for analysis in analyses:
            dl_insights = analysis['deep_learning_insights']['deep_learning_insights']
            all_ensemble_benefits.append(dl_insights['ensemble_benefit'])
            all_model_agreements.append(analysis['deep_learning_insights']['model_agreement_analysis']['average_agreement'])
            all_prediction_confidences.append(dl_insights['prediction_confidence'])
            all_model_diversities.append(dl_insights['model_diversity'])
        
        return {
            'ensemble_performance': {
                'average_benefit': np.mean(all_ensemble_benefits),
                'benefit_consistency': 1.0 - np.std(all_ensemble_benefits),
                'recommendation': self._get_ensemble_recommendation(np.mean(all_ensemble_benefits))
            },
            'model_agreement_analysis': {
                'average_agreement': np.mean(all_model_agreements),
                'agreement_consistency': 1.0 - np.std(all_model_agreements),
                'disagreement_rate': sum(1 for agree in all_model_agreements if agree < 0.7) / len(all_model_agreements)
            },
            'prediction_confidence': {
                'average_confidence': np.mean(all_prediction_confidences),
                'confidence_consistency': 1.0 - np.std(all_prediction_confidences),
                'low_confidence_rate': sum(1 for conf in all_prediction_confidences if conf < 0.6) / len(all_prediction_confidences)
            },
            'model_diversity': {
                'average_diversity': np.mean(all_model_diversities),
                'diversity_consistency': 1.0 - np.std(all_model_diversities),
                'recommendation': self._get_diversity_recommendation(np.mean(all_model_diversities))
            },
            'ai_optimization_opportunities': self._identify_ai_optimization_opportunities(analyses)
        }
    
    def _get_ensemble_recommendation(self, avg_benefit: float) -> str:
        """Get ensemble recommendation based on benefit score"""
        if avg_benefit > 0.8:
            return "Excellent ensemble performance - maintain current configuration"
        elif avg_benefit > 0.6:
            return "Good ensemble performance - consider fine-tuning weights"
        else:
            return "Poor ensemble performance - implement better model selection and weighting"
    
    def _get_diversity_recommendation(self, avg_diversity: float) -> str:
        """Get diversity recommendation based on diversity score"""
        if avg_diversity > 0.8:
            return "High model diversity - good for robust predictions"
        elif avg_diversity > 0.6:
            return "Moderate diversity - consider adding more model types"
        else:
            return "Low diversity - add more diverse models to ensemble"
    
    def _identify_ai_optimization_opportunities(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify AI optimization opportunities"""
        opportunities = []
        
        # Analyze model performance patterns
        low_confidence_convs = [analysis for analysis in analyses 
                              if analysis['quality_metrics']['average_confidence'] < 0.6]
        
        if len(low_confidence_convs) > len(analyses) * 0.3:
            opportunities.append({
                'area': 'Model Confidence',
                'issue': 'Widespread low confidence predictions',
                'solution': 'Implement confidence calibration and model fine-tuning',
                'priority': 'High',
                'expected_impact': 'Significant improvement in prediction reliability'
            })
        
        # Analyze ensemble effectiveness
        low_agreement_convs = [analysis for analysis in analyses 
                             if analysis['quality_metrics']['model_agreement_rate'] < 0.7]
        
        if len(low_agreement_convs) > len(analyses) * 0.4:
            opportunities.append({
                'area': 'Model Agreement',
                'issue': 'Poor model agreement across conversations',
                'solution': 'Implement consensus mechanisms and model selection strategies',
                'priority': 'Medium',
                'expected_impact': 'Improved prediction consistency'
            })
        
        return opportunities
    
    def _generate_learning_recommendations(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive learning recommendations"""
        
        # Aggregate recommendations from all conversations
        all_immediate_actions = []
        all_short_term_improvements = []
        all_long_term_strategies = []
        all_technical_recommendations = []
        
        for analysis in analyses:
            recommendations = analysis['learning_recommendations']
            all_immediate_actions.extend(recommendations['immediate_actions'])
            all_short_term_improvements.extend(recommendations['short_term_improvements'])
            all_long_term_strategies.extend(recommendations['long_term_strategy'])
            all_technical_recommendations.extend(recommendations['technical_recommendations'])
        
        # Consolidate and prioritize recommendations
        consolidated_recommendations = {
            'immediate_actions': self._consolidate_recommendations(all_immediate_actions),
            'short_term_improvements': self._consolidate_recommendations(all_short_term_improvements),
            'long_term_strategies': self._consolidate_recommendations(all_long_term_strategies),
            'technical_recommendations': self._consolidate_recommendations(all_technical_recommendations),
            'learning_priorities': self._generate_learning_priorities(analyses),
            'training_roadmap': self._generate_training_roadmap(analyses)
        }
        
        return consolidated_recommendations
    
    def _consolidate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Consolidate similar recommendations and remove duplicates"""
        consolidated = []
        seen_actions = set()
        
        for rec in recommendations:
            action_key = rec.get('action', rec.get('recommendation', '')).lower()
            if action_key not in seen_actions:
                consolidated.append(rec)
                seen_actions.add(action_key)
        
        return consolidated
    
    def _generate_learning_priorities(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate learning priorities based on analysis"""
        
        # Count critical issues
        critical_issues = self._identify_critical_issues(analyses)
        issue_counts = Counter(issue['issue'] for issue in critical_issues)
        
        priorities = {
            'critical': list(issue_counts.keys())[:3],  # Top 3 critical issues
            'high_impact_areas': [
                'Response Confidence',
                'Sentiment Management',
                'Model Agreement'
            ],
            'training_focus': [
                'Deep Learning Integration',
                'Ensemble Optimization',
                'Confidence Calibration'
            ]
        }
        
        return priorities
    
    def _generate_training_roadmap(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate training roadmap with timelines"""
        
        return {
            'phase_1_immediate': {
                'timeline': '1-2 weeks',
                'focus': 'Critical issue resolution',
                'activities': [
                    'Implement tone checking system',
                    'Deploy confidence improvement training',
                    'Set up real-time monitoring'
                ]
            },
            'phase_2_short_term': {
                'timeline': '1-2 months',
                'focus': 'Deep learning optimization',
                'activities': [
                    'Fine-tune transformer models',
                    'Implement ensemble consensus',
                    'Create training data pipeline'
                ]
            },
            'phase_3_long_term': {
                'timeline': '3-6 months',
                'focus': 'Advanced AI capabilities',
                'activities': [
                    'Develop custom domain models',
                    'Implement active learning',
                    'Create continuous improvement pipeline'
                ]
            }
        }
    
    def _generate_quality_analysis(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate quality analysis with deep learning insights"""
        
        quality_scores = [analysis['quality_metrics']['overall_quality_score'] for analysis in analyses]
        confidence_scores = [analysis['quality_metrics']['average_confidence'] for analysis in analyses]
        agreement_scores = [analysis['quality_metrics']['model_agreement_rate'] for analysis in analyses]
        
        return {
            'overall_quality_metrics': {
                'average_quality_score': np.mean(quality_scores),
                'quality_consistency': 1.0 - np.std(quality_scores),
                'quality_distribution': {
                    'excellent': sum(1 for score in quality_scores if score > 0.8),
                    'good': sum(1 for score in quality_scores if 0.6 <= score <= 0.8),
                    'average': sum(1 for score in quality_scores if 0.4 <= score < 0.6),
                    'poor': sum(1 for score in quality_scores if score < 0.4)
                }
            },
            'confidence_analysis': {
                'average_confidence': np.mean(confidence_scores),
                'confidence_consistency': 1.0 - np.std(confidence_scores),
                'low_confidence_rate': sum(1 for conf in confidence_scores if conf < 0.6) / len(confidence_scores)
            },
            'model_agreement_analysis': {
                'average_agreement': np.mean(agreement_scores),
                'agreement_consistency': 1.0 - np.std(agreement_scores),
                'disagreement_rate': sum(1 for agree in agreement_scores if agree < 0.7) / len(agreement_scores)
            }
        }
    
    def _generate_improvement_roadmap(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate improvement roadmap with specific actions"""
        
        return {
            'immediate_improvements': {
                'timeline': '1-2 weeks',
                'actions': [
                    'Deploy real-time tone checking',
                    'Implement confidence monitoring',
                    'Create quality alerts'
                ],
                'expected_impact': 'Eliminate critical issues'
            },
            'short_term_improvements': {
                'timeline': '1-2 months',
                'actions': [
                    'Optimize ensemble methods',
                    'Improve model agreement',
                    'Enhance training data quality'
                ],
                'expected_impact': 'Significant quality improvement'
            },
            'long_term_improvements': {
                'timeline': '3-6 months',
                'actions': [
                    'Develop custom AI models',
                    'Implement active learning',
                    'Create continuous improvement pipeline'
                ],
                'expected_impact': 'Industry-leading performance'
            }
        }
    
    def _generate_technical_insights(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate technical insights and recommendations"""
        
        return {
            'model_performance': {
                'ensemble_effectiveness': 'High' if np.mean([a['deep_learning_insights']['deep_learning_insights']['ensemble_benefit'] 
                                                           for a in analyses]) > 0.7 else 'Medium',
                'model_diversity': 'Good' if np.mean([a['deep_learning_insights']['deep_learning_insights']['model_diversity'] 
                                                    for a in analyses]) > 0.7 else 'Needs Improvement',
                'prediction_reliability': 'High' if np.mean([a['quality_metrics']['model_agreement_rate'] 
                                                           for a in analyses]) > 0.7 else 'Medium'
            },
            'infrastructure_recommendations': [
                'Deploy scalable model serving infrastructure',
                'Implement real-time monitoring and alerting',
                'Create automated model retraining pipeline',
                'Set up A/B testing for model improvements'
            ],
            'data_quality_insights': {
                'training_data_adequacy': 'Sufficient' if len(analyses) > 10 else 'Insufficient',
                'data_diversity': 'Good' if len(set(a['conversation_id'] for a in analyses)) > 5 else 'Limited',
                'quality_consistency': 'High' if np.std([a['quality_metrics']['overall_quality_score'] for a in analyses]) < 0.2 else 'Variable'
            }
        }
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save comprehensive report to file"""
        
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"enhanced_sentiment_report_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Enhanced report saved to: {filename}")
        return filename

def main():
    """Demo function for enhanced report generation"""
    print("🤖 SentilensAI - Enhanced Report Generator Demo")
    print("=" * 60)
    
    # Load sample conversations
    import glob
    result_files = glob.glob("sentiment_analysis_results_*.json")
    if not result_files:
        print("❌ No analysis results found!")
        return
    
    latest_file = max(result_files)
    with open(latest_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # Convert to conversation format
    conversations = []
    for conv in results['conversation_results']:
        conversation = {
            'conversation_id': conv['conversation_id'],
            'timestamp': conv['timestamp'],
            'messages': []
        }
        
        for msg in conv['message_analysis']:
            conversation['messages'].append({
                'user': msg['user_message'],
                'bot': msg['bot_message'],
                'timestamp': msg['timestamp']
            })
        
        conversations.append(conversation)
    
    # Generate enhanced report
    generator = EnhancedReportGenerator()
    report = generator.generate_comprehensive_report(conversations)
    
    # Save report
    filename = generator.save_report(report)
    
    # Display summary
    print(f"\n📊 Enhanced Report Summary:")
    print(f"   Total Conversations: {report['report_metadata']['total_conversations']}")
    print(f"   Average Quality Score: {report['executive_summary']['overall_performance']['average_quality_score']:.2f}")
    print(f"   Performance Grade: {report['executive_summary']['overall_performance']['performance_grade']}")
    print(f"   AI Readiness Score: {report['executive_summary']['deep_learning_metrics']['ai_readiness_score']:.2f}")
    
    print(f"\n🎓 Learning Recommendations:")
    print(f"   Immediate Actions: {len(report['learning_recommendations']['immediate_actions'])}")
    print(f"   Short-term Improvements: {len(report['learning_recommendations']['short_term_improvements'])}")
    print(f"   Long-term Strategies: {len(report['learning_recommendations']['long_term_strategies'])}")
    
    print(f"\n💾 Report saved to: {filename}")
    print(f"\n✅ Enhanced report generation completed!")
    print(f"🚀 Deep learning insights and recommendations ready!")

if __name__ == "__main__":
    main()
