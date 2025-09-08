"""
SentilensAI - Machine Learning Training Pipeline

This module provides comprehensive machine learning capabilities for training
custom sentiment analysis models specifically optimized for AI chatbot conversations.

Features:
- Multiple ML algorithms (Random Forest, SVM, Neural Networks, XGBoost, etc.)
- Advanced feature engineering for chatbot text
- Cross-validation and hyperparameter tuning
- Model comparison and evaluation
- Production-ready model persistence
- Real-time prediction capabilities

Author: Pravin Selvamuthu
Repository: https://github.com/kernelseed/sentilens-ai
"""

import os
import json
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import joblib

# Advanced ML libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# Visualization
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# LangChain integration
from langchain.schema import BaseMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Import our sentiment analyzer
from sentiment_analyzer import SentilensAIAnalyzer, SentimentResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentilensAITrainer:
    """
    Advanced machine learning trainer for sentiment analysis models
    specifically designed for AI chatbot conversations
    """
    
    def __init__(self, model_cache_dir: str = "./model_cache"):
        """
        Initialize the SentimentsAI trainer
        
        Args:
            model_cache_dir: Directory to cache trained models
        """
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(exist_ok=True)
        
        # Initialize components
        self.analyzer = SentilensAIAnalyzer()
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()
        self.vectorizer = None
        self.models = {}
        self.training_data = None
        self.feature_names = None
        
        # Initialize available models
        self._initialize_models()
        
        # Feature engineering parameters
        self.feature_params = {
            'max_features': 10000,
            'ngram_range': (1, 3),
            'min_df': 2,
            'max_df': 0.95,
            'stop_words': 'english'
        }
    
    def _initialize_models(self):
        """Initialize available machine learning models"""
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                n_jobs=-1
            ),
            'decision_tree': DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            'naive_bayes': MultinomialNB(alpha=1.0),
            'ada_boost': AdaBoostClassifier(
                n_estimators=50,
                learning_rate=1.0,
                random_state=42
            )
        }
        
        # Add advanced models if available
        if XGBOOST_AVAILABLE:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        
        if LIGHTGBM_AVAILABLE:
            self.models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        
        if CATBOOST_AVAILABLE:
            self.models['catboost'] = cb.CatBoostClassifier(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_seed=42,
                verbose=False
            )
    
    def create_synthetic_training_data(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Create synthetic training data for sentiment analysis
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with text and sentiment labels
        """
        logger.info(f"Creating {num_samples} synthetic training samples...")
        
        # Define sentiment categories and sample texts
        sentiment_data = {
            'positive': [
                "I love this chatbot! It's amazing and so helpful.",
                "This is exactly what I needed. Thank you so much!",
                "Great service! The bot understood me perfectly.",
                "Excellent! This chatbot is fantastic and very user-friendly.",
                "Perfect! I'm so happy with this experience.",
                "Wonderful! The bot provided exactly the right information.",
                "Outstanding service! I'm impressed with the quality.",
                "Brilliant! This is the best chatbot I've ever used.",
                "Fantastic! The response was quick and accurate.",
                "Superb! I'm delighted with the help I received."
            ],
            'negative': [
                "This chatbot is terrible. It doesn't understand anything.",
                "Worst experience ever. The bot is completely useless.",
                "This is awful. I'm frustrated and disappointed.",
                "Horrible service! The bot keeps giving wrong answers.",
                "Disgusting! This chatbot is a complete waste of time.",
                "Terrible! I hate this bot and its responses.",
                "Awful experience. The bot is stupid and unhelpful.",
                "Disappointing! This chatbot is broken and useless.",
                "Frustrating! The bot doesn't know what it's doing.",
                "Pathetic! This is the worst chatbot I've ever seen."
            ],
            'neutral': [
                "Can you help me with my account information?",
                "I need to check my order status.",
                "What are your business hours?",
                "How do I reset my password?",
                "I want to update my profile details.",
                "Can you provide more information about this product?",
                "I need assistance with my subscription.",
                "What is your return policy?",
                "How can I contact customer support?",
                "I have a question about my recent purchase."
            ]
        }
        
        # Generate synthetic data
        data = []
        samples_per_sentiment = num_samples // 3
        
        for sentiment, texts in sentiment_data.items():
            for i in range(samples_per_sentiment):
                # Select base text
                base_text = np.random.choice(texts)
                
                # Add variations
                variations = [
                    base_text,
                    base_text + " Please help me.",
                    "Hi, " + base_text.lower(),
                    base_text + " Thanks!",
                    "Hello, " + base_text.lower(),
                    base_text + " I appreciate it.",
                    "Hey, " + base_text.lower(),
                    base_text + " Could you assist?",
                    "Good morning, " + base_text.lower(),
                    base_text + " That would be great."
                ]
                
                text = np.random.choice(variations)
                data.append({
                    'text': text,
                    'sentiment': sentiment,
                    'confidence': np.random.uniform(0.6, 1.0),
                    'polarity': np.random.uniform(-1, 1) if sentiment == 'neutral' else (1 if sentiment == 'positive' else -1),
                    'subjectivity': np.random.uniform(0.3, 0.8),
                    'message_type': 'user' if i % 2 == 0 else 'bot',
                    'conversation_id': f'conv_{i//2}',
                    'timestamp': datetime.now()
                })
        
        # Add some mixed sentiment examples
        mixed_examples = [
            ("I'm not sure if this is good or bad.", "neutral"),
            ("It's okay, I guess.", "neutral"),
            ("This is fine, nothing special.", "neutral"),
            ("I have mixed feelings about this.", "neutral"),
            ("It's decent but could be better.", "neutral")
        ]
        
        for text, sentiment in mixed_examples:
            data.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': np.random.uniform(0.4, 0.7),
                'polarity': np.random.uniform(-0.3, 0.3),
                'subjectivity': np.random.uniform(0.5, 0.9),
                'message_type': 'user',
                'conversation_id': f'mixed_{len(data)}',
                'timestamp': datetime.now()
            })
        
        df = pd.DataFrame(data)
        logger.info(f"Created {len(df)} training samples")
        return df
    
    def extract_features(self, texts: List[str]) -> np.ndarray:
        """
        Extract comprehensive features from text data
        
        Args:
            texts: List of text strings
            
        Returns:
            Feature matrix
        """
        logger.info("Extracting features from text data...")
        
        # Initialize vectorizer if not already done
        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                max_features=self.feature_params['max_features'],
                ngram_range=self.feature_params['ngram_range'],
                min_df=self.feature_params['min_df'],
                max_df=self.feature_params['max_df'],
                stop_words=self.feature_params['stop_words']
            )
        
        # TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(texts).toarray()
        
        # Additional text features
        text_features = []
        for text in texts:
            features = []
            
            # Basic text statistics
            features.append(len(text))  # Text length
            features.append(len(text.split()))  # Word count
            features.append(len([c for c in text if c.isupper()]))  # Uppercase count
            features.append(len([c for c in text if c.isdigit()]))  # Digit count
            features.append(len([c for c in text if c in '!?']))  # Punctuation count
            
            # Sentiment features using our analyzer
            try:
                sentiment_result = self.analyzer.analyze_sentiment(text, method='ensemble')
                features.extend([
                    sentiment_result.polarity,
                    sentiment_result.confidence,
                    sentiment_result.subjectivity
                ])
                
                # Emotion features
                for emotion in ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust']:
                    features.append(sentiment_result.emotions.get(emotion, 0.0))
            except:
                features.extend([0.0] * 9)  # Default values if analysis fails
            
            # Text complexity features
            words = text.split()
            if words:
                avg_word_length = np.mean([len(word) for word in words])
                features.append(avg_word_length)
            else:
                features.append(0.0)
            
            text_features.append(features)
        
        text_features = np.array(text_features)
        
        # Combine all features
        all_features = np.hstack([tfidf_features, text_features])
        
        logger.info(f"Extracted {all_features.shape[1]} features from {len(texts)} texts")
        return all_features
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                   optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train a specific model
        
        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Target labels
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Training results dictionary
        """
        if model_name not in self.models:
            raise ValueError(f"Unknown model: {model_name}")
        
        logger.info(f"Training {model_name} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get base model
        model = self.models[model_name]
        
        # Optimize hyperparameters if requested
        if optimize_hyperparameters:
            model = self._optimize_hyperparameters(model, model_name, X_train_scaled, y_train)
        
        # Train model
        start_time = datetime.now()
        model.fit(X_train_scaled, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
        
        # Evaluate model
        results = self._evaluate_model(y_test, y_pred, y_pred_proba, model.classes_)
        results.update({
            'model_name': model_name,
            'training_time': training_time,
            'model': model,
            'feature_importance': self._get_feature_importance(model, model_name)
        })
        
        # Store trained model
        self.models[model_name] = model
        
        logger.info(f"Training completed for {model_name}")
        return results
    
    def _optimize_hyperparameters(self, model, model_name: str, X: np.ndarray, y: np.ndarray):
        """Optimize hyperparameters using GridSearchCV"""
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'extra_trees': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
                'kernel': ['rbf', 'linear']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            },
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'decision_tree': {
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'naive_bayes': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'ada_boost': {
                'n_estimators': [25, 50, 100],
                'learning_rate': [0.5, 1.0, 1.5]
            }
        }
        
        if XGBOOST_AVAILABLE and model_name == 'xgboost':
            param_grids['xgboost'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        if LIGHTGBM_AVAILABLE and model_name == 'lightgbm':
            param_grids['lightgbm'] = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        if CATBOOST_AVAILABLE and model_name == 'catboost':
            param_grids['catboost'] = {
                'iterations': [50, 100, 200],
                'depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
        
        if model_name in param_grids:
            logger.info(f"Optimizing hyperparameters for {model_name}...")
            grid_search = GridSearchCV(
                model, param_grids[model_name], 
                cv=3, scoring='f1_macro', n_jobs=-1, verbose=0
            )
            grid_search.fit(X, y)
            return grid_search.best_estimator_
        
        return model
    
    def _evaluate_model(self, y_true, y_pred, y_pred_proba, classes) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        results = {
            'accuracy': accuracy_score(y_true, y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'precision_micro': precision_score(y_true, y_pred, average='micro', zero_division=0),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_micro': recall_score(y_true, y_pred, average='micro', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'matthews_corrcoef': matthews_corrcoef(y_true, y_pred),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Add ROC AUC if probabilities are available
        if y_pred_proba is not None and len(classes) > 2:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
            except:
                results['roc_auc'] = 0.0
        elif y_pred_proba is not None and len(classes) == 2:
            try:
                results['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                results['roc_auc'] = 0.0
        else:
            results['roc_auc'] = 0.0
        
        return results
    
    def _get_feature_importance(self, model, model_name: str) -> Optional[Dict[str, float]]:
        """Get feature importance if available"""
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                if self.feature_names is not None:
                    return dict(zip(self.feature_names, importance))
                else:
                    return {f'feature_{i}': imp for i, imp in enumerate(importance)}
            elif hasattr(model, 'coef_'):
                # For linear models, use absolute coefficients
                coef = np.abs(model.coef_[0]) if len(model.coef_.shape) > 1 else np.abs(model.coef_)
                if self.feature_names is not None:
                    return dict(zip(self.feature_names, coef))
                else:
                    return {f'feature_{i}': imp for i, imp in enumerate(coef)}
        except:
            pass
        return None
    
    def compare_models(self, X: np.ndarray, y: np.ndarray, 
                      models_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple models using cross-validation
        
        Args:
            X: Feature matrix
            y: Target labels
            models_to_compare: List of model names to compare (None for all)
            
        Returns:
            Comparison results
        """
        if models_to_compare is None:
            models_to_compare = list(self.models.keys())
        
        logger.info(f"Comparing {len(models_to_compare)} models...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for model_name in models_to_compare:
            if model_name not in self.models:
                continue
            
            logger.info(f"Evaluating {model_name}...")
            model = self.models[model_name]
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1_macro')
            
            # Train and evaluate
            model.fit(X_scaled, y)
            y_pred = model.predict(X_scaled)
            
            results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'cv_scores': cv_scores.tolist(),
                'accuracy': accuracy_score(y, y_pred),
                'f1_macro': f1_score(y, y_pred, average='macro', zero_division=0),
                'training_time': 0  # Could be measured if needed
            }
        
        # Sort by F1 score
        sorted_results = dict(sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True))
        
        logger.info("Model comparison completed")
        return sorted_results
    
    def train_all_models(self, data: pd.DataFrame, optimize_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train all available models
        
        Args:
            data: Training data DataFrame
            optimize_hyperparameters: Whether to optimize hyperparameters
            
        Returns:
            Training results for all models
        """
        logger.info("Training all available models...")
        
        # Prepare data
        texts = data['text'].tolist()
        labels = data['sentiment'].tolist()
        
        # Extract features
        X = self.extract_features(texts)
        y = self.label_encoder.fit_transform(labels)
        
        # Store feature names for importance analysis
        if self.vectorizer is not None:
            tfidf_features = self.vectorizer.get_feature_names_out()
            additional_features = [
                'text_length', 'word_count', 'uppercase_count', 'digit_count', 
                'punctuation_count', 'polarity', 'confidence', 'subjectivity',
                'joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'avg_word_length'
            ]
            self.feature_names = list(tfidf_features) + additional_features
        
        # Train all models
        all_results = {}
        for model_name in self.models.keys():
            try:
                results = self.train_model(model_name, X, y, optimize_hyperparameters)
                all_results[model_name] = results
                logger.info(f"✅ {model_name}: F1={results['f1_macro']:.3f}, Accuracy={results['accuracy']:.3f}")
            except Exception as e:
                logger.error(f"❌ Failed to train {model_name}: {e}")
                all_results[model_name] = {'error': str(e)}
        
        # Store training data
        self.training_data = data
        
        logger.info("All models training completed")
        return all_results
    
    def predict_sentiment(self, text: str, model_name: str = 'random_forest') -> Dict[str, Any]:
        """
        Predict sentiment for a single text using trained model
        
        Args:
            text: Text to analyze
            model_name: Name of the model to use
            
        Returns:
            Prediction results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        if self.vectorizer is None:
            raise ValueError("No trained model found. Please train a model first.")
        
        # Extract features
        X = self.extract_features([text])
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        model = self.models[model_name]
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0] if hasattr(model, 'predict_proba') else None
        
        # Decode prediction
        sentiment = self.label_encoder.inverse_transform([prediction])[0]
        
        result = {
            'text': text,
            'sentiment': sentiment,
            'confidence': float(probabilities[prediction]) if probabilities is not None else 0.0,
            'probabilities': {
                label: float(prob) for label, prob in zip(self.label_encoder.classes_, probabilities)
            } if probabilities is not None else None,
            'model_used': model_name
        }
        
        return result
    
    def save_model(self, model_name: str, filepath: str):
        """Save trained model to file"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model_data = {
            'model': self.models[model_name],
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'vectorizer': self.vectorizer,
            'feature_names': self.feature_names,
            'feature_params': self.feature_params,
            'training_data_info': {
                'num_samples': len(self.training_data) if self.training_data is not None else 0,
                'features': X.shape[1] if hasattr(self, 'X') else 0
            } if self.training_data is not None else None
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.models['loaded'] = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.vectorizer = model_data['vectorizer']
        self.feature_names = model_data['feature_names']
        self.feature_params = model_data['feature_params']
        
        logger.info(f"Model loaded from {filepath}")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training configuration and available models"""
        return {
            'available_models': list(self.models.keys()),
            'xgboost_available': XGBOOST_AVAILABLE,
            'lightgbm_available': LIGHTGBM_AVAILABLE,
            'catboost_available': CATBOOST_AVAILABLE,
            'plotting_available': PLOTTING_AVAILABLE,
            'feature_params': self.feature_params,
            'training_data_samples': len(self.training_data) if self.training_data is not None else 0,
            'model_cache_dir': str(self.model_cache_dir)
        }


def main():
    """Demo function to showcase SentimentsAI ML training capabilities"""
    print("🤖 SentilensAI - Machine Learning Training Pipeline")
    print("=" * 60)
    
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
    training_data = trainer.create_synthetic_training_data(num_samples=500)
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
    
    # Test prediction
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
        except Exception as e:
            print(f"Prediction failed: {e}")
        print()
    
    # Save best model
    best_model = max(results.keys(), key=lambda k: results[k].get('f1_macro', 0) if 'error' not in results[k] else 0)
    if 'error' not in results[best_model]:
        model_path = f"sentiments_ai_{best_model}_model.pkl"
        trainer.save_model(best_model, model_path)
        print(f"💾 Best model ({best_model}) saved to {model_path}")
    
    print("\n✅ SentilensAI ML training demo completed!")
    print("🚀 Ready for production sentiment analysis!")


if __name__ == "__main__":
    main()
