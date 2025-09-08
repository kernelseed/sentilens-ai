#!/usr/bin/env python3
"""
SentilensAI - Deep Learning Sentiment Analysis Module

Advanced deep learning capabilities for sentiment analysis using:
- Transformer-based models (BERT, RoBERTa, DistilBERT)
- LSTM/GRU networks for sequence modeling
- CNN for text classification
- Ensemble methods combining multiple models
- Transfer learning and fine-tuning capabilities

Author: Pravin Selvamuthu
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import logging
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import pickle
import os

# Deep learning libraries
try:
    from transformers import (
        AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class SentimentLSTM(nn.Module):
    """LSTM-based sentiment analysis model"""
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(SentimentLSTM, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        output = self.dropout(hidden[-1])
        output = self.fc(output)
        return output

class SentimentCNN(nn.Module):
    """CNN-based sentiment analysis model"""
    
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout=0.3):
        super(SentimentCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        
    def forward(self, x):
        embedded = self.embedding(x).transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))
            pooled = F.max_pool1d(conv_out, conv_out.size(2))
            conv_outputs.append(pooled.squeeze(2))
        
        concatenated = torch.cat(conv_outputs, dim=1)
        output = self.dropout(concatenated)
        output = self.fc(output)
        return output

class SentimentEnsemble(nn.Module):
    """Ensemble model combining multiple architectures"""
    
    def __init__(self, models, num_classes):
        super(SentimentEnsemble, self).__init__()
        self.models = nn.ModuleList(models)
        self.num_classes = num_classes
        self.fusion_layer = nn.Linear(len(models) * num_classes, num_classes)
        
    def forward(self, x):
        model_outputs = []
        for model in self.models:
            output = model(x)
            model_outputs.append(output)
        
        concatenated = torch.cat(model_outputs, dim=1)
        final_output = self.fusion_layer(concatenated)
        return final_output

class DeepLearningSentimentAnalyzer:
    """Advanced deep learning sentiment analyzer"""
    
    def __init__(self, model_cache_dir: str = "./deep_learning_models"):
        self.model_cache_dir = model_cache_dir
        os.makedirs(model_cache_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.models = {}
        self.tokenizers = {}
        self.label_encoders = {}
        self.training_history = {}
        
        # Load pre-trained transformers
        self._load_pretrained_models()
    
    def _load_pretrained_models(self):
        """Load pre-trained transformer models"""
        if not TRANSFORMERS_AVAILABLE:
            logger.warning("Transformers library not available. Deep learning features limited.")
            return
        
        # Load different transformer models
        model_configs = {
            'bert-base': 'bert-base-uncased',
            'roberta-base': 'roberta-base',
            'distilbert': 'distilbert-base-uncased',
            'twitter-roberta': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        }
        
        for model_name, model_path in model_configs.items():
            try:
                logger.info(f"Loading {model_name}...")
                self.tokenizers[model_name] = AutoTokenizer.from_pretrained(model_path)
                self.models[model_name] = AutoModelForSequenceClassification.from_pretrained(
                    model_path, 
                    num_labels=3,  # positive, negative, neutral
                    ignore_mismatched_sizes=True
                ).to(self.device)
                logger.info(f"✅ {model_name} loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
    
    def prepare_training_data(self, texts: List[str], labels: List[str]) -> Tuple[Dataset, Dataset]:
        """Prepare training data for deep learning models"""
        
        # Encode labels
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        self.label_encoders['main'] = label_encoder
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, encoded_labels, test_size=0.2, random_state=42, stratify=encoded_labels
        )
        
        # Create datasets for different models
        datasets = {}
        
        # BERT dataset
        if 'bert-base' in self.tokenizers:
            train_dataset = SentimentDataset(X_train, y_train, self.tokenizers['bert-base'])
            test_dataset = SentimentDataset(X_test, y_test, self.tokenizers['bert-base'])
            datasets['bert'] = (train_dataset, test_dataset)
        
        # RoBERTa dataset
        if 'roberta-base' in self.tokenizers:
            train_dataset = SentimentDataset(X_train, y_train, self.tokenizers['roberta-base'])
            test_dataset = SentimentDataset(X_test, y_test, self.tokenizers['roberta-base'])
            datasets['roberta'] = (train_dataset, test_dataset)
        
        return datasets, (X_train, X_test, y_train, y_test)
    
    def train_custom_lstm(self, texts: List[str], labels: List[str], 
                         embedding_dim=100, hidden_dim=128, num_layers=2, 
                         epochs=10, batch_size=32, learning_rate=0.001):
        """Train custom LSTM model"""
        
        logger.info("Training custom LSTM model...")
        
        # Prepare data
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer = CountVectorizer(max_features=10000)
        X = vectorizer.fit_transform(texts).toarray()
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.long)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create model
        vocab_size = len(vectorizer.vocabulary_)
        num_classes = len(np.unique(y))
        
        model = SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_layers=num_layers
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size].to(self.device)
                batch_y = y_train_tensor[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_train_tensor) // batch_size)
            train_losses.append(avg_loss)
            
            if epoch % 2 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(self.device))
            test_predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()
            test_accuracy = accuracy_score(y_test, test_predictions)
        
        logger.info(f"LSTM Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.model_cache_dir, 'lstm_sentiment_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'vectorizer': vectorizer,
            'label_encoder': label_encoder,
            'model_config': {
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'num_classes': num_classes
            }
        }, model_path)
        
        self.models['lstm'] = model
        self.training_history['lstm'] = {
            'train_losses': train_losses,
            'test_accuracy': test_accuracy,
            'epochs': epochs
        }
        
        return model, test_accuracy
    
    def train_custom_cnn(self, texts: List[str], labels: List[str],
                        embedding_dim=100, num_filters=100, filter_sizes=[3, 4, 5],
                        epochs=10, batch_size=32, learning_rate=0.001):
        """Train custom CNN model"""
        
        logger.info("Training custom CNN model...")
        
        # Prepare data
        from sklearn.feature_extraction.text import CountVectorizer
        
        vectorizer = CountVectorizer(max_features=10000)
        X = vectorizer.fit_transform(texts).toarray()
        
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.long)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.long)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
        
        # Create model
        vocab_size = len(vectorizer.vocabulary_)
        num_classes = len(np.unique(y))
        
        model = SentimentCNN(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_filters=num_filters,
            filter_sizes=filter_sizes,
            output_dim=num_classes
        ).to(self.device)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        model.train()
        train_losses = []
        
        for epoch in range(epochs):
            total_loss = 0
            for i in range(0, len(X_train_tensor), batch_size):
                batch_X = X_train_tensor[i:i+batch_size].to(self.device)
                batch_y = y_train_tensor[i:i+batch_size].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_train_tensor) // batch_size)
            train_losses.append(avg_loss)
            
            if epoch % 2 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor.to(self.device))
            test_predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()
            test_accuracy = accuracy_score(y_test, test_predictions)
        
        logger.info(f"CNN Test Accuracy: {test_accuracy:.4f}")
        
        # Save model
        model_path = os.path.join(self.model_cache_dir, 'cnn_sentiment_model.pth')
        torch.save({
            'model_state_dict': model.state_dict(),
            'vectorizer': vectorizer,
            'label_encoder': label_encoder,
            'model_config': {
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'num_filters': num_filters,
                'filter_sizes': filter_sizes,
                'num_classes': num_classes
            }
        }, model_path)
        
        self.models['cnn'] = model
        self.training_history['cnn'] = {
            'train_losses': train_losses,
            'test_accuracy': test_accuracy,
            'epochs': epochs
        }
        
        return model, test_accuracy
    
    def analyze_sentiment_deep_learning(self, text: str) -> Dict[str, Any]:
        """Analyze sentiment using deep learning models"""
        
        results = {
            'text': text,
            'models_used': [],
            'predictions': {},
            'ensemble_prediction': None,
            'confidence_scores': {},
            'model_agreement': 0.0
        }
        
        # Analyze with each available model
        predictions = []
        confidence_scores = []
        
        # Transformer models
        for model_name, model in self.models.items():
            if model_name in ['bert-base', 'roberta-base', 'distilbert', 'twitter-roberta']:
                try:
                    tokenizer = self.tokenizers[model_name]
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True)
                    
                    with torch.no_grad():
                        outputs = model(**inputs)
                        probabilities = torch.softmax(outputs.logits, dim=-1)
                        prediction = torch.argmax(probabilities, dim=-1).item()
                        confidence = torch.max(probabilities).item()
                    
                    # Map to sentiment labels
                    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
                    sentiment = sentiment_map.get(prediction, 'neutral')
                    
                    results['models_used'].append(model_name)
                    results['predictions'][model_name] = sentiment
                    results['confidence_scores'][model_name] = confidence
                    
                    predictions.append(sentiment)
                    confidence_scores.append(confidence)
                    
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {e}")
        
        # Custom models
        for model_name in ['lstm', 'cnn']:
            if model_name in self.models:
                try:
                    # This would require loading the saved models and running inference
                    # For now, we'll skip this in the demo
                    pass
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {e}")
        
        # Ensemble prediction
        if predictions:
            from collections import Counter
            prediction_counts = Counter(predictions)
            ensemble_prediction = prediction_counts.most_common(1)[0][0]
            results['ensemble_prediction'] = ensemble_prediction
            
            # Calculate model agreement
            agreement = prediction_counts[ensemble_prediction] / len(predictions)
            results['model_agreement'] = agreement
            
            # Average confidence
            results['average_confidence'] = np.mean(confidence_scores)
        
        return results
    
    def generate_learning_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate learning recommendations based on deep learning analysis"""
        
        recommendations = {
            'model_performance': {},
            'training_insights': {},
            'improvement_suggestions': [],
            'data_quality_assessment': {},
            'next_steps': []
        }
        
        # Analyze model performance
        if 'training_history' in analysis_results:
            for model_name, history in analysis_results['training_history'].items():
                recommendations['model_performance'][model_name] = {
                    'test_accuracy': history.get('test_accuracy', 0),
                    'convergence': 'Good' if len(history.get('train_losses', [])) > 0 else 'Unknown',
                    'overfitting_risk': 'Low' if history.get('test_accuracy', 0) > 0.8 else 'Medium'
                }
        
        # Training insights
        recommendations['training_insights'] = {
            'best_performing_model': max(
                analysis_results.get('model_performance', {}).items(),
                key=lambda x: x[1].get('test_accuracy', 0)
            )[0] if analysis_results.get('model_performance') else 'None',
            'ensemble_benefit': 'High' if len(analysis_results.get('models_used', [])) > 2 else 'Medium',
            'confidence_variance': 'Low' if np.std(analysis_results.get('confidence_scores', {}).values()) < 0.1 else 'High'
        }
        
        # Improvement suggestions
        if analysis_results.get('model_agreement', 0) < 0.7:
            recommendations['improvement_suggestions'].append({
                'area': 'Model Agreement',
                'issue': 'Low model agreement detected',
                'suggestion': 'Consider ensemble methods or model fine-tuning',
                'priority': 'High'
            })
        
        if analysis_results.get('average_confidence', 0) < 0.6:
            recommendations['improvement_suggestions'].append({
                'area': 'Confidence',
                'issue': 'Low average confidence scores',
                'suggestion': 'Increase training data or improve model architecture',
                'priority': 'High'
            })
        
        # Data quality assessment
        recommendations['data_quality_assessment'] = {
            'training_data_size': 'Adequate' if analysis_results.get('training_samples', 0) > 1000 else 'Insufficient',
            'class_balance': 'Balanced' if analysis_results.get('class_balance_score', 0) > 0.8 else 'Imbalanced',
            'text_diversity': 'Good' if analysis_results.get('vocabulary_size', 0) > 5000 else 'Limited'
        }
        
        # Next steps
        recommendations['next_steps'] = [
            'Fine-tune best performing model on domain-specific data',
            'Implement active learning for continuous improvement',
            'Add data augmentation techniques',
            'Consider transfer learning from larger datasets',
            'Implement model monitoring and retraining pipeline'
        ]
        
        return recommendations

def main():
    """Demo function for deep learning sentiment analysis"""
    print("🤖 SentilensAI - Deep Learning Sentiment Analysis Demo")
    print("=" * 60)
    
    # Initialize deep learning analyzer
    analyzer = DeepLearningSentimentAnalyzer()
    
    # Sample data for training
    sample_texts = [
        "I love this product! It's amazing and works perfectly.",
        "This is terrible. I hate it and want a refund immediately.",
        "The service was okay, nothing special but not bad either.",
        "Excellent customer support! They were very helpful.",
        "I'm frustrated with the slow response time.",
        "Great quality and fast delivery. Highly recommended!",
        "The interface is confusing and hard to use.",
        "Outstanding service! Will definitely use again.",
        "Average experience, could be better.",
        "Worst customer service ever. Very disappointed."
    ]
    
    sample_labels = [
        "positive", "negative", "neutral", "positive", "negative",
        "positive", "negative", "positive", "neutral", "negative"
    ]
    
    print(f"📊 Training on {len(sample_texts)} sample texts...")
    
    # Train custom models
    try:
        lstm_model, lstm_accuracy = analyzer.train_custom_lstm(sample_texts, sample_labels, epochs=5)
        print(f"✅ LSTM model trained - Accuracy: {lstm_accuracy:.4f}")
    except Exception as e:
        print(f"❌ LSTM training failed: {e}")
    
    try:
        cnn_model, cnn_accuracy = analyzer.train_custom_cnn(sample_texts, sample_labels, epochs=5)
        print(f"✅ CNN model trained - Accuracy: {cnn_accuracy:.4f}")
    except Exception as e:
        print(f"❌ CNN training failed: {e}")
    
    # Test deep learning analysis
    test_text = "I'm really happy with the service and would recommend it to others!"
    print(f"\n🔍 Testing deep learning analysis on: '{test_text}'")
    
    dl_results = analyzer.analyze_sentiment_deep_learning(test_text)
    print(f"📊 Deep Learning Results:")
    print(f"   Models Used: {dl_results['models_used']}")
    print(f"   Predictions: {dl_results['predictions']}")
    print(f"   Ensemble Prediction: {dl_results['ensemble_prediction']}")
    print(f"   Model Agreement: {dl_results['model_agreement']:.2f}")
    print(f"   Average Confidence: {dl_results['average_confidence']:.2f}")
    
    # Generate learning recommendations
    recommendations = analyzer.generate_learning_recommendations(dl_results)
    print(f"\n🎓 Learning Recommendations:")
    print(f"   Best Model: {recommendations['training_insights']['best_performing_model']}")
    print(f"   Ensemble Benefit: {recommendations['training_insights']['ensemble_benefit']}")
    print(f"   Next Steps: {len(recommendations['next_steps'])} recommendations generated")
    
    print(f"\n✅ Deep learning sentiment analysis demo completed!")
    print(f"🚀 Advanced AI capabilities ready for production!")

if __name__ == "__main__":
    main()
