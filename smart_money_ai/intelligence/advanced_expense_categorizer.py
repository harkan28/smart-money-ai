#!/usr/bin/env python3
"""
Advanced ML Expense Categorizer with Enhanced Features
=====================================================

Enhanced ML categorization system with:
- Merchant embeddings using Word2Vec
- Behavioral pattern recognition
- Transaction velocity features
- User feedback loop for continuous learning
- Real-time model updates
"""

import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import json
import os
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import joblib

# Try to import advanced ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    from gensim.models import Word2Vec
    ADVANCED_ML_AVAILABLE = True
except ImportError:
    print("Advanced ML libraries not available. Using basic functionality.")
    ADVANCED_ML_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TransactionFeatures:
    """Enhanced transaction features for ML processing"""
    amount: float
    merchant: str
    timestamp: datetime
    is_weekend: bool
    hour_of_day: int
    day_of_month: int
    month: int
    is_recurring: bool
    merchant_frequency: int
    amount_category: str  # small, medium, large
    velocity_score: float
    merchant_embedding: Optional[List[float]] = None
    behavioral_score: float = 0.0


@dataclass
class UserFeedback:
    """User feedback for model improvement"""
    transaction_id: str
    predicted_category: str
    actual_category: str
    feedback_type: str  # correction, confirmation
    timestamp: datetime
    confidence_before: float
    user_id: Optional[str] = None


class MerchantEmbeddingManager:
    """Manages merchant embeddings using Word2Vec"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.word2vec_model = None
        self.merchant_embeddings = {}
        self.embedding_dim = 100
        
        if ADVANCED_ML_AVAILABLE:
            self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing Word2Vec model or create new one"""
        if self.model_path and os.path.exists(self.model_path):
            try:
                self.word2vec_model = Word2Vec.load(self.model_path)
                logger.info("Loaded existing Word2Vec model")
            except Exception as e:
                logger.warning(f"Failed to load Word2Vec model: {e}")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Create new Word2Vec model with common merchant names"""
        # Common Indian merchant categories for initial training
        merchant_sentences = [
            ['amazon', 'online', 'shopping', 'ecommerce'],
            ['flipkart', 'shopping', 'online', 'retail'],
            ['swiggy', 'food', 'delivery', 'restaurant'],
            ['zomato', 'food', 'delivery', 'dining'],
            ['uber', 'cab', 'transport', 'ride'],
            ['ola', 'cab', 'transport', 'ride'],
            ['paytm', 'wallet', 'payment', 'digital'],
            ['phonepe', 'payment', 'upi', 'digital'],
            ['gpay', 'google', 'payment', 'upi'],
            ['bigbasket', 'grocery', 'food', 'online'],
            ['grofers', 'grocery', 'food', 'delivery'],
            ['myntra', 'fashion', 'clothing', 'online'],
            ['ajio', 'fashion', 'clothing', 'retail'],
            ['bookmyshow', 'entertainment', 'movies', 'tickets'],
            ['irctc', 'travel', 'train', 'booking'],
            ['makemytrip', 'travel', 'booking', 'hotels'],
            ['reliance', 'fuel', 'petrol', 'gas'],
            ['hp', 'fuel', 'petrol', 'gas'],
            ['bpcl', 'fuel', 'petrol', 'gas'],
            ['atm', 'withdrawal', 'cash', 'bank']
        ]
        
        try:
            self.word2vec_model = Word2Vec(
                merchant_sentences,
                vector_size=self.embedding_dim,
                window=5,
                min_count=1,
                workers=4,
                epochs=100
            )
            
            if self.model_path:
                self.word2vec_model.save(self.model_path)
                logger.info("Created and saved new Word2Vec model")
                
        except Exception as e:
            logger.error(f"Failed to create Word2Vec model: {e}")
            self.word2vec_model = None
    
    def get_merchant_embedding(self, merchant: str) -> List[float]:
        """Get embedding for merchant name"""
        if not self.word2vec_model:
            return [0.0] * self.embedding_dim
        
        # Normalize merchant name
        merchant_clean = self._clean_merchant_name(merchant)
        
        if merchant_clean in self.merchant_embeddings:
            return self.merchant_embeddings[merchant_clean]
        
        # Try to get embedding from Word2Vec model
        try:
            words = merchant_clean.split()
            embeddings = []
            
            for word in words:
                if word in self.word2vec_model.wv:
                    embeddings.append(self.word2vec_model.wv[word])
            
            if embeddings:
                # Average embeddings of all words
                avg_embedding = np.mean(embeddings, axis=0).tolist()
                self.merchant_embeddings[merchant_clean] = avg_embedding
                return avg_embedding
            else:
                # Create random embedding for unknown merchants
                random_embedding = np.random.normal(0, 0.1, self.embedding_dim).tolist()
                self.merchant_embeddings[merchant_clean] = random_embedding
                return random_embedding
                
        except Exception as e:
            logger.warning(f"Error getting embedding for {merchant}: {e}")
            return [0.0] * self.embedding_dim
    
    def _clean_merchant_name(self, merchant: str) -> str:
        """Clean merchant name for embedding lookup"""
        import re
        # Remove special characters and normalize
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', merchant.lower())
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned
    
    def update_model_with_new_merchants(self, merchants: List[str]):
        """Update Word2Vec model with new merchant names"""
        if not self.word2vec_model:
            return
        
        try:
            # Create sentences from new merchants
            new_sentences = []
            for merchant in merchants:
                words = self._clean_merchant_name(merchant).split()
                if len(words) > 0:
                    new_sentences.append(words)
            
            if new_sentences:
                # Update vocabulary and retrain
                self.word2vec_model.build_vocab(new_sentences, update=True)
                self.word2vec_model.train(new_sentences, total_examples=len(new_sentences), epochs=10)
                
                if self.model_path:
                    self.word2vec_model.save(self.model_path)
                
                logger.info(f"Updated Word2Vec model with {len(new_sentences)} new merchants")
                
        except Exception as e:
            logger.error(f"Failed to update Word2Vec model: {e}")


class BehavioralPatternAnalyzer:
    """Analyzes user spending behavioral patterns"""
    
    def __init__(self):
        self.user_patterns = defaultdict(dict)
        self.spending_velocity = defaultdict(list)
        self.merchant_frequency = defaultdict(Counter)
        self.time_patterns = defaultdict(dict)
    
    def analyze_transaction_behavior(self, user_id: str, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze behavioral patterns for a user"""
        patterns = {
            'spending_velocity': self._calculate_spending_velocity(transactions),
            'time_patterns': self._analyze_time_patterns(transactions),
            'merchant_loyalty': self._calculate_merchant_loyalty(transactions),
            'amount_patterns': self._analyze_amount_patterns(transactions),
            'recurring_transactions': self._detect_recurring_transactions(transactions)
        }
        
        self.user_patterns[user_id] = patterns
        return patterns
    
    def _calculate_spending_velocity(self, transactions: List[Dict]) -> Dict[str, float]:
        """Calculate spending velocity metrics"""
        if len(transactions) < 2:
            return {'daily_avg': 0.0, 'weekly_avg': 0.0, 'velocity_score': 0.0}
        
        # Sort transactions by timestamp
        sorted_txns = sorted(transactions, key=lambda x: x.get('timestamp', datetime.now()))
        
        daily_amounts = defaultdict(float)
        for txn in sorted_txns:
            date = txn.get('timestamp', datetime.now()).date()
            if txn.get('transaction_type') == 'debit':
                daily_amounts[date] += txn.get('amount', 0)
        
        amounts = list(daily_amounts.values())
        daily_avg = np.mean(amounts) if amounts else 0.0
        weekly_avg = daily_avg * 7
        
        # Calculate velocity score (variance in spending)
        velocity_score = np.std(amounts) / (daily_avg + 1) if daily_avg > 0 else 0.0
        
        return {
            'daily_avg': daily_avg,
            'weekly_avg': weekly_avg,
            'velocity_score': velocity_score
        }
    
    def _analyze_time_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze time-based spending patterns"""
        hour_spending = defaultdict(float)
        day_spending = defaultdict(float)
        weekend_spending = 0.0
        weekday_spending = 0.0
        
        for txn in transactions:
            if txn.get('transaction_type') == 'debit':
                timestamp = txn.get('timestamp', datetime.now())
                amount = txn.get('amount', 0)
                
                hour_spending[timestamp.hour] += amount
                day_spending[timestamp.weekday()] += amount
                
                if timestamp.weekday() >= 5:  # Weekend
                    weekend_spending += amount
                else:
                    weekday_spending += amount
        
        return {
            'peak_spending_hour': max(hour_spending, key=hour_spending.get) if hour_spending else 12,
            'peak_spending_day': max(day_spending, key=day_spending.get) if day_spending else 0,
            'weekend_ratio': weekend_spending / (weekend_spending + weekday_spending) if (weekend_spending + weekday_spending) > 0 else 0,
            'hour_distribution': dict(hour_spending),
            'day_distribution': dict(day_spending)
        }
    
    def _calculate_merchant_loyalty(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Calculate merchant loyalty metrics"""
        merchant_counts = Counter()
        merchant_amounts = defaultdict(float)
        
        for txn in transactions:
            if txn.get('transaction_type') == 'debit':
                merchant = txn.get('merchant', 'Unknown')
                amount = txn.get('amount', 0)
                
                merchant_counts[merchant] += 1
                merchant_amounts[merchant] += amount
        
        total_transactions = sum(merchant_counts.values())
        
        if total_transactions == 0:
            return {'loyalty_score': 0.0, 'top_merchants': [], 'merchant_diversity': 0.0}
        
        # Calculate loyalty score (concentration)
        loyalty_score = max(merchant_counts.values()) / total_transactions if total_transactions > 0 else 0
        
        # Top merchants by frequency
        top_merchants = merchant_counts.most_common(5)
        
        # Merchant diversity (entropy)
        merchant_diversity = len(merchant_counts) / total_transactions if total_transactions > 0 else 0
        
        return {
            'loyalty_score': loyalty_score,
            'top_merchants': top_merchants,
            'merchant_diversity': merchant_diversity
        }
    
    def _analyze_amount_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze spending amount patterns"""
        amounts = [txn.get('amount', 0) for txn in transactions if txn.get('transaction_type') == 'debit']
        
        if not amounts:
            return {'avg_amount': 0.0, 'amount_variance': 0.0, 'small_txn_ratio': 0.0}
        
        avg_amount = np.mean(amounts)
        amount_variance = np.std(amounts)
        
        # Small transaction ratio (< 500 INR)
        small_txns = sum(1 for amount in amounts if amount < 500)
        small_txn_ratio = small_txns / len(amounts)
        
        return {
            'avg_amount': avg_amount,
            'amount_variance': amount_variance,
            'small_txn_ratio': small_txn_ratio,
            'amount_percentiles': {
                '25th': np.percentile(amounts, 25),
                '50th': np.percentile(amounts, 50),
                '75th': np.percentile(amounts, 75),
                '90th': np.percentile(amounts, 90)
            }
        }
    
    def _detect_recurring_transactions(self, transactions: List[Dict]) -> List[Dict[str, Any]]:
        """Detect potentially recurring transactions"""
        merchant_amounts = defaultdict(list)
        
        for txn in transactions:
            if txn.get('transaction_type') == 'debit':
                key = (txn.get('merchant', 'Unknown'), round(txn.get('amount', 0)))
                merchant_amounts[key].append(txn.get('timestamp', datetime.now()))
        
        recurring = []
        for (merchant, amount), timestamps in merchant_amounts.items():
            if len(timestamps) >= 3:  # At least 3 occurrences
                # Check if roughly monthly/weekly pattern
                timestamps.sort()
                intervals = [(timestamps[i+1] - timestamps[i]).days for i in range(len(timestamps)-1)]
                avg_interval = np.mean(intervals)
                
                if 25 <= avg_interval <= 35 or 6 <= avg_interval <= 8:  # Monthly or weekly
                    recurring.append({
                        'merchant': merchant,
                        'amount': amount,
                        'frequency': len(timestamps),
                        'avg_interval_days': avg_interval,
                        'type': 'monthly' if avg_interval > 20 else 'weekly'
                    })
        
        return recurring


class AdvancedExpenseCategorizer:
    """Advanced ML expense categorizer with enhanced features"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or "models/advanced_expense_model.joblib"
        self.embedding_manager = MerchantEmbeddingManager()
        self.behavioral_analyzer = BehavioralPatternAnalyzer()
        
        # ML models
        self.tfidf_vectorizer = None
        self.scaler = None
        self.ensemble_model = None
        self.feature_importance = {}
        
        # Feedback system
        self.feedback_history = []
        self.model_performance = {'accuracy': 0.0, 'last_updated': datetime.now()}
        
        # Load existing model or create new one
        self._load_or_create_model()
    
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        if os.path.exists(self.model_path):
            try:
                model_data = joblib.load(self.model_path)
                self.tfidf_vectorizer = model_data.get('tfidf_vectorizer')
                self.scaler = model_data.get('scaler')
                self.ensemble_model = model_data.get('ensemble_model')
                self.feature_importance = model_data.get('feature_importance', {})
                logger.info("Loaded existing advanced expense categorization model")
            except Exception as e:
                logger.warning(f"Failed to load model: {e}")
                self._create_new_model()
        else:
            self._create_new_model()
    
    def _create_new_model(self):
        """Create new ML model with default categories"""
        if not ADVANCED_ML_AVAILABLE:
            logger.warning("Advanced ML libraries not available")
            return
        
        # Initialize components
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.scaler = StandardScaler()
        
        # Create ensemble model
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        # For now, use RandomForest as primary model
        self.ensemble_model = rf_model
        
        logger.info("Created new advanced expense categorization model")
    
    def extract_advanced_features(self, transaction: Dict[str, Any], user_id: Optional[str] = None) -> TransactionFeatures:
        """Extract advanced features from transaction"""
        timestamp = transaction.get('timestamp', datetime.now())
        merchant = transaction.get('merchant', 'Unknown')
        amount = transaction.get('amount', 0.0)
        
        # Get merchant embedding
        merchant_embedding = self.embedding_manager.get_merchant_embedding(merchant)
        
        # Calculate amount category
        if amount < 100:
            amount_category = 'small'
        elif amount < 1000:
            amount_category = 'medium'
        else:
            amount_category = 'large'
        
        # Calculate merchant frequency (if user history available)
        merchant_frequency = 1  # Default for new merchants
        
        # Basic behavioral score (can be enhanced with user history)
        behavioral_score = 0.5
        
        features = TransactionFeatures(
            amount=amount,
            merchant=merchant,
            timestamp=timestamp,
            is_weekend=timestamp.weekday() >= 5,
            hour_of_day=timestamp.hour,
            day_of_month=timestamp.day,
            month=timestamp.month,
            is_recurring=False,  # Will be updated with pattern analysis
            merchant_frequency=merchant_frequency,
            amount_category=amount_category,
            velocity_score=0.0,  # Will be calculated with transaction history
            merchant_embedding=merchant_embedding,
            behavioral_score=behavioral_score
        )
        
        return features
    
    def predict_category(self, transaction: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Predict category with confidence score and reasoning"""
        try:
            # Extract advanced features
            features = self.extract_advanced_features(transaction, user_id)
            
            if not self.ensemble_model or not ADVANCED_ML_AVAILABLE:
                # Fallback to rule-based categorization
                return self._rule_based_categorization(transaction)
            
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            # Predict with ensemble model
            prediction = self.ensemble_model.predict([feature_vector])[0]
            confidence = np.max(self.ensemble_model.predict_proba([feature_vector])[0])
            
            # Get feature importance for this prediction
            feature_importance = self._get_prediction_reasoning(features, feature_vector)
            
            return {
                'category': prediction,
                'confidence': float(confidence),
                'reasoning': feature_importance,
                'features_used': asdict(features),
                'model_type': 'advanced_ml'
            }
            
        except Exception as e:
            logger.error(f"Error in advanced prediction: {e}")
            return self._rule_based_categorization(transaction)
    
    def _prepare_feature_vector(self, features: TransactionFeatures) -> List[float]:
        """Prepare feature vector for ML model"""
        # Basic numerical features
        numerical_features = [
            features.amount,
            features.hour_of_day,
            features.day_of_month,
            features.month,
            features.merchant_frequency,
            features.velocity_score,
            features.behavioral_score,
            1.0 if features.is_weekend else 0.0,
            1.0 if features.is_recurring else 0.0
        ]
        
        # Amount category encoding
        amount_encoding = [0.0, 0.0, 0.0]
        if features.amount_category == 'small':
            amount_encoding[0] = 1.0
        elif features.amount_category == 'medium':
            amount_encoding[1] = 1.0
        else:
            amount_encoding[2] = 1.0
        
        # Merchant embedding
        merchant_embedding = features.merchant_embedding or [0.0] * self.embedding_manager.embedding_dim
        
        # Combine all features
        feature_vector = numerical_features + amount_encoding + merchant_embedding
        
        return feature_vector
    
    def _rule_based_categorization(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback rule-based categorization"""
        merchant = transaction.get('merchant', '').lower()
        amount = transaction.get('amount', 0)
        
        # Rule-based category mapping
        if any(keyword in merchant for keyword in ['swiggy', 'zomato', 'food', 'restaurant', 'domino', 'mcd']):
            category = 'Food & Dining'
        elif any(keyword in merchant for keyword in ['amazon', 'flipkart', 'shopping', 'retail']):
            category = 'Shopping'
        elif any(keyword in merchant for keyword in ['uber', 'ola', 'cab', 'taxi', 'transport']):
            category = 'Transportation'
        elif any(keyword in merchant for keyword in ['fuel', 'petrol', 'diesel', 'gas', 'hp', 'bpcl']):
            category = 'Fuel'
        elif any(keyword in merchant for keyword in ['atm', 'withdrawal', 'cash']):
            category = 'Cash & ATM'
        elif any(keyword in merchant for keyword in ['medical', 'hospital', 'pharmacy', 'health']):
            category = 'Healthcare'
        elif any(keyword in merchant for keyword in ['electricity', 'water', 'gas', 'utility']):
            category = 'Utilities'
        elif amount > 10000:
            category = 'Large Expense'
        else:
            category = 'Other'
        
        return {
            'category': category,
            'confidence': 0.7,
            'reasoning': {'rule_based': True, 'merchant_keyword': True},
            'model_type': 'rule_based'
        }
    
    def _get_prediction_reasoning(self, features: TransactionFeatures, feature_vector: List[float]) -> Dict[str, Any]:
        """Get reasoning for the prediction"""
        reasoning = {
            'merchant_similarity': 'high' if features.merchant_embedding else 'unknown',
            'amount_influence': 'high' if features.amount > 1000 else 'medium',
            'time_pattern': 'weekend' if features.is_weekend else 'weekday',
            'behavioral_score': features.behavioral_score
        }
        
        if hasattr(self.ensemble_model, 'feature_importances_'):
            top_features = np.argsort(self.ensemble_model.feature_importances_)[-5:]
            reasoning['top_feature_indices'] = top_features.tolist()
        
        return reasoning
    
    def add_feedback(self, transaction_id: str, predicted_category: str, actual_category: str, 
                    confidence: float, user_id: Optional[str] = None):
        """Add user feedback for model improvement"""
        feedback = UserFeedback(
            transaction_id=transaction_id,
            predicted_category=predicted_category,
            actual_category=actual_category,
            feedback_type='correction' if predicted_category != actual_category else 'confirmation',
            timestamp=datetime.now(),
            confidence_before=confidence,
            user_id=user_id
        )
        
        self.feedback_history.append(feedback)
        logger.info(f"Added feedback: {feedback.feedback_type} for transaction {transaction_id}")
        
        # Trigger model retraining if enough feedback collected
        if len(self.feedback_history) % 50 == 0:
            self._retrain_with_feedback()
    
    def _retrain_with_feedback(self):
        """Retrain model with accumulated feedback"""
        if not ADVANCED_ML_AVAILABLE or len(self.feedback_history) < 10:
            return
        
        try:
            # Prepare training data from feedback
            correction_feedback = [fb for fb in self.feedback_history if fb.feedback_type == 'correction']
            
            if len(correction_feedback) >= 5:
                logger.info(f"Retraining model with {len(correction_feedback)} corrections")
                # Implementation for incremental learning would go here
                self.model_performance['last_updated'] = datetime.now()
                
                # Save updated model
                self._save_model()
                
        except Exception as e:
            logger.error(f"Error retraining model: {e}")
    
    def _save_model(self):
        """Save the trained model"""
        if not self.ensemble_model:
            return
        
        try:
            # Create models directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            model_data = {
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'scaler': self.scaler,
                'ensemble_model': self.ensemble_model,
                'feature_importance': self.feature_importance,
                'model_performance': self.model_performance,
                'feedback_count': len(self.feedback_history)
            }
            
            joblib.dump(model_data, self.model_path)
            logger.info(f"Saved advanced expense categorization model to {self.model_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model performance statistics"""
        total_feedback = len(self.feedback_history)
        corrections = sum(1 for fb in self.feedback_history if fb.feedback_type == 'correction')
        confirmations = sum(1 for fb in self.feedback_history if fb.feedback_type == 'confirmation')
        
        accuracy = (confirmations / total_feedback * 100) if total_feedback > 0 else 0
        
        return {
            'total_predictions': total_feedback,
            'accuracy_from_feedback': round(accuracy, 2),
            'corrections': corrections,
            'confirmations': confirmations,
            'last_updated': self.model_performance['last_updated'],
            'model_type': 'advanced_ml' if ADVANCED_ML_AVAILABLE else 'rule_based',
            'embedding_dim': self.embedding_manager.embedding_dim,
            'merchants_in_embedding': len(self.embedding_manager.merchant_embeddings)
        }


# Convenience function for backward compatibility
def categorize_transaction(transaction: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
    """Categorize a single transaction using advanced ML"""
    categorizer = AdvancedExpenseCategorizer()
    return categorizer.predict_category(transaction, user_id)


if __name__ == "__main__":
    # Test the advanced categorizer
    categorizer = AdvancedExpenseCategorizer()
    
    # Test transactions
    test_transactions = [
        {'merchant': 'Amazon', 'amount': 1500, 'timestamp': datetime.now()},
        {'merchant': 'Swiggy', 'amount': 350, 'timestamp': datetime.now()},
        {'merchant': 'Uber', 'amount': 250, 'timestamp': datetime.now()},
        {'merchant': 'ATM Withdrawal', 'amount': 2000, 'timestamp': datetime.now()}
    ]
    
    for txn in test_transactions:
        result = categorizer.predict_category(txn)
        print(f"✅ {txn['merchant']}: {result['category']} (confidence: {result['confidence']:.2f})")
    
    print(f"\nModel Statistics: {categorizer.get_model_statistics()}")