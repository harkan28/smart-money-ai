#!/usr/bin/env python3
"""
Enhanced ML Expense Categorizer - Backend Core
==============================================

Advanced backend ML system with:
- Merchant embeddings using Word2Vec/TF-IDF
- Behavioral pattern recognition
- Transaction velocity analysis
- User feedback loop for continuous learning
- Performance optimization
"""

import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Core ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Advanced embeddings (optional)
try:
    from gensim.models import Word2Vec
    WORD2VEC_AVAILABLE = True
except ImportError:
    WORD2VEC_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TransactionPattern:
    """Transaction pattern data structure"""
    merchant: str
    amount: float
    category: str
    frequency: int
    last_seen: datetime
    confidence: float


class MerchantEmbeddingEngine:
    """Advanced merchant embedding system with fallback"""
    
    def __init__(self, embedding_size: int = 100):
        self.embedding_size = embedding_size
        self.word2vec_model = None
        self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        self.merchant_embeddings = {}
        self.is_trained = False
        
    def train_embeddings(self, merchants: List[str]) -> bool:
        """Train merchant embeddings with multiple fallbacks"""
        try:
            # Method 1: Word2Vec (if available and sufficient data)
            if WORD2VEC_AVAILABLE and len(merchants) > 20:
                return self._train_word2vec(merchants)
            
            # Method 2: TF-IDF based embeddings
            return self._train_tfidf_embeddings(merchants)
            
        except Exception as e:
            logger.error(f"Error training embeddings: {e}")
            return self._create_hash_embeddings(merchants)
    
    def _train_word2vec(self, merchants: List[str]) -> bool:
        """Train Word2Vec embeddings"""
        try:
            # Prepare corpus
            corpus = []
            for merchant in merchants:
                tokens = merchant.lower().replace('-', ' ').replace('_', ' ').split()
                if tokens:
                    corpus.append(tokens)
            
            if len(corpus) < 10:
                return self._train_tfidf_embeddings(merchants)
            
            # Train Word2Vec
            self.word2vec_model = Word2Vec(
                sentences=corpus,
                vector_size=self.embedding_size,
                window=3,
                min_count=1,
                workers=2,
                epochs=10
            )
            
            # Create embeddings
            for merchant in merchants:
                self.merchant_embeddings[merchant] = self._get_word2vec_embedding(merchant)
            
            self.is_trained = True
            logger.info(f"Trained Word2Vec embeddings for {len(merchants)} merchants")
            return True
            
        except Exception as e:
            logger.error(f"Word2Vec training failed: {e}")
            return self._train_tfidf_embeddings(merchants)
    
    def _train_tfidf_embeddings(self, merchants: List[str]) -> bool:
        """Train TF-IDF based embeddings"""
        try:
            if SKLEARN_AVAILABLE:
                # Fit TF-IDF
                self.tfidf_vectorizer.fit(merchants)
                
                # Create embeddings
                tfidf_matrix = self.tfidf_vectorizer.transform(merchants)
                
                for i, merchant in enumerate(merchants):
                    # Convert sparse to dense and pad/truncate to desired size
                    tfidf_vector = tfidf_matrix[i].toarray().flatten()
                    
                    if len(tfidf_vector) > self.embedding_size:
                        embedding = tfidf_vector[:self.embedding_size]
                    else:
                        embedding = np.pad(tfidf_vector, (0, max(0, self.embedding_size - len(tfidf_vector))))
                    
                    self.merchant_embeddings[merchant] = embedding
                
                self.is_trained = True
                logger.info(f"Trained TF-IDF embeddings for {len(merchants)} merchants")
                return True
            else:
                return self._create_hash_embeddings(merchants)
                
        except Exception as e:
            logger.error(f"TF-IDF training failed: {e}")
            return self._create_hash_embeddings(merchants)
    
    def _create_hash_embeddings(self, merchants: List[str]) -> bool:
        """Create hash-based embeddings as final fallback"""
        try:
            for merchant in merchants:
                # Create deterministic hash-based embedding
                merchant_hash = hash(merchant.lower()) % (2**31)
                np.random.seed(merchant_hash)
                embedding = np.random.randn(self.embedding_size)
                self.merchant_embeddings[merchant] = embedding
            
            self.is_trained = True
            logger.info(f"Created hash-based embeddings for {len(merchants)} merchants")
            return True
            
        except Exception as e:
            logger.error(f"Hash embedding creation failed: {e}")
            return False
    
    def _get_word2vec_embedding(self, merchant: str) -> np.ndarray:
        """Get Word2Vec embedding for merchant"""
        tokens = merchant.lower().replace('-', ' ').replace('_', ' ').split()
        embeddings = []
        
        for token in tokens:
            if token in self.word2vec_model.wv:
                embeddings.append(self.word2vec_model.wv[token])
        
        if embeddings:
            return np.mean(embeddings, axis=0)
        else:
            return np.zeros(self.embedding_size)
    
    def get_embedding(self, merchant: str) -> np.ndarray:
        """Get embedding for merchant with smart fallback"""
        if merchant in self.merchant_embeddings:
            return self.merchant_embeddings[merchant]
        
        # Generate embedding for new merchant
        if self.word2vec_model and WORD2VEC_AVAILABLE:
            embedding = self._get_word2vec_embedding(merchant)
        elif SKLEARN_AVAILABLE and hasattr(self.tfidf_vectorizer, 'vocabulary_'):
            # TF-IDF embedding for new merchant
            tfidf_vector = self.tfidf_vectorizer.transform([merchant]).toarray().flatten()
            if len(tfidf_vector) > self.embedding_size:
                embedding = tfidf_vector[:self.embedding_size]
            else:
                embedding = np.pad(tfidf_vector, (0, max(0, self.embedding_size - len(tfidf_vector))))
        else:
            # Hash-based fallback
            merchant_hash = hash(merchant.lower()) % (2**31)
            np.random.seed(merchant_hash)
            embedding = np.random.randn(self.embedding_size)
        
        self.merchant_embeddings[merchant] = embedding
        return embedding


class BehavioralAnalysisEngine:
    """Analyze spending behavior patterns"""
    
    def __init__(self):
        self.transaction_patterns = {}
        self.velocity_window = 30  # days
        self.pattern_cache = {}
    
    def analyze_spending_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Comprehensive behavioral pattern analysis"""
        if not transactions:
            return self._get_default_patterns()
        
        try:
            df = pd.DataFrame(transactions)
            
            patterns = {
                'velocity': self._calculate_velocity_metrics(df),
                'temporal': self._analyze_temporal_patterns(df),
                'merchant_loyalty': self._calculate_loyalty_metrics(df),
                'amount_patterns': self._analyze_amount_behaviors(df),
                'recurring_detection': self._detect_recurring_transactions(df),
                'anomaly_scores': self._calculate_anomaly_scores(df)
            }
            
            # Cache results
            cache_key = f"patterns_{len(transactions)}_{datetime.now().date()}"
            self.pattern_cache[cache_key] = patterns
            
            return patterns
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {e}")
            return self._get_default_patterns()
    
    def _calculate_velocity_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate transaction velocity and frequency metrics"""
        velocity = {}
        
        if 'timestamp' in df.columns and len(df) > 1:
            # Convert timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp']).sort_values('timestamp')
            
            if len(df) > 1:
                # Time-based metrics
                time_diffs = df['timestamp'].diff().dt.total_seconds() / 3600  # hours
                total_period = (df['timestamp'].max() - df['timestamp'].min()).days
                
                velocity.update({
                    'avg_hours_between_transactions': float(time_diffs.mean()) if not pd.isna(time_diffs.mean()) else 24.0,
                    'transactions_per_day': len(df) / max(1, total_period),
                    'peak_activity_score': self._calculate_peak_activity(df),
                    'consistency_score': 1.0 / (1.0 + time_diffs.std()) if not pd.isna(time_diffs.std()) else 0.5
                })
        
        return velocity
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze when transactions typically occur"""
        temporal = {}
        
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            
            if len(df) > 0:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['day_of_month'] = df['timestamp'].dt.day
                
                temporal.update({
                    'peak_hour': int(df['hour'].mode().iloc[0]) if len(df['hour'].mode()) > 0 else 12,
                    'peak_day_of_week': int(df['day_of_week'].mode().iloc[0]) if len(df['day_of_week'].mode()) > 0 else 1,
                    'weekend_ratio': len(df[df['day_of_week'] >= 5]) / len(df),
                    'morning_activity': len(df[df['hour'] < 12]) / len(df),
                    'evening_activity': len(df[df['hour'] >= 18]) / len(df),
                    'month_end_concentration': len(df[df['day_of_month'] >= 25]) / len(df)
                })
        
        return temporal
    
    def _calculate_loyalty_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate merchant loyalty and diversity metrics"""
        loyalty = {}
        
        if 'merchant' in df.columns:
            merchant_counts = df['merchant'].value_counts()
            total_transactions = len(df)
            
            loyalty.update({
                'unique_merchants': len(merchant_counts),
                'diversity_index': len(merchant_counts) / total_transactions,
                'top_merchant_ratio': merchant_counts.iloc[0] / total_transactions if len(merchant_counts) > 0 else 0,
                'loyalty_concentration': (merchant_counts.head(3).sum() / total_transactions) if len(merchant_counts) >= 3 else 1.0
            })
        
        return loyalty
    
    def _analyze_amount_behaviors(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze spending amount patterns"""
        amount_patterns = {}
        
        if 'amount' in df.columns:
            amounts = df['amount']
            
            amount_patterns.update({
                'round_number_tendency': len(amounts[amounts % 100 == 0]) / len(amounts),
                'small_transaction_ratio': len(amounts[amounts < 500]) / len(amounts),
                'large_transaction_ratio': len(amounts[amounts > 5000]) / len(amounts),
                'amount_consistency': 1.0 / (1.0 + amounts.std() / amounts.mean()) if amounts.mean() > 0 else 0.5,
                'preferred_ranges': self._identify_preferred_amount_ranges(amounts)
            })
        
        return amount_patterns
    
    def _detect_recurring_transactions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect recurring payment patterns"""
        recurring = {'count': 0, 'patterns': []}
        
        if 'merchant' in df.columns and 'amount' in df.columns:
            # Group by merchant and amount
            merchant_amounts = df.groupby(['merchant', 'amount']).size()
            
            # Find recurring patterns (same merchant + amount, frequency >= 2)
            recurring_patterns = merchant_amounts[merchant_amounts >= 2]
            
            recurring.update({
                'count': len(recurring_patterns),
                'patterns': [
                    {
                        'merchant': merchant,
                        'amount': float(amount),
                        'frequency': int(freq)
                    }
                    for (merchant, amount), freq in recurring_patterns.items()
                ],
                'subscription_likelihood': min(1.0, len(recurring_patterns) / max(1, len(merchant_amounts) * 0.1))
            })
        
        return recurring
    
    def _calculate_anomaly_scores(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate various anomaly scores"""
        anomalies = {}
        
        if 'amount' in df.columns:
            amounts = df['amount']
            mean_amount = amounts.mean()
            std_amount = amounts.std()
            
            if std_amount > 0:
                # Z-score based anomaly detection
                z_scores = abs((amounts - mean_amount) / std_amount)
                anomaly_count = len(z_scores[z_scores > 2.0])  # 2 standard deviations
                
                anomalies.update({
                    'high_amount_anomalies': anomaly_count,
                    'anomaly_ratio': anomaly_count / len(amounts),
                    'max_anomaly_score': float(z_scores.max()) if len(z_scores) > 0 else 0.0,
                    'spending_volatility': float(std_amount / mean_amount) if mean_amount > 0 else 0.0
                })
        
        return anomalies
    
    def _calculate_peak_activity(self, df: pd.DataFrame) -> float:
        """Calculate peak activity concentration score"""
        if 'timestamp' in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            hourly_counts = df['hour'].value_counts()
            if len(hourly_counts) > 0:
                return float(hourly_counts.max() / len(df))
        return 0.0
    
    def _identify_preferred_amount_ranges(self, amounts: pd.Series) -> List[Dict[str, Any]]:
        """Identify preferred spending amount ranges"""
        ranges = [
            {'range': '0-500', 'count': 0},
            {'range': '500-1000', 'count': 0},
            {'range': '1000-5000', 'count': 0},
            {'range': '5000+', 'count': 0}
        ]
        
        total = len(amounts)
        if total > 0:
            ranges[0]['count'] = len(amounts[amounts < 500]) / total
            ranges[1]['count'] = len(amounts[(amounts >= 500) & (amounts < 1000)]) / total
            ranges[2]['count'] = len(amounts[(amounts >= 1000) & (amounts < 5000)]) / total
            ranges[3]['count'] = len(amounts[amounts >= 5000]) / total
        
        return ranges
    
    def _get_default_patterns(self) -> Dict[str, Any]:
        """Return default patterns when analysis fails"""
        return {
            'velocity': {'transactions_per_day': 1.0, 'avg_hours_between_transactions': 24.0},
            'temporal': {'peak_hour': 12, 'weekend_ratio': 0.3},
            'merchant_loyalty': {'diversity_index': 0.5, 'unique_merchants': 5},
            'amount_patterns': {'round_number_tendency': 0.3, 'small_transaction_ratio': 0.6},
            'recurring_detection': {'count': 0, 'patterns': []},
            'anomaly_scores': {'anomaly_ratio': 0.1, 'spending_volatility': 0.5}
        }


class UserFeedbackSystem:
    """Manage user feedback for continuous model improvement"""
    
    def __init__(self, feedback_file: str = "data/user_feedback/feedback.json"):
        self.feedback_file = feedback_file
        self.feedback_data = []
        self.model_accuracy_tracking = {}
        self.load_feedback()
    
    def load_feedback(self):
        """Load existing feedback data"""
        try:
            os.makedirs(os.path.dirname(self.feedback_file), exist_ok=True)
            
            if os.path.exists(self.feedback_file):
                with open(self.feedback_file, 'r') as f:
                    data = json.load(f)
                    self.feedback_data = data.get('feedback', [])
                    self.model_accuracy_tracking = data.get('accuracy_tracking', {})
                logger.info(f"Loaded {len(self.feedback_data)} feedback entries")
            
        except Exception as e:
            logger.error(f"Error loading feedback: {e}")
            self.feedback_data = []
            self.model_accuracy_tracking = {}
    
    def add_feedback(self, transaction_data: Dict, predicted_category: str, 
                    actual_category: str, user_id: str = "anonymous"):
        """Add user feedback entry"""
        feedback_entry = {
            'id': len(self.feedback_data) + 1,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'transaction': {
                'merchant': transaction_data.get('merchant', ''),
                'amount': transaction_data.get('amount', 0),
                'description': transaction_data.get('description', '')
            },
            'prediction': {
                'category': predicted_category,
                'actual_category': actual_category,
                'is_correct': predicted_category == actual_category
            }
        }
        
        self.feedback_data.append(feedback_entry)
        self._update_accuracy_tracking(predicted_category, actual_category)
        self.save_feedback()
        
        logger.info(f"Feedback added: {predicted_category} -> {actual_category}")
    
    def _update_accuracy_tracking(self, predicted: str, actual: str):
        """Update accuracy tracking metrics"""
        if predicted not in self.model_accuracy_tracking:
            self.model_accuracy_tracking[predicted] = {'total': 0, 'correct': 0}
        
        self.model_accuracy_tracking[predicted]['total'] += 1
        if predicted == actual:
            self.model_accuracy_tracking[predicted]['correct'] += 1
    
    def save_feedback(self):
        """Save feedback data to file"""
        try:
            data = {
                'feedback': self.feedback_data,
                'accuracy_tracking': self.model_accuracy_tracking,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.feedback_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving feedback: {e}")
    
    def get_accuracy_report(self) -> Dict[str, Any]:
        """Generate accuracy report from feedback"""
        total_feedback = len(self.feedback_data)
        correct_predictions = sum(1 for fb in self.feedback_data if fb['prediction']['is_correct'])
        
        overall_accuracy = correct_predictions / total_feedback if total_feedback > 0 else 0
        
        category_accuracy = {}
        for category, stats in self.model_accuracy_tracking.items():
            accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            category_accuracy[category] = {
                'accuracy': accuracy,
                'total_predictions': stats['total'],
                'correct_predictions': stats['correct']
            }
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_feedback_entries': total_feedback,
            'category_accuracy': category_accuracy,
            'feedback_volume_trend': self._calculate_feedback_trend()
        }
    
    def _calculate_feedback_trend(self) -> List[Dict[str, Any]]:
        """Calculate feedback volume trend over time"""
        if not self.feedback_data:
            return []
        
        try:
            # Group feedback by date
            daily_counts = {}
            for feedback in self.feedback_data:
                date = feedback['timestamp'][:10]  # Extract date part
                daily_counts[date] = daily_counts.get(date, 0) + 1
            
            # Convert to trend data
            trend = [
                {'date': date, 'feedback_count': count}
                for date, count in sorted(daily_counts.items())
            ]
            
            return trend[-30:]  # Last 30 days
            
        except Exception as e:
            logger.error(f"Error calculating feedback trend: {e}")
            return []
    
    def get_training_data(self) -> List[Dict]:
        """Extract training data from feedback"""
        training_data = []
        
        for feedback in self.feedback_data:
            training_data.append({
                'merchant': feedback['transaction']['merchant'],
                'amount': feedback['transaction']['amount'],
                'description': feedback['transaction']['description'],
                'category': feedback['prediction']['actual_category']
            })
        
        return training_data


class EnhancedExpenseCategorizer:
    """Enhanced backend expense categorization engine"""
    
    CATEGORIES = [
        'FOOD_DINING', 'TRANSPORTATION', 'SHOPPING', 'ENTERTAINMENT',
        'HEALTHCARE', 'UTILITIES', 'EDUCATION', 'TRAVEL', 'RENT',
        'GROCERIES', 'CASH_WITHDRAWAL', 'SUBSCRIPTION', 'INVESTMENT',
        'TRANSFER', 'MISCELLANEOUS'
    ]
    
    def __init__(self, model_path: str = "models/enhanced_categorizer.joblib"):
        self.model_path = model_path
        self.model = None
        self.is_trained = False
        
        # Initialize components
        self.embedding_engine = MerchantEmbeddingEngine()
        self.behavioral_engine = BehavioralAnalysisEngine()
        self.feedback_system = UserFeedbackSystem()
        
        # Feature extraction components
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
            self.amount_scaler = StandardScaler()
            self.label_encoder = LabelEncoder()
        
        # Enhanced category keywords for rule-based fallback
        self.category_keywords = {
            'FOOD_DINING': ['food', 'restaurant', 'cafe', 'dining', 'zomato', 'swiggy', 'dominos', 'mcdonald', 'pizza', 'biryani', 'meal', 'lunch', 'dinner'],
            'TRANSPORTATION': ['uber', 'ola', 'taxi', 'metro', 'bus', 'train', 'flight', 'fuel', 'petrol', 'diesel', 'parking', 'toll', 'auto', 'rickshaw', 'cab'],
            'SHOPPING': ['amazon', 'flipkart', 'myntra', 'shopping', 'mall', 'store', 'purchase', 'retail', 'clothing', 'fashion', 'electronics', 'buy'],
            'ENTERTAINMENT': ['movie', 'cinema', 'netflix', 'spotify', 'prime', 'youtube', 'hotstar', 'entertainment', 'game', 'bookmyshow', 'pvr', 'music'],
            'HEALTHCARE': ['hospital', 'doctor', 'medicine', 'pharmacy', 'apollo', 'medical', 'health', 'clinic', 'treatment', 'drug', 'dentist'],
            'UTILITIES': ['electricity', 'water', 'gas', 'internet', 'phone', 'mobile', 'airtel', 'jio', 'vodafone', 'utility', 'bill', 'recharge'],
            'EDUCATION': ['school', 'college', 'university', 'course', 'book', 'education', 'learning', 'training', 'tuition', 'fee', 'exam'],
            'TRAVEL': ['hotel', 'booking', 'flight', 'train', 'travel', 'vacation', 'trip', 'oyo', 'makemytrip', 'cleartrip', 'goibibo'],
            'RENT': ['rent', 'house', 'apartment', 'accommodation', 'housing', 'lease', 'maintenance', 'deposit'],
            'GROCERIES': ['grocery', 'supermarket', 'vegetables', 'fruits', 'milk', 'bread', 'bigbasket', 'grofers', 'fresh', 'market'],
            'SUBSCRIPTION': ['subscription', 'monthly', 'yearly', 'premium', 'pro', 'plus', 'recurring', 'renewal'],
            'CASH_WITHDRAWAL': ['atm', 'cash', 'withdrawal', 'withdraw', 'cdm'],
            'INVESTMENT': ['mutual', 'fund', 'sip', 'equity', 'stock', 'bond', 'investment', 'zerodha', 'groww'],
            'TRANSFER': ['transfer', 'send', 'receive', 'upi', 'neft', 'imps', 'rtgs', 'paytm', 'gpay', 'phonepe']
        }
        
        # Load existing model if available
        self.load_model()
    
    def categorize_transaction(self, merchant: str, amount: float, 
                             description: str = "",
                             user_transaction_history: Optional[List[Dict]] = None,
                             user_id: str = "anonymous") -> Dict[str, Any]:
        """Main categorization method with comprehensive analysis"""
        try:
            # Extract behavioral patterns if history provided
            behavioral_patterns = {}
            if user_transaction_history:
                behavioral_patterns = self.behavioral_engine.analyze_spending_patterns(user_transaction_history)
            
            # Get category prediction
            if self.is_trained and SKLEARN_AVAILABLE:
                result = self._ml_categorization(merchant, amount, description, behavioral_patterns)
            else:
                result = self._rule_based_categorization(merchant, amount, description)
            
            # Add metadata
            result.update({
                'merchant': merchant,
                'amount': amount,
                'description': description,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'model_version': 'enhanced_v1.0',
                'features_used': self._get_feature_summary(behavioral_patterns)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error in transaction categorization: {e}")
            return self._fallback_categorization(merchant, amount)
    
    def _ml_categorization(self, merchant: str, amount: float, 
                          description: str, behavioral_patterns: Dict) -> Dict[str, Any]:
        """ML-based categorization using trained model"""
        try:
            # Extract features
            features = self._extract_ml_features(merchant, amount, description, behavioral_patterns)
            
            # Predict
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            
            category = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            # Get alternatives
            top_indices = np.argsort(probabilities)[-3:][::-1]
            alternatives = [
                {
                    'category': self.label_encoder.inverse_transform([idx])[0],
                    'confidence': float(probabilities[idx])
                }
                for idx in top_indices
            ]
            
            return {
                'category': category,
                'confidence': confidence,
                'alternatives': alternatives,
                'method': 'ml_model',
                'behavioral_insights': self._generate_behavioral_insights(behavioral_patterns)
            }
            
        except Exception as e:
            logger.error(f"ML categorization failed: {e}")
            return self._rule_based_categorization(merchant, amount, description)
    
    def _rule_based_categorization(self, merchant: str, amount: float, description: str) -> Dict[str, Any]:
        """Enhanced rule-based categorization"""
        merchant_lower = merchant.lower()
        description_lower = description.lower()
        combined_text = f"{merchant_lower} {description_lower}"
        
        # Score categories
        category_scores = {}
        
        for category, keywords in self.category_keywords.items():
            score = 0
            for keyword in keywords:
                if keyword in combined_text:
                    # Weight by keyword length and position
                    weight = len(keyword) / 10
                    if keyword in merchant_lower:
                        weight *= 1.5  # Merchant name is more important
                    score += weight
            
            # Amount-based scoring adjustments
            score += self._get_amount_category_score(category, amount)
            
            category_scores[category] = score
        
        # Get best category
        sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_categories[0][1] > 0:
            category = sorted_categories[0][0]
            confidence = min(0.95, max(0.4, sorted_categories[0][1] / 3))
        else:
            category = 'MISCELLANEOUS'
            confidence = 0.5
        
        # Create alternatives
        alternatives = [
            {'category': cat, 'confidence': min(0.95, max(0.1, score / 3))}
            for cat, score in sorted_categories[:3]
        ]
        
        return {
            'category': category,
            'confidence': confidence,
            'alternatives': alternatives,
            'method': 'rule_based',
            'behavioral_insights': {}
        }
    
    def _get_amount_category_score(self, category: str, amount: float) -> float:
        """Get category score boost based on amount patterns"""
        if category == 'CASH_WITHDRAWAL' and amount >= 1000:
            return 0.3
        elif category == 'SUBSCRIPTION' and 100 <= amount <= 2000:
            return 0.2
        elif category == 'GROCERIES' and 500 <= amount <= 5000:
            return 0.15
        elif category == 'ENTERTAINMENT' and amount <= 1000:
            return 0.1
        elif category == 'RENT' and amount >= 10000:
            return 0.2
        elif category == 'UTILITIES' and 200 <= amount <= 3000:
            return 0.1
        return 0
    
    def _extract_ml_features(self, merchant: str, amount: float, 
                           description: str, behavioral_patterns: Dict) -> np.ndarray:
        """Extract comprehensive features for ML model"""
        features = []
        
        # Basic features
        features.extend([
            amount,
            np.log1p(amount),  # Log transform
            len(merchant),
            merchant.count(' '),
            1 if amount % 100 == 0 else 0,  # Round number
        ])
        
        # Behavioral features
        if behavioral_patterns:
            velocity = behavioral_patterns.get('velocity', {})
            temporal = behavioral_patterns.get('temporal', {})
            loyalty = behavioral_patterns.get('merchant_loyalty', {})
            
            features.extend([
                velocity.get('transactions_per_day', 1.0),
                temporal.get('weekend_ratio', 0.3),
                loyalty.get('diversity_index', 0.5),
                behavioral_patterns.get('recurring_detection', {}).get('subscription_likelihood', 0.0),
                behavioral_patterns.get('anomaly_scores', {}).get('anomaly_ratio', 0.1)
            ])
        else:
            features.extend([1.0, 0.3, 0.5, 0.0, 0.1])  # Default values
        
        # Merchant embedding
        try:
            embedding = self.embedding_engine.get_embedding(merchant)
            features.extend(embedding[:20])  # Use first 20 dimensions
        except:
            features.extend([0.0] * 20)  # Fallback
        
        return np.array(features)
    
    def _generate_behavioral_insights(self, patterns: Dict) -> Dict[str, Any]:
        """Generate human-readable behavioral insights"""
        insights = {}
        
        if patterns:
            velocity = patterns.get('velocity', {})
            temporal = patterns.get('temporal', {})
            recurring = patterns.get('recurring_detection', {})
            
            insights.update({
                'spending_frequency': 'high' if velocity.get('transactions_per_day', 1) > 3 else 'moderate' if velocity.get('transactions_per_day', 1) > 1 else 'low',
                'weekend_spender': temporal.get('weekend_ratio', 0.3) > 0.4,
                'has_subscriptions': recurring.get('count', 0) > 0,
                'loyalty_level': 'high' if patterns.get('merchant_loyalty', {}).get('loyalty_concentration', 0.5) > 0.7 else 'moderate'
            })
        
        return insights
    
    def _get_feature_summary(self, behavioral_patterns: Dict) -> Dict[str, Any]:
        """Get summary of features used"""
        return {
            'embedding_available': self.embedding_engine.is_trained,
            'behavioral_analysis': bool(behavioral_patterns),
            'ml_model_active': self.is_trained,
            'feedback_entries': len(self.feedback_system.feedback_data)
        }
    
    def _fallback_categorization(self, merchant: str, amount: float) -> Dict[str, Any]:
        """Final fallback categorization"""
        return {
            'category': 'MISCELLANEOUS',
            'confidence': 0.3,
            'alternatives': [
                {'category': 'MISCELLANEOUS', 'confidence': 0.3},
                {'category': 'SHOPPING', 'confidence': 0.2},
                {'category': 'FOOD_DINING', 'confidence': 0.1}
            ],
            'method': 'fallback',
            'behavioral_insights': {},
            'merchant': merchant,
            'amount': amount,
            'timestamp': datetime.now().isoformat()
        }
    
    def add_user_feedback(self, transaction_data: Dict, predicted_category: str, 
                         actual_category: str, user_id: str = "anonymous"):
        """Add user feedback to improve model"""
        self.feedback_system.add_feedback(
            transaction_data, predicted_category, actual_category, user_id
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        accuracy_report = self.feedback_system.get_accuracy_report()
        
        return {
            'model_accuracy': accuracy_report,
            'embedding_stats': {
                'merchants_embedded': len(self.embedding_engine.merchant_embeddings),
                'embedding_method': 'word2vec' if self.embedding_engine.word2vec_model else 'tfidf' if SKLEARN_AVAILABLE else 'hash'
            },
            'categorization_stats': {
                'total_categories': len(self.CATEGORIES),
                'ml_model_trained': self.is_trained,
                'fallback_method': 'rule_based'
            },
            'system_health': {
                'sklearn_available': SKLEARN_AVAILABLE,
                'word2vec_available': WORD2VEC_AVAILABLE,
                'feedback_system_active': True
            }
        }
    
    def load_model(self) -> bool:
        """Load pre-trained model"""
        try:
            if os.path.exists(self.model_path) and SKLEARN_AVAILABLE:
                model_data = joblib.load(self.model_path)
                self.model = model_data.get('model')
                self.label_encoder = model_data.get('label_encoder')
                self.is_trained = True
                logger.info("Loaded pre-trained model")
                return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        
        return False
    
    def save_model(self) -> bool:
        """Save trained model"""
        try:
            if self.model and SKLEARN_AVAILABLE:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                
                model_data = {
                    'model': self.model,
                    'label_encoder': self.label_encoder,
                    'categories': self.CATEGORIES,
                    'version': 'enhanced_v1.0',
                    'timestamp': datetime.now().isoformat()
                }
                
                joblib.dump(model_data, self.model_path)
                logger.info(f"Model saved to {self.model_path}")
                return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
        
        return False


def main():
    """Demo the enhanced categorizer backend"""
    print("🤖 Enhanced ML Expense Categorizer - Backend Demo")
    print("=" * 60)
    
    # Initialize categorizer
    categorizer = EnhancedExpenseCategorizer()
    
    # Sample transaction history for behavioral analysis
    transaction_history = [
        {"merchant": "ZOMATO", "amount": 450, "timestamp": "2025-01-10T19:30:00"},
        {"merchant": "UBER", "amount": 350, "timestamp": "2025-01-10T09:15:00"},
        {"merchant": "NETFLIX.COM", "amount": 799, "timestamp": "2025-01-01T12:00:00"},
        {"merchant": "NETFLIX.COM", "amount": 799, "timestamp": "2024-12-01T12:00:00"},
        {"merchant": "AMAZON.IN", "amount": 2500, "timestamp": "2025-01-08T14:30:00"},
        {"merchant": "APOLLO PHARMACY", "amount": 850, "timestamp": "2025-01-09T16:45:00"},
    ]
    
    # Test transactions
    test_transactions = [
        {"merchant": "ZOMATO BANGALORE", "amount": 650, "description": "Food delivery order"},
        {"merchant": "OLA CABS", "amount": 280, "description": "Cab booking"},
        {"merchant": "NETFLIX.COM", "amount": 799, "description": "Monthly subscription"},
        {"merchant": "APOLLO PHARMACY", "amount": 1200, "description": "Medicine purchase"},
        {"merchant": "ATM WITHDRAWAL SBI", "amount": 5000, "description": "Cash withdrawal"},
        {"merchant": "UNKNOWN MERCHANT", "amount": 1500, "description": "Some transaction"},
    ]
    
    print(f"\n🧪 Testing Enhanced Categorization:")
    print("-" * 50)
    
    for txn in test_transactions:
        result = categorizer.categorize_transaction(
            merchant=txn["merchant"],
            amount=txn["amount"],
            description=txn["description"],
            user_transaction_history=transaction_history,
            user_id="demo_user"
        )
        
        print(f"\n💰 {txn['merchant']}: ₹{txn['amount']}")
        print(f"   📂 Category: {result['category']} ({result['confidence']:.2f} confidence)")
        print(f"   🔧 Method: {result['method']}")
        
        if result.get('behavioral_insights'):
            insights = result['behavioral_insights']
            print(f"   🎯 Insights: {', '.join([f'{k}: {v}' for k, v in insights.items()])}")
        
        # Add some feedback for demo
        if txn["merchant"].startswith("ZOMATO"):
            categorizer.add_user_feedback(
                transaction_data=txn,
                predicted_category=result['category'],
                actual_category="FOOD_DINING",
                user_id="demo_user"
            )
    
    # Performance metrics
    print(f"\n📊 Performance Metrics:")
    print("-" * 30)
    metrics = categorizer.get_performance_metrics()
    
    print(f"Model Accuracy: {metrics['model_accuracy']['overall_accuracy']:.2f}")
    print(f"Feedback Entries: {metrics['model_accuracy']['total_feedback_entries']}")
    print(f"Merchants Embedded: {metrics['embedding_stats']['merchants_embedded']}")
    print(f"Embedding Method: {metrics['embedding_stats']['embedding_method']}")
    print(f"ML Model Active: {metrics['categorization_stats']['ml_model_trained']}")
    
    print(f"\n✅ Enhanced categorizer backend demo completed!")


if __name__ == "__main__":
    main()