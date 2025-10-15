#!/usr/bin/env python3
"""
Enhanced Expense Categorizer with Advanced ML Features
======================================================

Main expense categorization module that integrates:
- Advanced ML categorization with merchant embeddings
- Behavioral pattern analysis
- User feedback loop
- Real-time model updates
"""

# Import the advanced categorizer
try:
    from .advanced_expense_categorizer import (
        AdvancedExpenseCategorizer,
        MerchantEmbeddingManager,
        BehavioralPatternAnalyzer,
        TransactionFeatures,
        UserFeedback,
        categorize_transaction
    )
    ADVANCED_AVAILABLE = True
except ImportError:
    try:
        from advanced_expense_categorizer import (
            AdvancedExpenseCategorizer,
            MerchantEmbeddingManager,
            BehavioralPatternAnalyzer,
            TransactionFeatures,
            UserFeedback,
            categorize_transaction
        )
        ADVANCED_AVAILABLE = True
    except ImportError:
        print("Advanced categorizer not available, using basic version")
        ADVANCED_AVAILABLE = False

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpenseCategorizer:
    """Enhanced expense categorizer with backward compatibility"""
    
    def __init__(self, model_path: Optional[str] = None):
        if ADVANCED_AVAILABLE:
            self.categorizer = AdvancedExpenseCategorizer(model_path)
            self.version = "advanced"
        else:
            self.categorizer = BasicExpenseCategorizer()
            self.version = "basic"
        
        logger.info(f"Initialized {self.version} expense categorizer")
    
    def categorize(self, transaction: Dict[str, Any], user_id: Optional[str] = None) -> Dict[str, Any]:
        """Categorize transaction with enhanced features"""
        if ADVANCED_AVAILABLE:
            return self.categorizer.predict_category(transaction, user_id)
        else:
            return self.categorizer.categorize(transaction)
    
    def add_feedback(self, transaction_id: str, predicted_category: str, 
                    actual_category: str, confidence: float, user_id: Optional[str] = None):
        """Add user feedback for model improvement"""
        if ADVANCED_AVAILABLE and hasattr(self.categorizer, 'add_feedback'):
            self.categorizer.add_feedback(transaction_id, predicted_category, 
                                        actual_category, confidence, user_id)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get categorization statistics"""
        if ADVANCED_AVAILABLE and hasattr(self.categorizer, 'get_model_statistics'):
            return self.categorizer.get_model_statistics()
        else:
            return {'version': self.version, 'features': 'basic'}


class BasicExpenseCategorizer:
    """Basic rule-based categorizer as fallback"""
    
    def __init__(self):
        self.categories = {
            'Food & Dining': ['food', 'restaurant', 'swiggy', 'zomato', 'domino', 'mcd', 'kfc', 'pizza'],
            'Shopping': ['amazon', 'flipkart', 'myntra', 'ajio', 'shopping', 'retail', 'mall'],
            'Transportation': ['uber', 'ola', 'cab', 'taxi', 'metro', 'bus', 'auto'],
            'Fuel': ['petrol', 'diesel', 'fuel', 'gas', 'hp', 'bpcl', 'ioc', 'shell'],
            'Utilities': ['electricity', 'water', 'gas', 'internet', 'mobile', 'recharge'],
            'Healthcare': ['hospital', 'medical', 'pharmacy', 'doctor', 'health'],
            'Entertainment': ['movie', 'cinema', 'bookmyshow', 'netflix', 'amazon prime'],
            'Cash & ATM': ['atm', 'withdrawal', 'cash'],
            'Transfer': ['transfer', 'neft', 'imps', 'rtgs'],
            'Investment': ['mutual fund', 'sip', 'investment', 'stock', 'equity']
        }
    
    def categorize(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Basic rule-based categorization"""
        merchant = transaction.get('merchant', '').lower()
        amount = transaction.get('amount', 0)
        
        # Check each category for keyword matches
        for category, keywords in self.categories.items():
            if any(keyword in merchant for keyword in keywords):
                confidence = 0.8 if len([k for k in keywords if k in merchant]) > 1 else 0.6
                return {
                    'category': category,
                    'confidence': confidence,
                    'reasoning': {'keyword_match': True},
                    'model_type': 'rule_based'
                }
        
        # Default categorization based on amount
        if amount > 10000:
            category = 'Large Expense'
            confidence = 0.5
        else:
            category = 'Other'
            confidence = 0.3
        
        return {
            'category': category,
            'confidence': confidence,
            'reasoning': {'default_rule': True},
            'model_type': 'rule_based'
        }


# Legacy function for backward compatibility
def extract_transaction_info(transaction_text: str) -> dict:
    """Legacy function for backward compatibility"""
    # Basic transaction info extraction
    import re
    
    amount_match = re.search(r'Rs\.?\s*(\d+\.?\d*)', transaction_text)
    merchant_match = re.search(r'(at|to|from)\s+([A-Za-z0-9\s]+)', transaction_text)
    
    amount = float(amount_match.group(1)) if amount_match else 0.0
    merchant = merchant_match.group(2).strip() if merchant_match else "Unknown"
    
    transaction = {
        'amount': amount,
        'merchant': merchant,
        'timestamp': datetime.now()
    }
    
    categorizer = ExpenseCategorizer()
    result = categorizer.categorize(transaction)
    
    return {
        'amount': amount,
        'merchant': merchant,
        'category': result['category'],
        'confidence': result['confidence']
    }


# Export main classes and functions
__all__ = [
    'ExpenseCategorizer',
    'BasicExpenseCategorizer',
    'extract_transaction_info',
    'categorize_transaction'
]

if ADVANCED_AVAILABLE:
    __all__.extend([
        'AdvancedExpenseCategorizer',
        'MerchantEmbeddingManager',
        'BehavioralPatternAnalyzer',
        'TransactionFeatures',
        'UserFeedback'
    ])

import os
import joblib
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Advanced feature extraction for expense categorization"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.amount_scaler = StandardScaler()
        self.is_fitted = False
        
        # Category keywords for feature engineering
        self.category_keywords = {
            'FOOD_DINING': ['food', 'restaurant', 'cafe', 'dining', 'meal', 'lunch', 'dinner', 'breakfast',
                           'zomato', 'swiggy', 'uber eats', 'dominos', 'mcdonald', 'kfc', 'pizza'],
            'TRANSPORTATION': ['uber', 'ola', 'taxi', 'metro', 'bus', 'train', 'flight', 'fuel', 'petrol',
                             'diesel', 'parking', 'toll', 'auto', 'rickshaw', 'transport'],
            'SHOPPING': ['amazon', 'flipkart', 'myntra', 'shopping', 'mall', 'store', 'purchase',
                        'buy', 'order', 'retail', 'clothing', 'fashion', 'electronics'],
            'ENTERTAINMENT': ['movie', 'cinema', 'netflix', 'spotify', 'amazon prime', 'youtube',
                            'hotstar', 'entertainment', 'game', 'fun', 'recreation', 'bookmyshow'],
            'HEALTHCARE': ['hospital', 'doctor', 'medicine', 'pharmacy', 'apollo', 'medical',
                          'health', 'clinic', 'treatment', 'drug', 'tablet', 'checkup'],
            'UTILITIES': ['electricity', 'water', 'gas', 'internet', 'phone', 'mobile', 'airtel',
                         'jio', 'vodafone', 'utility', 'bill', 'recharge', 'broadband'],
            'EDUCATION': ['school', 'college', 'university', 'course', 'book', 'education',
                         'learning', 'training', 'tuition', 'fee', 'study', 'exam'],
            'TRAVEL': ['hotel', 'booking', 'flight', 'train', 'travel', 'vacation', 'trip',
                      'oyo', 'make my trip', 'cleartrip', 'goibibo', 'tourism'],
            'RENT': ['rent', 'house', 'apartment', 'accommodation', 'housing', 'lease'],
            'GROCERIES': ['grocery', 'supermarket', 'vegetables', 'fruits', 'milk', 'bread',
                         'big basket', 'grofers', 'fresh', 'market', 'food items'],
            'CASH_WITHDRAWAL': ['atm', 'cash', 'withdrawal', 'withdraw', 'cdm']
        }
    
    def fit(self, merchants: List[str], amounts: List[float]) -> 'FeatureExtractor':
        """Fit the feature extractor on training data"""
        try:
            # Fit TF-IDF vectorizer
            self.tfidf_vectorizer.fit(merchants)
            
            # Fit amount scaler
            amounts_array = np.array(amounts).reshape(-1, 1)
            self.amount_scaler.fit(amounts_array)
            
            self.is_fitted = True
            logger.info("Feature extractor fitted successfully")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting feature extractor: {e}")
            raise
    
    def transform(self, merchants: List[str], amounts: List[float], 
                  descriptions: Optional[List[str]] = None) -> np.ndarray:
        """Transform input data into feature vectors"""
        if not self.is_fitted:
            raise ValueError("Feature extractor not fitted. Call fit() first.")
        
        try:
            n_samples = len(merchants)
            
            # TF-IDF features from merchant names
            tfidf_features = self.tfidf_vectorizer.transform(merchants).toarray()
            
            # Amount features (normalized)
            amounts_array = np.array(amounts).reshape(-1, 1)
            amount_features = self.amount_scaler.transform(amounts_array)
            
            # Amount-based features
            amount_log = np.log1p(amounts_array).flatten()
            amount_categories = self._categorize_amounts(amounts)
            
            # Category keyword features
            keyword_features = self._extract_keyword_features(merchants, descriptions)
            
            # Merchant length and complexity features
            merchant_features = self._extract_merchant_features(merchants)
            
            # Time-based features (if available)
            time_features = np.zeros((n_samples, 3))  # placeholder for hour, day, month
            
            # Combine all features
            all_features = np.hstack([
                tfidf_features,                    # TF-IDF features
                amount_features,                   # Normalized amount
                amount_log.reshape(-1, 1),         # Log amount
                amount_categories.reshape(-1, 1),  # Amount category
                keyword_features,                  # Category keywords
                merchant_features,                 # Merchant characteristics
                time_features                      # Time features
            ])
            
            logger.debug(f"Extracted {all_features.shape[1]} features for {n_samples} samples")
            return all_features
            
        except Exception as e:
            logger.error(f"Error transforming features: {e}")
            raise
    
    def fit_transform(self, merchants: List[str], amounts: List[float],
                     descriptions: Optional[List[str]] = None) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(merchants, amounts).transform(merchants, amounts, descriptions)
    
    def _categorize_amounts(self, amounts: List[float]) -> np.ndarray:
        """Categorize amounts into bins"""
        amounts_array = np.array(amounts)
        
        # Define amount bins
        bins = [0, 100, 500, 1000, 5000, 10000, np.inf]
        categories = np.digitize(amounts_array, bins)
        
        return categories
    
    def _extract_keyword_features(self, merchants: List[str], 
                                 descriptions: Optional[List[str]] = None) -> np.ndarray:
        """Extract category-specific keyword features"""
        n_samples = len(merchants)
        n_categories = len(self.category_keywords)
        keyword_features = np.zeros((n_samples, n_categories))
        
        for i, merchant in enumerate(merchants):
            text = merchant.lower()
            if descriptions and i < len(descriptions):
                text += " " + descriptions[i].lower()
            
            for j, (category, keywords) in enumerate(self.category_keywords.items()):
                # Count keyword matches
                matches = sum(1 for keyword in keywords if keyword in text)
                keyword_features[i, j] = matches
        
        return keyword_features
    
    def _extract_merchant_features(self, merchants: List[str]) -> np.ndarray:
        """Extract merchant name characteristics"""
        features = []
        
        for merchant in merchants:
            merchant_features = [
                len(merchant),                          # Length
                merchant.count(' '),                    # Number of spaces
                sum(1 for c in merchant if c.isupper()), # Upper case count
                sum(1 for c in merchant if c.isdigit()), # Digit count
                1 if '@' in merchant else 0,            # Has @ (UPI)
                1 if merchant.isupper() else 0,         # All uppercase
                len(set(merchant.lower()))              # Unique characters
            ]
            features.append(merchant_features)
        
        return np.array(features)


class ExpenseCategorizer:
    """Main ML expense categorization model"""
    
    CATEGORIES = [
        'FOOD_DINING', 'TRANSPORTATION', 'SHOPPING', 'ENTERTAINMENT',
        'HEALTHCARE', 'UTILITIES', 'EDUCATION', 'TRAVEL', 'RENT',
        'GROCERIES', 'CASH_WITHDRAWAL', 'MISCELLANEOUS'
    ]
    
    def __init__(self, model_path: Optional[str] = None, 
                 feature_extractor_path: Optional[str] = None):
        """Initialize the expense categorizer"""
        self.model = None
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.is_trained = False
        
        # Load pre-trained model if available
        if model_path and os.path.exists(model_path):
            self.load_model(model_path, feature_extractor_path)
    
    def train(self, training_data: List[Dict]) -> Dict[str, Any]:
        """Train the expense categorization model"""
        try:
            logger.info("Starting model training...")
            
            # Prepare training data
            merchants = [item['merchant'] for item in training_data]
            amounts = [item['amount'] for item in training_data]
            categories = [item['category'] for item in training_data]
            descriptions = [item.get('description', '') for item in training_data]
            
            # Fit feature extractor
            X = self.feature_extractor.fit_transform(merchants, amounts, descriptions)
            
            # Encode labels
            y = self.label_encoder.fit_transform(categories)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train ensemble model
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            self.model.fit(X_train, y_train)
            
            # Evaluate model
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X, y, cv=5)
            
            self.is_trained = True
            
            training_results = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'n_features': X.shape[1],
                'n_samples': len(training_data)
            }
            
            logger.info(f"Model training completed. Test accuracy: {test_score:.3f}")
            return training_results
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def categorize_expense(self, merchant: str, amount: float, 
                          description: str = "") -> Dict[str, Any]:
        """Categorize a single expense"""
        if not self.is_trained:
            # Return mock result for demo
            return self._mock_categorization(merchant, amount)
        
        try:
            # Extract features
            X = self.feature_extractor.transform([merchant], [amount], [description])
            
            # Predict
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]
            
            # Decode prediction
            category = self.label_encoder.inverse_transform([prediction])[0]
            confidence = float(max(probabilities))
            
            # Get alternative predictions
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
                'merchant': merchant,
                'amount': amount
            }
            
        except Exception as e:
            logger.error(f"Error categorizing expense: {e}")
            return self._mock_categorization(merchant, amount)
    
    def _mock_categorization(self, merchant: str, amount: float) -> Dict[str, Any]:
        """Mock categorization for demo purposes"""
        merchant_lower = merchant.lower()
        
        # Simple rule-based categorization for demo
        if any(word in merchant_lower for word in ['zomato', 'swiggy', 'food', 'restaurant']):
            category = 'FOOD_DINING'
            confidence = 0.85
        elif any(word in merchant_lower for word in ['uber', 'ola', 'taxi']):
            category = 'TRANSPORTATION'
            confidence = 0.82
        elif any(word in merchant_lower for word in ['amazon', 'flipkart', 'shopping']):
            category = 'SHOPPING'
            confidence = 0.78
        elif any(word in merchant_lower for word in ['netflix', 'entertainment']):
            category = 'ENTERTAINMENT'
            confidence = 0.75
        elif any(word in merchant_lower for word in ['pharmacy', 'hospital', 'apollo']):
            category = 'HEALTHCARE'
            confidence = 0.72
        elif 'atm' in merchant_lower or 'cash' in merchant_lower:
            category = 'CASH_WITHDRAWAL'
            confidence = 0.90
        else:
            category = 'MISCELLANEOUS'
            confidence = 0.55
        
        # Add some randomness to confidence
        import random
        confidence += random.uniform(-0.15, 0.10)
        confidence = max(0.3, min(0.95, confidence))
        
        return {
            'category': category,
            'confidence': confidence,
            'alternatives': [
                {'category': category, 'confidence': confidence},
                {'category': 'MISCELLANEOUS', 'confidence': max(0.1, 1 - confidence - 0.1)},
                {'category': 'SHOPPING', 'confidence': max(0.05, 1 - confidence - 0.2)}
            ],
            'merchant': merchant,
            'amount': amount
        }
    
    def batch_categorize(self, expenses: List[Dict]) -> List[Dict]:
        """Categorize multiple expenses"""
        results = []
        for expense in expenses:
            result = self.categorize_expense(
                expense['merchant'],
                expense['amount'],
                expense.get('description', '')
            )
            results.append(result)
        return results
    
    def save_model(self, model_path: str, feature_extractor_path: str):
        """Save trained model and feature extractor"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        try:
            # Create directories if they don't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(feature_extractor_path), exist_ok=True)
            
            # Save model components
            model_data = {
                'model': self.model,
                'label_encoder': self.label_encoder,
                'categories': self.CATEGORIES
            }
            joblib.dump(model_data, model_path)
            joblib.dump(self.feature_extractor, feature_extractor_path)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Feature extractor saved to {feature_extractor_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, model_path: str, feature_extractor_path: str):
        """Load pre-trained model and feature extractor"""
        try:
            if os.path.exists(model_path) and os.path.exists(feature_extractor_path):
                # Load model components
                model_data = joblib.load(model_path)
                self.model = model_data['model']
                self.label_encoder = model_data['label_encoder']
                
                # Load feature extractor
                self.feature_extractor = joblib.load(feature_extractor_path)
                
                self.is_trained = True
                logger.info("Model loaded successfully")
            else:
                logger.warning("Model files not found. Using mock categorization.")
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.warning("Falling back to mock categorization")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and statistics"""
        if not self.is_trained:
            return {"status": "not_trained", "using_mock": True}
        
        return {
            "status": "trained",
            "categories": self.CATEGORIES,
            "n_categories": len(self.CATEGORIES),
            "model_type": type(self.model).__name__,
            "feature_count": self.feature_extractor.tfidf_vectorizer.max_features,
            "using_mock": False
        }


def main():
    """Demo function"""
    print("🤖 ML Expense Categorizer Demo")
    print("=" * 50)
    
    # Initialize categorizer
    categorizer = ExpenseCategorizer()
    
    # Test cases
    test_expenses = [
        {"merchant": "ZOMATO", "amount": 450},
        {"merchant": "UBER INDIA", "amount": 350},
        {"merchant": "AMAZON PAY", "amount": 2500},
        {"merchant": "APOLLO PHARMACY", "amount": 1200},
        {"merchant": "NETFLIX", "amount": 599},
    ]
    
    print("\n🧪 Testing expense categorization:")
    print("-" * 40)
    
    for expense in test_expenses:
        result = categorizer.categorize_expense(expense["merchant"], expense["amount"])
        print(f"\n💰 {expense['merchant']}: ₹{expense['amount']}")
        print(f"   Category: {result['category']}")
        print(f"   Confidence: {result['confidence']:.2f}")
    
    print(f"\n📊 Model Info:")
    info = categorizer.get_model_info()
    for key, value in info.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    main()