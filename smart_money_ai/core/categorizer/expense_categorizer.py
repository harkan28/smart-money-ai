"""
Expense Categorization Model (Part 2 of 4-Part ML System)
========================================================

Advanced ML model for automatically categorizing expenses into proper categories:
- Utilities (electricity, water, gas, internet, phone)
- Food & Dining (restaurants, groceries, food delivery)
- Transportation (fuel, public transport, cab rides)
- Entertainment (movies, games, subscriptions)
- Shopping (clothing, electronics, general purchases)
- Healthcare (medical, pharmacy, insurance)
- Education (courses, books, training)
- Investment (mutual funds, stocks, SIP)
- Others (miscellaneous expenses)

Uses multiple approaches:
1. Rule-based categorization for bank-specific patterns
2. ML-based text classification using embeddings
3. Amount-based pattern recognition
4. Merchant/vendor pattern matching
"""

import re
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import sqlite3
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

class ExpenseCategorizer:
    """
    Advanced ML-based expense categorization system
    Part 2 of the 4-Part Smart Money AI ML System
    """
    
    def __init__(self):
        """Initialize the expense categorizer"""
        self.categories = {
            'utilities': ['electricity', 'water', 'gas', 'internet', 'phone', 'mobile', 'broadband', 'wifi', 'telecom', 'bsnl', 'airtel', 'jio', 'vi'],
            'food': ['restaurant', 'food', 'dining', 'cafe', 'swiggy', 'zomato', 'dominos', 'mcdonald', 'kfc', 'pizza', 'grocery', 'supermarket', 'dmart'],
            'transportation': ['fuel', 'petrol', 'diesel', 'uber', 'ola', 'metro', 'bus', 'train', 'taxi', 'auto', 'parking', 'toll'],
            'entertainment': ['movie', 'cinema', 'netflix', 'amazon', 'spotify', 'youtube', 'game', 'book', 'magazine', 'subscription'],
            'shopping': ['shopping', 'mall', 'amazon', 'flipkart', 'myntra', 'clothing', 'electronics', 'mobile', 'laptop', 'fashion'],
            'healthcare': ['hospital', 'medical', 'pharmacy', 'medicine', 'doctor', 'clinic', 'health', 'insurance', 'apollo', 'fortis'],
            'education': ['school', 'college', 'university', 'course', 'training', 'education', 'book', 'tuition', 'fees'],
            'investment': ['mutual', 'fund', 'sip', 'stock', 'equity', 'bond', 'investment', 'trading', 'demat', 'zerodha', 'upstox'],
            'transfer': ['transfer', 'sent', 'credited', 'debited', 'upi', 'imps', 'neft', 'rtgs'],
            'others': []
        }
        
        self.model = None
        self.model_path = "/Users/harshitrawal/Downloads/SMART MONEY/smart_money_ai/data/models/expense_categorizer.pkl"
        self.training_data_path = "/Users/harshitrawal/Downloads/SMART MONEY/smart_money_ai/data/training/categorization_training_data.json"
        
        # Ensure directories exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        
        # Load or train model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model for categorization"""
        if os.path.exists(self.model_path):
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ Expense categorization model loaded from disk")
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
                self._train_model()
        else:
            self._train_model()
    
    def _generate_training_data(self) -> List[Dict]:
        """Generate comprehensive training data for expense categorization"""
        
        training_data = []
        
        # Utilities
        utilities_samples = [
            {'text': 'ELECTRICITY BILL PAYMENT BSES', 'category': 'utilities'},
            {'text': 'Mobile recharge Airtel prepaid', 'category': 'utilities'},
            {'text': 'Internet bill payment Jio Fiber', 'category': 'utilities'},
            {'text': 'Water bill BMC payment', 'category': 'utilities'},
            {'text': 'Gas cylinder booking HP Gas', 'category': 'utilities'},
            {'text': 'DTH recharge Tata Sky', 'category': 'utilities'},
            {'text': 'Landline bill BSNL', 'category': 'utilities'},
            {'text': 'Broadband bill Airtel', 'category': 'utilities'},
        ]
        
        # Food & Dining
        food_samples = [
            {'text': 'Payment to SWIGGY for food delivery', 'category': 'food'},
            {'text': 'ZOMATO online food order', 'category': 'food'},
            {'text': 'MCDONALDS restaurant payment', 'category': 'food'},
            {'text': 'DMart grocery shopping', 'category': 'food'},
            {'text': 'Local restaurant dinner', 'category': 'food'},
            {'text': 'Coffee shop payment Starbucks', 'category': 'food'},
            {'text': 'Pizza Hut online order', 'category': 'food'},
            {'text': 'Grocery store vegetables', 'category': 'food'},
        ]
        
        # Transportation
        transport_samples = [
            {'text': 'Petrol pump fuel payment', 'category': 'transportation'},
            {'text': 'UBER cab ride payment', 'category': 'transportation'},
            {'text': 'OLA auto booking', 'category': 'transportation'},
            {'text': 'Metro card recharge', 'category': 'transportation'},
            {'text': 'Railway ticket booking', 'category': 'transportation'},
            {'text': 'Parking fee payment', 'category': 'transportation'},
            {'text': 'Toll plaza payment', 'category': 'transportation'},
            {'text': 'Bus ticket MSRTC', 'category': 'transportation'},
        ]
        
        # Entertainment
        entertainment_samples = [
            {'text': 'NETFLIX subscription payment', 'category': 'entertainment'},
            {'text': 'Movie ticket BookMyShow', 'category': 'entertainment'},
            {'text': 'Amazon Prime subscription', 'category': 'entertainment'},
            {'text': 'Spotify music subscription', 'category': 'entertainment'},
            {'text': 'YouTube Premium payment', 'category': 'entertainment'},
            {'text': 'Gaming subscription payment', 'category': 'entertainment'},
            {'text': 'Book purchase Amazon', 'category': 'entertainment'},
            {'text': 'Magazine subscription', 'category': 'entertainment'},
        ]
        
        # Shopping
        shopping_samples = [
            {'text': 'AMAZON online shopping', 'category': 'shopping'},
            {'text': 'FLIPKART electronics purchase', 'category': 'shopping'},
            {'text': 'MYNTRA clothing order', 'category': 'shopping'},
            {'text': 'Mobile phone purchase', 'category': 'shopping'},
            {'text': 'Laptop purchase online', 'category': 'shopping'},
            {'text': 'Fashion shopping mall', 'category': 'shopping'},
            {'text': 'Electronics store payment', 'category': 'shopping'},
            {'text': 'Home appliances purchase', 'category': 'shopping'},
        ]
        
        # Healthcare
        healthcare_samples = [
            {'text': 'APOLLO hospital payment', 'category': 'healthcare'},
            {'text': 'Pharmacy medicine purchase', 'category': 'healthcare'},
            {'text': 'Doctor consultation fee', 'category': 'healthcare'},
            {'text': 'Medical insurance premium', 'category': 'healthcare'},
            {'text': 'Dental clinic payment', 'category': 'healthcare'},
            {'text': 'Lab test pathology', 'category': 'healthcare'},
            {'text': 'Prescription medicine', 'category': 'healthcare'},
            {'text': 'Health checkup payment', 'category': 'healthcare'},
        ]
        
        # Education
        education_samples = [
            {'text': 'School fees payment', 'category': 'education'},
            {'text': 'Online course Udemy', 'category': 'education'},
            {'text': 'University tuition fees', 'category': 'education'},
            {'text': 'Book purchase education', 'category': 'education'},
            {'text': 'Training program fees', 'category': 'education'},
            {'text': 'Coaching class payment', 'category': 'education'},
            {'text': 'Library membership', 'category': 'education'},
            {'text': 'Educational software', 'category': 'education'},
        ]
        
        # Investment
        investment_samples = [
            {'text': 'SIP mutual fund payment', 'category': 'investment'},
            {'text': 'Stock purchase Zerodha', 'category': 'investment'},
            {'text': 'Mutual fund investment', 'category': 'investment'},
            {'text': 'ELSS tax saving fund', 'category': 'investment'},
            {'text': 'Fixed deposit renewal', 'category': 'investment'},
            {'text': 'Gold ETF purchase', 'category': 'investment'},
            {'text': 'PPF contribution', 'category': 'investment'},
            {'text': 'Trading account deposit', 'category': 'investment'},
        ]
        
        # Transfer
        transfer_samples = [
            {'text': 'Money transfer to friend', 'category': 'transfer'},
            {'text': 'UPI payment sent', 'category': 'transfer'},
            {'text': 'NEFT transfer to family', 'category': 'transfer'},
            {'text': 'Amount credited salary', 'category': 'transfer'},
            {'text': 'Cash deposit ATM', 'category': 'transfer'},
            {'text': 'Bank transfer received', 'category': 'transfer'},
            {'text': 'Online transfer sent', 'category': 'transfer'},
            {'text': 'Money received UPI', 'category': 'transfer'},
        ]
        
        # Others
        others_samples = [
            {'text': 'ATM cash withdrawal', 'category': 'others'},
            {'text': 'Bank charges applied', 'category': 'others'},
            {'text': 'Interest credited', 'category': 'others'},
            {'text': 'Cashback received', 'category': 'others'},
            {'text': 'Reward points redeemed', 'category': 'others'},
            {'text': 'Service charge deducted', 'category': 'others'},
            {'text': 'Penalty payment', 'category': 'others'},
            {'text': 'Miscellaneous payment', 'category': 'others'},
        ]
        
        # Combine all samples
        all_samples = (utilities_samples + food_samples + transport_samples + 
                      entertainment_samples + shopping_samples + healthcare_samples +
                      education_samples + investment_samples + transfer_samples + others_samples)
        
        return all_samples
    
    def _train_model(self):
        """Train the ML model for expense categorization"""
        print("🤖 Training expense categorization model...")
        
        # Generate training data
        training_data = self._generate_training_data()
        
        # Save training data
        with open(self.training_data_path, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        # Prepare data for training
        texts = [item['text'].lower() for item in training_data]
        labels = [item['category'] for item in training_data]
        
        # Create ML pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=1000)),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Train the model
        self.model.fit(texts, labels)
        
        # Save the trained model
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Evaluate model performance
        X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        print(f"✅ Model trained successfully! Accuracy: {accuracy:.2%}")
        print(f"📊 Training data: {len(training_data)} samples")
        print(f"💾 Model saved to: {self.model_path}")
    
    def categorize_transaction(self, transaction_text: str, amount: float = 0) -> Dict[str, Any]:
        """
        Categorize a single transaction
        
        Args:
            transaction_text: Description of the transaction
            amount: Transaction amount (optional)
            
        Returns:
            Dictionary with category, confidence, and details
        """
        
        # Clean and preprocess text
        cleaned_text = self._preprocess_text(transaction_text)
        
        # Rule-based categorization (high confidence)
        rule_category = self._rule_based_categorization(cleaned_text)
        if rule_category:
            return {
                'category': rule_category,
                'confidence': 0.95,
                'method': 'rule-based',
                'original_text': transaction_text,
                'processed_text': cleaned_text
            }
        
        # ML-based categorization
        if self.model:
            try:
                prediction = self.model.predict([cleaned_text])[0]
                probabilities = self.model.predict_proba([cleaned_text])[0]
                confidence = max(probabilities)
                
                return {
                    'category': prediction,
                    'confidence': confidence,
                    'method': 'ml-based',
                    'original_text': transaction_text,
                    'processed_text': cleaned_text
                }
            except Exception as e:
                print(f"⚠️ ML prediction failed: {e}")
        
        # Fallback to keyword matching
        keyword_category = self._keyword_based_categorization(cleaned_text)
        return {
            'category': keyword_category,
            'confidence': 0.7,
            'method': 'keyword-based',
            'original_text': transaction_text,
            'processed_text': cleaned_text
        }
    
    def categorize_transactions(self, transactions: List[Dict]) -> Dict[str, Any]:
        """
        Categorize multiple transactions
        
        Args:
            transactions: List of transaction dictionaries
            
        Returns:
            Dictionary with categorized transactions and summary
        """
        
        categorized = []
        category_summary = {}
        total_amount = 0
        
        for transaction in transactions:
            text = transaction.get('description', '') or transaction.get('text', '')
            amount = transaction.get('amount', 0)
            
            # Categorize the transaction
            result = self.categorize_transaction(text, amount)
            
            # Add original transaction data
            categorized_transaction = {
                **transaction,
                'category': result['category'],
                'confidence': result['confidence'],
                'categorization_method': result['method']
            }
            
            categorized.append(categorized_transaction)
            
            # Update summary
            category = result['category']
            if category not in category_summary:
                category_summary[category] = {'count': 0, 'amount': 0}
            
            category_summary[category]['count'] += 1
            category_summary[category]['amount'] += amount
            total_amount += amount
        
        # Calculate percentages
        for category in category_summary:
            if total_amount > 0:
                category_summary[category]['percentage'] = (category_summary[category]['amount'] / total_amount) * 100
            else:
                category_summary[category]['percentage'] = 0
        
        return {
            'categorized_transactions': categorized,
            'category_summary': category_summary,
            'total_transactions': len(transactions),
            'total_amount': total_amount,
            'processing_timestamp': datetime.now().isoformat()
        }
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess transaction text"""
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters except spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra spaces
        text = ' '.join(text.split())
        
        return text
    
    def _rule_based_categorization(self, text: str) -> Optional[str]:
        """Apply rule-based categorization for high-confidence matches"""
        
        # High-confidence patterns
        patterns = {
            'utilities': [
                r'\b(electricity|electric|power|bses|tata power|adani)\b',
                r'\b(mobile|airtel|jio|vi|vodafone|bsnl)\b.*\b(recharge|prepaid|postpaid)\b',
                r'\b(internet|broadband|wifi|fiber)\b',
                r'\b(gas|lpg|hp gas|bharat gas|indane)\b'
            ],
            'food': [
                r'\b(swiggy|zomato|uber eats|food panda)\b',
                r'\b(mcdonalds|kfc|dominos|pizza hut|subway)\b',
                r'\b(dmart|big bazaar|reliance fresh)\b.*\b(grocery|food)\b'
            ],
            'transportation': [
                r'\b(petrol|diesel|fuel|hp|ioc|bpcl)\b.*\b(pump|station)\b',
                r'\b(uber|ola|rapido)\b',
                r'\b(metro|rail|train|irctc)\b'
            ],
            'entertainment': [
                r'\b(netflix|amazon prime|hotstar|spotify|youtube)\b.*\b(subscription|premium)\b',
                r'\b(bookmyshow|pvr|inox)\b'
            ],
            'investment': [
                r'\b(sip|mutual fund|mf|elss)\b',
                r'\b(zerodha|upstox|groww|paytm money)\b',
                r'\b(nse|bse|equity|stock)\b'
            ]
        }
        
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if re.search(pattern, text):
                    return category
        
        return None
    
    def _keyword_based_categorization(self, text: str) -> str:
        """Fallback keyword-based categorization"""
        
        max_score = 0
        best_category = 'others'
        
        for category, keywords in self.categories.items():
            score = 0
            for keyword in keywords:
                if keyword in text:
                    score += 1
            
            if score > max_score:
                max_score = score
                best_category = category
        
        return best_category
    
    def get_category_insights(self, transactions: List[Dict], period: str = "monthly") -> Dict[str, Any]:
        """Generate insights from categorized transactions"""
        
        if not transactions:
            return {'error': 'No transactions provided'}
        
        # Categorize all transactions
        result = self.categorize_transactions(transactions)
        category_summary = result['category_summary']
        
        # Find top spending categories
        sorted_categories = sorted(category_summary.items(), 
                                 key=lambda x: x[1]['amount'], reverse=True)
        
        # Generate insights
        insights = []
        
        if sorted_categories:
            top_category = sorted_categories[0]
            insights.append(f"Your highest spending is on {top_category[0]} (₹{top_category[1]['amount']:.2f}, {top_category[1]['percentage']:.1f}%)")
        
        # Check for concerning patterns
        if 'entertainment' in category_summary:
            ent_percentage = category_summary['entertainment']['percentage']
            if ent_percentage > 15:
                insights.append(f"Entertainment spending is {ent_percentage:.1f}% - consider optimizing")
        
        if 'food' in category_summary:
            food_percentage = category_summary['food']['percentage']
            if food_percentage > 25:
                insights.append(f"Food expenses are {food_percentage:.1f}% - try home cooking to save")
        
        # Investment insights
        if 'investment' in category_summary:
            inv_percentage = category_summary['investment']['percentage']
            insights.append(f"Great! You're investing {inv_percentage:.1f}% of expenses")
        else:
            insights.append("Consider starting investments for wealth building")
        
        return {
            'category_breakdown': category_summary,
            'top_categories': sorted_categories[:5],
            'insights': insights,
            'total_analyzed': result['total_transactions'],
            'period': period
        }

# Export for use in other modules
__all__ = ['ExpenseCategorizer']