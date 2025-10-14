"""
Inference Pipeline for Expense Categorization ML Model

This module provides a high-level interface for making predictions on new transactions
with confidence scores and category recommendations.
"""

import pandas as pd
import numpy as np
import os
import logging
from typing import Dict, List, Any, Optional, Union
import json
from datetime import datetime

from .preprocessor import TransactionPreprocessor
from .feature_extractor import FeatureExtractor
from .model import ExpenseCategoryModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpenseCategorizer:
    """
    A complete inference pipeline for expense categorization.
    
    This class provides a simple interface for categorizing expenses using
    the trained ML model with preprocessing and feature extraction.
    """
    
    def __init__(self, model_path: str = None, feature_extractor_path: str = None):
        """
        Initialize the expense categorizer.
        
        Args:
            model_path (str): Path to the saved model
            feature_extractor_path (str): Path to the saved feature extractor
        """
        self.preprocessor = TransactionPreprocessor()
        self.feature_extractor = None
        self.model = None
        self.is_loaded = False
        
        # Default paths
        self.default_model_path = "models/expense_category_model.joblib"
        self.default_feature_extractor_path = "models/feature_extractor.joblib"
        
        # Load models if paths provided
        if model_path and feature_extractor_path:
            self.load_models(model_path, feature_extractor_path)
    
    def load_models(self, model_path: str, feature_extractor_path: str):
        """
        Load the trained model and feature extractor.
        
        Args:
            model_path (str): Path to the saved model
            feature_extractor_path (str): Path to the saved feature extractor
        """
        try:
            logger.info("Loading trained models...")
            
            # Load feature extractor
            self.feature_extractor = FeatureExtractor.load(feature_extractor_path)
            logger.info("Feature extractor loaded successfully")
            
            # Load model
            self.model = ExpenseCategoryModel.load_model(model_path)
            logger.info("Model loaded successfully")
            
            self.is_loaded = True
            logger.info("All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.is_loaded = False
            raise
    
    def predict_single_transaction(self, merchant: str, description: str, 
                                 amount: float = None) -> Dict[str, Any]:
        """
        Predict category for a single transaction.
        
        Args:
            merchant (str): Merchant name
            description (str): Transaction description
            amount (float): Transaction amount (optional)
            
        Returns:
            Dict[str, Any]: Prediction result with confidence
        """
        if not self.is_loaded:
            raise ValueError("Models must be loaded before making predictions")
        
        # Create transaction dataframe
        transaction_data = {
            'merchant': [merchant],
            'description': [description],
            'amount': [amount if amount is not None else 0]
        }
        
        df = pd.DataFrame(transaction_data)
        
        # Preprocess
        processed_df = self.preprocessor.preprocess_dataframe(df)
        
        # Extract features
        X = self.feature_extractor.transform(processed_df)
        
        # Make prediction
        predictions = self.model.predict_with_confidence(X, self.feature_extractor.label_encoder)
        
        result = predictions[0]
        result.update({
            'input': {
                'merchant': merchant,
                'description': description,
                'amount': amount
            },
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def predict_batch_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Predict categories for multiple transactions.
        
        Args:
            transactions (List[Dict]): List of transaction dictionaries
            
        Returns:
            List[Dict[str, Any]]: List of prediction results
        """
        if not self.is_loaded:
            raise ValueError("Models must be loaded before making predictions")
        
        if not transactions:
            return []
        
        # Convert to dataframe
        df = pd.DataFrame(transactions)
        
        # Ensure required columns exist
        required_columns = ['merchant', 'description']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in transactions")
        
        # Add amount column if missing
        if 'amount' not in df.columns:
            df['amount'] = 0
        
        # Preprocess
        processed_df = self.preprocessor.preprocess_dataframe(df)
        
        # Extract features
        X = self.feature_extractor.transform(processed_df)
        
        # Make predictions
        predictions = self.model.predict_with_confidence(X, self.feature_extractor.label_encoder)
        
        # Add input data and timestamp to results
        for i, pred in enumerate(predictions):
            pred.update({
                'input': transactions[i],
                'timestamp': datetime.now().isoformat()
            })
        
        return predictions
    
    def categorize_expense(self, merchant: str, description: str, 
                         amount: float = None, return_alternatives: bool = True) -> Dict[str, Any]:
        """
        High-level function to categorize an expense with detailed output.
        
        Args:
            merchant (str): Merchant name
            description (str): Transaction description
            amount (float): Transaction amount
            return_alternatives (bool): Whether to include alternative categories
            
        Returns:
            Dict[str, Any]: Detailed categorization result
        """
        prediction = self.predict_single_transaction(merchant, description, amount)
        
        # Format result for user-friendly output
        result = {
            'category': prediction['predicted_category'],
            'confidence': round(prediction['confidence'], 3),
            'confidence_level': self._get_confidence_level(prediction['confidence'])
        }
        
        if return_alternatives and len(prediction['top_predictions']) > 1:
            alternatives = []
            for alt in prediction['top_predictions'][1:]:  # Skip the top prediction
                alternatives.append({
                    'category': alt['category'],
                    'probability': round(alt['probability'], 3)
                })
            result['alternatives'] = alternatives
        
        return result
    
    def _get_confidence_level(self, confidence: float) -> str:
        """
        Convert confidence score to descriptive level.
        
        Args:
            confidence (float): Confidence score
            
        Returns:
            str: Confidence level description
        """
        if confidence >= 0.9:
            return "Very High"
        elif confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        elif confidence >= 0.3:
            return "Low"
        else:
            return "Very Low"
    
    def analyze_spending_pattern(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze spending patterns from a list of transactions.
        
        Args:
            transactions (List[Dict]): List of transactions
            
        Returns:
            Dict[str, Any]: Spending pattern analysis
        """
        if not transactions:
            return {"error": "No transactions provided"}
        
        # Get predictions for all transactions
        predictions = self.predict_batch_transactions(transactions)
        
        # Analyze patterns
        categories = [pred['predicted_category'] for pred in predictions]
        amounts = [trans.get('amount', 0) for trans in transactions]
        
        # Category distribution by count
        category_counts = {}
        category_amounts = {}
        
        for i, category in enumerate(categories):
            category_counts[category] = category_counts.get(category, 0) + 1
            category_amounts[category] = category_amounts.get(category, 0) + amounts[i]
        
        # Calculate percentages
        total_transactions = len(transactions)
        total_amount = sum(amounts)
        
        category_analysis = {}
        for category in category_counts:
            category_analysis[category] = {
                'transaction_count': category_counts[category],
                'transaction_percentage': round((category_counts[category] / total_transactions) * 100, 2),
                'total_amount': round(category_amounts[category], 2),
                'amount_percentage': round((category_amounts[category] / total_amount) * 100, 2) if total_amount > 0 else 0,
                'average_amount': round(category_amounts[category] / category_counts[category], 2)
            }
        
        # Find top categories
        top_by_count = sorted(category_analysis.items(), key=lambda x: x[1]['transaction_count'], reverse=True)
        top_by_amount = sorted(category_analysis.items(), key=lambda x: x[1]['total_amount'], reverse=True)
        
        analysis = {
            'summary': {
                'total_transactions': total_transactions,
                'total_amount': round(total_amount, 2),
                'unique_categories': len(category_analysis),
                'average_transaction_amount': round(total_amount / total_transactions, 2) if total_transactions > 0 else 0
            },
            'category_breakdown': category_analysis,
            'top_categories_by_count': [{'category': cat, **data} for cat, data in top_by_count[:5]],
            'top_categories_by_amount': [{'category': cat, **data} for cat, data in top_by_amount[:5]]
        }
        
        return analysis
    
    def get_budget_recommendations(self, transactions: List[Dict[str, Any]], 
                                 monthly_income: float = None) -> Dict[str, Any]:
        """
        Generate budget recommendations based on spending patterns.
        
        Args:
            transactions (List[Dict]): List of transactions
            monthly_income (float): Monthly income for percentage-based recommendations
            
        Returns:
            Dict[str, Any]: Budget recommendations
        """
        analysis = self.analyze_spending_pattern(transactions)
        
        if 'error' in analysis:
            return analysis
        
        recommendations = {
            'spending_analysis': analysis,
            'recommendations': []
        }
        
        # Standard budget allocation percentages (50/30/20 rule as baseline)
        recommended_allocations = {
            'FOOD_DINING': 15,  # 15% of income
            'TRANSPORTATION': 10,
            'UTILITIES': 8,
            'HEALTHCARE': 5,
            'EDUCATION': 5,
            'ENTERTAINMENT': 7,
            'SHOPPING': 10,
            'INVESTMENT': 20,
            'MISCELLANEOUS': 20
        }
        
        # Analyze current spending vs recommendations
        for category, data in analysis['category_breakdown'].items():
            current_amount = data['total_amount']
            current_percentage = data['amount_percentage']
            
            recommended_percentage = recommended_allocations.get(category, 10)
            
            if monthly_income:
                recommended_amount = (monthly_income * recommended_percentage) / 100
                difference = current_amount - recommended_amount
                
                if difference > 0:
                    recommendation_type = "REDUCE"
                    message = f"Consider reducing {category.lower().replace('_', ' ')} spending by ${difference:.0f}"
                else:
                    recommendation_type = "OK" if abs(difference) < recommended_amount * 0.1 else "INCREASE"
                    message = f"{category.lower().replace('_', ' ').title()} spending is within recommended range"
            else:
                recommendation_type = "ANALYZE"
                message = f"{category.lower().replace('_', ' ').title()} represents {current_percentage:.1f}% of your spending"
            
            recommendations['recommendations'].append({
                'category': category,
                'current_amount': current_amount,
                'current_percentage': current_percentage,
                'recommended_percentage': recommended_percentage,
                'recommendation_type': recommendation_type,
                'message': message
            })
        
        return recommendations
    
    def export_predictions(self, predictions: List[Dict[str, Any]], 
                          filepath: str, format: str = 'json'):
        """
        Export predictions to file.
        
        Args:
            predictions (List[Dict]): List of predictions
            filepath (str): Output file path
            format (str): Export format ('json' or 'csv')
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        if format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(predictions, f, indent=2)
        elif format.lower() == 'csv':
            # Flatten predictions for CSV
            flattened = []
            for pred in predictions:
                row = {
                    'merchant': pred['input']['merchant'],
                    'description': pred['input']['description'],
                    'amount': pred['input'].get('amount', 0),
                    'predicted_category': pred['predicted_category'],
                    'confidence': pred['confidence'],
                    'timestamp': pred['timestamp']
                }
                flattened.append(row)
            
            df = pd.DataFrame(flattened)
            df.to_csv(filepath, index=False)
        else:
            raise ValueError("Format must be 'json' or 'csv'")
        
        logger.info(f"Predictions exported to {filepath}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            Dict[str, Any]: Model information
        """
        if not self.is_loaded:
            return {"error": "Models not loaded"}
        
        return {
            'model_summary': self.model.get_model_summary(),
            'feature_summary': self.feature_extractor.get_feature_summary(),
            'categories': list(self.feature_extractor.label_encoder.classes_),
            'is_loaded': self.is_loaded
        }


def create_sample_transactions() -> List[Dict[str, Any]]:
    """Create sample transactions for testing."""
    return [
        {"merchant": "Zomato", "description": "food delivery order", "amount": 450},
        {"merchant": "Uber", "description": "cab ride to office", "amount": 150},
        {"merchant": "Amazon", "description": "online shopping electronics", "amount": 15000},
        {"merchant": "Netflix", "description": "streaming subscription", "amount": 800},
        {"merchant": "Airtel", "description": "mobile recharge", "amount": 500},
        {"merchant": "Apollo Hospital", "description": "medical consultation", "amount": 1200},
        {"merchant": "Byju's", "description": "online course", "amount": 5000},
        {"merchant": "Zerodha", "description": "stock trading", "amount": 10000},
        {"merchant": "Unknown", "description": "cash withdrawal", "amount": 2000}
    ]


if __name__ == "__main__":
    # Example usage (will work after models are trained)
    print("Expense Categorizer - Inference Pipeline")
    print("Note: This requires trained models to be available")
    
    # Create sample transactions
    sample_transactions = create_sample_transactions()
    print(f"\nCreated {len(sample_transactions)} sample transactions for testing")
    
    # Display sample transactions
    for i, trans in enumerate(sample_transactions, 1):
        print(f"{i}. {trans['merchant']}: {trans['description']} - ₹{trans['amount']}")