"""
Data Preprocessing Module for Expense Categorization ML Model

This module handles all data cleaning, preprocessing, and feature engineering
for transaction data to prepare it for machine learning model training.
"""

import pandas as pd
import numpy as np
import re
import string
from typing import List, Dict, Tuple, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransactionPreprocessor:
    """
    A comprehensive preprocessing class for transaction data.
    
    This class handles text cleaning, normalization, and feature engineering
    for transaction descriptions and merchant names.
    """
    
    def __init__(self):
        """Initialize the preprocessor with necessary NLTK components."""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Banking and payment-specific stopwords to remove
        self.banking_stopwords = {
            'upi', 'inr', 'rs', 'rupees', 'payment', 'transaction', 'txn',
            'transfer', 'debit', 'credit', 'account', 'acc', 'bank',
            'online', 'mobile', 'app', 'wallet', 'pay', 'paid', 'bill',
            'purchase', 'buy', 'bought', 'order', 'booking', 'booking',
            'service', 'charge', 'fee', 'amount', 'total', 'balance',
            'card', 'visa', 'mastercard', 'amex', 'american', 'express',
            'pos', 'atm', 'withdrawal', 'deposit', 'refund', 'reversal'
        }
        
        # Common merchant name variations and their standardized forms
        self.merchant_mapping = {
            'zomato': ['zomato', 'zomato india', 'zomato food', 'zomato delivery'],
            'swiggy': ['swiggy', 'swiggy food', 'swiggy delivery', 'swiggy india'],
            'amazon': ['amazon', 'amazon.in', 'amazon india', 'amazon pay', 'amzn'],
            'flipkart': ['flipkart', 'flipkart india', 'fkrt'],
            'uber': ['uber', 'uber india', 'uber technologies'],
            'ola': ['ola', 'ola cabs', 'ola mobility'],
            'netflix': ['netflix', 'netflix.com', 'netflix india'],
            'spotify': ['spotify', 'spotify india', 'spotify premium'],
            'paytm': ['paytm', 'paytm payments'],
            'gpay': ['google pay', 'googlepay', 'gpay', 'google'],
            'phonepe': ['phonepe', 'phone pe', 'phonepe india'],
            'bigbasket': ['bigbasket', 'big basket', 'bb'],
            'grofers': ['grofers', 'blinkit', 'grofers india'],
            'myntra': ['myntra', 'myntra designs'],
            'ajio': ['ajio', 'ajio.com'],
            'nykaa': ['nykaa', 'nykaa.com'],
            'bookmyshow': ['bookmyshow', 'book my show', 'bms'],
            'irctc': ['irctc', 'indian railway'],
            'makemytrip': ['makemytrip', 'make my trip', 'mmt'],
            'goibibo': ['goibibo', 'goibibo.com'],
            'airtel': ['airtel', 'bharti airtel'],
            'jio': ['jio', 'reliance jio', 'jiophone'],
            'vodafone': ['vodafone', 'vodafone idea', 'vi']
        }
        
        # Predefined expense categories
        self.categories = [
            'FOOD_DINING',
            'TRANSPORTATION', 
            'SHOPPING',
            'ENTERTAINMENT',
            'UTILITIES',
            'HEALTHCARE',
            'EDUCATION',
            'INVESTMENT',
            'MISCELLANEOUS'
        ]
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned and normalized text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to lowercase
        text = str(text).lower().strip()
        
        # Remove special characters and numbers (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove banking-specific terms
        words = text.split()
        words = [word for word in words if word not in self.banking_stopwords]
        text = ' '.join(words)
        
        return text
    
    def preprocess_merchant_name(self, merchant: str) -> str:
        """
        Standardize merchant names using predefined mappings.
        
        Args:
            merchant (str): Raw merchant name
            
        Returns:
            str: Standardized merchant name
        """
        if pd.isna(merchant) or merchant is None:
            return "unknown"
        
        merchant_clean = self.clean_text(merchant)
        
        # Check for known merchant mappings
        for standard_name, variations in self.merchant_mapping.items():
            for variation in variations:
                if variation in merchant_clean:
                    return standard_name
        
        return merchant_clean
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract meaningful keywords from text using NLP techniques.
        
        Args:
            text (str): Input text
            
        Returns:
            List[str]: List of extracted keywords
        """
        if not text or pd.isna(text):
            return []
        
        # Tokenize
        tokens = word_tokenize(str(text).lower())
        
        # Remove stopwords and punctuation
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and token not in string.punctuation
                 and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return tokens
    
    def create_features_from_text(self, description: str, merchant: str) -> Dict[str, any]:
        """
        Create engineered features from text data.
        
        Args:
            description (str): Transaction description
            merchant (str): Merchant name
            
        Returns:
            Dict: Dictionary of engineered features
        """
        features = {}
        
        # Clean text
        desc_clean = self.clean_text(description)
        merchant_clean = self.preprocess_merchant_name(merchant)
        
        # Combine description and merchant for better context
        combined_text = f"{desc_clean} {merchant_clean}"
        
        # Extract keywords
        keywords = self.extract_keywords(combined_text)
        
        # Text length features
        features['desc_length'] = len(desc_clean)
        features['merchant_length'] = len(merchant_clean)
        features['keyword_count'] = len(keywords)
        
        # Check for category-specific keywords
        category_keywords = {
            'food': ['food', 'restaurant', 'cafe', 'delivery', 'dining', 'meal', 'lunch', 'dinner', 'breakfast'],
            'transport': ['cab', 'taxi', 'uber', 'ola', 'bus', 'train', 'metro', 'fuel', 'petrol', 'diesel'],
            'shopping': ['shopping', 'store', 'mall', 'retail', 'clothes', 'fashion', 'electronics'],
            'entertainment': ['movie', 'cinema', 'show', 'music', 'game', 'streaming', 'subscription'],
            'utilities': ['electricity', 'gas', 'water', 'internet', 'phone', 'mobile', 'recharge'],
            'healthcare': ['medical', 'doctor', 'hospital', 'pharmacy', 'medicine', 'health'],
            'education': ['school', 'college', 'university', 'course', 'book', 'education', 'study']
        }
        
        for category, category_words in category_keywords.items():
            features[f'has_{category}_keywords'] = any(word in combined_text for word in category_words)
        
        # Store processed text for vectorization
        features['processed_text'] = combined_text
        features['keywords'] = ' '.join(keywords)
        
        return features
    
    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess entire dataframe of transactions.
        
        Args:
            df (pd.DataFrame): Input dataframe with transaction data
            
        Returns:
            pd.DataFrame: Preprocessed dataframe with engineered features
        """
        logger.info(f"Preprocessing {len(df)} transactions...")
        
        # Create a copy to avoid modifying original data
        df_processed = df.copy()
        
        # Handle missing values
        df_processed['description'] = df_processed['description'].fillna('')
        df_processed['merchant'] = df_processed['merchant'].fillna('unknown')
        df_processed['amount'] = pd.to_numeric(df_processed['amount'], errors='coerce').fillna(0)
        
        # Extract features for each row
        feature_list = []
        for idx, row in df_processed.iterrows():
            features = self.create_features_from_text(row['description'], row['merchant'])
            feature_list.append(features)
        
        # Convert features to dataframe and merge
        feature_df = pd.DataFrame(feature_list)
        df_processed = pd.concat([df_processed, feature_df], axis=1)
        
        # Add amount-based features
        if 'amount' in df_processed.columns:
            df_processed['amount_log'] = np.log1p(df_processed['amount'])
            df_processed['amount_category'] = pd.cut(df_processed['amount'], 
                                                   bins=[0, 100, 500, 1000, 5000, float('inf')],
                                                   labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        logger.info("Preprocessing completed successfully!")
        return df_processed
    
    def create_sample_data(self) -> pd.DataFrame:
        """
        Create sample transaction data for testing and training.
        
        Returns:
            pd.DataFrame: Sample transaction dataframe
        """
        sample_data = [
            # Food & Dining
            {"merchant": "Zomato", "description": "food delivery order", "amount": 450, "category": "FOOD_DINING"},
            {"merchant": "Swiggy", "description": "restaurant food delivery", "amount": 350, "category": "FOOD_DINING"},
            {"merchant": "McDonald's", "description": "fast food purchase", "amount": 250, "category": "FOOD_DINING"},
            {"merchant": "Starbucks", "description": "coffee and snacks", "amount": 300, "category": "FOOD_DINING"},
            {"merchant": "KFC", "description": "chicken meal", "amount": 400, "category": "FOOD_DINING"},
            {"merchant": "Dominos", "description": "pizza delivery", "amount": 600, "category": "FOOD_DINING"},
            {"merchant": "Cafe Coffee Day", "description": "coffee shop", "amount": 200, "category": "FOOD_DINING"},
            
            # Transportation
            {"merchant": "Uber", "description": "cab ride to office", "amount": 150, "category": "TRANSPORTATION"},
            {"merchant": "Ola", "description": "taxi booking", "amount": 120, "category": "TRANSPORTATION"},
            {"merchant": "Indian Oil", "description": "petrol fuel", "amount": 2000, "category": "TRANSPORTATION"},
            {"merchant": "Metro", "description": "metro card recharge", "amount": 100, "category": "TRANSPORTATION"},
            {"merchant": "BMTC", "description": "bus ticket", "amount": 50, "category": "TRANSPORTATION"},
            {"merchant": "IRCTC", "description": "train ticket booking", "amount": 800, "category": "TRANSPORTATION"},
            
            # Shopping
            {"merchant": "Amazon", "description": "online shopping electronics", "amount": 15000, "category": "SHOPPING"},
            {"merchant": "Flipkart", "description": "mobile phone purchase", "amount": 25000, "category": "SHOPPING"},
            {"merchant": "Myntra", "description": "clothing and fashion", "amount": 2500, "category": "SHOPPING"},
            {"merchant": "Ajio", "description": "fashion apparel", "amount": 1800, "category": "SHOPPING"},
            {"merchant": "Nykaa", "description": "cosmetics and beauty", "amount": 1200, "category": "SHOPPING"},
            {"merchant": "BigBasket", "description": "grocery shopping", "amount": 800, "category": "SHOPPING"},
            {"merchant": "DMart", "description": "retail store purchase", "amount": 1500, "category": "SHOPPING"},
            
            # Entertainment
            {"merchant": "Netflix", "description": "streaming subscription", "amount": 800, "category": "ENTERTAINMENT"},
            {"merchant": "Spotify", "description": "music subscription", "amount": 500, "category": "ENTERTAINMENT"},
            {"merchant": "BookMyShow", "description": "movie ticket booking", "amount": 600, "category": "ENTERTAINMENT"},
            {"merchant": "YouTube", "description": "premium subscription", "amount": 400, "category": "ENTERTAINMENT"},
            {"merchant": "Amazon Prime", "description": "video subscription", "amount": 1000, "category": "ENTERTAINMENT"},
            {"merchant": "Hotstar", "description": "streaming service", "amount": 900, "category": "ENTERTAINMENT"},
            
            # Utilities
            {"merchant": "Airtel", "description": "mobile recharge", "amount": 500, "category": "UTILITIES"},
            {"merchant": "Jio", "description": "phone bill payment", "amount": 600, "category": "UTILITIES"},
            {"merchant": "BSES", "description": "electricity bill", "amount": 2500, "category": "UTILITIES"},
            {"merchant": "Indane Gas", "description": "lpg cylinder", "amount": 800, "category": "UTILITIES"},
            {"merchant": "Vodafone", "description": "internet bill", "amount": 1000, "category": "UTILITIES"},
            {"merchant": "Tata Sky", "description": "dth recharge", "amount": 400, "category": "UTILITIES"},
            
            # Healthcare
            {"merchant": "Apollo Hospital", "description": "medical consultation", "amount": 1200, "category": "HEALTHCARE"},
            {"merchant": "MedPlus", "description": "pharmacy medicines", "amount": 800, "category": "HEALTHCARE"},
            {"merchant": "Practo", "description": "doctor appointment", "amount": 500, "category": "HEALTHCARE"},
            {"merchant": "Max Hospital", "description": "health checkup", "amount": 3000, "category": "HEALTHCARE"},
            {"merchant": "1mg", "description": "online medicine", "amount": 600, "category": "HEALTHCARE"},
            
            # Education
            {"merchant": "Byju's", "description": "online course", "amount": 5000, "category": "EDUCATION"},
            {"merchant": "Unacademy", "description": "educational subscription", "amount": 2000, "category": "EDUCATION"},
            {"merchant": "Coursera", "description": "online learning", "amount": 4000, "category": "EDUCATION"},
            {"merchant": "Udemy", "description": "course purchase", "amount": 1500, "category": "EDUCATION"},
            {"merchant": "Khan Academy", "description": "education donation", "amount": 1000, "category": "EDUCATION"},
            
            # Investment
            {"merchant": "Zerodha", "description": "stock trading", "amount": 10000, "category": "INVESTMENT"},
            {"merchant": "Groww", "description": "mutual fund", "amount": 5000, "category": "INVESTMENT"},
            {"merchant": "SBI", "description": "fixed deposit", "amount": 50000, "category": "INVESTMENT"},
            {"merchant": "HDFC Bank", "description": "investment account", "amount": 25000, "category": "INVESTMENT"},
            
            # Miscellaneous
            {"merchant": "Unknown", "description": "cash withdrawal", "amount": 2000, "category": "MISCELLANEOUS"},
            {"merchant": "ATM", "description": "cash deposit", "amount": 5000, "category": "MISCELLANEOUS"},
            {"merchant": "Government", "description": "tax payment", "amount": 15000, "category": "MISCELLANEOUS"},
            {"merchant": "Insurance", "description": "policy premium", "amount": 8000, "category": "MISCELLANEOUS"}
        ]
        
        return pd.DataFrame(sample_data)
    
    def get_category_distribution(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Get distribution of categories in the dataset.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Dict[str, int]: Category distribution
        """
        if 'category' in df.columns:
            return df['category'].value_counts().to_dict()
        else:
            return {}


if __name__ == "__main__":
    # Example usage
    preprocessor = TransactionPreprocessor()
    
    # Create sample data
    sample_df = preprocessor.create_sample_data()
    print("Sample data created:")
    print(sample_df.head())
    
    # Preprocess the data
    processed_df = preprocessor.preprocess_dataframe(sample_df)
    print("\nProcessed data shape:", processed_df.shape)
    print("\nProcessed columns:", processed_df.columns.tolist())
    
    # Show category distribution
    distribution = preprocessor.get_category_distribution(processed_df)
    print("\nCategory distribution:", distribution)