"""
Feature Extraction Module for Expense Categorization ML Model

This module handles feature engineering, text vectorization, and numerical feature
scaling for the machine learning model.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import logging
from typing import List, Dict, Tuple, Optional, Union
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """
    A comprehensive feature extraction class for transaction data.
    
    This class handles text vectorization using TF-IDF, numerical feature scaling,
    and categorical encoding for machine learning model training.
    """
    
    def __init__(self, max_features: int = 5000, min_df: int = 2, max_df: float = 0.95):
        """
        Initialize the feature extractor.
        
        Args:
            max_features (int): Maximum number of features for TF-IDF
            min_df (int): Minimum document frequency for TF-IDF
            max_df (float): Maximum document frequency for TF-IDF
        """
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        # Initialize vectorizers and scalers
        self.text_vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=(1, 2),  # Include both unigrams and bigrams
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b'  # Only alphabetic tokens with 2+ chars
        )
        
        self.keyword_vectorizer = TfidfVectorizer(
            max_features=1000,
            min_df=1,
            max_df=0.95,
            ngram_range=(1, 1),  # Only unigrams for keywords
            stop_words='english',
            lowercase=True
        )
        
        self.numerical_scaler = StandardScaler()
        self.categorical_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        self.label_encoder = LabelEncoder()
        
        # Feature names for tracking
        self.text_feature_names = []
        self.keyword_feature_names = []
        self.numerical_feature_names = []
        self.categorical_feature_names = []
        self.boolean_feature_names = []
        
        # Fitted status
        self.is_fitted = False
        
    def extract_text_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract TF-IDF features from processed text.
        
        Args:
            df (pd.DataFrame): DataFrame with processed text
            
        Returns:
            np.ndarray: TF-IDF feature matrix
        """
        if 'processed_text' not in df.columns:
            raise ValueError("Column 'processed_text' not found in dataframe")
        
        # Fill missing values
        text_data = df['processed_text'].fillna('').astype(str)
        
        if not self.is_fitted:
            # Fit and transform
            text_features = self.text_vectorizer.fit_transform(text_data)
            self.text_feature_names = [f"text_{name}" for name in self.text_vectorizer.get_feature_names_out()]
        else:
            # Only transform
            text_features = self.text_vectorizer.transform(text_data)
        
        return text_features.toarray()
    
    def extract_keyword_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract TF-IDF features from keywords.
        
        Args:
            df (pd.DataFrame): DataFrame with keywords
            
        Returns:
            np.ndarray: Keyword TF-IDF feature matrix
        """
        if 'keywords' not in df.columns:
            raise ValueError("Column 'keywords' not found in dataframe")
        
        # Fill missing values
        keyword_data = df['keywords'].fillna('').astype(str)
        
        if not self.is_fitted:
            # Fit and transform
            keyword_features = self.keyword_vectorizer.fit_transform(keyword_data)
            self.keyword_feature_names = [f"keyword_{name}" for name in self.keyword_vectorizer.get_feature_names_out()]
        else:
            # Only transform
            keyword_features = self.keyword_vectorizer.transform(keyword_data)
        
        return keyword_features.toarray()
    
    def extract_numerical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and scale numerical features.
        
        Args:
            df (pd.DataFrame): DataFrame with numerical features
            
        Returns:
            np.ndarray: Scaled numerical feature matrix
        """
        numerical_columns = [
            'amount', 'amount_log', 'desc_length', 'merchant_length', 'keyword_count'
        ]
        
        # Select only available columns
        available_numerical = [col for col in numerical_columns if col in df.columns]
        
        if not available_numerical:
            logger.warning("No numerical features found")
            return np.array([]).reshape(len(df), 0)
        
        # Extract numerical data and handle missing values
        numerical_data = df[available_numerical].fillna(0).astype(float)
        
        if not self.is_fitted:
            # Fit and transform
            numerical_features = self.numerical_scaler.fit_transform(numerical_data)
            self.numerical_feature_names = [f"num_{name}" for name in available_numerical]
        else:
            # Only transform
            numerical_features = self.numerical_scaler.transform(numerical_data)
        
        return numerical_features
    
    def extract_categorical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and encode categorical features.
        
        Args:
            df (pd.DataFrame): DataFrame with categorical features
            
        Returns:
            np.ndarray: One-hot encoded categorical feature matrix
        """
        categorical_columns = ['amount_category']
        
        # Select only available columns
        available_categorical = [col for col in categorical_columns if col in df.columns]
        
        if not available_categorical:
            logger.warning("No categorical features found")
            return np.array([]).reshape(len(df), 0)
        
        # Extract categorical data and handle missing values
        categorical_data = df[available_categorical].copy()
        for col in available_categorical:
            if categorical_data[col].dtype.name == 'category':
                # Add 'unknown' to categories if not present
                if 'unknown' not in categorical_data[col].cat.categories:
                    categorical_data[col] = categorical_data[col].cat.add_categories(['unknown'])
                categorical_data[col] = categorical_data[col].fillna('unknown')
            else:
                categorical_data[col] = categorical_data[col].fillna('unknown')
        
        categorical_data = categorical_data.astype(str)
        
        if not self.is_fitted:
            # Fit and transform
            categorical_features = self.categorical_encoder.fit_transform(categorical_data)
            # Generate feature names
            cat_names = []
            for i, col in enumerate(available_categorical):
                categories = self.categorical_encoder.categories_[i]
                cat_names.extend([f"cat_{col}_{cat}" for cat in categories])
            self.categorical_feature_names = cat_names
        else:
            # Only transform
            categorical_features = self.categorical_encoder.transform(categorical_data)
        
        return categorical_features
    
    def extract_boolean_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract boolean features.
        
        Args:
            df (pd.DataFrame): DataFrame with boolean features
            
        Returns:
            np.ndarray: Boolean feature matrix
        """
        boolean_columns = [
            'has_food_keywords', 'has_transport_keywords', 'has_shopping_keywords',
            'has_entertainment_keywords', 'has_utilities_keywords', 'has_healthcare_keywords',
            'has_education_keywords'
        ]
        
        # Select only available columns
        available_boolean = [col for col in boolean_columns if col in df.columns]
        
        if not available_boolean:
            logger.warning("No boolean features found")
            return np.array([]).reshape(len(df), 0)
        
        # Extract boolean data and convert to int
        boolean_data = df[available_boolean].fillna(False).astype(int)
        
        if not self.is_fitted:
            self.boolean_feature_names = [f"bool_{name}" for name in available_boolean]
        
        return boolean_data.values
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit the feature extractors and transform the data.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            Tuple[np.ndarray, List[str]]: Feature matrix and feature names
        """
        logger.info("Fitting feature extractors and transforming data...")
        
        # Extract different types of features
        text_features = self.extract_text_features(df)
        keyword_features = self.extract_keyword_features(df)
        numerical_features = self.extract_numerical_features(df)
        categorical_features = self.extract_categorical_features(df)
        boolean_features = self.extract_boolean_features(df)
        
        # Combine all features
        feature_matrices = []
        all_feature_names = []
        
        if text_features.shape[1] > 0:
            feature_matrices.append(text_features)
            all_feature_names.extend(self.text_feature_names)
        
        if keyword_features.shape[1] > 0:
            feature_matrices.append(keyword_features)
            all_feature_names.extend(self.keyword_feature_names)
        
        if numerical_features.shape[1] > 0:
            feature_matrices.append(numerical_features)
            all_feature_names.extend(self.numerical_feature_names)
        
        if categorical_features.shape[1] > 0:
            feature_matrices.append(categorical_features)
            all_feature_names.extend(self.categorical_feature_names)
        
        if boolean_features.shape[1] > 0:
            feature_matrices.append(boolean_features)
            all_feature_names.extend(self.boolean_feature_names)
        
        # Concatenate all features
        if feature_matrices:
            X = np.hstack(feature_matrices)
        else:
            raise ValueError("No features could be extracted from the data")
        
        self.is_fitted = True
        
        logger.info(f"Feature extraction completed. Shape: {X.shape}")
        logger.info(f"Total features: {len(all_feature_names)}")
        
        return X, all_feature_names
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted extractors.
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            np.ndarray: Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Feature extractors must be fitted before transform")
        
        logger.info("Transforming new data...")
        
        # Extract different types of features
        text_features = self.extract_text_features(df)
        keyword_features = self.extract_keyword_features(df)
        numerical_features = self.extract_numerical_features(df)
        categorical_features = self.extract_categorical_features(df)
        boolean_features = self.extract_boolean_features(df)
        
        # Combine all features
        feature_matrices = []
        
        if text_features.shape[1] > 0:
            feature_matrices.append(text_features)
        
        if keyword_features.shape[1] > 0:
            feature_matrices.append(keyword_features)
        
        if numerical_features.shape[1] > 0:
            feature_matrices.append(numerical_features)
        
        if categorical_features.shape[1] > 0:
            feature_matrices.append(categorical_features)
        
        if boolean_features.shape[1] > 0:
            feature_matrices.append(boolean_features)
        
        # Concatenate all features
        if feature_matrices:
            X = np.hstack(feature_matrices)
        else:
            raise ValueError("No features could be extracted from the data")
        
        logger.info(f"Transform completed. Shape: {X.shape}")
        
        return X
    
    def encode_labels(self, labels: pd.Series) -> np.ndarray:
        """
        Encode category labels.
        
        Args:
            labels (pd.Series): Category labels
            
        Returns:
            np.ndarray: Encoded labels
        """
        encoded_labels = self.label_encoder.fit_transform(labels)
        return encoded_labels
    
    def decode_labels(self, encoded_labels: np.ndarray) -> List[str]:
        """
        Decode encoded labels back to categories.
        
        Args:
            encoded_labels (np.ndarray): Encoded labels
            
        Returns:
            List[str]: Decoded category labels
        """
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def get_feature_importance_mapping(self, feature_importance: np.ndarray) -> Dict[str, float]:
        """
        Create a mapping of feature names to their importance scores.
        
        Args:
            feature_importance (np.ndarray): Feature importance scores
            
        Returns:
            Dict[str, float]: Feature name to importance mapping
        """
        all_feature_names = (self.text_feature_names + 
                           self.keyword_feature_names + 
                           self.numerical_feature_names + 
                           self.categorical_feature_names + 
                           self.boolean_feature_names)
        
        if len(feature_importance) != len(all_feature_names):
            logger.warning(f"Mismatch in feature importance length: {len(feature_importance)} vs {len(all_feature_names)}")
            return {}
        
        return dict(zip(all_feature_names, feature_importance))
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted feature extractor.
        
        Args:
            filepath (str): Path to save the feature extractor
        """
        if not self.is_fitted:
            logger.warning("Feature extractor is not fitted. Saving unfitted extractor.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the entire feature extractor object
        joblib.dump(self, filepath)
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """
        Load a fitted feature extractor.
        
        Args:
            filepath (str): Path to the saved feature extractor
            
        Returns:
            FeatureExtractor: Loaded feature extractor
        """
        extractor = joblib.load(filepath)
        extractor.is_fitted = True  # Set fitted flag when loading
        logger.info(f"Feature extractor loaded from {filepath}")
        return extractor
    
    def get_feature_summary(self) -> Dict[str, any]:
        """
        Get a summary of extracted features.
        
        Returns:
            Dict[str, any]: Feature summary
        """
        return {
            'text_features': len(self.text_feature_names),
            'keyword_features': len(self.keyword_feature_names),
            'numerical_features': len(self.numerical_feature_names),
            'categorical_features': len(self.categorical_feature_names),
            'boolean_features': len(self.boolean_feature_names),
            'total_features': (len(self.text_feature_names) + 
                             len(self.keyword_feature_names) + 
                             len(self.numerical_feature_names) + 
                             len(self.categorical_feature_names) + 
                             len(self.boolean_feature_names)),
            'is_fitted': self.is_fitted,
            'label_classes': list(self.label_encoder.classes_) if hasattr(self.label_encoder, 'classes_') else []
        }


if __name__ == "__main__":
    # Example usage
    from .preprocessor import TransactionPreprocessor
    
    # Create sample data
    preprocessor = TransactionPreprocessor()
    sample_df = preprocessor.create_sample_data()
    processed_df = preprocessor.preprocess_dataframe(sample_df)
    
    # Extract features
    feature_extractor = FeatureExtractor()
    X, feature_names = feature_extractor.fit_transform(processed_df)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_names)}")
    print("\nFeature summary:")
    print(feature_extractor.get_feature_summary())
    
    # Encode labels
    y = feature_extractor.encode_labels(processed_df['category'])
    print(f"\nLabel classes: {feature_extractor.label_encoder.classes_}")
    print(f"Encoded labels shape: {y.shape}")