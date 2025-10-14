"""
Comprehensive Testing Framework for Expense Categorization ML Model

This module provides extensive testing capabilities for all components
of the machine learning system.
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np
import tempfile
import logging

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.preprocessor import TransactionPreprocessor
from src.feature_extractor import FeatureExtractor
from src.model import ExpenseCategoryModel
from src.inference import ExpenseCategorizer, create_sample_transactions

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestTransactionPreprocessor(unittest.TestCase):
    """Test cases for TransactionPreprocessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = TransactionPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning functionality."""
        # Test normal text
        result = self.preprocessor.clean_text("Zomato Food Delivery - INR 450")
        self.assertIn("zomato", result)
        self.assertNotIn("INR", result)  # Should remove banking terms
        
        # Test with special characters
        result = self.preprocessor.clean_text("Amazon@#$%^& Purchase 123")
        self.assertIn("amazon", result)
        self.assertNotIn("@", result)
        self.assertNotIn("123", result)
        
        # Test empty input
        result = self.preprocessor.clean_text("")
        self.assertEqual(result, "")
        
        # Test None input
        result = self.preprocessor.clean_text(None)
        self.assertEqual(result, "")
    
    def test_preprocess_merchant_name(self):
        """Test merchant name preprocessing."""
        # Test known merchant
        result = self.preprocessor.preprocess_merchant_name("Zomato India")
        self.assertEqual(result, "zomato")
        
        # Test unknown merchant
        result = self.preprocessor.preprocess_merchant_name("Random Merchant")
        self.assertIn("random", result)
        
        # Test None input
        result = self.preprocessor.preprocess_merchant_name(None)
        self.assertEqual(result, "unknown")
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        keywords = self.preprocessor.extract_keywords("Zomato food delivery order")
        self.assertIn("zomato", keywords)
        self.assertIn("food", keywords)
        self.assertIn("delivery", keywords)
        
        # Test empty input
        keywords = self.preprocessor.extract_keywords("")
        self.assertEqual(keywords, [])
    
    def test_create_features_from_text(self):
        """Test feature creation from text."""
        features = self.preprocessor.create_features_from_text(
            "Zomato food delivery", "Zomato"
        )
        
        self.assertIn('desc_length', features)
        self.assertIn('merchant_length', features)
        self.assertIn('keyword_count', features)
        self.assertIn('has_food_keywords', features)
        self.assertIn('processed_text', features)
        
        # Check food keywords detection
        self.assertTrue(features['has_food_keywords'])
    
    def test_preprocess_dataframe(self):
        """Test dataframe preprocessing."""
        # Create sample data
        sample_df = self.preprocessor.create_sample_data()
        processed_df = self.preprocessor.preprocess_dataframe(sample_df)
        
        # Check that new columns are added
        self.assertIn('processed_text', processed_df.columns)
        self.assertIn('desc_length', processed_df.columns)
        self.assertIn('has_food_keywords', processed_df.columns)
        
        # Check that data is not empty
        self.assertGreater(len(processed_df), 0)
    
    def test_get_category_distribution(self):
        """Test category distribution calculation."""
        sample_df = self.preprocessor.create_sample_data()
        distribution = self.preprocessor.get_category_distribution(sample_df)
        
        self.assertIsInstance(distribution, dict)
        self.assertGreater(len(distribution), 0)


class TestFeatureExtractor(unittest.TestCase):
    """Test cases for FeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.feature_extractor = FeatureExtractor()
        self.preprocessor = TransactionPreprocessor()
        
        # Create sample processed data
        sample_df = self.preprocessor.create_sample_data()
        self.processed_df = self.preprocessor.preprocess_dataframe(sample_df)
    
    def test_extract_text_features(self):
        """Test text feature extraction."""
        text_features = self.feature_extractor.extract_text_features(self.processed_df)
        
        self.assertIsInstance(text_features, np.ndarray)
        self.assertEqual(text_features.shape[0], len(self.processed_df))
        self.assertGreater(text_features.shape[1], 0)
    
    def test_extract_numerical_features(self):
        """Test numerical feature extraction."""
        numerical_features = self.feature_extractor.extract_numerical_features(self.processed_df)
        
        self.assertIsInstance(numerical_features, np.ndarray)
        self.assertEqual(numerical_features.shape[0], len(self.processed_df))
    
    def test_extract_boolean_features(self):
        """Test boolean feature extraction."""
        boolean_features = self.feature_extractor.extract_boolean_features(self.processed_df)
        
        self.assertIsInstance(boolean_features, np.ndarray)
        self.assertEqual(boolean_features.shape[0], len(self.processed_df))
    
    def test_fit_transform(self):
        """Test fit and transform functionality."""
        X, feature_names = self.feature_extractor.fit_transform(self.processed_df)
        
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(feature_names, list)
        self.assertEqual(X.shape[0], len(self.processed_df))
        self.assertEqual(X.shape[1], len(feature_names))
        self.assertTrue(self.feature_extractor.is_fitted)
    
    def test_encode_decode_labels(self):
        """Test label encoding and decoding."""
        labels = self.processed_df['category']
        
        # Test encoding
        encoded_labels = self.feature_extractor.encode_labels(labels)
        self.assertIsInstance(encoded_labels, np.ndarray)
        self.assertEqual(len(encoded_labels), len(labels))
        
        # Test decoding
        decoded_labels = self.feature_extractor.decode_labels(encoded_labels)
        self.assertEqual(list(decoded_labels), list(labels))
    
    def test_transform(self):
        """Test transform on new data."""
        # First fit the extractor
        X_train, _ = self.feature_extractor.fit_transform(self.processed_df)
        
        # Transform new data (same data for testing)
        X_test = self.feature_extractor.transform(self.processed_df)
        
        self.assertEqual(X_train.shape, X_test.shape)
    
    def test_save_and_load(self):
        """Test saving and loading feature extractor."""
        # Fit the extractor
        self.feature_extractor.fit_transform(self.processed_df)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            self.feature_extractor.save(tmp.name)
            
            # Load from file
            loaded_extractor = FeatureExtractor.load(tmp.name)
            
            # Test that loaded extractor works
            X_original = self.feature_extractor.transform(self.processed_df)
            X_loaded = loaded_extractor.transform(self.processed_df)
            
            np.testing.assert_array_almost_equal(X_original, X_loaded)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_get_feature_summary(self):
        """Test feature summary generation."""
        self.feature_extractor.fit_transform(self.processed_df)
        summary = self.feature_extractor.get_feature_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('total_features', summary)
        self.assertIn('is_fitted', summary)
        self.assertTrue(summary['is_fitted'])


class TestExpenseCategoryModel(unittest.TestCase):
    """Test cases for ExpenseCategoryModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = ExpenseCategoryModel(random_state=42)
        self.preprocessor = TransactionPreprocessor()
        self.feature_extractor = FeatureExtractor()
        
        # Create sample data
        sample_df = self.preprocessor.create_sample_data()
        processed_df = self.preprocessor.preprocess_dataframe(sample_df)
        self.X, self.feature_names = self.feature_extractor.fit_transform(processed_df)
        self.y = self.feature_extractor.encode_labels(processed_df['category'])
    
    def test_prepare_data(self):
        """Test data preparation for training."""
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y, test_size=0.3)
        
        # Check shapes
        self.assertEqual(len(X_train) + len(X_test), len(self.X))
        self.assertEqual(len(y_train) + len(y_test), len(self.y))
        
        # Check that test size is approximately correct
        test_ratio = len(X_test) / len(self.X)
        self.assertAlmostEqual(test_ratio, 0.3, delta=0.1)
    
    def test_train_single_model(self):
        """Test training a single model."""
        X_train, X_test, y_train, y_test = self.model.prepare_data(self.X, self.y)
        
        # Train random forest (fastest for testing)
        trained_model = self.model.train_single_model(
            X_train, y_train, 'random_forest', tune_hyperparams=False
        )
        
        self.assertIsNotNone(trained_model)
        
        # Test predictions
        y_pred = trained_model.predict(X_test)
        self.assertEqual(len(y_pred), len(y_test))
    
    def test_predict_with_confidence(self):
        """Test prediction with confidence scores."""
        # First train the model
        training_results = self.model.train(
            self.X, self.y, self.feature_names, 
            self.feature_extractor.label_encoder, 
            model_name='random_forest'
        )
        
        # Test predictions
        predictions = self.model.predict_with_confidence(
            self.X[:5], self.feature_extractor.label_encoder
        )
        
        self.assertEqual(len(predictions), 5)
        
        for pred in predictions:
            self.assertIn('predicted_category', pred)
            self.assertIn('confidence', pred)
            self.assertIn('top_predictions', pred)
            self.assertGreaterEqual(pred['confidence'], 0)
            self.assertLessEqual(pred['confidence'], 1)
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        # Train the model first
        self.model.train(
            self.X, self.y, self.feature_names,
            self.feature_extractor.label_encoder,
            model_name='random_forest'
        )
        
        importance = self.model.get_feature_importance(top_n=10)
        
        self.assertIsInstance(importance, dict)
        self.assertLessEqual(len(importance), 10)
    
    def test_save_and_load_model(self):
        """Test model saving and loading."""
        # Train the model
        self.model.train(
            self.X, self.y, self.feature_names,
            self.feature_extractor.label_encoder,
            model_name='random_forest'
        )
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
            self.model.save_model(tmp.name)
            
            # Load from file
            loaded_model = ExpenseCategoryModel.load_model(tmp.name)
            
            # Test that loaded model works
            pred_original = self.model.predict(self.X[:5])
            pred_loaded = loaded_model.predict(self.X[:5])
            
            np.testing.assert_array_equal(pred_original, pred_loaded)
            
            # Clean up
            os.unlink(tmp.name)
    
    def test_get_model_summary(self):
        """Test model summary generation."""
        summary = self.model.get_model_summary()
        
        self.assertIsInstance(summary, dict)
        self.assertIn('is_trained', summary)
        self.assertFalse(summary['is_trained'])  # Not trained yet


class TestExpenseCategorizer(unittest.TestCase):
    """Test cases for ExpenseCategorizer inference pipeline."""
    
    def setUp(self):
        """Set up test fixtures by training a model."""
        # Note: This requires actual model files to exist
        # In a real scenario, you would train models first
        pass
    
    def test_create_sample_transactions(self):
        """Test sample transaction creation."""
        transactions = create_sample_transactions()
        
        self.assertIsInstance(transactions, list)
        self.assertGreater(len(transactions), 0)
        
        for trans in transactions:
            self.assertIn('merchant', trans)
            self.assertIn('description', trans)
            self.assertIn('amount', trans)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline."""
    
    def test_complete_pipeline(self):
        """Test the complete pipeline from data to predictions."""
        # Step 1: Preprocess data
        preprocessor = TransactionPreprocessor()
        sample_df = preprocessor.create_sample_data()
        processed_df = preprocessor.preprocess_dataframe(sample_df)
        
        # Step 2: Extract features
        feature_extractor = FeatureExtractor()
        X, feature_names = feature_extractor.fit_transform(processed_df)
        y = feature_extractor.encode_labels(processed_df['category'])
        
        # Step 3: Train model
        model = ExpenseCategoryModel(random_state=42)
        training_results = model.train(
            X, y, feature_names, feature_extractor.label_encoder,
            model_name='random_forest'
        )
        
        # Step 4: Test predictions
        predictions = model.predict_with_confidence(X[:5], feature_extractor.label_encoder)
        
        # Verify results
        self.assertIsInstance(training_results, dict)
        self.assertEqual(len(predictions), 5)
        self.assertTrue(model.is_trained)
    
    def test_end_to_end_workflow(self):
        """Test end-to-end workflow with minimal data."""
        # Create minimal sample data
        data = [
            {"merchant": "Zomato", "description": "food delivery", "amount": 500, "category": "FOOD_DINING"},
            {"merchant": "Uber", "description": "cab ride", "amount": 200, "category": "TRANSPORTATION"},
            {"merchant": "Amazon", "description": "shopping", "amount": 1000, "category": "SHOPPING"},
        ]
        
        # Duplicate data to have enough samples
        extended_data = data * 20  # 60 samples total
        df = pd.DataFrame(extended_data)
        
        # Run pipeline
        preprocessor = TransactionPreprocessor()
        processed_df = preprocessor.preprocess_dataframe(df)
        
        feature_extractor = FeatureExtractor()
        X, feature_names = feature_extractor.fit_transform(processed_df)
        y = feature_extractor.encode_labels(processed_df['category'])
        
        model = ExpenseCategoryModel(random_state=42)
        results = model.train(X, y, feature_names, feature_extractor.label_encoder, 'random_forest')
        
        # Test new prediction
        new_transaction = pd.DataFrame([{
            "merchant": "Zomato", "description": "dinner order", "amount": 600
        }])
        
        processed_new = preprocessor.preprocess_dataframe(new_transaction)
        X_new = feature_extractor.transform(processed_new)
        prediction = model.predict_with_confidence(X_new, feature_extractor.label_encoder)
        
        # Verify prediction
        self.assertEqual(len(prediction), 1)
        self.assertIn('predicted_category', prediction[0])


def run_performance_tests():
    """Run performance tests to ensure the system meets requirements."""
    logger.info("Running performance tests...")
    
    import time
    
    # Test preprocessing speed
    preprocessor = TransactionPreprocessor()
    large_df = preprocessor.create_sample_data()
    
    # Duplicate to create larger dataset
    large_df = pd.concat([large_df] * 100, ignore_index=True)  # 4000+ transactions
    
    start_time = time.time()
    processed_df = preprocessor.preprocess_dataframe(large_df)
    preprocessing_time = time.time() - start_time
    
    logger.info(f"Preprocessing {len(large_df)} transactions took {preprocessing_time:.2f} seconds")
    
    # Test feature extraction speed
    feature_extractor = FeatureExtractor()
    
    start_time = time.time()
    X, feature_names = feature_extractor.fit_transform(processed_df)
    feature_extraction_time = time.time() - start_time
    
    logger.info(f"Feature extraction took {feature_extraction_time:.2f} seconds")
    logger.info(f"Created {X.shape[1]} features for {X.shape[0]} samples")
    
    # Performance assertions
    assert preprocessing_time < 30, f"Preprocessing too slow: {preprocessing_time:.2f}s"
    assert feature_extraction_time < 30, f"Feature extraction too slow: {feature_extraction_time:.2f}s"
    assert X.shape[1] > 0, "No features extracted"
    
    logger.info("✅ Performance tests passed!")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("EXPENSE CATEGORIZATION ML MODEL - COMPREHENSIVE TESTING")
    logger.info("=" * 60)
    
    # Run unit tests
    logger.info("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance tests
    run_performance_tests()
    
    logger.info("\n" + "=" * 60)
    logger.info("ALL TESTS COMPLETED!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()