"""
Main Training Script for Expense Categorization ML Model

This script orchestrates the complete training pipeline from data preprocessing
to model training and evaluation.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json
from typing import Dict, List, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.preprocessor import TransactionPreprocessor
from src.feature_extractor import FeatureExtractor
from src.model import ExpenseCategoryModel
from src.inference import ExpenseCategorizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainingPipeline:
    """
    Complete training pipeline for the expense categorization model.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the training pipeline.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.preprocessor = TransactionPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.model = ExpenseCategoryModel(random_state=random_state)
        
        # Training results
        self.training_results = {}
        self.trained_data = None
        
    def create_enhanced_sample_data(self, num_samples_per_category: int = 50) -> pd.DataFrame:
        """
        Create enhanced sample data with more examples per category.
        
        Args:
            num_samples_per_category (int): Number of samples per category
            
        Returns:
            pd.DataFrame: Enhanced sample dataset
        """
        logger.info(f"Creating enhanced sample data with {num_samples_per_category} samples per category...")
        
        # Extended sample data for better training
        extended_data = []
        
        # Food & Dining
        food_data = [
            {"merchant": "Zomato", "description": "food delivery order biryani", "amount": 450, "category": "FOOD_DINING"},
            {"merchant": "Swiggy", "description": "restaurant food delivery pizza", "amount": 350, "category": "FOOD_DINING"},
            {"merchant": "McDonald's", "description": "fast food burger meal", "amount": 250, "category": "FOOD_DINING"},
            {"merchant": "Starbucks", "description": "coffee and snacks latte", "amount": 300, "category": "FOOD_DINING"},
            {"merchant": "KFC", "description": "chicken bucket meal", "amount": 400, "category": "FOOD_DINING"},
            {"merchant": "Dominos", "description": "pizza delivery large", "amount": 600, "category": "FOOD_DINING"},
            {"merchant": "Cafe Coffee Day", "description": "coffee shop cappuccino", "amount": 200, "category": "FOOD_DINING"},
            {"merchant": "Haldiram", "description": "snacks and sweets", "amount": 300, "category": "FOOD_DINING"},
            {"merchant": "Subway", "description": "sandwich meal", "amount": 350, "category": "FOOD_DINING"},
            {"merchant": "Barbeque Nation", "description": "dinner buffet", "amount": 800, "category": "FOOD_DINING"},
            {"merchant": "Local Restaurant", "description": "lunch thali", "amount": 200, "category": "FOOD_DINING"},
            {"merchant": "Food Court", "description": "mall food court", "amount": 400, "category": "FOOD_DINING"},
        ]
        
        # Transportation
        transport_data = [
            {"merchant": "Uber", "description": "cab ride to office", "amount": 150, "category": "TRANSPORTATION"},
            {"merchant": "Ola", "description": "taxi booking airport", "amount": 500, "category": "TRANSPORTATION"},
            {"merchant": "Indian Oil", "description": "petrol fuel pump", "amount": 2000, "category": "TRANSPORTATION"},
            {"merchant": "Metro", "description": "metro card recharge", "amount": 100, "category": "TRANSPORTATION"},
            {"merchant": "BMTC", "description": "bus ticket daily", "amount": 50, "category": "TRANSPORTATION"},
            {"merchant": "IRCTC", "description": "train ticket booking", "amount": 800, "category": "TRANSPORTATION"},
            {"merchant": "Auto Rickshaw", "description": "auto ride local", "amount": 80, "category": "TRANSPORTATION"},
            {"merchant": "Rapido", "description": "bike taxi", "amount": 60, "category": "TRANSPORTATION"},
            {"merchant": "Parking", "description": "mall parking fee", "amount": 40, "category": "TRANSPORTATION"},
            {"merchant": "Toll Plaza", "description": "highway toll", "amount": 200, "category": "TRANSPORTATION"},
        ]
        
        # Shopping
        shopping_data = [
            {"merchant": "Amazon", "description": "online shopping electronics mobile", "amount": 15000, "category": "SHOPPING"},
            {"merchant": "Flipkart", "description": "laptop computer purchase", "amount": 45000, "category": "SHOPPING"},
            {"merchant": "Myntra", "description": "clothing fashion shirt", "amount": 1500, "category": "SHOPPING"},
            {"merchant": "Ajio", "description": "fashion apparel jeans", "amount": 2000, "category": "SHOPPING"},
            {"merchant": "Nykaa", "description": "cosmetics beauty products", "amount": 800, "category": "SHOPPING"},
            {"merchant": "BigBasket", "description": "grocery shopping vegetables", "amount": 1200, "category": "SHOPPING"},
            {"merchant": "DMart", "description": "retail store household", "amount": 2000, "category": "SHOPPING"},
            {"merchant": "Reliance Digital", "description": "electronics store", "amount": 8000, "category": "SHOPPING"},
            {"merchant": "Westside", "description": "clothing store", "amount": 3000, "category": "SHOPPING"},
            {"merchant": "Lifestyle", "description": "fashion retail", "amount": 2500, "category": "SHOPPING"},
        ]
        
        # Entertainment
        entertainment_data = [
            {"merchant": "Netflix", "description": "streaming subscription monthly", "amount": 650, "category": "ENTERTAINMENT"},
            {"merchant": "Spotify", "description": "music subscription premium", "amount": 400, "category": "ENTERTAINMENT"},
            {"merchant": "BookMyShow", "description": "movie ticket booking", "amount": 300, "category": "ENTERTAINMENT"},
            {"merchant": "YouTube", "description": "premium subscription", "amount": 400, "category": "ENTERTAINMENT"},
            {"merchant": "Amazon Prime", "description": "video subscription annual", "amount": 1000, "category": "ENTERTAINMENT"},
            {"merchant": "Hotstar", "description": "streaming service sports", "amount": 900, "category": "ENTERTAINMENT"},
            {"merchant": "PVR Cinemas", "description": "movie tickets multiplex", "amount": 600, "category": "ENTERTAINMENT"},
            {"merchant": "Gaming", "description": "online game purchase", "amount": 1500, "category": "ENTERTAINMENT"},
            {"merchant": "Concert", "description": "music concert ticket", "amount": 2000, "category": "ENTERTAINMENT"},
            {"merchant": "Amusement Park", "description": "theme park entry", "amount": 800, "category": "ENTERTAINMENT"},
        ]
        
        # Utilities
        utilities_data = [
            {"merchant": "Airtel", "description": "mobile recharge postpaid", "amount": 600, "category": "UTILITIES"},
            {"merchant": "Jio", "description": "phone bill payment", "amount": 500, "category": "UTILITIES"},
            {"merchant": "BSES", "description": "electricity bill monthly", "amount": 2500, "category": "UTILITIES"},
            {"merchant": "Indane Gas", "description": "lpg cylinder refill", "amount": 900, "category": "UTILITIES"},
            {"merchant": "Vodafone", "description": "internet broadband bill", "amount": 1200, "category": "UTILITIES"},
            {"merchant": "Tata Sky", "description": "dth recharge cable", "amount": 400, "category": "UTILITIES"},
            {"merchant": "Water Board", "description": "water bill payment", "amount": 300, "category": "UTILITIES"},
            {"merchant": "Dish TV", "description": "satellite tv recharge", "amount": 350, "category": "UTILITIES"},
            {"merchant": "MTNL", "description": "landline bill", "amount": 200, "category": "UTILITIES"},
            {"merchant": "Internet Provider", "description": "wifi bill monthly", "amount": 1000, "category": "UTILITIES"},
        ]
        
        # Healthcare
        healthcare_data = [
            {"merchant": "Apollo Hospital", "description": "medical consultation doctor", "amount": 1000, "category": "HEALTHCARE"},
            {"merchant": "MedPlus", "description": "pharmacy medicines", "amount": 600, "category": "HEALTHCARE"},
            {"merchant": "Practo", "description": "doctor appointment online", "amount": 400, "category": "HEALTHCARE"},
            {"merchant": "Max Hospital", "description": "health checkup full body", "amount": 3000, "category": "HEALTHCARE"},
            {"merchant": "1mg", "description": "online medicine delivery", "amount": 800, "category": "HEALTHCARE"},
            {"merchant": "Fortis Hospital", "description": "specialist consultation", "amount": 1500, "category": "HEALTHCARE"},
            {"merchant": "Dental Clinic", "description": "dental treatment", "amount": 2000, "category": "HEALTHCARE"},
            {"merchant": "Eye Clinic", "description": "eye checkup", "amount": 800, "category": "HEALTHCARE"},
            {"merchant": "Lab Test", "description": "blood test", "amount": 500, "category": "HEALTHCARE"},
            {"merchant": "Physiotherapy", "description": "therapy session", "amount": 600, "category": "HEALTHCARE"},
        ]
        
        # Education
        education_data = [
            {"merchant": "Byju's", "description": "online learning course", "amount": 8000, "category": "EDUCATION"},
            {"merchant": "Unacademy", "description": "competitive exam prep", "amount": 5000, "category": "EDUCATION"},
            {"merchant": "Coursera", "description": "online certification course", "amount": 3000, "category": "EDUCATION"},
            {"merchant": "Udemy", "description": "skill development course", "amount": 2000, "category": "EDUCATION"},
            {"merchant": "Khan Academy", "description": "educational donation", "amount": 1000, "category": "EDUCATION"},
            {"merchant": "School Fee", "description": "tuition fee payment", "amount": 15000, "category": "EDUCATION"},
            {"merchant": "Book Store", "description": "educational books", "amount": 1500, "category": "EDUCATION"},
            {"merchant": "Coaching Center", "description": "entrance exam coaching", "amount": 20000, "category": "EDUCATION"},
            {"merchant": "University", "description": "semester fee", "amount": 25000, "category": "EDUCATION"},
            {"merchant": "Online Tutorial", "description": "programming course", "amount": 4000, "category": "EDUCATION"},
        ]
        
        # Investment
        investment_data = [
            {"merchant": "Zerodha", "description": "stock market trading", "amount": 10000, "category": "INVESTMENT"},
            {"merchant": "Groww", "description": "mutual fund sip", "amount": 5000, "category": "INVESTMENT"},
            {"merchant": "SBI", "description": "fixed deposit", "amount": 50000, "category": "INVESTMENT"},
            {"merchant": "HDFC Bank", "description": "recurring deposit", "amount": 10000, "category": "INVESTMENT"},
            {"merchant": "Upstox", "description": "equity trading", "amount": 15000, "category": "INVESTMENT"},
            {"merchant": "Coin", "description": "mutual fund investment", "amount": 8000, "category": "INVESTMENT"},
            {"merchant": "PPF", "description": "public provident fund", "amount": 12000, "category": "INVESTMENT"},
            {"merchant": "Insurance", "description": "life insurance premium", "amount": 6000, "category": "INVESTMENT"},
            {"merchant": "ELSS", "description": "tax saving fund", "amount": 7000, "category": "INVESTMENT"},
            {"merchant": "Gold", "description": "digital gold purchase", "amount": 3000, "category": "INVESTMENT"},
        ]
        
        # Miscellaneous
        misc_data = [
            {"merchant": "ATM", "description": "cash withdrawal", "amount": 2000, "category": "MISCELLANEOUS"},
            {"merchant": "Government", "description": "tax payment income", "amount": 15000, "category": "MISCELLANEOUS"},
            {"merchant": "Bank", "description": "account maintenance charge", "amount": 500, "category": "MISCELLANEOUS"},
            {"merchant": "Post Office", "description": "postal service", "amount": 100, "category": "MISCELLANEOUS"},
            {"merchant": "Courier", "description": "package delivery", "amount": 200, "category": "MISCELLANEOUS"},
            {"merchant": "Charity", "description": "donation", "amount": 1000, "category": "MISCELLANEOUS"},
            {"merchant": "Gift", "description": "gift purchase", "amount": 2000, "category": "MISCELLANEOUS"},
            {"merchant": "Miscellaneous", "description": "other expenses", "amount": 500, "category": "MISCELLANEOUS"},
            {"merchant": "Service Charge", "description": "various service fees", "amount": 300, "category": "MISCELLANEOUS"},
            {"merchant": "Unknown", "description": "unidentified transaction", "amount": 1000, "category": "MISCELLANEOUS"},
        ]
        
        # Combine all data
        all_categories_data = [
            food_data, transport_data, shopping_data, entertainment_data,
            utilities_data, healthcare_data, education_data, investment_data, misc_data
        ]
        
        # Replicate data to reach desired sample count
        for category_data in all_categories_data:
            samples_needed = max(num_samples_per_category, len(category_data))
            for i in range(samples_needed):
                # Cycle through available samples with slight variations
                base_sample = category_data[i % len(category_data)].copy()
                
                # Add slight amount variation
                if i >= len(category_data):
                    amount_variation = np.random.uniform(0.8, 1.2)
                    base_sample['amount'] = int(base_sample['amount'] * amount_variation)
                
                extended_data.append(base_sample)
        
        df = pd.DataFrame(extended_data)
        logger.info(f"Created enhanced dataset with {len(df)} transactions")
        logger.info(f"Category distribution: {df['category'].value_counts().to_dict()}")
        
        return df
    
    def run_training_pipeline(self, use_enhanced_data: bool = True, 
                            num_samples_per_category: int = 50) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Args:
            use_enhanced_data (bool): Whether to use enhanced sample data
            num_samples_per_category (int): Number of samples per category
            
        Returns:
            Dict: Training results
        """
        logger.info("Starting training pipeline...")
        start_time = datetime.now()
        
        try:
            # Step 1: Create or load data
            if use_enhanced_data:
                raw_data = self.create_enhanced_sample_data(num_samples_per_category)
            else:
                raw_data = self.preprocessor.create_sample_data()
            
            # Step 2: Preprocess data
            logger.info("Preprocessing data...")
            processed_data = self.preprocessor.preprocess_dataframe(raw_data)
            
            # Step 3: Extract features
            logger.info("Extracting features...")
            X, feature_names = self.feature_extractor.fit_transform(processed_data)
            y = self.feature_extractor.encode_labels(processed_data['category'])
            
            # Step 4: Train model
            logger.info("Training models...")
            training_results = self.model.train(
                X=X, 
                y=y, 
                feature_names=feature_names,
                label_encoder=self.feature_extractor.label_encoder,
                model_name='auto'  # Compare multiple models
            )
            
            # Step 5: Save models
            logger.info("Saving trained models...")
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = os.path.join(model_dir, "expense_category_model.joblib")
            feature_extractor_path = os.path.join(model_dir, "feature_extractor.joblib")
            
            self.model.save_model(model_path)
            self.feature_extractor.save(feature_extractor_path)
            
            # Step 6: Create training summary
            end_time = datetime.now()
            training_duration = (end_time - start_time).total_seconds()
            
            summary = {
                'training_info': {
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': training_duration,
                    'dataset_size': len(processed_data),
                    'num_features': X.shape[1],
                    'num_categories': len(np.unique(y)),
                    'categories': list(self.feature_extractor.label_encoder.classes_)
                },
                'model_results': training_results,
                'best_model': {
                    'type': self.model.model_type,
                    'metrics': self.model.validation_metrics
                },
                'feature_summary': self.feature_extractor.get_feature_summary(),
                'file_paths': {
                    'model': model_path,
                    'feature_extractor': feature_extractor_path
                }
            }
            
            # Save training summary
            summary_path = os.path.join(model_dir, "training_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            self.training_results = summary
            self.trained_data = processed_data
            
            logger.info(f"Training completed successfully in {training_duration:.2f} seconds!")
            logger.info(f"Best model: {self.model.model_type}")
            logger.info(f"Validation F1 Score: {self.model.validation_metrics['f1_score']:.4f}")
            logger.info(f"Training summary saved to: {summary_path}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise
    
    def test_inference(self, num_test_samples: int = 10) -> Dict[str, Any]:
        """
        Test the inference pipeline with sample transactions.
        
        Args:
            num_test_samples (int): Number of test samples
            
        Returns:
            Dict: Test results
        """
        logger.info("Testing inference pipeline...")
        
        # Create test transactions
        from src.inference import create_sample_transactions
        test_transactions = create_sample_transactions()
        
        # Load the inference pipeline
        model_path = "models/expense_category_model.joblib"
        feature_extractor_path = "models/feature_extractor.joblib"
        
        categorizer = ExpenseCategorizer(model_path, feature_extractor_path)
        
        # Test single predictions
        logger.info("Testing single predictions...")
        single_results = []
        for i, trans in enumerate(test_transactions[:num_test_samples]):
            result = categorizer.categorize_expense(
                merchant=trans['merchant'],
                description=trans['description'],
                amount=trans['amount']
            )
            single_results.append({
                'input': trans,
                'prediction': result
            })
            logger.info(f"Test {i+1}: {trans['merchant']} -> {result['category']} ({result['confidence']:.3f})")
        
        # Test batch predictions
        logger.info("Testing batch predictions...")
        batch_results = categorizer.predict_batch_transactions(test_transactions)
        
        # Test spending analysis
        logger.info("Testing spending analysis...")
        analysis = categorizer.analyze_spending_pattern(test_transactions)
        
        test_summary = {
            'single_predictions': single_results,
            'batch_predictions': batch_results,
            'spending_analysis': analysis,
            'model_info': categorizer.get_model_info()
        }
        
        logger.info("Inference testing completed successfully!")
        return test_summary


def main():
    """Main function to run the training pipeline."""
    logger.info("=" * 60)
    logger.info("EXPENSE CATEGORIZATION ML MODEL - TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Initialize training pipeline
    pipeline = ModelTrainingPipeline(random_state=42)
    
    try:
        # Run training
        training_results = pipeline.run_training_pipeline(
            use_enhanced_data=True,
            num_samples_per_category=60  # Create robust training set
        )
        
        # Test inference
        test_results = pipeline.test_inference(num_test_samples=5)
        
        # Display final results
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Best Model: {training_results['best_model']['type']}")
        logger.info(f"Accuracy: {training_results['best_model']['metrics']['accuracy']:.4f}")
        logger.info(f"F1 Score: {training_results['best_model']['metrics']['f1_score']:.4f}")
        logger.info(f"Total Features: {training_results['feature_summary']['total_features']}")
        logger.info(f"Categories: {len(training_results['training_info']['categories'])}")
        
        print("\n🎉 Your ML model is ready!")
        print(f"📊 Model Accuracy: {training_results['best_model']['metrics']['accuracy']:.1%}")
        print(f"🎯 F1 Score: {training_results['best_model']['metrics']['f1_score']:.1%}")
        print(f"📁 Model saved to: {training_results['file_paths']['model']}")
        print(f"🔧 Feature extractor saved to: {training_results['file_paths']['feature_extractor']}")
        
        return training_results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()