"""
Incremental Learning System for Expense Categorization

This module enables the ML model to learn from user corrections and new expenses,
continuously improving its accuracy over time.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import joblib
from sklearn.metrics import accuracy_score, classification_report

from .preprocessor import TransactionPreprocessor
from .feature_extractor import FeatureExtractor
from .model import ExpenseCategoryModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalLearner:
    """
    System for continuous learning from user feedback and new data.
    """
    
    def __init__(self, data_dir: str = "data/user_feedback"):
        """
        Initialize the incremental learning system.
        
        Args:
            data_dir: Directory to store user feedback data
        """
        self.data_dir = data_dir
        self.feedback_file = os.path.join(data_dir, "user_feedback.jsonl")
        self.corrections_file = os.path.join(data_dir, "user_corrections.jsonl")
        self.learning_stats_file = os.path.join(data_dir, "learning_stats.json")
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Learning thresholds
        self.confidence_threshold = 0.7  # Below this, ask for user confirmation
        self.retrain_threshold = 50      # Retrain after this many corrections
        self.min_category_samples = 10   # Minimum samples needed per category
        
        # Initialize stats
        self.learning_stats = self.load_learning_stats()
    
    def load_learning_stats(self) -> Dict[str, Any]:
        """Load learning statistics."""
        if os.path.exists(self.learning_stats_file):
            with open(self.learning_stats_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'total_feedback': 0,
                'total_corrections': 0,
                'total_retrains': 0,
                'accuracy_improvements': [],
                'last_retrain_date': None,
                'category_feedback_count': {},
                'model_version': 1
            }
    
    def save_learning_stats(self):
        """Save learning statistics."""
        with open(self.learning_stats_file, 'w') as f:
            json.dump(self.learning_stats, f, indent=2)
    
    def record_prediction(self, merchant: str, description: str, amount: float,
                         predicted_category: str, confidence: float,
                         user_confirmed: bool = True, actual_category: str = None) -> Dict[str, Any]:
        """
        Record a prediction and user feedback.
        
        Args:
            merchant: Merchant name
            description: Transaction description
            amount: Transaction amount
            predicted_category: Model's prediction
            confidence: Prediction confidence
            user_confirmed: Whether user confirmed the prediction
            actual_category: Actual category if different from prediction
            
        Returns:
            Dictionary with feedback record
        """
        feedback_record = {
            'timestamp': datetime.now().isoformat(),
            'merchant': merchant,
            'description': description,
            'amount': amount,
            'predicted_category': predicted_category,
            'confidence': confidence,
            'user_confirmed': user_confirmed,
            'actual_category': actual_category if actual_category else predicted_category,
            'needs_learning': not user_confirmed or confidence < self.confidence_threshold
        }
        
        # Save feedback
        with open(self.feedback_file, 'a') as f:
            f.write(json.dumps(feedback_record) + '\n')
        
        # Update stats
        self.learning_stats['total_feedback'] += 1
        
        # Record correction if needed
        if not user_confirmed and actual_category:
            self.record_correction(merchant, description, amount, 
                                 predicted_category, actual_category, confidence)
        
        # Update category feedback count
        category = actual_category if actual_category else predicted_category
        self.learning_stats['category_feedback_count'][category] = \
            self.learning_stats['category_feedback_count'].get(category, 0) + 1
        
        self.save_learning_stats()
        
        logger.info(f"Recorded feedback: {predicted_category} -> {category} "
                   f"(confidence: {confidence:.3f}, confirmed: {user_confirmed})")
        
        return feedback_record
    
    def record_correction(self, merchant: str, description: str, amount: float,
                         predicted_category: str, actual_category: str, 
                         confidence: float) -> Dict[str, Any]:
        """
        Record a user correction for learning.
        
        Args:
            merchant: Merchant name
            description: Transaction description
            amount: Transaction amount
            predicted_category: Model's incorrect prediction
            actual_category: User's correct category
            confidence: Original prediction confidence
            
        Returns:
            Correction record
        """
        correction_record = {
            'timestamp': datetime.now().isoformat(),
            'merchant': merchant,
            'description': description,
            'amount': amount,
            'predicted_category': predicted_category,
            'actual_category': actual_category,
            'confidence': confidence,
            'correction_type': 'user_correction'
        }
        
        # Save correction
        with open(self.corrections_file, 'a') as f:
            f.write(json.dumps(correction_record) + '\n')
        
        # Update stats
        self.learning_stats['total_corrections'] += 1
        self.save_learning_stats()
        
        logger.info(f"Recorded correction: {predicted_category} -> {actual_category} "
                   f"(confidence was: {confidence:.3f})")
        
        # Check if we should retrain
        if self.should_retrain():
            logger.info("Retrain threshold reached. Triggering incremental learning...")
            self.trigger_incremental_learning()
        
        return correction_record
    
    def should_retrain(self) -> bool:
        """
        Determine if the model should be retrained.
        
        Returns:
            True if retraining is recommended
        """
        corrections_since_retrain = self.get_corrections_since_last_retrain()
        
        # Retrain if we have enough corrections
        if len(corrections_since_retrain) >= self.retrain_threshold:
            return True
        
        # Retrain if we have consistent errors in a category
        category_errors = {}
        for correction in corrections_since_retrain:
            actual = correction['actual_category']
            category_errors[actual] = category_errors.get(actual, 0) + 1
        
        # If any category has more than 10 corrections, retrain
        if any(count >= 10 for count in category_errors.values()):
            return True
        
        return False
    
    def get_corrections_since_last_retrain(self) -> List[Dict[str, Any]]:
        """Get all corrections since the last retrain."""
        if not os.path.exists(self.corrections_file):
            return []
        
        corrections = []
        last_retrain_date = self.learning_stats.get('last_retrain_date')
        
        with open(self.corrections_file, 'r') as f:
            for line in f:
                correction = json.loads(line.strip())
                if not last_retrain_date or correction['timestamp'] > last_retrain_date:
                    corrections.append(correction)
        
        return corrections
    
    def get_all_user_data(self) -> pd.DataFrame:
        """
        Get all user feedback data as a training dataset.
        
        Returns:
            DataFrame with user-provided training data
        """
        if not os.path.exists(self.feedback_file):
            return pd.DataFrame()
        
        feedback_data = []
        
        with open(self.feedback_file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                feedback_data.append({
                    'merchant': record['merchant'],
                    'description': record['description'],
                    'amount': record['amount'],
                    'category': record['actual_category'],
                    'source': 'user_feedback'
                })
        
        return pd.DataFrame(feedback_data)
    
    def trigger_incremental_learning(self, model_path: str = "models/expense_category_model.joblib",
                                   feature_extractor_path: str = "models/feature_extractor.joblib",
                                   original_data_path: str = "data/enhanced_training_data.csv"):
        """
        Trigger incremental learning with user feedback.
        
        Args:
            model_path: Path to current model
            feature_extractor_path: Path to current feature extractor
            original_data_path: Path to original training data
        """
        logger.info("🚀 Starting incremental learning process...")
        
        try:
            # Load user feedback data
            user_data = self.get_all_user_data()
            
            if len(user_data) < self.min_category_samples:
                logger.warning(f"Not enough user data ({len(user_data)} samples). "
                              f"Need at least {self.min_category_samples}.")
                return
            
            logger.info(f"📊 User feedback data: {len(user_data)} samples")
            logger.info(f"📋 User categories: {user_data['category'].value_counts().to_dict()}")
            
            # Load original training data
            if os.path.exists(original_data_path):
                original_data = pd.read_csv(original_data_path)
                logger.info(f"📊 Original training data: {len(original_data)} samples")
            else:
                # Fallback to basic data if no enhanced data available
                from .preprocessor import TransactionPreprocessor
                preprocessor = TransactionPreprocessor()
                original_data = preprocessor.create_sample_data()
                logger.warning("Using basic sample data as original training data")
            
            # Combine datasets, giving higher weight to user corrections
            user_data_weighted = pd.concat([user_data] * 3, ignore_index=True)  # 3x weight
            combined_data = pd.concat([original_data, user_data_weighted], ignore_index=True)
            combined_data = combined_data.sample(frac=1).reset_index(drop=True)  # Shuffle
            
            logger.info(f"📊 Combined training data: {len(combined_data)} samples")
            
            # Preprocess data
            preprocessor = TransactionPreprocessor()
            processed_data = preprocessor.preprocess_dataframe(combined_data)
            
            # Handle categorical column issue
            if 'amount_category' in processed_data.columns:
                processed_data['amount_category'] = processed_data['amount_category'].astype(str)
            
            # Extract features
            feature_extractor = FeatureExtractor()
            X, feature_names = feature_extractor.fit_transform(processed_data)
            y = feature_extractor.encode_labels(processed_data['category'])
            
            # Train new model
            model = ExpenseCategoryModel(random_state=42)
            results = model.train(X, y, feature_names, feature_extractor.label_encoder, 
                                model_name='random_forest')
            
            # Calculate improvement
            old_accuracy = self.get_previous_accuracy()
            new_accuracy = model.validation_metrics['accuracy']
            improvement = new_accuracy - old_accuracy if old_accuracy else 0
            
            # Save improved models
            model.save_model(model_path)
            feature_extractor.save(feature_extractor_path)
            
            # Update learning stats
            self.learning_stats['total_retrains'] += 1
            self.learning_stats['last_retrain_date'] = datetime.now().isoformat()
            self.learning_stats['accuracy_improvements'].append({
                'date': datetime.now().isoformat(),
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
                'improvement': improvement,
                'user_samples': len(user_data)
            })
            self.learning_stats['model_version'] += 1
            
            self.save_learning_stats()
            
            logger.info(f"✅ Incremental learning completed!")
            logger.info(f"📈 Model accuracy: {old_accuracy:.3f} → {new_accuracy:.3f} "
                       f"(+{improvement:.3f})")
            logger.info(f"🔄 Model version: {self.learning_stats['model_version']}")
            
            return {
                'success': True,
                'old_accuracy': old_accuracy,
                'new_accuracy': new_accuracy,
                'improvement': improvement,
                'model_version': self.learning_stats['model_version']
            }
            
        except Exception as e:
            logger.error(f"❌ Incremental learning failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_previous_accuracy(self) -> float:
        """Get the accuracy from the previous model version."""
        if self.learning_stats['accuracy_improvements']:
            return self.learning_stats['accuracy_improvements'][-1]['new_accuracy']
        return 0.85  # Default baseline
    
    def get_learning_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get recommendations for improving the model.
        
        Returns:
            List of learning recommendations
        """
        recommendations = []
        
        # Get recent low-confidence predictions
        low_confidence_count = self.count_low_confidence_predictions()
        if low_confidence_count > 10:
            recommendations.append({
                'type': 'low_confidence',
                'message': f"You have {low_confidence_count} low-confidence predictions. "
                          "Consider providing feedback to improve accuracy.",
                'action': 'provide_feedback'
            })
        
        # Check for categories with many corrections
        corrections = self.get_corrections_since_last_retrain()
        category_errors = {}
        for correction in corrections:
            actual = correction['actual_category']
            category_errors[actual] = category_errors.get(actual, 0) + 1
        
        for category, count in category_errors.items():
            if count >= 5:
                recommendations.append({
                    'type': 'category_errors',
                    'message': f"Category '{category}' has {count} corrections. "
                              "Model may need more examples for this category.",
                    'action': 'add_examples',
                    'category': category
                })
        
        # Check if retrain is recommended
        if self.should_retrain():
            recommendations.append({
                'type': 'retrain_ready',
                'message': f"Model is ready for retraining with {len(corrections)} new corrections.",
                'action': 'retrain_model'
            })
        
        return recommendations
    
    def count_low_confidence_predictions(self, days: int = 7) -> int:
        """Count low-confidence predictions in recent days."""
        if not os.path.exists(self.feedback_file):
            return 0
        
        count = 0
        cutoff_date = (datetime.now() - pd.Timedelta(days=days)).isoformat()
        
        with open(self.feedback_file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                if (record['timestamp'] > cutoff_date and 
                    record['confidence'] < self.confidence_threshold):
                    count += 1
        
        return count
    
    def export_learning_report(self) -> Dict[str, Any]:
        """
        Export a comprehensive learning report.
        
        Returns:
            Detailed learning report
        """
        report = {
            'summary': self.learning_stats,
            'recommendations': self.get_learning_recommendations(),
            'recent_accuracy_trend': [],
            'category_performance': {},
            'learning_progress': {}
        }
        
        # Calculate accuracy trend
        if self.learning_stats['accuracy_improvements']:
            trend = []
            for improvement in self.learning_stats['accuracy_improvements'][-5:]:  # Last 5
                trend.append({
                    'date': improvement['date'][:10],  # Date only
                    'accuracy': improvement['new_accuracy']
                })
            report['recent_accuracy_trend'] = trend
        
        # Category-wise performance
        if os.path.exists(self.corrections_file):
            corrections = self.get_corrections_since_last_retrain()
            category_performance = {}
            
            for correction in corrections:
                predicted = correction['predicted_category']
                actual = correction['actual_category']
                
                if actual not in category_performance:
                    category_performance[actual] = {'errors': 0, 'low_confidence': 0}
                
                category_performance[actual]['errors'] += 1
                if correction['confidence'] < self.confidence_threshold:
                    category_performance[actual]['low_confidence'] += 1
            
            report['category_performance'] = category_performance
        
        # Learning progress
        report['learning_progress'] = {
            'total_interactions': self.learning_stats['total_feedback'],
            'correction_rate': (self.learning_stats['total_corrections'] / 
                              max(1, self.learning_stats['total_feedback'])) * 100,
            'retrain_frequency': self.learning_stats['total_retrains'],
            'model_version': self.learning_stats['model_version']
        }
        
        return report


class SmartLearningInterface:
    """
    User-friendly interface for incremental learning.
    """
    
    def __init__(self, learner: IncrementalLearner = None):
        """Initialize the learning interface."""
        self.learner = learner or IncrementalLearner()
    
    def ask_for_feedback(self, merchant: str, description: str, amount: float,
                        predicted_category: str, confidence: float) -> Dict[str, Any]:
        """
        Interactive method to ask user for feedback on a prediction.
        
        Args:
            merchant: Merchant name
            description: Transaction description
            amount: Transaction amount
            predicted_category: Model's prediction
            confidence: Prediction confidence
            
        Returns:
            User feedback record
        """
        print(f"\n🤖 AI Prediction for: {merchant} - {description} (₹{amount})")
        print(f"📊 Predicted Category: {predicted_category}")
        print(f"🎯 Confidence: {confidence:.1%}")
        
        if confidence < self.learner.confidence_threshold:
            print(f"⚠️  Low confidence prediction. Please verify!")
        
        print(f"\nIs this prediction correct? (y/n): ", end="")
        user_input = input().strip().lower()
        
        if user_input in ['y', 'yes', '1', 'true']:
            # User confirmed prediction
            feedback = self.learner.record_prediction(
                merchant, description, amount, predicted_category, 
                confidence, user_confirmed=True
            )
            print("✅ Thank you for confirming!")
            
        else:
            # User wants to correct
            print(f"\n📝 Available categories:")
            categories = [
                'FOOD_DINING', 'TRANSPORTATION', 'SHOPPING', 'ENTERTAINMENT',
                'UTILITIES', 'HEALTHCARE', 'EDUCATION', 'INVESTMENT', 'MISCELLANEOUS'
            ]
            
            for i, category in enumerate(categories, 1):
                print(f"   {i}. {category.replace('_', ' ').title()}")
            
            print(f"\nEnter the correct category number (1-{len(categories)}): ", end="")
            try:
                choice = int(input().strip())
                if 1 <= choice <= len(categories):
                    actual_category = categories[choice - 1]
                    
                    feedback = self.learner.record_prediction(
                        merchant, description, amount, predicted_category,
                        confidence, user_confirmed=False, actual_category=actual_category
                    )
                    
                    print(f"✅ Thank you! I've learned that this should be: {actual_category.replace('_', ' ').title()}")
                    
                    # Show recommendations if any
                    recommendations = self.learner.get_learning_recommendations()
                    if recommendations:
                        print(f"\n💡 Learning Recommendations:")
                        for rec in recommendations[:2]:  # Show top 2
                            print(f"   • {rec['message']}")
                
                else:
                    print("❌ Invalid choice. Skipping feedback.")
                    return {}
                    
            except ValueError:
                print("❌ Invalid input. Skipping feedback.")
                return {}
        
        return feedback
    
    def batch_learning_session(self, transactions: List[Dict[str, Any]],
                              categorizer) -> Dict[str, Any]:
        """
        Conduct a batch learning session with multiple transactions.
        
        Args:
            transactions: List of transactions to learn from
            categorizer: ExpenseCategorizer instance
            
        Returns:
            Learning session summary
        """
        print(f"🎓 Starting Batch Learning Session")
        print(f"📊 Processing {len(transactions)} transactions...")
        print("=" * 50)
        
        feedback_count = 0
        corrections_count = 0
        
        for i, transaction in enumerate(transactions, 1):
            print(f"\n📝 Transaction {i}/{len(transactions)}")
            
            # Get prediction
            result = categorizer.categorize_expense(
                merchant=transaction['merchant'],
                description=transaction['description'],
                amount=transaction.get('amount', 0)
            )
            
            # Ask for feedback
            feedback = self.ask_for_feedback(
                transaction['merchant'],
                transaction['description'], 
                transaction.get('amount', 0),
                result['category'],
                result['confidence']
            )
            
            if feedback:
                feedback_count += 1
                if not feedback.get('user_confirmed', True):
                    corrections_count += 1
            
            # Ask if user wants to continue
            if i < len(transactions):
                print(f"\nContinue to next transaction? (y/n): ", end="")
                if input().strip().lower() not in ['y', 'yes', '1', 'true']:
                    break
        
        # Session summary
        summary = {
            'transactions_processed': i,
            'feedback_provided': feedback_count,
            'corrections_made': corrections_count,
            'learning_recommendations': self.learner.get_learning_recommendations()
        }
        
        print(f"\n🎉 Learning Session Complete!")
        print(f"📊 Processed: {summary['transactions_processed']} transactions")
        print(f"📝 Feedback: {summary['feedback_provided']} responses")
        print(f"🔧 Corrections: {summary['corrections_made']} corrections")
        
        if summary['learning_recommendations']:
            print(f"\n💡 Recommendations:")
            for rec in summary['learning_recommendations']:
                print(f"   • {rec['message']}")
        
        return summary


if __name__ == "__main__":
    # Example usage
    learner = IncrementalLearner()
    interface = SmartLearningInterface(learner)
    
    # Generate learning report
    report = learner.export_learning_report()
    print("📊 Learning Report:", json.dumps(report, indent=2))