"""
Machine Learning Model Training Module for Expense Categorization

This module handles model training, hyperparameter tuning, evaluation, and 
model selection for the expense categorization system.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExpenseCategoryModel:
    """
    A comprehensive machine learning model for expense categorization.
    
    This class handles model training, evaluation, hyperparameter tuning,
    and prediction with confidence scores.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the expense category model.
        
        Args:
            random_state (int): Random state for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_model = None
        self.feature_names = []
        self.is_trained = False
        self.model_type = None
        self.training_metrics = {}
        self.validation_metrics = {}
        
        # Available models
        self.available_models = {
            'random_forest': RandomForestClassifier(random_state=random_state),
            'xgboost': xgb.XGBClassifier(random_state=random_state, eval_metric='mlogloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=random_state, verbose=-1),
            'gradient_boost': GradientBoostingClassifier(random_state=random_state),
            'logistic_regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'svm': SVC(random_state=random_state, probability=True)
        }
        
        # Hyperparameter grids for tuning
        self.param_grids = {
            'random_forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            },
            'xgboost': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [100, 200, 300],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'num_leaves': [31, 50, 100],
                'subsample': [0.8, 0.9, 1.0]
            },
            'gradient_boost': {
                'n_estimators': [100, 200],
                'max_depth': [6, 8, 10],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            },
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2', 'elasticnet'],
                'solver': ['liblinear', 'saga'],
                'class_weight': ['balanced']
            }
        }
    
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training by splitting into train and test sets.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            test_size (float): Proportion of test set
            
        Returns:
            Tuple: X_train, X_test, y_train, y_test
        """
        logger.info(f"Splitting data into train ({1-test_size:.1%}) and test ({test_size:.1%}) sets...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, 
            stratify=y  # Ensure balanced split across categories
        )
        
        logger.info(f"Training set shape: {X_train.shape}")
        logger.info(f"Test set shape: {X_test.shape}")
        logger.info(f"Training labels distribution: {np.bincount(y_train)}")
        logger.info(f"Test labels distribution: {np.bincount(y_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_single_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                          model_name: str, tune_hyperparams: bool = True) -> Any:
        """
        Train a single model with optional hyperparameter tuning.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            model_name (str): Name of the model to train
            tune_hyperparams (bool): Whether to perform hyperparameter tuning
            
        Returns:
            Any: Trained model
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available. Choose from {list(self.available_models.keys())}")
        
        logger.info(f"Training {model_name} model...")
        
        base_model = self.available_models[model_name]
        
        if tune_hyperparams and model_name in self.param_grids:
            logger.info(f"Performing hyperparameter tuning for {model_name}...")
            
            # Use stratified k-fold for cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=self.param_grids[model_name],
                cv=cv,
                scoring='f1_weighted',  # Use weighted F1 for imbalanced classes
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            
            logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
            logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            if model_name in ['random_forest', 'gradient_boost']:
                # Set class_weight for tree-based models
                base_model.set_params(class_weight='balanced')
            elif model_name == 'logistic_regression':
                base_model.set_params(class_weight='balanced')
            
            best_model = base_model
            best_model.fit(X_train, y_train)
        
        return best_model
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      label_encoder: Any) -> Dict[str, float]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model (Any): Trained model
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test labels
            label_encoder (Any): Label encoder for category names
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Calculate AUC for multiclass (one-vs-rest)
        try:
            auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except:
            auc = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }
        
        # Print detailed classification report
        category_names = label_encoder.classes_
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(y_test, y_pred, target_names=category_names))
        
        return metrics
    
    def train_and_compare_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                                y_train: np.ndarray, y_test: np.ndarray,
                                label_encoder: Any, models_to_compare: List[str] = None) -> Dict[str, Dict]:
        """
        Train and compare multiple models.
        
        Args:
            X_train (np.ndarray): Training features
            X_test (np.ndarray): Test features
            y_train (np.ndarray): Training labels
            y_test (np.ndarray): Test labels
            label_encoder (Any): Label encoder
            models_to_compare (List[str]): List of models to compare
            
        Returns:
            Dict[str, Dict]: Results for each model
        """
        if models_to_compare is None:
            models_to_compare = ['random_forest', 'xgboost', 'lightgbm']
        
        results = {}
        trained_models = {}
        
        for model_name in models_to_compare:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training and evaluating {model_name}")
                logger.info(f"{'='*50}")
                
                # Train model
                trained_model = self.train_single_model(X_train, y_train, model_name, tune_hyperparams=True)
                trained_models[model_name] = trained_model
                
                # Evaluate model
                metrics = self.evaluate_model(trained_model, X_test, y_test, label_encoder)
                results[model_name] = {
                    'model': trained_model,
                    'metrics': metrics
                }
                
                logger.info(f"\n{model_name} Results:")
                for metric, value in metrics.items():
                    logger.info(f"{metric.capitalize()}: {value:.4f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Find best model based on F1 score
        best_model_name = max(results.keys(), key=lambda x: results[x]['metrics']['f1_score'])
        best_model = results[best_model_name]['model']
        
        logger.info(f"\nBest model: {best_model_name}")
        logger.info(f"Best F1 Score: {results[best_model_name]['metrics']['f1_score']:.4f}")
        
        # Store best model
        self.best_model = best_model
        self.model_type = best_model_name
        self.is_trained = True
        self.validation_metrics = results[best_model_name]['metrics']
        
        return results
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
              label_encoder: Any, model_name: str = 'auto') -> Dict[str, Any]:
        """
        Train the expense categorization model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels
            feature_names (List[str]): Names of features
            label_encoder (Any): Label encoder
            model_name (str): Model to train or 'auto' for comparison
            
        Returns:
            Dict[str, Any]: Training results
        """
        self.feature_names = feature_names
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        if model_name == 'auto':
            # Compare multiple models and select the best
            results = self.train_and_compare_models(
                X_train, X_test, y_train, y_test, label_encoder,
                models_to_compare=['random_forest', 'xgboost', 'lightgbm']
            )
            self.model = self.best_model
        else:
            # Train specific model
            logger.info(f"Training {model_name} model...")
            trained_model = self.train_single_model(X_train, y_train, model_name, tune_hyperparams=True)
            metrics = self.evaluate_model(trained_model, X_test, y_test, label_encoder)
            
            self.model = trained_model
            self.best_model = trained_model
            self.model_type = model_name
            self.is_trained = True
            self.validation_metrics = metrics
            
            results = {
                model_name: {
                    'model': trained_model,
                    'metrics': metrics
                }
            }
        
        # Store training data for feature importance analysis
        self.training_metrics = {
            'train_size': len(X_train),
            'test_size': len(X_test),
            'num_features': X.shape[1],
            'num_classes': len(np.unique(y))
        }
        
        logger.info("\nTraining completed successfully!")
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: np.ndarray, label_encoder: Any) -> List[Dict[str, Any]]:
        """
        Make predictions with confidence scores and category names.
        
        Args:
            X (np.ndarray): Feature matrix
            label_encoder (Any): Label encoder
            
        Returns:
            List[Dict[str, Any]]: Predictions with confidence scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Get predictions and probabilities
        y_pred = self.predict(X)
        y_pred_proba = self.predict_proba(X)
        
        # Convert to category names
        predicted_categories = label_encoder.inverse_transform(y_pred)
        
        results = []
        for i in range(len(y_pred)):
            # Get confidence (maximum probability)
            confidence = float(np.max(y_pred_proba[i]))
            
            # Get top 3 predictions with probabilities
            top_indices = np.argsort(y_pred_proba[i])[::-1][:3]
            top_predictions = []
            
            for idx in top_indices:
                category = label_encoder.classes_[idx]
                probability = float(y_pred_proba[i][idx])
                top_predictions.append({
                    'category': category,
                    'probability': probability
                })
            
            results.append({
                'predicted_category': predicted_categories[i],
                'confidence': confidence,
                'top_predictions': top_predictions
            })
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            Dict[str, float]: Feature importance mapping
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get feature importance based on model type
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # For linear models, use absolute values of coefficients
            importance = np.mean(np.abs(self.model.coef_), axis=0)
        else:
            logger.warning("Model does not support feature importance")
            return {}
        
        # Create feature importance mapping
        feature_importance = dict(zip(self.feature_names, importance))
        
        # Sort by importance and return top N
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        return dict(sorted_features[:top_n])
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """
        Plot feature importance.
        
        Args:
            top_n (int): Number of top features to plot
            save_path (str): Path to save the plot
        """
        feature_importance = self.get_feature_importance(top_n)
        
        if not feature_importance:
            logger.warning("No feature importance available")
            return
        
        # Create plot
        plt.figure(figsize=(12, 8))
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        plt.barh(range(len(features)), importance_values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importance - {self.model_type}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, filepath: str, include_metadata: bool = True):
        """
        Save the trained model.
        
        Args:
            filepath (str): Path to save the model
            include_metadata (bool): Whether to save additional metadata
        """
        if not self.is_trained:
            logger.warning("Model is not trained. Saving untrained model.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model
        model_data = {
            'model': self.model,
            'best_model': self.best_model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'random_state': self.random_state
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ExpenseCategoryModel':
        """
        Load a saved model.
        
        Args:
            filepath (str): Path to the saved model
            
        Returns:
            ExpenseCategoryModel: Loaded model instance
        """
        model_data = joblib.load(filepath)
        
        # Create new instance
        instance = cls(random_state=model_data.get('random_state', 42))
        
        # Restore model state
        instance.model = model_data['model']
        instance.best_model = model_data['best_model']
        instance.model_type = model_data['model_type']
        instance.feature_names = model_data['feature_names']
        instance.is_trained = model_data['is_trained']
        instance.training_metrics = model_data['training_metrics']
        instance.validation_metrics = model_data['validation_metrics']
        
        logger.info(f"Model loaded from {filepath}")
        return instance
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the trained model.
        
        Returns:
            Dict[str, Any]: Model summary
        """
        return {
            'model_type': self.model_type,
            'is_trained': self.is_trained,
            'num_features': len(self.feature_names),
            'training_metrics': self.training_metrics,
            'validation_metrics': self.validation_metrics,
            'feature_importance_available': hasattr(self.model, 'feature_importances_') if self.model else False
        }


if __name__ == "__main__":
    # Example usage will be demonstrated in the testing module
    pass