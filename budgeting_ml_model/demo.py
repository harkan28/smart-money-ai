"""
Demo Script for Expense Categorization ML Model

This script demonstrates the complete functionality of the ML model
including training, prediction, and analysis capabilities.
"""

import os
import sys
import pandas as pd
import json

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from train_model import ModelTrainingPipeline
from src.inference import ExpenseCategorizer, create_sample_transactions

def demo_training():
    """Demonstrate the training pipeline."""
    print("🚀 TRAINING EXPENSE CATEGORIZATION MODEL")
    print("=" * 50)
    
    # Initialize and run training
    pipeline = ModelTrainingPipeline(random_state=42)
    results = pipeline.run_training_pipeline(
        use_enhanced_data=True,
        num_samples_per_category=40
    )
    
    print(f"\n✅ Training completed!")
    print(f"📊 Best Model: {results['best_model']['type']}")
    print(f"🎯 Accuracy: {results['best_model']['metrics']['accuracy']:.1%}")
    print(f"🎯 F1 Score: {results['best_model']['metrics']['f1_score']:.1%}")
    
    return results

def demo_predictions():
    """Demonstrate prediction capabilities."""
    print("\n🔮 TESTING PREDICTION CAPABILITIES")
    print("=" * 50)
    
    # Load trained model
    try:
        categorizer = ExpenseCategorizer(
            model_path="models/expense_category_model.joblib",
            feature_extractor_path="models/feature_extractor.joblib"
        )
        print("✅ Models loaded successfully!")
    except:
        print("❌ Models not found. Please run training first.")
        return
    
    # Test individual predictions
    print("\n📱 INDIVIDUAL TRANSACTION PREDICTIONS:")
    test_transactions = [
        {"merchant": "Zomato", "description": "biryani delivery", "amount": 400},
        {"merchant": "Uber", "description": "airport cab", "amount": 600},
        {"merchant": "Amazon", "description": "laptop purchase", "amount": 45000},
        {"merchant": "Netflix", "description": "monthly subscription", "amount": 650},
        {"merchant": "HDFC Bank", "description": "mutual fund investment", "amount": 10000}
    ]
    
    for i, trans in enumerate(test_transactions, 1):
        result = categorizer.categorize_expense(
            merchant=trans['merchant'],
            description=trans['description'],
            amount=trans['amount']
        )
        
        print(f"{i}. {trans['merchant']}: {trans['description']} (₹{trans['amount']})")
        print(f"   → {result['category']} ({result['confidence']:.1%} confidence)")
        
        if 'alternatives' in result:
            alt_text = ', '.join([f"{alt['category']} ({alt['probability']:.1%})" for alt in result['alternatives'][:2]])
            print(f"   Alternatives: {alt_text}")
        print()

def demo_spending_analysis():
    """Demonstrate spending pattern analysis."""
    print("\n📊 SPENDING PATTERN ANALYSIS")
    print("=" * 50)
    
    # Load model
    try:
        categorizer = ExpenseCategorizer(
            model_path="models/expense_category_model.joblib",
            feature_extractor_path="models/feature_extractor.joblib"
        )
    except:
        print("❌ Models not found. Please run training first.")
        return
    
    # Create realistic monthly spending data
    monthly_transactions = [
        # Food & Dining
        {"merchant": "Zomato", "description": "lunch delivery", "amount": 350},
        {"merchant": "Swiggy", "description": "dinner order", "amount": 450},
        {"merchant": "Cafe Coffee Day", "description": "coffee meeting", "amount": 200},
        {"merchant": "McDonald's", "description": "quick meal", "amount": 300},
        
        # Transportation
        {"merchant": "Uber", "description": "office commute", "amount": 150},
        {"merchant": "Ola", "description": "weekend trip", "amount": 400},
        {"merchant": "Indian Oil", "description": "petrol refill", "amount": 2000},
        
        # Shopping
        {"merchant": "Amazon", "description": "electronics purchase", "amount": 8000},
        {"merchant": "Myntra", "description": "clothing shopping", "amount": 2500},
        {"merchant": "BigBasket", "description": "monthly groceries", "amount": 3000},
        
        # Entertainment
        {"merchant": "Netflix", "description": "streaming subscription", "amount": 650},
        {"merchant": "BookMyShow", "description": "movie tickets", "amount": 600},
        
        # Utilities
        {"merchant": "Airtel", "description": "mobile bill", "amount": 800},
        {"merchant": "BSES", "description": "electricity bill", "amount": 2500},
        
        # Healthcare
        {"merchant": "Apollo Hospital", "description": "health checkup", "amount": 2000},
        
        # Investment
        {"merchant": "Zerodha", "description": "mutual fund SIP", "amount": 5000},
        {"merchant": "SBI", "description": "fixed deposit", "amount": 10000},
    ]
    
    # Analyze spending patterns
    analysis = categorizer.analyze_spending_pattern(monthly_transactions)
    
    print(f"💰 Total Spending: ₹{analysis['summary']['total_amount']:,.0f}")
    print(f"📈 Total Transactions: {analysis['summary']['total_transactions']}")
    print(f"💳 Average Transaction: ₹{analysis['summary']['average_transaction_amount']:.0f}")
    
    print(f"\n🏆 TOP SPENDING CATEGORIES:")
    for i, category in enumerate(analysis['top_categories_by_amount'][:5], 1):
        print(f"{i}. {category['category'].replace('_', ' ').title()}: ₹{category['total_amount']:,.0f} ({category['amount_percentage']:.1f}%)")
    
    # Generate budget recommendations
    recommendations = categorizer.get_budget_recommendations(
        monthly_transactions, 
        monthly_income=50000  # Assuming ₹50,000 monthly income
    )
    
    print(f"\n💡 BUDGET RECOMMENDATIONS:")
    for rec in recommendations['recommendations'][:5]:
        if rec['recommendation_type'] in ['REDUCE', 'INCREASE']:
            print(f"• {rec['message']}")

def demo_model_info():
    """Demonstrate model information display."""
    print("\n🔧 MODEL INFORMATION")
    print("=" * 50)
    
    try:
        categorizer = ExpenseCategorizer(
            model_path="models/expense_category_model.joblib",
            feature_extractor_path="models/feature_extractor.joblib"
        )
        
        model_info = categorizer.get_model_info()
        
        print(f"🤖 Model Type: {model_info['model_summary']['model_type']}")
        print(f"🎯 Validation Accuracy: {model_info['model_summary']['validation_metrics']['accuracy']:.1%}")
        print(f"📊 Total Features: {model_info['feature_summary']['total_features']}")
        print(f"🏷️  Categories: {len(model_info['categories'])}")
        print(f"📋 Available Categories:")
        for category in model_info['categories']:
            print(f"   • {category.replace('_', ' ').title()}")
            
    except Exception as e:
        print(f"❌ Error loading model info: {e}")

def demo_real_world_usage():
    """Demonstrate real-world usage scenarios."""
    print("\n🌍 REAL-WORLD USAGE SCENARIOS")
    print("=" * 50)
    
    try:
        categorizer = ExpenseCategorizer(
            model_path="models/expense_category_model.joblib",
            feature_extractor_path="models/feature_extractor.joblib"
        )
    except:
        print("❌ Models not found. Please run training first.")
        return
    
    # Scenario 1: Bank SMS parsing
    print("📱 Scenario 1: Processing Bank SMS Notifications")
    bank_sms_data = [
        {"merchant": "UPI-ZOMATO", "description": "UPI Payment to ZOMATO", "amount": 450},
        {"merchant": "PAYTM", "description": "Bill Payment ELECTRICITY", "amount": 2500},
        {"merchant": "ATM-WITHDRAW", "description": "Cash Withdrawal", "amount": 5000},
        {"merchant": "AMAZON PAY", "description": "Online Purchase", "amount": 1200},
    ]
    
    for sms in bank_sms_data:
        result = categorizer.categorize_expense(
            merchant=sms['merchant'],
            description=sms['description'],
            amount=sms['amount']
        )
        print(f"   {sms['description']} → {result['category']} ({result['confidence_level']} Confidence)")
    
    # Scenario 2: Credit card statement processing
    print(f"\n💳 Scenario 2: Credit Card Statement Processing")
    cc_transactions = [
        {"merchant": "NETFLIX.COM", "description": "Subscription Payment", "amount": 650},
        {"merchant": "SHELL PETROL", "description": "Fuel Purchase", "amount": 3000},
        {"merchant": "APOLLO PHARMACY", "description": "Medicine Purchase", "amount": 800},
        {"merchant": "SWIGGY", "description": "Food Delivery", "amount": 600},
    ]
    
    predictions = categorizer.predict_batch_transactions(cc_transactions)
    for i, (trans, pred) in enumerate(zip(cc_transactions, predictions)):
        print(f"   {trans['merchant']}: ₹{trans['amount']} → {pred['predicted_category']}")
    
    print(f"\n🏦 Scenario 3: Monthly Budget Tracking")
    # This would integrate with banking APIs or manual input
    print("   Integration points:")
    print("   • Bank SMS parsing")
    print("   • Credit card statement import")
    print("   • UPI transaction history")
    print("   • Manual expense entry")
    print("   • Receipt scanning (future enhancement)")

def main():
    """Run the complete demonstration."""
    print("🎯 EXPENSE CATEGORIZATION ML MODEL - COMPLETE DEMO")
    print("=" * 60)
    print("This demo showcases a production-ready ML model for automatic expense categorization")
    print("=" * 60)
    
    try:
        # Check if models exist
        if not os.path.exists("models/expense_category_model.joblib"):
            print("🔄 Models not found. Training new models...")
            demo_training()
        else:
            print("✅ Using existing trained models")
        
        # Run demonstrations
        demo_predictions()
        demo_spending_analysis()
        demo_model_info()
        demo_real_world_usage()
        
        print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("💡 Next Steps:")
        print("1. Integrate with your banking/payment APIs")
        print("2. Customize categories for your specific needs")
        print("3. Retrain with your historical transaction data")
        print("4. Deploy as a web service or mobile app backend")
        print("5. Add investment recommendation features")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please check the error logs and ensure all dependencies are installed.")

if __name__ == "__main__":
    main()