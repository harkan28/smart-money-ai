#!/usr/bin/env python3
"""
Smart Money - Comprehensive Integration System
Connects SMS parsing, ML categorization, and intelligent budgeting
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import json
import logging

# Add both systems to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'SMS PARSING SYSTEM'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'budgeting_ml_model', 'src'))

try:
    from sms_parser.core_parser import SMSParser
    sys.path.append(os.path.join(os.path.dirname(__file__), 'budgeting_ml_model'))
    from src.inference import ExpenseCategorizer
    from src.incremental_learning import IncrementalLearner
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("Please ensure both SMS parsing and ML model systems are set up")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartMoneyIntegrator:
    """Comprehensive Smart Money system integrating all components."""
    
    def __init__(self):
        """Initialize all system components."""
        self.sms_parser = None
        self.ml_categorizer = None
        self.learner = None
        self.transactions_db = []
        self.budgets = {}
        
        self.initialize_components()
    
    def initialize_components(self):
        """Initialize SMS parser, ML model, and learning system."""
        try:
            # Initialize SMS parser
            self.sms_parser = SMSParser()
            logger.info("✅ SMS Parser initialized")
            
            # Initialize ML categorizer (if models exist)
            model_path = "budgeting_ml_model/models/expense_category_model.joblib"
            feature_path = "budgeting_ml_model/models/feature_extractor.joblib"
            
            if os.path.exists(model_path) and os.path.exists(feature_path):
                self.ml_categorizer = ExpenseCategorizer(model_path, feature_path)
                logger.info("✅ ML Categorizer initialized")
            else:
                logger.warning("⚠️ ML models not found. Please train the model first.")
            
            # Initialize incremental learner
            self.learner = IncrementalLearner()
            logger.info("✅ Incremental Learner initialized")
            
        except Exception as e:
            logger.error(f"❌ Initialization error: {e}")
    
    def process_bank_sms(self, sms_text: str, sender: str) -> Dict:
        """Process bank SMS and categorize the transaction."""
        try:
            # Parse SMS
            if self.sms_parser:
                transaction_data = self.sms_parser.parse_sms(sms_text, sender)
                if not transaction_data:
                    return {"status": "failed", "error": "SMS parsing failed"}
                
                # Convert Transaction object to dict
                transaction_dict = {
                    'amount': transaction_data.amount,
                    'merchant': transaction_data.merchant,
                    'account_number': transaction_data.account_number,
                    'transaction_type': transaction_data.transaction_type,
                    'bank_name': transaction_data.bank_name,
                    'description': f"{transaction_data.merchant} transaction"
                }
            else:
                # Fallback manual parsing
                transaction_dict = self.manual_sms_parse(sms_text)
            
            # Categorize using ML
            if self.ml_categorizer and transaction_dict:
                merchant = transaction_dict.get('merchant', 'Unknown')
                description = transaction_dict.get('description', sms_text)
                amount = transaction_dict.get('amount', 0)
                
                ml_result = self.ml_categorizer.categorize_expense(
                    merchant=merchant,
                    description=description,
                    amount=amount
                )
                
                # Combine SMS data with ML categorization
                result = {
                    **transaction_dict,
                    'category': ml_result['category'],
                    'confidence': ml_result['confidence'],
                    'alternatives': ml_result.get('alternatives', []),
                    'timestamp': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                return {"success": True, "transaction": result}
                
                # Store for learning
                self.store_transaction(result)
                
                return result
            
            return {"status": "failed", "error": "ML categorizer not available"}
            
        except Exception as e:
            logger.error(f"❌ SMS processing error: {e}")
            return {"status": "failed", "error": str(e)}
    
    def manual_sms_parse(self, sms_text: str) -> Dict:
        """Manual SMS parsing as fallback."""
        import re
        
        # Extract amount
        amount_match = re.search(r'Rs\.?\s*(\d+(?:,\d+)*(?:\.\d+)?)', sms_text)
        amount = float(amount_match.group(1).replace(',', '')) if amount_match else 0
        
        # Extract merchant (basic pattern)
        merchant_patterns = [
            r'at\s+([A-Z][A-Z\s]+)',
            r'to\s+([A-Z][A-Z\s]+)',
            r'from\s+([A-Z][A-Z\s]+)'
        ]
        
        merchant = "Unknown"
        for pattern in merchant_patterns:
            match = re.search(pattern, sms_text, re.IGNORECASE)
            if match:
                merchant = match.group(1).strip()
                break
        
        return {
            'merchant': merchant,
            'amount': amount,
            'description': sms_text[:100],  # First 100 chars
            'raw_sms': sms_text
        }
    
    def store_transaction(self, transaction: Dict):
        """Store transaction for analysis and learning."""
        self.transactions_db.append(transaction)
    
    def process_user_feedback(self, transaction_id: str, correct_category: str):
        """Process user feedback for incremental learning."""
        try:
            # Find the transaction
            transaction = None
            for t in self.transactions_db:
                if t.get('id') == transaction_id:
                    transaction = t
                    break
            
            if not transaction:
                return {"status": "failed", "error": "Transaction not found"}
            
            # Record feedback for learning
            if self.learner:
                self.learner.record_prediction(
                    merchant=transaction['merchant'],
                    description=transaction['description'],
                    amount=transaction['amount'],
                    predicted_category=transaction['predicted_category'],
                    confidence=transaction['confidence'],
                    user_confirmed=(correct_category == transaction['predicted_category']),
                    correct_category=correct_category if correct_category != transaction['predicted_category'] else None
                )
            
            # Check if retraining is needed
            if self.learner and self.learner.should_retrain():
                logger.info("🔄 Triggering incremental learning...")
                self.learner.trigger_incremental_learning()
            
            return {"status": "success", "message": "Feedback recorded"}
            
        except Exception as e:
            logger.error(f"❌ Feedback processing error: {e}")
            return {"status": "failed", "error": str(e)}
    
    def analyze_spending_patterns(self, days: int = 30) -> Dict:
        """Analyze spending patterns from recent transactions."""
        try:
            # Filter recent transactions
            cutoff_date = datetime.now() - timedelta(days=days)
            recent_transactions = [
                t for t in self.transactions_db 
                if datetime.fromisoformat(t['timestamp']) > cutoff_date
            ]
            
            if not recent_transactions:
                return {"status": "no_data", "message": "No recent transactions found"}
            
            # Calculate spending by category
            category_spending = {}
            total_spending = 0
            
            for transaction in recent_transactions:
                category = transaction.get('predicted_category', 'MISCELLANEOUS')
                amount = transaction.get('amount', 0)
                
                category_spending[category] = category_spending.get(category, 0) + amount
                total_spending += amount
            
            # Calculate percentages
            category_percentages = {
                cat: (amount / total_spending) * 100 
                for cat, amount in category_spending.items()
            }
            
            # Find top categories
            top_categories = sorted(
                category_spending.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:5]
            
            return {
                "status": "success",
                "analysis_period": f"{days} days",
                "total_transactions": len(recent_transactions),
                "total_spending": round(total_spending, 2),
                "category_spending": category_spending,
                "category_percentages": category_percentages,
                "top_categories": top_categories,
                "average_transaction": round(total_spending / len(recent_transactions), 2)
            }
            
        except Exception as e:
            logger.error(f"❌ Analysis error: {e}")
            return {"status": "failed", "error": str(e)}
    
    def suggest_budget(self, income: float) -> Dict:
        """Suggest intelligent budget based on spending patterns and income."""
        try:
            # Get spending analysis
            analysis = self.analyze_spending_patterns(90)  # 3 months
            
            if analysis['status'] != 'success':
                # Use default budgeting rules
                return self.default_budget_suggestion(income)
            
            # Calculate budget based on 50/30/20 rule with AI adjustments
            total_spending = analysis['total_spending']
            monthly_spending = total_spending / 3  # 3 months average
            
            # Needs (50% of income)
            needs_budget = income * 0.5
            needs_categories = ['FOOD_DINING', 'TRANSPORTATION', 'UTILITIES', 'HEALTHCARE']
            
            # Wants (30% of income)
            wants_budget = income * 0.3
            wants_categories = ['ENTERTAINMENT', 'SHOPPING', 'MISCELLANEOUS']
            
            # Savings/Investments (20% of income)
            savings_budget = income * 0.2
            
            # Distribute budgets based on spending patterns
            category_budgets = {}
            category_spending = analysis.get('category_spending', {})
            
            # Allocate needs budget
            needs_total_spending = sum(category_spending.get(cat, 0) for cat in needs_categories)
            if needs_total_spending > 0:
                for category in needs_categories:
                    proportion = category_spending.get(category, 0) / needs_total_spending
                    category_budgets[category] = round(needs_budget * proportion, 2)
            
            # Allocate wants budget
            wants_total_spending = sum(category_spending.get(cat, 0) for cat in wants_categories)
            if wants_total_spending > 0:
                for category in wants_categories:
                    proportion = category_spending.get(category, 0) / wants_total_spending
                    category_budgets[category] = round(wants_budget * proportion, 2)
            
            # Investment/Education budget
            category_budgets['INVESTMENT'] = round(savings_budget * 0.8, 2)
            category_budgets['EDUCATION'] = round(savings_budget * 0.2, 2)
            
            return {
                "status": "success",
                "monthly_income": income,
                "total_budget": income,
                "category_budgets": category_budgets,
                "budget_rules": {
                    "needs": f"₹{needs_budget:,.2f} (50%)",
                    "wants": f"₹{wants_budget:,.2f} (30%)",
                    "savings": f"₹{savings_budget:,.2f} (20%)"
                },
                "recommendations": self.generate_budget_recommendations(category_budgets, analysis)
            }
            
        except Exception as e:
            logger.error(f"❌ Budget suggestion error: {e}")
            return {"status": "failed", "error": str(e)}
    
    def default_budget_suggestion(self, income: float) -> Dict:
        """Default budget suggestion when no spending history is available."""
        return {
            "status": "success",
            "monthly_income": income,
            "category_budgets": {
                "FOOD_DINING": round(income * 0.25, 2),
                "TRANSPORTATION": round(income * 0.10, 2),
                "UTILITIES": round(income * 0.08, 2),
                "HEALTHCARE": round(income * 0.07, 2),
                "ENTERTAINMENT": round(income * 0.10, 2),
                "SHOPPING": round(income * 0.15, 2),
                "EDUCATION": round(income * 0.05, 2),
                "INVESTMENT": round(income * 0.15, 2),
                "MISCELLANEOUS": round(income * 0.05, 2)
            },
            "note": "Default budget based on standard recommendations. Will improve as we learn your spending patterns."
        }
    
    def generate_budget_recommendations(self, budgets: Dict, analysis: Dict) -> List[str]:
        """Generate personalized budget recommendations."""
        recommendations = []
        
        # Check if spending exceeds suggested budgets
        category_spending = analysis.get('category_spending', {})
        for category, budget in budgets.items():
            spent = category_spending.get(category, 0)
            if spent > budget * 1.2:  # 20% over budget
                recommendations.append(
                    f"Consider reducing {category.lower().replace('_', ' ')} expenses by ₹{spent - budget:,.0f}"
                )
        
        # Investment recommendations
        if budgets.get('INVESTMENT', 0) > 0:
            recommendations.append("Great! You have budget allocated for investments. Consider SIP in mutual funds.")
        
        # Savings recommendations
        total_needs = sum(budgets.get(cat, 0) for cat in ['FOOD_DINING', 'TRANSPORTATION', 'UTILITIES', 'HEALTHCARE'])
        if total_needs < budgets.get('ENTERTAINMENT', 0) + budgets.get('SHOPPING', 0):
            recommendations.append("Consider increasing emergency fund allocation.")
        
        return recommendations[:5]  # Top 5 recommendations


def main():
    """Demo the integrated Smart Money system."""
    print("🎯 Smart Money - Integrated Financial Assistant")
    print("=" * 50)
    
    # Initialize system
    integrator = SmartMoneyIntegrator()
    
    # Demo SMS processing
    sample_sms = "Dear Customer, Rs.450 has been debited from your account for UPI payment to ZOMATO on 14-Oct-24. Available balance: Rs.12,550"
    
    print("\n📱 Processing Sample SMS:")
    print(f"SMS: {sample_sms[:80]}...")
    
    result = integrator.process_bank_sms(sample_sms, "HDFC-BANK")
    print(f"Result: {json.dumps(result, indent=2)}")
    
    # Demo spending analysis
    print("\n📊 Spending Analysis:")
    analysis = integrator.analyze_spending_patterns()
    print(f"Analysis: {json.dumps(analysis, indent=2)}")
    
    # Demo budget suggestion
    print("\n💰 Budget Suggestion:")
    budget = integrator.suggest_budget(50000)  # ₹50K monthly income
    print(f"Budget: {json.dumps(budget, indent=2)}")
    
    print("\n🎉 Smart Money integration demo completed!")


if __name__ == "__main__":
    main()