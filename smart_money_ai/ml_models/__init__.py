"""
SMART MONEY AI - 4-PART ML SYSTEM ARCHITECTURE
=============================================

This system consists of 4 independent ML models that work together:

1. SMS PARSING MODEL - Extract transaction data from banking SMS
2. EXPENSE CATEGORIZATION MODEL - Automatically categorize expenses
3. SAVINGS & BUDGETING MODEL - Monthly savings analysis and budget optimization
4. INVESTMENT RECOMMENDATION MODEL - Stock, mutual fund, gold/silver recommendations

Each model can operate independently and will be integrated via API when frontend is ready.
"""

from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
import os

class SmartMoneyMLSystem:
    """
    Main orchestrator for the 4-part ML system
    Coordinates between all models while keeping them independent
    """
    
    def __init__(self):
        self.data_dir = "/Users/harshitrawal/Downloads/SMART MONEY/smart_money_ai/data"
        self.models_dir = "/Users/harshitrawal/Downloads/SMART MONEY/smart_money_ai/ml_models"
        
        # Initialize all 4 models
        self.sms_parser = None
        self.expense_categorizer = None
        self.savings_advisor = None
        self.investment_advisor = None
        
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(f"{self.models_dir}/sms_parser", exist_ok=True)
        os.makedirs(f"{self.models_dir}/expense_categorizer", exist_ok=True)
        os.makedirs(f"{self.models_dir}/savings_advisor", exist_ok=True)
        os.makedirs(f"{self.models_dir}/investment_advisor", exist_ok=True)
    
    def initialize_models(self):
        """Initialize all 4 ML models"""
        from .ml_models.sms_parser.sms_parsing_model import SMSParsingModel
        from .ml_models.expense_categorizer.categorization_model import ExpenseCategorizer
        from .ml_models.savings_advisor.savings_model import SavingsAdvisor
        from .ml_models.investment_advisor.investment_model import InvestmentAdvisor
        
        self.sms_parser = SMSParsingModel()
        self.expense_categorizer = ExpenseCategorizer()
        self.savings_advisor = SavingsAdvisor()
        self.investment_advisor = InvestmentAdvisor()
        
        print("✅ All 4 ML models initialized successfully!")
    
    def process_user_financial_data(self, user_id: str, sms_data: List[str] = None, 
                                  manual_transactions: List[Dict] = None) -> Dict[str, Any]:
        """
        Complete pipeline processing user financial data through all 4 models
        
        Args:
            user_id: Unique user identifier
            sms_data: List of SMS messages to parse
            manual_transactions: Manually added transactions
            
        Returns:
            Complete financial analysis and recommendations
        """
        
        results = {
            'user_id': user_id,
            'timestamp': datetime.now().isoformat(),
            'sms_parsing': {},
            'expense_categorization': {},
            'savings_analysis': {},
            'investment_recommendations': {}
        }
        
        # Model 1: SMS Parsing
        if sms_data:
            results['sms_parsing'] = self.sms_parser.parse_sms_batch(sms_data)
        
        # Model 2: Expense Categorization
        all_transactions = []
        if results['sms_parsing'].get('transactions'):
            all_transactions.extend(results['sms_parsing']['transactions'])
        if manual_transactions:
            all_transactions.extend(manual_transactions)
        
        if all_transactions:
            results['expense_categorization'] = self.expense_categorizer.categorize_transactions(all_transactions)
        
        # Model 3: Savings & Budgeting Analysis
        if results['expense_categorization'].get('categorized_expenses'):
            results['savings_analysis'] = self.savings_advisor.analyze_savings_potential(
                user_id, results['expense_categorization']['categorized_expenses']
            )
        
        # Model 4: Investment Recommendations
        financial_profile = self._build_financial_profile(results)
        results['investment_recommendations'] = self.investment_advisor.get_investment_recommendations(
            user_id, financial_profile
        )
        
        return results
    
    def _build_financial_profile(self, results: Dict) -> Dict:
        """Build comprehensive financial profile for investment recommendations"""
        
        profile = {
            'monthly_income': 0,
            'monthly_expenses': 0,
            'savings_rate': 0,
            'expense_categories': {},
            'spending_behavior': 'moderate',
            'risk_tolerance': 'medium'
        }
        
        # Extract from categorized expenses
        if results.get('expense_categorization', {}).get('categorized_expenses'):
            expenses = results['expense_categorization']['categorized_expenses']
            profile['monthly_expenses'] = sum(t.get('amount', 0) for t in expenses)
            
            # Category breakdown
            for transaction in expenses:
                category = transaction.get('category', 'other')
                profile['expense_categories'][category] = profile['expense_categories'].get(category, 0) + transaction.get('amount', 0)
        
        # Extract from savings analysis
        if results.get('savings_analysis'):
            savings_data = results['savings_analysis']
            profile['savings_rate'] = savings_data.get('savings_rate', 0)
            profile['spending_behavior'] = savings_data.get('spending_behavior', 'moderate')
        
        return profile
    
    def get_model_status(self) -> Dict[str, bool]:
        """Check status of all 4 models"""
        return {
            'sms_parser': self.sms_parser is not None,
            'expense_categorizer': self.expense_categorizer is not None,
            'savings_advisor': self.savings_advisor is not None,
            'investment_advisor': self.investment_advisor is not None
        }

# Export the main system class
__all__ = ['SmartMoneyMLSystem']