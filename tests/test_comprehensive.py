#!/usr/bin/env python3
"""
Comprehensive Test Suite for Smart Money AI
==========================================

Tests all major components and integration workflows
"""

import sys
import os
import pytest
import unittest
from datetime import datetime
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    from src.core.smart_money_ai import SmartMoneyAI, SmartMoneyUser
    from src.parsers.sms_parser import SMSParser, Transaction
    from src.ml_models.expense_categorizer import ExpenseCategorizer
    from src.utils.config_manager import ConfigManager
    from src.utils.data_manager import DataManager
except ImportError:
    # Fallback to direct imports
    from core.smart_money_ai import SmartMoneyAI, SmartMoneyUser
    from parsers.sms_parser import SMSParser, Transaction
    from ml_models.expense_categorizer import ExpenseCategorizer
    from utils.config_manager import ConfigManager
    from utils.data_manager import DataManager


class TestSMSParser(unittest.TestCase):
    """Test SMS parsing functionality"""
    
    def setUp(self):
        self.parser = SMSParser()
    
    def test_hdfc_transaction_parsing(self):
        """Test HDFC bank SMS parsing"""
        sms = "HDFC Bank: Rs 2500 debited from A/c **1234 on 15-Oct-25 at AMAZON PAY for UPI txn. Avl bal Rs 25000"
        sender = "HDFCBK"
        
        transaction = self.parser.parse_sms(sms, sender)
        
        self.assertIsNotNone(transaction)
        self.assertEqual(transaction.amount, 2500.0)
        self.assertIn("AMAZON", transaction.merchant.upper())
        self.assertEqual(transaction.bank_name, "HDFC")
        self.assertEqual(transaction.transaction_type, "debit")
    
    def test_sbi_transaction_parsing(self):
        """Test SBI bank SMS parsing"""
        sms = "SBI: Rs 450 debited from A/c **5678 on 15-Oct-25 at ZOMATO for UPI txn. Avl bal Rs 15000"
        sender = "SBIINB"
        
        transaction = self.parser.parse_sms(sms, sender)
        
        self.assertIsNotNone(transaction)
        self.assertEqual(transaction.amount, 450.0)
        self.assertIn("ZOMATO", transaction.merchant.upper())
        self.assertEqual(transaction.bank_name, "SBI")
    
    def test_unknown_bank_fallback(self):
        """Test fallback parsing for unknown banks"""
        sms = "Unknown Bank: Rs 1000 debited for purchase"
        sender = "UNKNOWN"
        
        transaction = self.parser.parse_sms(sms, sender)
        
        # Should either parse with fallback or return None gracefully
        if transaction:
            self.assertEqual(transaction.amount, 1000.0)


class TestExpenseCategorizer(unittest.TestCase):
    """Test ML expense categorization"""
    
    def setUp(self):
        self.categorizer = ExpenseCategorizer()
    
    def test_food_categorization(self):
        """Test food-related expense categorization"""
        result = self.categorizer.categorize_expense("ZOMATO", 450.0)
        
        self.assertIsInstance(result, dict)
        self.assertIn('category', result)
        self.assertIn('confidence', result)
        self.assertEqual(result['category'], 'FOOD_DINING')
        self.assertGreater(result['confidence'], 0.5)
    
    def test_transportation_categorization(self):
        """Test transportation expense categorization"""
        result = self.categorizer.categorize_expense("UBER INDIA", 350.0)
        
        self.assertEqual(result['category'], 'TRANSPORTATION')
        self.assertGreater(result['confidence'], 0.5)
    
    def test_shopping_categorization(self):
        """Test shopping expense categorization"""
        result = self.categorizer.categorize_expense("AMAZON PAY", 2500.0)
        
        self.assertEqual(result['category'], 'SHOPPING')
        self.assertGreater(result['confidence'], 0.5)
    
    def test_healthcare_categorization(self):
        """Test healthcare expense categorization"""
        result = self.categorizer.categorize_expense("APOLLO PHARMACY", 1200.0)
        
        self.assertEqual(result['category'], 'HEALTHCARE')
        self.assertGreater(result['confidence'], 0.5)
    
    def test_batch_categorization(self):
        """Test batch expense categorization"""
        expenses = [
            {"merchant": "ZOMATO", "amount": 450},
            {"merchant": "UBER", "amount": 350},
            {"merchant": "AMAZON", "amount": 2500}
        ]
        
        results = self.categorizer.batch_categorize(expenses)
        
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertIn('category', result)
            self.assertIn('confidence', result)


class TestConfigManager(unittest.TestCase):
    """Test configuration management"""
    
    def setUp(self):
        self.config = ConfigManager()
    
    def test_default_config_loading(self):
        """Test default configuration values"""
        self.assertTrue(self.config.is_demo_mode())
        self.assertEqual(self.config.get('log_level'), 'INFO')
        self.assertIsNotNone(self.config.get('ml_model_path'))
    
    def test_config_get_set(self):
        """Test configuration get/set operations"""
        test_key = "test_setting"
        test_value = "test_value"
        
        self.config.set(test_key, test_value)
        self.assertEqual(self.config.get(test_key), test_value)
    
    def test_model_paths(self):
        """Test model path configuration"""
        paths = self.config.get_model_paths()
        
        self.assertIn('model_path', paths)
        self.assertIn('feature_extractor_path', paths)
    
    def test_config_validation(self):
        """Test configuration validation"""
        is_valid = self.config.validate_config()
        self.assertTrue(is_valid)


class TestDataManager(unittest.TestCase):
    """Test data management functionality"""
    
    def setUp(self):
        self.data_manager = DataManager("test_data/")
    
    def test_user_save_load(self):
        """Test user data save and load"""
        user_data = {
            "user_id": "test_001",
            "name": "Test User",
            "email": "test@example.com",
            "phone": "+91-1234567890",
            "age": 30,
            "annual_income": 600000,
            "risk_profile": "moderate",
            "investment_timeline": 15,
            "financial_goals": ["wealth_creation"]
        }
        
        # Save user
        success = self.data_manager.save_user(user_data)
        self.assertTrue(success)
        
        # Load user
        loaded_user = self.data_manager.load_user("test_001")
        self.assertIsNotNone(loaded_user)
        self.assertEqual(loaded_user['name'], "Test User")
        self.assertEqual(loaded_user['email'], "test@example.com")
    
    def test_transaction_save_load(self):
        """Test transaction data save and load"""
        transaction_data = {
            "transaction_id": "test_txn_001",
            "user_id": "test_001",
            "amount": 1000.0,
            "merchant": "TEST MERCHANT",
            "category": "TESTING",
            "transaction_type": "debit",
            "bank_name": "TEST BANK",
            "account_number": "****1234",
            "timestamp": datetime.now()
        }
        
        # Save transaction
        success = self.data_manager.save_transaction(transaction_data)
        self.assertTrue(success)
        
        # Load transactions
        transactions = self.data_manager.load_user_transactions("test_001")
        self.assertGreater(len(transactions), 0)
        
        # Find our test transaction
        test_txn = next((t for t in transactions if t['transaction_id'] == 'test_txn_001'), None)
        self.assertIsNotNone(test_txn)
        self.assertEqual(test_txn['amount'], 1000.0)
    
    def test_database_stats(self):
        """Test database statistics"""
        stats = self.data_manager.get_database_stats()
        
        self.assertIn('total_users', stats)
        self.assertIn('total_transactions', stats)
        self.assertIn('database_size_bytes', stats)
        self.assertIsInstance(stats['total_users'], int)


class TestSmartMoneyAI(unittest.TestCase):
    """Test main Smart Money AI integration"""
    
    def setUp(self):
        self.smart_money = SmartMoneyAI()
    
    def test_user_profile_creation(self):
        """Test user profile creation"""
        user_data = {
            "user_id": "integration_test_001",
            "name": "Integration Test User",
            "email": "integration@test.com",
            "phone": "+91-9876543210",
            "age": 28,
            "annual_income": 600000,
            "risk_profile": "moderate",
            "investment_timeline": 15,
            "financial_goals": ["wealth_creation", "retirement_planning"]
        }
        
        user = self.smart_money.create_user_profile(user_data)
        
        self.assertIsInstance(user, SmartMoneyUser)
        self.assertEqual(user.user_id, "integration_test_001")
        self.assertEqual(user.name, "Integration Test User")
        self.assertIn(user.user_id, self.smart_money.users)
    
    def test_manual_transaction_addition(self):
        """Test manual transaction addition"""
        # Create user first
        user_data = {
            "user_id": "transaction_test_001",
            "name": "Transaction Test User",
            "email": "transaction@test.com",
            "phone": "+91-9876543210",
            "age": 30,
            "annual_income": 500000
        }
        
        user = self.smart_money.create_user_profile(user_data)
        
        # Add transaction
        transaction_data = {
            "amount": 5000.0,
            "merchant": "TEST GROCERY STORE",
            "category": "GROCERIES",
            "description": "Monthly grocery shopping"
        }
        
        result = self.smart_money.add_manual_transaction(user.user_id, transaction_data)
        
        self.assertEqual(result['status'], 'success')
        self.assertIn('transaction', result)
        self.assertIn('ml_prediction', result)
    
    def test_comprehensive_insights_generation(self):
        """Test comprehensive insights generation"""
        # Create user and add transactions
        user_data = {
            "user_id": "insights_test_001",
            "name": "Insights Test User",
            "email": "insights@test.com",
            "phone": "+91-9876543210",
            "age": 35,
            "annual_income": 800000
        }
        
        user = self.smart_money.create_user_profile(user_data)
        
        # Add multiple transactions
        transactions = [
            {"amount": 8000, "merchant": "RENT PAYMENT", "category": "RENT"},
            {"amount": 12000, "merchant": "GROCERY STORE", "category": "GROCERIES"},
            {"amount": 3000, "merchant": "NETFLIX", "category": "ENTERTAINMENT"},
            {"amount": 2500, "merchant": "ZOMATO", "category": "FOOD_DINING"}
        ]
        
        for txn in transactions:
            self.smart_money.add_manual_transaction(user.user_id, txn)
        
        # Generate insights
        insights = self.smart_money.generate_comprehensive_insights(user.user_id)
        
        self.assertEqual(insights.user_id, user.user_id)
        self.assertIsNotNone(insights.transaction_summary)
        self.assertIsNotNone(insights.spending_analysis)
        self.assertIsNotNone(insights.investment_recommendations)
        self.assertGreater(insights.transaction_summary['total_transactions'], 0)
    
    def test_dashboard_data_generation(self):
        """Test dashboard data generation"""
        # Create user
        user_data = {
            "user_id": "dashboard_test_001",
            "name": "Dashboard Test User",
            "email": "dashboard@test.com",
            "phone": "+91-9876543210",
            "age": 32,
            "annual_income": 700000
        }
        
        user = self.smart_money.create_user_profile(user_data)
        
        # Add some transactions
        self.smart_money.add_manual_transaction(user.user_id, {
            "amount": 10000, "merchant": "TEST MERCHANT", "category": "SHOPPING"
        })
        
        # Get dashboard data
        dashboard_data = self.smart_money.get_user_dashboard_data(user.user_id)
        
        self.assertIn('user_profile', dashboard_data)
        self.assertIn('insights', dashboard_data)
        self.assertIn('system_status', dashboard_data)
        self.assertEqual(dashboard_data['user_profile']['user_id'], user.user_id)


class TestIntegrationWorkflows(unittest.TestCase):
    """Test end-to-end integration workflows"""
    
    def test_complete_sms_to_insights_workflow(self):
        """Test complete workflow from SMS to insights"""
        smart_money = SmartMoneyAI()
        
        # Create user
        user_data = {
            "user_id": "workflow_test_001",
            "name": "Workflow Test User",
            "email": "workflow@test.com",
            "phone": "+91-9876543210",
            "age": 29,
            "annual_income": 600000
        }
        
        user = smart_money.create_user_profile(user_data)
        
        # Process SMS transactions
        sms_transactions = [
            ("HDFC Bank: Rs 2500 debited from A/c **1234 on 15-Oct-25 at AMAZON PAY for UPI txn. Avl bal Rs 25000", "HDFCBK"),
            ("SBI: Rs 450 debited from A/c **5678 on 15-Oct-25 at ZOMATO for UPI txn. Avl bal Rs 15000", "SBIINB"),
            ("ICICI Bank: Rs 350 debited from A/c **9012 on 15-Oct-25 at UBER INDIA for UPI txn. Avl bal Rs 8000", "ICICIBK")
        ]
        
        processed_count = 0
        for sms, sender in sms_transactions:
            result = smart_money.process_sms_transaction(user.user_id, sms, sender)
            if result and result.get('status') == 'success':
                processed_count += 1
        
        # Should have processed at least some transactions
        self.assertGreater(processed_count, 0)
        
        # Generate insights
        insights = smart_money.generate_comprehensive_insights(user.user_id)
        
        # Verify insights contain expected data
        self.assertGreater(insights.transaction_summary['total_transactions'], 0)
        self.assertIsInstance(insights.investment_recommendations, list)
        self.assertGreater(len(insights.optimization_suggestions), 0)


def run_tests():
    """Run all tests with detailed output"""
    print("🧪 Running Smart Money AI Test Suite")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSMSParser,
        TestExpenseCategorizer,
        TestConfigManager,
        TestDataManager,
        TestSmartMoneyAI,
        TestIntegrationWorkflows
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if not result.failures and not result.errors:
        print("\n🎉 ALL TESTS PASSED!")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)