#!/usr/bin/env python3
"""
Smart Money AI - Simple Test Runner
==================================

Basic test runner to validate the structured system
"""

import sys
import os
from pathlib import Path

# Add src to path
CURRENT_DIR = Path(__file__).parent
SRC_DIR = CURRENT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

def test_imports():
    """Test that all modules can be imported"""
    print("🧪 Testing Module Imports...")
    
    try:
        from parsers.sms_parser import SMSParser
        print("✅ SMS Parser import successful")
    except Exception as e:
        print(f"❌ SMS Parser import failed: {e}")
        return False
    
    try:
        from ml_models.expense_categorizer import ExpenseCategorizer
        print("✅ ML Categorizer import successful")
    except Exception as e:
        print(f"❌ ML Categorizer import failed: {e}")
        return False
    
    try:
        from utils.config_manager import ConfigManager
        print("✅ Config Manager import successful")
    except Exception as e:
        print(f"❌ Config Manager import failed: {e}")
        return False
    
    try:
        from utils.data_manager import DataManager
        print("✅ Data Manager import successful")
    except Exception as e:
        print(f"❌ Data Manager import failed: {e}")
        return False
    
    try:
        from core.smart_money_ai import SmartMoneyAI
        print("✅ Core Smart Money AI import successful")
    except Exception as e:
        print(f"❌ Core Smart Money AI import failed: {e}")
        return False
    
    return True

def test_sms_parsing():
    """Test SMS parsing functionality"""
    print("\n🧪 Testing SMS Parsing...")
    
    try:
        from parsers.sms_parser import SMSParser
        
        parser = SMSParser()
        
        # Test cases
        test_cases = [
            ("HDFC Bank: Rs 2500 debited from A/c **1234 on 15-Oct-25 at AMAZON PAY for UPI txn. Avl bal Rs 25000", "HDFCBK"),
            ("SBI: Rs 450 debited from A/c **5678 on 15-Oct-25 at ZOMATO for UPI txn. Avl bal Rs 15000", "SBIINB"),
        ]
        
        success_count = 0
        for sms, sender in test_cases:
            transaction = parser.parse_sms(sms, sender)
            if transaction and transaction.amount > 0:
                success_count += 1
                print(f"  ✅ Parsed: ₹{transaction.amount} at {transaction.merchant}")
            else:
                print(f"  ❌ Failed to parse: {sms[:50]}...")
        
        success_rate = (success_count / len(test_cases)) * 100
        print(f"📊 SMS Parsing Success Rate: {success_rate:.1f}%")
        
        return success_rate >= 50  # At least 50% should work
        
    except Exception as e:
        print(f"❌ SMS parsing test failed: {e}")
        return False

def test_ml_categorization():
    """Test ML expense categorization"""
    print("\n🧪 Testing ML Categorization...")
    
    try:
        from ml_models.expense_categorizer import ExpenseCategorizer
        
        categorizer = ExpenseCategorizer()
        
        # Test cases
        test_cases = [
            ("ZOMATO", 450.0, "FOOD_DINING"),
            ("UBER INDIA", 350.0, "TRANSPORTATION"),
            ("AMAZON PAY", 2500.0, "SHOPPING"),
            ("APOLLO PHARMACY", 1200.0, "HEALTHCARE"),
            ("ATM WITHDRAWAL", 5000.0, "CASH_WITHDRAWAL")
        ]
        
        success_count = 0
        for merchant, amount, expected_category in test_cases:
            result = categorizer.categorize_expense(merchant, amount)
            if result['category'] == expected_category:
                success_count += 1
                print(f"  ✅ {merchant}: {result['category']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"  ⚠️  {merchant}: Expected {expected_category}, got {result['category']}")
        
        success_rate = (success_count / len(test_cases)) * 100
        print(f"📊 ML Categorization Accuracy: {success_rate:.1f}%")
        
        return success_rate >= 60  # At least 60% should be correct
        
    except Exception as e:
        print(f"❌ ML categorization test failed: {e}")
        return False

def test_data_management():
    """Test data management functionality"""
    print("\n🧪 Testing Data Management...")
    
    try:
        from utils.data_manager import DataManager
        
        # Create test data manager
        data_manager = DataManager("test_data_validation/")
        
        # Test user save/load
        user_data = {
            "user_id": "test_validation_001",
            "name": "Test User",
            "email": "test@validation.com",
            "phone": "+91-1234567890",
            "age": 30,
            "annual_income": 600000,
            "risk_profile": "moderate",
            "investment_timeline": 15,
            "financial_goals": ["wealth_creation"]
        }
        
        # Save user
        save_success = data_manager.save_user(user_data)
        if save_success:
            print("  ✅ User data save successful")
        else:
            print("  ❌ User data save failed")
            return False
        
        # Load user
        loaded_user = data_manager.load_user("test_validation_001")
        if loaded_user and loaded_user['name'] == "Test User":
            print("  ✅ User data load successful")
        else:
            print("  ❌ User data load failed")
            return False
        
        # Get database stats
        stats = data_manager.get_database_stats()
        if stats and 'total_users' in stats:
            print(f"  ✅ Database stats: {stats['total_users']} users")
        else:
            print("  ❌ Database stats failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Data management test failed: {e}")
        return False

def test_configuration():
    """Test configuration management"""
    print("\n🧪 Testing Configuration Management...")
    
    try:
        from utils.config_manager import ConfigManager
        
        config = ConfigManager()
        
        # Test basic config operations
        test_key = "test_validation_key"
        test_value = "test_validation_value"
        
        config.set(test_key, test_value)
        retrieved_value = config.get(test_key)
        
        if retrieved_value == test_value:
            print("  ✅ Config set/get successful")
        else:
            print("  ❌ Config set/get failed")
            return False
        
        # Test validation
        is_valid = config.validate_config()
        if is_valid:
            print("  ✅ Config validation successful")
        else:
            print("  ❌ Config validation failed")
            return False
        
        # Test demo mode
        demo_mode = config.is_demo_mode()
        print(f"  ✅ Demo mode: {demo_mode}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_integration():
    """Test basic integration workflow"""
    print("\n🧪 Testing Integration Workflow...")
    
    try:
        from core.smart_money_ai import SmartMoneyAI
        
        # Initialize system
        smart_money = SmartMoneyAI()
        print("  ✅ Smart Money AI initialized")
        
        # Create user
        user_data = {
            "user_id": "integration_test_001",
            "name": "Integration Test User",
            "email": "integration@test.com",
            "phone": "+91-9876543210",
            "age": 28,
            "annual_income": 600000
        }
        
        user = smart_money.create_user_profile(user_data)
        if user.user_id == "integration_test_001":
            print("  ✅ User profile creation successful")
        else:
            print("  ❌ User profile creation failed")
            return False
        
        # Add transaction
        transaction_data = {
            "amount": 5000.0,
            "merchant": "TEST MERCHANT",
            "category": "TESTING"
        }
        
        result = smart_money.add_manual_transaction(user.user_id, transaction_data)
        if result['status'] == 'success':
            print("  ✅ Transaction addition successful")
        else:
            print("  ❌ Transaction addition failed")
            return False
        
        # Generate insights
        insights = smart_money.generate_comprehensive_insights(user.user_id)
        if insights and insights.transaction_summary['total_transactions'] > 0:
            print("  ✅ Insights generation successful")
        else:
            print("  ❌ Insights generation failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Run all validation tests"""
    print("🚀 Smart Money AI - System Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Module Imports", test_imports),
        ("SMS Parsing", test_sms_parsing),
        ("ML Categorization", test_ml_categorization),
        ("Data Management", test_data_management),
        ("Configuration", test_configuration),
        ("Integration Workflow", test_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 VALIDATION RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:.<30} {status}")
    
    print(f"\nOverall Success Rate: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL VALIDATION TESTS PASSED!")
        print("✅ Smart Money AI structured system is working perfectly!")
        print("✅ Ready for production deployment!")
    elif passed >= total * 0.8:
        print("\n✅ MOST VALIDATION TESTS PASSED!")
        print("⚠️  Some minor issues detected, but system is functional")
    else:
        print("\n⚠️  VALIDATION ISSUES DETECTED!")
        print("❌ System needs attention before production use")
    
    return passed >= total * 0.8  # 80% pass rate required

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)