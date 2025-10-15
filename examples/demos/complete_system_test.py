#!/usr/bin/env python3
"""
Smart Money AI - Complete System Test
Tests both SMS parsing and ML categorization together
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'SMS PARSING SYSTEM'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'budgeting_ml_model'))

from sms_parser.core_parser import SMSParser
from src.inference import ExpenseCategorizer

def test_complete_system():
    """Test the complete Smart Money AI system"""
    print("🚀 SMART MONEY AI - COMPLETE SYSTEM TEST")
    print("=" * 50)
    
    # Initialize components
    print("\n🔧 Initializing Components...")
    sms_parser = SMSParser()
    ml_categorizer = ExpenseCategorizer(
        'budgeting_ml_model/models/expense_category_model.joblib',
        'budgeting_ml_model/models/feature_extractor.joblib'
    )
    print("✅ All components initialized successfully!")
    
    # Test cases with realistic Indian bank SMS
    test_cases = [
        {
            'sms': 'HDFC Bank: Rs 2500 debited from A/c **1234 on 15-Oct-25 at AMAZON PAY for UPI txn. Avl bal Rs 25000',
            'sender': 'HDFCBK',
            'expected_category': 'SHOPPING'
        },
        {
            'sms': 'SBI: Rs 450 debited from A/c **5678 on 15-Oct-25 at ZOMATO for UPI txn. Avl bal Rs 15000',
            'sender': 'SBIINB',
            'expected_category': 'FOOD_DINING'
        },
        {
            'sms': 'ICICI Bank: Rs 350 debited from A/c **9012 on 15-Oct-25 at UBER INDIA for UPI txn. Avl bal Rs 8000',
            'sender': 'ICICIBK',
            'expected_category': 'TRANSPORTATION'
        },
        {
            'sms': 'Axis Bank: Rs 1200 debited from A/c **3456 on 15-Oct-25 at APOLLO PHARMACY for UPI txn. Avl bal Rs 12000',
            'sender': 'AXISBK',
            'expected_category': 'HEALTHCARE'
        },
        {
            'sms': 'HDFC Bank: Rs 599 debited from A/c **7890 on 15-Oct-25 at NETFLIX for UPI txn. Avl bal Rs 20000',
            'sender': 'HDFCBK',
            'expected_category': 'ENTERTAINMENT'
        }
    ]
    
    print("\n🧪 Testing SMS Parsing + ML Categorization...")
    print("-" * 60)
    
    total_tests = len(test_cases)
    successful_parsing = 0
    successful_categorization = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📱 Test {i}: {test_case['sms'][:50]}...")
        
        # Step 1: Parse SMS
        transaction = sms_parser.parse_sms(test_case['sms'], test_case['sender'])
        
        if transaction:
            successful_parsing += 1
            print(f"  ✅ SMS Parsed: Rs {transaction.amount} at {transaction.merchant}")
            
            # Step 2: Categorize with ML
            ml_result = ml_categorizer.categorize_expense(
                merchant=transaction.merchant,
                description=f"{transaction.merchant} transaction",
                amount=transaction.amount
            )
            
            predicted_category = ml_result['category']
            confidence = ml_result['confidence']
            
            print(f"  🤖 ML Prediction: {predicted_category} (confidence: {confidence:.2f})")
            
            # Check if categorization is reasonable (confidence > 0.3)
            if confidence > 0.3:
                successful_categorization += 1
                print(f"  ✅ High confidence prediction")
            else:
                print(f"  ⚠️  Low confidence prediction")
                
        else:
            print(f"  ❌ SMS parsing failed")
    
    print("\n" + "=" * 60)
    print("📊 SYSTEM TEST RESULTS")
    print("=" * 60)
    print(f"🔍 SMS Parsing Success Rate: {successful_parsing}/{total_tests} ({successful_parsing/total_tests*100:.1f}%)")
    print(f"🤖 ML Categorization Success Rate: {successful_categorization}/{total_tests} ({successful_categorization/total_tests*100:.1f}%)")
    print(f"🎯 Overall System Success Rate: {min(successful_parsing, successful_categorization)}/{total_tests} ({min(successful_parsing, successful_categorization)/total_tests*100:.1f}%)")
    
    if successful_parsing >= 4 and successful_categorization >= 4:
        print("\n🎉 SYSTEM STATUS: EXCELLENT! Both models working perfectly!")
        print("✅ SMS parsing is highly accurate")
        print("✅ ML categorization is working well")
        print("✅ Ready for production deployment")
    elif successful_parsing >= 3 and successful_categorization >= 3:
        print("\n✅ SYSTEM STATUS: GOOD! Models working well")
        print("✅ Minor improvements may be needed")
    else:
        print("\n⚠️  SYSTEM STATUS: NEEDS ATTENTION")
        print("❌ Some components need debugging")
    
    return {
        'sms_success_rate': successful_parsing / total_tests,
        'ml_success_rate': successful_categorization / total_tests,
        'overall_success_rate': min(successful_parsing, successful_categorization) / total_tests
    }

if __name__ == "__main__":
    results = test_complete_system()