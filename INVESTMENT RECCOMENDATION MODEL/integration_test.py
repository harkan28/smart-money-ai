#!/usr/bin/env python3
"""
Smart Money AI Integration Test
Tests the complete integration of all components in the Smart Money AI system
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_smart_money_integration():
    """Comprehensive integration test for Smart Money AI"""
    
    print("🧪 Smart Money AI Integration Test")
    print("=" * 50)
    
    # Test Results Storage
    test_results = {
        'total_tests': 0,
        'passed_tests': 0,
        'failed_tests': 0,
        'test_details': []
    }
    
    def run_test(test_name: str, test_function, *args) -> bool:
        """Run a test and record results"""
        test_results['total_tests'] += 1
        try:
            result = test_function(*args)
            if result:
                test_results['passed_tests'] += 1
                test_results['test_details'].append(f"✅ {test_name}: PASSED")
                print(f"✅ {test_name}: PASSED")
                return True
            else:
                test_results['failed_tests'] += 1
                test_results['test_details'].append(f"❌ {test_name}: FAILED")
                print(f"❌ {test_name}: FAILED")
                return False
        except Exception as e:
            test_results['failed_tests'] += 1
            test_results['test_details'].append(f"❌ {test_name}: ERROR - {str(e)}")
            print(f"❌ {test_name}: ERROR - {str(e)}")
            return False
    
    # Test 1: Investment Engine Module
    print("\n📈 Testing Investment Engine...")
    def test_investment_engine():
        try:
            from investment_engine import InvestmentRecommendationEngine
            engine = InvestmentRecommendationEngine()
            
            sample_profile = {
                'user_id': 'test_user',
                'age': 30,
                'income_monthly': 50000,
                'risk_tolerance': 'moderate',
                'investment_goals': ['wealth_building'],
                'current_savings': 100000
            }
            
            recommendations = engine.generate_recommendations(
                user_profile=sample_profile,
                current_spending=30000,
                available_amount=20000
            )
            
            return 'recommendations' in recommendations and len(recommendations['recommendations']) > 0
        except Exception as e:
            logger.error(f"Investment engine test failed: {e}")
            return False
    
    run_test("Investment Engine Initialization", test_investment_engine)
    
    # Test 2: Behavioral Analyzer Module
    print("\n🧠 Testing Behavioral Analyzer...")
    def test_behavioral_analyzer():
        try:
            from behavioral_analyzer import BehavioralFinanceAnalyzer
            analyzer = BehavioralFinanceAnalyzer()
            
            # Create sample transaction data
            sample_data = pd.DataFrame([
                {'amount': 1500, 'category': 'FOOD_DINING', 'timestamp': datetime.now()},
                {'amount': 5000, 'category': 'SHOPPING', 'timestamp': datetime.now() - timedelta(hours=2)},
                {'amount': 800, 'category': 'TRANSPORTATION', 'timestamp': datetime.now() - timedelta(days=1)},
            ])
            
            analysis = analyzer.analyze_behavior(sample_data)
            return 'emotional_spending' in analysis and 'financial_stress_indicators' in analysis
        except Exception as e:
            logger.error(f"Behavioral analyzer test failed: {e}")
            return False
    
    run_test("Behavioral Analyzer Initialization", test_behavioral_analyzer)
    
    # Test 3: Predictive Analytics Module
    print("\n🔮 Testing Predictive Analytics...")
    def test_predictive_analytics():
        try:
            from predictive_analytics import ExpensePredictionEngine
            engine = ExpensePredictionEngine()
            
            # Create sample historical data
            sample_data = []
            for i in range(30):
                sample_data.append({
                    'amount': 1000 + np.random.normal(0, 200),
                    'category': 'MISCELLANEOUS',
                    'timestamp': datetime.now() - timedelta(days=i)
                })
            
            df = pd.DataFrame(sample_data)
            training_scores = engine.train_models(df)
            
            return isinstance(training_scores, dict) and len(training_scores) > 0
        except Exception as e:
            logger.error(f"Predictive analytics test failed: {e}")
            return False
    
    run_test("Predictive Analytics Initialization", test_predictive_analytics)
    
    # Test 4: Complete System Integration
    print("\n🔗 Testing Complete System Integration...")
    def test_complete_integration():
        try:
            from smart_money_complete import SmartMoneyAI
            
            # Initialize system
            smart_money = SmartMoneyAI()
            
            # Create user profile
            user_data = {
                'user_id': 'integration_test_user',
                'age': 25,
                'income_monthly': 60000,
                'risk_tolerance': 'moderate',
                'current_savings': 150000
            }
            
            profile = smart_money.create_user_profile(user_data)
            
            # Add sample transactions
            transactions = [
                {'amount': 2000, 'category': 'GROCERIES', 'description': 'Weekly groceries'},
                {'amount': 1500, 'category': 'TRANSPORTATION', 'description': 'Fuel'},
                {'amount': 3000, 'category': 'ENTERTAINMENT', 'description': 'Movies and dinner'},
            ]
            
            for trans in transactions:
                smart_money.add_manual_transaction(trans, profile.user_id)
            
            # Generate insights
            insights = smart_money.generate_comprehensive_insights(profile.user_id)
            
            return (insights.transaction_summary['total_transactions'] == 3 and
                   insights.risk_assessment['risk_level'] in ['Low', 'Moderate', 'High'])
            
        except Exception as e:
            logger.error(f"Complete integration test failed: {e}")
            return False
    
    run_test("Complete System Integration", test_complete_integration)
    
    # Test 5: Data Processing Pipeline
    print("\n🔄 Testing Data Processing Pipeline...")
    def test_data_pipeline():
        try:
            # Test SMS parsing simulation
            sample_sms = "Dear Customer, Rs.2500 debited from your account at GROCERY STORE on 15-Oct-23."
            
            from smart_money_complete import SmartMoneyAI
            smart_money = SmartMoneyAI()
            
            # Create user
            user_data = {'user_id': 'pipeline_test_user', 'income_monthly': 50000}
            profile = smart_money.create_user_profile(user_data)
            
            # Process SMS
            transaction = smart_money.process_sms_transaction(sample_sms, profile.user_id)
            
            return (transaction is not None and 
                   transaction.amount > 0 and 
                   transaction.category is not None)
            
        except Exception as e:
            logger.error(f"Data pipeline test failed: {e}")
            return False
    
    run_test("Data Processing Pipeline", test_data_pipeline)
    
    # Test 6: Error Handling and Resilience
    print("\n🛡️ Testing Error Handling...")
    def test_error_handling():
        try:
            from smart_money_complete import SmartMoneyAI
            smart_money = SmartMoneyAI()
            
            # Test with invalid user data
            try:
                invalid_data = {'user_id': '', 'income_monthly': -1000}
                profile = smart_money.create_user_profile(invalid_data)
                # Should handle gracefully
            except Exception:
                pass  # Expected to fail
            
            # Test insights with no data
            user_data = {'user_id': 'error_test_user', 'income_monthly': 50000}
            profile = smart_money.create_user_profile(user_data)
            insights = smart_money.generate_comprehensive_insights(profile.user_id)
            
            return insights is not None  # Should return default insights
            
        except Exception as e:
            logger.error(f"Error handling test failed: {e}")
            return False
    
    run_test("Error Handling and Resilience", test_error_handling)
    
    # Test 7: Performance Test
    print("\n⚡ Testing Performance...")
    def test_performance():
        try:
            from smart_money_complete import SmartMoneyAI
            import time
            
            start_time = time.time()
            
            smart_money = SmartMoneyAI()
            user_data = {'user_id': 'perf_test_user', 'income_monthly': 50000}
            profile = smart_money.create_user_profile(user_data)
            
            # Add 50 transactions
            for i in range(50):
                trans_data = {
                    'amount': 1000 + i * 10,
                    'category': 'MISCELLANEOUS',
                    'description': f'Transaction {i}'
                }
                smart_money.add_manual_transaction(trans_data, profile.user_id)
            
            # Generate insights
            insights = smart_money.generate_comprehensive_insights(profile.user_id)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            print(f"  Processing time for 50 transactions: {processing_time:.2f} seconds")
            return processing_time < 30  # Should complete within 30 seconds
            
        except Exception as e:
            logger.error(f"Performance test failed: {e}")
            return False
    
    run_test("Performance Test", test_performance)
    
    # Test 8: System Status and Health Check
    print("\n🔧 Testing System Health...")
    def test_system_health():
        try:
            from smart_money_complete import SmartMoneyAI
            smart_money = SmartMoneyAI()
            
            status = smart_money.get_system_status()
            
            return ('timestamp' in status and 
                   'components' in status and 
                   'data_summary' in status)
            
        except Exception as e:
            logger.error(f"System health test failed: {e}")
            return False
    
    run_test("System Health Check", test_system_health)
    
    # Test Summary
    print(f"\n📋 Test Summary")
    print(f"=" * 50)
    print(f"Total Tests: {test_results['total_tests']}")
    print(f"Passed: {test_results['passed_tests']}")
    print(f"Failed: {test_results['failed_tests']}")
    print(f"Success Rate: {(test_results['passed_tests']/test_results['total_tests']*100):.1f}%")
    
    print(f"\n📝 Detailed Results:")
    for detail in test_results['test_details']:
        print(f"  {detail}")
    
    # Overall Assessment
    success_rate = test_results['passed_tests'] / test_results['total_tests'] * 100
    
    if success_rate >= 90:
        print(f"\n🎉 EXCELLENT: System integration is highly successful!")
        print(f"🚀 Smart Money AI is ready for production deployment!")
    elif success_rate >= 75:
        print(f"\n✅ GOOD: System integration is mostly successful!")
        print(f"🔧 Minor improvements needed for optimal performance.")
    elif success_rate >= 50:
        print(f"\n⚠️  FAIR: System integration has some issues.")
        print(f"🛠️  Significant improvements needed before deployment.")
    else:
        print(f"\n❌ POOR: System integration needs major fixes.")
        print(f"🚨 Critical issues must be resolved before deployment.")
    
    print(f"\n🏁 Integration test completed!")
    
    return test_results

if __name__ == "__main__":
    test_results = test_smart_money_integration()
    
    # Exit with appropriate code
    if test_results['failed_tests'] == 0:
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some tests failed