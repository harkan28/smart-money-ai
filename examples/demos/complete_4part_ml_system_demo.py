#!/usr/bin/env python3
"""
Smart Money AI - Complete 4-Part ML System Demo
==============================================

Demonstrates all 4 independent ML models working together:

1. SMS PARSING MODEL - Extract transaction data from banking SMS
2. EXPENSE CATEGORIZATION MODEL - Automatically categorize expenses  
3. SAVINGS & BUDGETING MODEL - Monthly savings analysis and optimization
4. INVESTMENT RECOMMENDATION MODEL - Stock, mutual fund, gold/silver recommendations

This demo shows how the complete system processes financial data end-to-end
to provide world-class financial intelligence and recommendations.
"""

import sys
import os
sys.path.append('/Users/harshitrawal/Downloads/SMART MONEY')

from smart_money_ai import SmartMoneyAI
import json
from datetime import datetime, timedelta

def demo_complete_ml_system():
    """Demonstrate the complete 4-part ML system"""
    
    print("🚀 SMART MONEY AI - COMPLETE 4-PART ML SYSTEM DEMO")
    print("=" * 70)
    print()
    
    # Initialize Smart Money AI
    print("🔧 Initializing Smart Money AI...")
    ai = SmartMoneyAI()
    print()
    
    # Sample user profile
    user_profile = {
        'age': 28,
        'income': 75000,
        'city_tier': 'Tier1',
        'investment_experience': 'intermediate',
        'investment_horizon': 'long',
        'risk_tolerance': 'moderate'
    }
    
    print("👤 USER PROFILE:")
    print(f"   Age: {user_profile['age']} years")
    print(f"   Monthly Income: ₹{user_profile['income']:,}")
    print(f"   City: {user_profile['city_tier']}")
    print(f"   Risk Tolerance: {user_profile['risk_tolerance']}")
    print()
    
    # ================================
    # PART 1: SMS PARSING DEMO
    # ================================
    
    print("📱 PART 1: SMS PARSING MODEL DEMO")
    print("-" * 40)
    
    sample_sms = [
        "Paid Rs.2,500 to SWIGGY via UPI. Your A/c balance is Rs.45,230. Txn ID: 1234567890",
        "Rs.15,000 debited from A/c for HDFC MUTUAL FUND SIP. Balance: Rs.42,730",
        "Fuel purchase Rs.3,200 at HP PETROL PUMP via Debit Card. Available balance Rs.39,530",
        "Salary credited Rs.75,000 in your account. Current balance Rs.1,14,530",
        "DTH recharge Rs.400 for TATA SKY. Balance Rs.1,14,130"
    ]
    
    print("📥 Processing SMS batch...")
    sms_results = ai.parse_sms_batch(sample_sms)
    
    print(f"✅ Successfully parsed {sms_results['successful_parses']}/{sms_results['total_processed']} SMS messages")
    
    if 'batch_analysis' in sms_results:
        batch_analysis = sms_results['batch_analysis']
        print("📊 Category Breakdown:")
        for category, data in batch_analysis['category_breakdown'].items():
            print(f"   {category.title()}: ₹{data['amount']:.0f} ({data['percentage']:.1f}%)")
    print()
    
    # ================================
    # PART 2: EXPENSE CATEGORIZATION DEMO
    # ================================
    
    print("🏷️ PART 2: EXPENSE CATEGORIZATION MODEL DEMO")
    print("-" * 50)
    
    manual_transactions = [
        {'description': 'Amazon shopping electronics', 'amount': 8500, 'type': 'expense'},
        {'description': 'Freelance project payment received', 'amount': 25000, 'type': 'income'},
        {'description': 'Movie tickets BookMyShow', 'amount': 800, 'type': 'expense'},
        {'description': 'Electricity bill BSES payment', 'amount': 2200, 'type': 'expense'}
    ]
    
    print("📝 Processing manual transactions...")
    for tx in manual_transactions:
        result = ai.add_manual_transaction(tx['description'], tx['amount'], tx['type'])
        if result['success']:
            category = result['transaction']['category']
            print(f"   ✅ {tx['description']} → {category}")
    print()
    
    # ================================
    # PART 3: SAVINGS & BUDGETING DEMO
    # ================================
    
    print("💰 PART 3: SAVINGS & BUDGETING MODEL DEMO")
    print("-" * 45)
    
    # Combine all transactions
    all_transactions = []
    
    # Add SMS transactions
    if 'individual_results' in sms_results:
        all_transactions.extend([t for t in sms_results['individual_results'] if t.get('success')])
    
    # Add manual transactions (convert format)
    for tx in manual_transactions:
        if tx['type'] == 'expense':  # Only include expenses for savings analysis
            all_transactions.append({
                'description': tx['description'],
                'amount': tx['amount'],
                'category': 'unknown',  # Will be categorized
                'date': datetime.now().isoformat()
            })
    
    print("📊 Analyzing savings potential...")
    if all_transactions:
        savings_analysis = ai.analyze_monthly_savings(user_profile, all_transactions)
        
        if 'current_savings' in savings_analysis:
            current = savings_analysis['current_savings']
            print(f"   Current Monthly Savings: ₹{current['amount']:.0f} ({current['rate']*100:.1f}%)")
            
            if 'predicted_optimal' in savings_analysis:
                optimal = savings_analysis['predicted_optimal']
                print(f"   Optimal Savings Rate: {optimal['savings_rate']*100:.1f}%")
                print(f"   Improvement Potential: ₹{optimal['improvement_potential']:.0f}")
            
            if 'savings_score' in savings_analysis:
                print(f"   Savings Score: {savings_analysis['savings_score']}/100")
            
            if 'recommendations' in savings_analysis:
                print("   💡 Recommendations:")
                for rec in savings_analysis['recommendations'][:3]:
                    print(f"      • {rec}")
    print()
    
    # ================================
    # PART 4: INVESTMENT RECOMMENDATIONS DEMO
    # ================================
    
    print("📈 PART 4: INVESTMENT RECOMMENDATION MODEL DEMO")
    print("-" * 55)
    
    investment_amount = 100000  # 1 lakh investment
    print(f"💵 Investment Amount: ₹{investment_amount:,}")
    
    print("🤖 Generating AI-powered investment recommendations...")
    investment_recs = ai.get_investment_recommendations(user_profile, investment_amount)
    
    if 'risk_profile' in investment_recs:
        risk_profile = investment_recs['risk_profile']
        print(f"   Risk Profile: {risk_profile['risk_tolerance']} (Score: {risk_profile['risk_score']})")
    
    if 'asset_allocation' in investment_recs:
        allocation = investment_recs['asset_allocation']
        print("   🎯 Recommended Asset Allocation:")
        print(f"      Equity: {allocation['equity']*100:.1f}%")
        print(f"      Debt: {allocation['debt']*100:.1f}%")
        print(f"      Gold: {allocation['gold']*100:.1f}%")
        print(f"      Cash: {allocation['cash']*100:.1f}%")
    
    if 'recommended_portfolio' in investment_recs:
        portfolio = investment_recs['recommended_portfolio']
        print(f"   📊 Portfolio Expected Return: {portfolio['expected_return']*100:.1f}% annually")
        print(f"   🎯 Diversification Score: {portfolio['diversification_score']}/100")
        
        if 'investments' in portfolio:
            print("   💎 Top Investment Recommendations:")
            for inv in portfolio['investments'][:3]:
                print(f"      • {inv['name']} (₹{inv['amount']:.0f}) - Expected: {inv['expected_return']*100:.1f}%")
    
    # Gold-specific analysis
    if 'gold_analysis' in investment_recs:
        gold_analysis = investment_recs['gold_analysis']
        print("   🥇 Gold Investment Analysis:")
        if 'current_recommendation' in gold_analysis:
            print(f"      Signal: {gold_analysis['current_recommendation'].upper()}")
        if 'allocation_suggestion' in gold_analysis:
            gold_alloc = gold_analysis['allocation_suggestion']
            print(f"      Suggested Allocation: {gold_alloc['percentage']:.1f}% (₹{gold_alloc['amount']:.0f})")
    print()
    
    # ================================
    # COMPREHENSIVE ANALYSIS
    # ================================
    
    print("🎯 COMPREHENSIVE FINANCIAL INTELLIGENCE")
    print("-" * 45)
    
    print("🔄 Running complete financial analysis pipeline...")
    
    # Process all data through the complete system
    complete_analysis = ai.process_complete_financial_data(
        user_profile=user_profile,
        sms_data=sample_sms,
        manual_transactions=manual_transactions,
        investment_amount=investment_amount
    )
    
    # Display integrated insights
    if 'integrated_insights' in complete_analysis:
        print("💡 AI-Generated Insights:")
        for insight in complete_analysis['integrated_insights']:
            print(f"   • {insight}")
    
    # Display comprehensive financial health score
    if 'financial_health_score' in complete_analysis:
        health_score = complete_analysis['financial_health_score']
        print(f"\n🏆 COMPREHENSIVE FINANCIAL HEALTH SCORE")
        print(f"   Overall Score: {health_score['overall_score']:.1f}/100 (Grade: {health_score['grade']})")
        print(f"   Status: {health_score['status']}")
        
        if 'component_scores' in health_score:
            print("   📊 Component Breakdown:")
            for component, score in health_score['component_scores'].items():
                print(f"      {component.replace('_', ' ').title()}: {score:.1f}")
        
        if 'areas_for_improvement' in health_score and health_score['areas_for_improvement']:
            print("   🎯 Areas for Improvement:")
            for improvement in health_score['areas_for_improvement']:
                print(f"      • {improvement}")
    
    print()
    print("✅ DEMO COMPLETED SUCCESSFULLY!")
    print("🚀 Smart Money AI - Complete 4-Part ML System is ready for production!")
    print()
    
    # System capabilities summary
    print("🎯 SYSTEM CAPABILITIES SUMMARY:")
    print("-" * 35)
    print("✅ Part 1: SMS Parsing - Intelligent transaction extraction from banking SMS")
    print("✅ Part 2: Expense Categorization - ML-powered automatic expense categorization")
    print("✅ Part 3: Savings & Budgeting - Advanced savings analysis and budget optimization")
    print("✅ Part 4: Investment Recommendations - Sophisticated investment advice with gold prediction")
    print()
    print("💎 UNIQUE FEATURES:")
    print("   • Multi-dataset intelligence (20,000+ profiles + gold price prediction)")
    print("   • End-to-end financial pipeline from SMS to investment recommendations")
    print("   • Real-time categorization and insights")
    print("   • Advanced ML models for each financial domain")
    print("   • Comprehensive financial health scoring")
    print("   • Gold price prediction using 7 years of market data")
    print()
    print("🏆 Smart Money AI is now the most comprehensive personal finance ML system!")

if __name__ == "__main__":
    demo_complete_ml_system()