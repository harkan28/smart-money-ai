#!/usr/bin/env python3
"""
Smart Money AI - Clean System Demo
Demonstrate the reorganized, unified system capabilities
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from smart_money_ai import SmartMoneyAI
import json

def demo_clean_unified_system():
    """Demonstrate the clean, unified Smart Money AI system"""
    
    print("🎯 SMART MONEY AI - CLEAN UNIFIED SYSTEM DEMO")
    print("=" * 70)
    print("🏗️  NOW FEATURING: Clean Architecture + Dual Dataset Intelligence")
    print("📁 Organized Structure | 🧠 Unified Interface | 🗄️ Optimized Data")
    print()
    
    # Initialize the unified system
    print("🚀 Initializing Smart Money AI...")
    ai = SmartMoneyAI()
    print("✅ System ready with clean, organized architecture!")
    print()
    
    # Demo 1: SMS Parsing
    print("📱 DEMO 1: SMS TRANSACTION PARSING")
    print("-" * 40)
    
    sample_sms = "Your account has been debited with Rs 2,500 at BigBasket on 15-Oct-2025. Available balance: Rs 45,320"
    
    parsed_transaction = ai.parse_sms(sample_sms)
    
    if parsed_transaction['status'] == 'success':
        print(f"✅ SMS Parsed Successfully!")
        print(f"   💰 Amount: ₹{parsed_transaction['amount']:,.2f}")
        print(f"   🏪 Merchant: {parsed_transaction['merchant']}")
        print(f"   📊 Category: {parsed_transaction['category'].title()}")
        print(f"   🔄 Type: {parsed_transaction['type'].title()}")
    
    print()
    
    # Demo 2: Spending Analysis with Demographic Intelligence
    print("📊 DEMO 2: INTELLIGENT SPENDING ANALYSIS")
    print("-" * 45)
    
    user_profile = {
        'age': 29,
        'income': 85000,
        'city_tier': 'Tier_1',
        'dependents': 0
    }
    
    monthly_expenses = {
        'groceries': 6000,
        'transport': 4500,
        'eating_out': 4000,
        'entertainment': 2500,
        'utilities': 3000
    }
    
    print(f"👤 User: {user_profile['age']} years, ₹{user_profile['income']:,}/month, {user_profile['city_tier']}")
    
    spending_analysis = ai.analyze_spending(user_profile, monthly_expenses)
    
    if spending_analysis['status'] == 'success':
        print(f"🎯 Demographic Segment: {spending_analysis['segment']}")
        print(f"📈 Overall Assessment: {spending_analysis['overall_assessment']}")
        
        # Show spending insights
        overspend_categories = []
        for category, comparison in spending_analysis['comparisons'].items():
            if comparison['difference_percentage'] > 15:
                overspend_categories.append({
                    'category': category,
                    'excess': comparison['difference_percentage']
                })
        
        if overspend_categories:
            print(f"⚠️  Areas to Optimize:")
            for cat in overspend_categories[:3]:
                print(f"   • {cat['category'].title()}: +{cat['excess']:.1f}% above peers")
        else:
            print(f"✅ Excellent spending control across all categories!")
    
    print()
    
    # Demo 3: Investment Intelligence with Behavioral Profiling
    print("💎 DEMO 3: BEHAVIORAL INVESTMENT INTELLIGENCE")
    print("-" * 45)
    
    investment_profile = {
        **user_profile,
        'monthly_investment': 7000,
        'goal': 'wealth_generation',
        'duration_years': 18
    }
    
    investment_recommendations = ai.get_investment_recommendations(investment_profile)
    
    print(f"🎯 Risk Profile: {investment_recommendations['risk_profile'].title()}")
    print(f"💰 Investment Category: {investment_recommendations['investment_category'].replace('_', ' ').title()}")
    print(f"📈 Expected Returns: {investment_recommendations['expected_returns']}")
    print(f"💎 Projected Wealth: ₹{investment_recommendations['projected_wealth']:,.0f} in {investment_profile['duration_years']} years")
    
    # Show portfolio allocation
    print(f"📊 Recommended Portfolio:")
    for asset, percentage in investment_recommendations['portfolio_allocation'].items():
        print(f"   • {asset.title()}: {percentage}%")
    
    print()
    
    # Demo 4: Smart Budget Creation
    print("💰 DEMO 4: AI-POWERED SMART BUDGETING")
    print("-" * 40)
    
    smart_budget = ai.create_smart_budget(user_profile, [])
    
    if smart_budget['status'] == 'success':
        budget_data = smart_budget['budget']
        metadata = smart_budget['metadata']
        
        print(f"📋 Smart Budget for ₹{metadata['total_income']:,}/month income:")
        print(f"💡 Total Allocated: ₹{metadata['total_allocated']:,.0f} ({metadata['allocation_percentage']:.1f}%)")
        
        # Show top budget categories
        sorted_budget = sorted(budget_data.items(), key=lambda x: x[1]['amount'], reverse=True)
        print(f"🏆 Top Budget Allocations:")
        for category, details in sorted_budget[:5]:
            print(f"   • {category.title()}: ₹{details['amount']:,.0f} ({details['percentage']:.1f}%) - {details['priority']} priority")
        
        # Show recommendations
        recommendations = smart_budget['recommendations']
        if recommendations:
            print(f"💡 Smart Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                print(f"   {i}. {rec}")
    
    print()
    
    # Demo 5: Comprehensive Financial Health Score
    print("🏥 DEMO 5: COMPREHENSIVE FINANCIAL HEALTH ASSESSMENT")
    print("-" * 55)
    
    investment_goals = {
        'monthly_investment': 7000,
        'goal': 'wealth_generation',
        'duration_years': 18
    }
    
    health_assessment = ai.get_financial_health_score(user_profile, monthly_expenses, investment_goals)
    
    print(f"📊 FINANCIAL HEALTH SCORE: {health_assessment['score']}/100 ({health_assessment['grade']})")
    print(f"📈 Assessment: {health_assessment['summary']}")
    
    print(f"💡 Key Strengths:")
    for i, factor in enumerate(health_assessment['factors'][:4], 1):
        print(f"   {i}. {factor}")
    
    print()
    
    # Demo 6: System Statistics
    print("📈 DEMO 6: SYSTEM INTELLIGENCE STATISTICS")
    print("-" * 45)
    
    system_stats = ai.get_system_stats()
    
    if 'error' not in system_stats:
        print(f"🗄️  DATASET INTEGRATION:")
        print(f"   • Personal Finance Records: {system_stats['personal_finance_records']:,}")
        print(f"   • Spending Segments: {system_stats['spending_segments']}")
        print(f"   • Investment Profiles: {system_stats['behavioral_profiles']}")
        print(f"   • Total Intelligence: {system_stats['total_intelligence_profiles']:,} profiles")
        
        print(f"\n🎯 SYSTEM CAPABILITIES:")
        for capability in system_stats['system_capabilities'][:4]:
            print(f"   ✅ {capability}")
    else:
        print(f"📊 Estimated Intelligence: {system_stats['estimated_profiles']} profiles")
        print(f"🎯 Status: {system_stats['system_status']}")
    
    print()
    
    # Show clean architecture benefits
    print("🏗️  CLEAN ARCHITECTURE BENEFITS")
    print("=" * 45)
    print("✅ ORGANIZED STRUCTURE: Core, Intelligence, Data, API separated")
    print("✅ UNIFIED INTERFACE: Single SmartMoneyAI class for all features")
    print("✅ OPTIMIZED DATA: Consolidated datasets with efficient access")
    print("✅ CLEAN IMPORTS: Proper module hierarchy and dependencies")
    print("✅ PRODUCTION READY: Scalable, maintainable codebase")
    print("✅ DEVELOPER FRIENDLY: Easy to extend and modify")
    
    print()
    print("🎉 SYSTEM TRANSFORMATION COMPLETE!")
    print("=" * 50)
    print("🏆 Smart Money AI is now a clean, organized, world-class financial intelligence system!")
    print("📁 All clutter removed, datasets merged, code organized")
    print("🚀 Ready for production deployment with enterprise-grade architecture!")

if __name__ == "__main__":
    demo_clean_unified_system()