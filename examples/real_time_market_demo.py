#!/usr/bin/env python3
"""
🔥 Smart Money AI - Real-Time Market Intelligence Demo
=====================================================

Complete demonstration of the enhanced 4-Part ML System with live market data
integration using Finnhub API for real-time investment recommendations.

Features Demonstrated:
- Real-time stock analysis with ML predictions
- Live market overview and sentiment analysis
- Enhanced investment recommendations with current market data
- Portfolio monitoring with real-time alerts
- Integration with existing Smart Money AI models

Usage:
    python real_time_market_demo.py
"""

import sys
import os
from datetime import datetime
import json

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the enhanced Smart Money AI and configuration
from smart_money_ai import SmartMoneyAI
from config.settings import get_config, setup_config

def main():
    """Comprehensive demo of Smart Money AI with real-time market intelligence"""
    
    print("🚀 Smart Money AI - Real-Time Market Intelligence Demo")
    print("=" * 70)
    
    # Get configuration
    config = get_config()
    
    # Check if API key is configured
    if not config.api_config.finnhub_api_key:
        print("⚠️  Finnhub API key not configured!")
        print("Please set your API key in config/config.json or environment variable FINNHUB_API_KEY")
        
        # Use demo API key for testing
        API_KEY = "d3o93nhr01qmj830b3mgd3o93nhr01qmj830b3n0"
        setup_config(finnhub_api_key=API_KEY)
        config = get_config()
        print("🔧 Using demo API key for testing...")
    
    try:
        # Initialize the enhanced system
        smart_money = SmartMoneyAI(finnhub_api_key=config.api_config.finnhub_api_key)
        
        print("\n📊 SYSTEM STATUS:")
        print("-" * 30)
        
        # Check what's available
        has_real_time = smart_money.real_time_analyzer is not None
        print(f"Real-Time Market Data: {'✅ ACTIVE' if has_real_time else '❌ Not Available'}")
        print(f"ML Investment Model: {'✅ Available' if smart_money.investment_ml_model else '❌ Not Available'}")
        print(f"Expense Categorizer: {'✅ Available' if smart_money.expense_categorizer else '❌ Not Available'}")
        print(f"Savings Model: {'✅ Available' if smart_money.savings_model else '❌ Not Available'}")
        
        # Sample user profile for testing
        user_profile = {
            'age': 28,
            'monthly_income': 150000,
            'location': 'Mumbai',
            'risk_tolerance': 'moderate',
            'investment_experience': 'intermediate',
            'financial_goals': ['wealth_building', 'retirement_planning'],
            'time_horizon': '15+ years'
        }
        
        print(f"\n👤 USER PROFILE:")
        print("-" * 30)
        print(f"Age: {user_profile['age']}")
        print(f"Monthly Income: ₹{user_profile['monthly_income']:,}")
        print(f"Risk Tolerance: {user_profile['risk_tolerance']}")
        print(f"Investment Experience: {user_profile['investment_experience']}")
        
        # Demo 1: Market Overview
        if has_real_time:
            print(f"\n📈 LIVE MARKET OVERVIEW:")
            print("-" * 30)
            
            market_overview = smart_money.get_market_overview()
            
            if 'error' not in market_overview:
                print(f"Market Sentiment: {market_overview.get('market_sentiment', 'Unknown')}")
                
                # Show major indices
                indices = market_overview.get('indices', {})
                if indices:
                    print("\nMajor Indices:")
                    for symbol, data in list(indices.items())[:4]:  # Show top 4
                        if data and 'current_price' in data:
                            price = data['current_price']
                            change = data.get('change_percent', 0)
                            print(f"  {symbol}: ${price:.2f} ({change:+.2f}%)")
                
                # Show top news
                top_news = market_overview.get('top_news', [])
                if top_news:
                    print(f"\nTop Market News:")
                    for i, news in enumerate(top_news[:2], 1):
                        headline = news.get('headline', 'No headline')[:60] + "..."
                        print(f"  {i}. {headline}")
            else:
                print(f"Error: {market_overview['error']}")
        
        # Demo 2: Real-Time Stock Analysis
        demo_stocks = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
        
        if has_real_time:
            print(f"\n🎯 REAL-TIME STOCK ANALYSIS:")
            print("-" * 30)
            
            for symbol in demo_stocks[:2]:  # Analyze first 2 stocks
                print(f"\n📊 {symbol} Analysis:")
                
                analysis = smart_money.get_real_time_stock_analysis(symbol)
                
                if 'error' not in analysis:
                    # Quote data
                    quote = analysis.get('real_time_data', {}).get('quote', {})
                    if quote:
                        price = quote.get('current_price', 0)
                        change = quote.get('change_percent', 0)
                        print(f"  Current Price: ${price:.2f} ({change:+.2f}%)")
                    
                    # Investment recommendation
                    recommendation = analysis.get('investment_recommendation', {})
                    if recommendation:
                        action = recommendation.get('action', 'Hold')
                        confidence = recommendation.get('confidence', 0)
                        target = recommendation.get('target_price', 0)
                        
                        print(f"  Recommendation: {action}")
                        print(f"  Confidence: {confidence:.0f}%")
                        print(f"  Target Price: ${target:.2f}")
                        
                        # Show top reasons
                        reasons = recommendation.get('reasons', [])
                        if reasons:
                            print(f"  Key Reason: {reasons[0]}")
                else:
                    print(f"  Error: {analysis['error']}")
        
        # Demo 3: Enhanced Investment Recommendations
        print(f"\n💎 ENHANCED INVESTMENT RECOMMENDATIONS:")
        print("-" * 30)
        
        investment_amount = 500000  # 5 Lakh investment
        
        if has_real_time:
            # Get enhanced recommendations with real-time data
            enhanced_recs = smart_money.get_real_time_investment_recommendations(
                user_profile, 
                investment_amount,
                symbols_to_analyze=['AAPL', 'MSFT', 'TSLA']
            )
            
            if 'error' not in enhanced_recs:
                print(f"Investment Amount: ₹{investment_amount:,}")
                
                # Show market conditions
                market_conditions = enhanced_recs.get('market_conditions', {})
                if market_conditions:
                    market_sentiment = market_conditions.get('market_sentiment', 'Unknown')
                    print(f"Market Conditions: {market_sentiment}")
                
                # Show enhanced recommendations
                enhanced = enhanced_recs.get('enhanced_recommendations', {})
                if enhanced:
                    action = enhanced.get('action', 'Hold')
                    confidence = enhanced.get('confidence', 0)
                    timing = enhanced.get('market_timing', 'Neutral')
                    risk = enhanced.get('risk_assessment', 'Medium')
                    
                    print(f"Enhanced Action: {action}")
                    print(f"Confidence: {confidence:.0f}%")
                    print(f"Market Timing: {timing}")
                    print(f"Risk Assessment: {risk}")
                    
                    # Show specific recommendations
                    recs = enhanced.get('recommendations', [])
                    if recs:
                        print(f"\nSpecific Recommendations:")
                        for i, rec in enumerate(recs[:3], 1):
                            print(f"  {i}. {rec}")
            else:
                print(f"Error: {enhanced_recs['error']}")
        else:
            # Use standard ML recommendations
            standard_recs = smart_money.get_investment_recommendations(user_profile, investment_amount)
            
            if 'recommendation' in standard_recs:
                print(f"Investment Amount: ₹{investment_amount:,}")
                print(f"ML Recommendation: {standard_recs['recommendation']}")
                
                portfolio = standard_recs.get('recommended_portfolio', {})
                if portfolio:
                    print(f"Risk Profile: {portfolio.get('risk_tolerance', 'Unknown')}")
                    
                    allocations = portfolio.get('allocations', {})
                    if allocations:
                        print(f"\nRecommended Allocation:")
                        for category, percentage in allocations.items():
                            print(f"  {category}: {percentage:.1f}%")
        
        # Demo 4: Portfolio Monitoring
        if has_real_time:
            print(f"\n📊 PORTFOLIO MONITORING:")
            print("-" * 30)
            
            portfolio_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
            portfolio_analysis = smart_money.monitor_investment_portfolio(portfolio_symbols)
            
            if 'error' not in portfolio_analysis:
                summary = portfolio_analysis.get('summary', {})
                if summary:
                    avg_change = summary.get('average_change', 0)
                    positive_performers = summary.get('positive_performers', 0)
                    total_symbols = len(portfolio_symbols)
                    sentiment = summary.get('overall_sentiment', 'Neutral')
                    
                    print(f"Portfolio Performance:")
                    print(f"  Average Change: {avg_change:+.2f}%")
                    print(f"  Positive Performers: {positive_performers}/{total_symbols}")
                    print(f"  Overall Sentiment: {sentiment}")
                
                # Show individual performances
                performance = portfolio_analysis.get('portfolio_performance', {})
                if performance:
                    print(f"\nTop Performers:")
                    # Sort by change percentage
                    sorted_performance = sorted(
                        performance.items(), 
                        key=lambda x: x[1].get('change_percent', 0), 
                        reverse=True
                    )
                    
                    for symbol, data in sorted_performance[:3]:
                        price = data.get('price', 0)
                        change = data.get('change_percent', 0)
                        status = data.get('status', 'Unknown')
                        print(f"  {symbol}: ${price:.2f} ({change:+.2f}%) - {status}")
                
                # Show alerts
                alerts = portfolio_analysis.get('alerts', [])
                if alerts:
                    print(f"\n🚨 Portfolio Alerts:")
                    for alert in alerts[:3]:
                        print(f"  • {alert.get('message', 'No message')}")
            else:
                print(f"Error: {portfolio_analysis['error']}")
        
        # Demo 5: SMS Processing (existing functionality)
        print(f"\n📱 SMS TRANSACTION PROCESSING:")
        print("-" * 30)
        
        sample_sms = "Your A/c XXXXXX1234 debited for Rs.2,500.00 on 15-JUL-23 at Coffee House Mumbai. Available balance: Rs.45,230.50"
        
        sms_result = smart_money.parse_sms(sample_sms)
        
        if sms_result.get('success'):
            print(f"Transaction Amount: ₹{sms_result.get('amount', 0):,.2f}")
            print(f"Transaction Type: {sms_result.get('transaction_type', 'Unknown')}")
            print(f"Merchant: {sms_result.get('merchant', 'Unknown')}")
            print(f"Location: {sms_result.get('location', 'Unknown')}")
            
            if 'category' in sms_result:
                print(f"Auto-Category: {sms_result['category']}")
                print(f"Confidence: {sms_result.get('category_confidence', 0)*100:.1f}%")
        else:
            print(f"SMS parsing failed: {sms_result.get('error', 'Unknown error')}")
        
        # Demo 6: System Integration Summary
        print(f"\n🎯 SYSTEM INTEGRATION SUMMARY:")
        print("-" * 30)
        
        capabilities = config.get_capabilities()
        status_list = [
            f"{'✅' if capabilities['ml_models'] else '❌'} SMS Transaction Parsing with Auto-Categorization",
            f"{'✅' if capabilities['ml_models'] else '❌'} Expense Analysis with ML Categorization",
            f"{'✅' if capabilities['ml_models'] else '❌'} Smart Budget Creation with Savings Optimization",
            f"{'✅' if capabilities['ml_models'] else '❌'} Advanced Investment Recommendations with Risk Profiling",
            f"{'✅' if capabilities['real_time_data'] else '❌'} Real-Time Market Data Integration",
            f"{'✅' if capabilities['portfolio_monitoring'] else '❌'} Live Investment Portfolio Monitoring",
            f"{'✅' if capabilities['real_time_data'] else '❌'} Market Sentiment Analysis",
            f"{'✅' if capabilities['real_time_data'] else '❌'} Enhanced Investment Timing Intelligence"
        ]
        
        for status in status_list:
            print(f"  {status}")
        
        system_status = "FULLY OPERATIONAL WITH LIVE DATA" if has_real_time else "OPERATIONAL (OFFLINE MODE)"
        print(f"\n🚀 SYSTEM STATUS: {system_status}")
        print(f"💡 Your Smart Money AI system is ready for comprehensive financial intelligence!")
        
        if has_real_time:
            print(f"\n🔥 REAL-TIME CAPABILITIES ACTIVE:")
            print(f"   • Live stock quotes and analysis")
            print(f"   • Real-time market sentiment")
            print(f"   • Enhanced ML predictions with current data")
            print(f"   • Portfolio monitoring with alerts")
            print(f"   • Market timing optimization")
        
        print(f"\n📊 Ready to provide world-class financial intelligence!")
        
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        print(f"💡 Make sure you have:")
        print(f"   • Valid Finnhub API key in config/config.json")
        print(f"   • Internet connection")
        print(f"   • All required dependencies installed")
        print(f"\nTry running: pip install requests pandas numpy")

if __name__ == "__main__":
    main()