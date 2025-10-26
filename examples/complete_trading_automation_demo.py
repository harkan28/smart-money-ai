#!/usr/bin/env python3
"""
🔥 Smart Money AI - Complete Trading Automation Demo
===================================================

Advanced demonstration of Smart Money AI with:
1. Real-time market intelligence (Finnhub API)
2. Automated trading capabilities (Breeze API)
3. AI-powered portfolio management
4. Risk management and monitoring
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
    """Comprehensive demo of Smart Money AI with trading automation"""
    
    print("🚀 Smart Money AI - Complete Trading Automation Demo")
    print("=" * 70)
    
    # Get configuration
    config = get_config()
    
    # Check configuration status
    print("\n🔧 CONFIGURATION STATUS:")
    print("-" * 30)
    print(f"Real-Time Data: {'✅' if config.is_real_time_enabled() else '❌'}")
    print(f"Automated Trading: {'✅' if config.is_trading_enabled() else '❌'}")
    
    # Setup demo keys if not configured
    if not config.api_config.finnhub_api_key:
        print("\n⚠️ Setting up demo configuration...")
        setup_config(
            finnhub_api_key="d3o93nhr01qmj830b3mgd3o93nhr01qmj830b3n0",
            breeze_app_key="demo_app_key",
            breeze_secret_key="demo_secret_key"
        )
        config = get_config()
    
    try:
        print("\n🔧 INITIALIZING SMART MONEY AI SYSTEM...")
        print("-" * 50)
        
        # Initialize Smart Money AI with all capabilities
        smart_money = SmartMoneyAI(
            finnhub_api_key=config.api_config.finnhub_api_key,
            breeze_app_key=config.api_config.breeze_app_key,
            breeze_secret_key=config.api_config.breeze_secret_key,
            breeze_session_token=config.api_config.breeze_session_token
        )
        
        print("\n📊 SYSTEM CAPABILITIES:")
        print("-" * 30)
        
        # Check available capabilities
        capabilities = config.get_capabilities()
        has_real_time = smart_money.real_time_analyzer is not None
        has_trading = smart_money.breeze_integration is not None
        has_ml_models = all([
            smart_money.expense_categorizer,
            smart_money.savings_model,
            smart_money.investment_ml_model
        ])
        
        print(f"Real-Time Market Data: {'✅ ACTIVE' if has_real_time else '❌ Not Available'}")
        print(f"Automated Trading: {'✅ ACTIVE' if has_trading else '❌ Not Available'}")
        print(f"ML Investment Models: {'✅ ACTIVE' if has_ml_models else '❌ Partial'}")
        print(f"Risk Management: {'✅ ACTIVE' if has_trading else '❌ Not Available'}")
        
        # Sample user profile for testing
        user_profile = {
            'age': 32,
            'monthly_income': 180000,
            'location': 'Mumbai',
            'risk_tolerance': 'moderate',
            'investment_experience': 'intermediate',
            'financial_goals': ['wealth_building', 'retirement_planning'],
            'time_horizon': '10+ years',
            'current_investments': 500000
        }
        
        print(f"\n👤 DEMO USER PROFILE:")
        print("-" * 30)
        print(f"Age: {user_profile['age']}")
        print(f"Monthly Income: ₹{user_profile['monthly_income']:,}")
        print(f"Risk Tolerance: {user_profile['risk_tolerance']}")
        print(f"Current Investments: ₹{user_profile['current_investments']:,}")
        
        # Demo 1: Market Intelligence
        if has_real_time:
            print(f"\n📈 LIVE MARKET INTELLIGENCE:")
            print("-" * 30)
            
            market_overview = smart_money.get_market_overview()
            
            if 'error' not in market_overview:
                print(f"Market Sentiment: {market_overview.get('market_sentiment', 'Unknown')}")
                
                # Show major indices
                indices = market_overview.get('indices', {})
                if indices:
                    print(f"\nMajor Indices Performance:")
                    for symbol, data in list(indices.items())[:3]:
                        if data and 'current_price' in data:
                            price = data['current_price']
                            change = data.get('change_percent', 0)
                            print(f"  {symbol}: ${price:.2f} ({change:+.2f}%)")
        
        # Demo 2: AI Investment Recommendations
        print(f"\n🤖 AI INVESTMENT RECOMMENDATIONS:")
        print("-" * 30)
        
        investment_amount = 100000  # 1 Lakh investment
        
        if has_real_time:
            # Get enhanced recommendations with real-time data
            recommendations = smart_money.get_real_time_investment_recommendations(
                user_profile, 
                investment_amount,
                symbols_to_analyze=['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ITC']
            )
            
            if 'error' not in recommendations:
                enhanced = recommendations.get('enhanced_recommendations', {})
                print(f"Investment Amount: ₹{investment_amount:,}")
                print(f"AI Recommendation: {enhanced.get('action', 'Hold')}")
                print(f"Confidence Level: {enhanced.get('confidence', 0):.0f}%")
                print(f"Risk Assessment: {enhanced.get('risk_assessment', 'Medium')}")
                
                # Show specific stock recommendations
                recs = enhanced.get('recommendations', [])
                if recs:
                    print(f"\nSpecific Recommendations:")
                    for i, rec in enumerate(recs[:3], 1):
                        print(f"  {i}. {rec}")
        
        # Demo 3: Trading Automation (Demo Mode)
        if has_trading:
            print(f"\n🔥 AUTOMATED TRADING DEMO:")
            print("-" * 30)
            
            print("⚠️ NOTE: This is demonstration mode - no actual trades will be executed")
            
            # Authentication demo
            print("\n1. Authentication:")
            if config.api_config.breeze_session_token:
                auth_result = smart_money.authenticate_trading_account(config.api_config.breeze_session_token)
                if 'error' not in auth_result:
                    print("   ✅ Trading account authenticated")
                    print(f"   User: {auth_result.get('customer_details', {}).get('idirect_user_name', 'Demo User')}")
                else:
                    print(f"   ❌ Authentication failed (Demo): {auth_result['error']}")
            else:
                print("   ⚠️ No session token configured - using demo mode")
            
            # Portfolio analysis demo
            print("\n2. Live Portfolio Analysis:")
            portfolio_analysis = smart_money.get_live_portfolio_analysis()
            if 'error' not in portfolio_analysis:
                print(f"   Portfolio Value: ₹{portfolio_analysis.get('total_portfolio_value', 0):,.2f}")
                print(f"   Diversification Score: {portfolio_analysis.get('diversification_score', 0)}/100")
                print(f"   AI Portfolio Rating: {portfolio_analysis.get('overall_portfolio_rating', 'Unknown')}")
            else:
                print(f"   ❌ Portfolio analysis failed (Demo): {portfolio_analysis['error']}")
            
        else:
            print(f"\n💡 TRADING AUTOMATION NOT AVAILABLE:")
            print("-" * 30)
            print("To enable automated trading, configure in config/config.json:")
            print("• Breeze API app_key")
            print("• Breeze API secret_key")
            print("• Valid session_token")
        
        # Demo 4: Traditional ML Features (Always Available)
        print(f"\n📱 CORE ML CAPABILITIES DEMO:")
        print("-" * 30)
        
        # SMS parsing demo
        sample_sms = "Your A/c XXXXXX5678 debited for Rs.15,500.00 on 16-OCT-25 at AMAZON INDIA Mumbai. Available balance: Rs.87,230.50"
        
        sms_result = smart_money.parse_sms(sample_sms)
        if sms_result.get('success'):
            print(f"SMS Parsing: ✅ Working")
            print(f"  Amount: ₹{sms_result.get('amount', 0):,.2f}")
            print(f"  Merchant: {sms_result.get('merchant', 'Unknown')}")
            if 'category' in sms_result:
                print(f"  Auto-Category: {sms_result['category']} ({sms_result.get('category_confidence', 0)*100:.0f}%)")
        else:
            print(f"SMS Parsing: ❌ {sms_result.get('error', 'Failed')}")
        
        # Show trading configuration
        print(f"\n⚙️ TRADING CONFIGURATION:")
        print("-" * 30)
        print(f"Max Position Size: {config.trading_config.max_position_size*100:.1f}%")
        print(f"Stop Loss: {config.trading_config.stop_loss_percentage*100:.1f}%")
        print(f"Take Profit: {config.trading_config.take_profit_percentage*100:.1f}%")
        print(f"Max Daily Trades: {config.trading_config.max_daily_trades}")
        print(f"Risk Tolerance: {config.trading_config.risk_tolerance}")
        
        # System status
        if has_real_time and has_trading:
            status = "COMPLETE AUTOMATION SYSTEM ACTIVE"
            icon = "🔥"
        elif has_real_time:
            status = "INTELLIGENCE SYSTEM ACTIVE"  
            icon = "📊"
        else:
            status = "CORE ML SYSTEM ACTIVE"
            icon = "🤖"
        
        print(f"\n{icon} SYSTEM STATUS: {status}")
        
        if has_real_time and has_trading:
            print(f"\n🚀 READY FOR COMPLETE AUTOMATED TRADING!")
            print(f"Your Smart Money AI can now:")
            print(f"  • Analyze markets in real-time")
            print(f"  • Generate AI-powered recommendations")
            print(f"  • Execute trades automatically")
            print(f"  • Monitor and optimize your portfolio")
            print(f"  • Provide comprehensive risk management")
        elif has_real_time:
            print(f"\n📈 READY FOR INTELLIGENT MARKET ANALYSIS!")
            print(f"Add Breeze API credentials for automated trading")
        else:
            print(f"\n💡 READY FOR SMART FINANCIAL MANAGEMENT!")
            print(f"Add API credentials for live market data and trading")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo Error: {e}")
        print(f"💡 Make sure you have:")
        print(f"   • Valid API keys configured in config/config.json")
        print(f"   • Internet connection")
        print(f"   • All required dependencies installed")

if __name__ == "__main__":
    main()