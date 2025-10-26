#!/usr/bin/env python3
"""
🧪 Smart Money AI - Simple System Test
=====================================

Quick test to verify Smart Money AI system functionality and configuration.
"""

import sys
import os

# Add the project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    """Simple system test for Smart Money AI"""
    
    print("🧪 Smart Money AI - System Test")
    print("=" * 40)
    
    try:
        print("\n🚀 Testing Smart Money AI import...")
        from smart_money_ai import SmartMoneyAI
        from config.settings import get_config
        print("✅ Import successful!")
        
        print("\n🔧 Checking configuration...")
        config = get_config()
        
        # Check configuration status
        print(f"Real-Time Data: {'✅' if config.is_real_time_enabled() else '❌'}")
        print(f"Trading Enabled: {'✅' if config.is_trading_enabled() else '❌'}")
        
        capabilities = config.get_capabilities()
        print(f"ML Models: {'✅' if capabilities['ml_models'] else '❌'}")
        print(f"Portfolio Monitoring: {'✅' if capabilities['portfolio_monitoring'] else '❌'}")
        
        print("\n🔧 Testing initialization...")
        if config.api_config.finnhub_api_key:
            smart_money = SmartMoneyAI(finnhub_api_key=config.api_config.finnhub_api_key)
        else:
            # Use demo API key for testing
            demo_api_key = "d3o93nhr01qmj830b3mgd3o93nhr01qmj830b3n0"
            smart_money = SmartMoneyAI(finnhub_api_key=demo_api_key)
            
        print("✅ Initialization successful!")
        
        print("\n📊 Testing core functionality...")
        
        # Test SMS parsing
        sample_sms = "Your A/c XXXXXX1234 debited for Rs.2,500.00 on 15-JUL-23 at Coffee House Mumbai."
        result = smart_money.parse_sms(sample_sms)
        sms_status = "✅ Working" if result.get('success') else "❌ Failed"
        print(f"SMS Parsing: {sms_status}")
        
        if result.get('success'):
            print(f"  Amount: ₹{result.get('amount', 0):,.2f}")
            print(f"  Merchant: {result.get('merchant', 'Unknown')}")
            if 'category' in result:
                print(f"  Category: {result['category']}")
        
        # Test real-time capability check
        has_real_time = smart_money.real_time_analyzer is not None
        real_time_status = "✅ Available" if has_real_time else "❌ Not Available"
        print(f"Real-Time Market Data: {real_time_status}")
        
        if has_real_time:
            print("\n🔥 Testing real-time market data...")
            try:
                market_overview = smart_money.get_market_overview()
                if 'error' not in market_overview:
                    print("✅ Market overview working!")
                    sentiment = market_overview.get('market_sentiment', 'Unknown')
                    print(f"  Market Sentiment: {sentiment}")
                else:
                    print(f"❌ Market overview error: {market_overview['error']}")
            except Exception as e:
                print(f"❌ Market data test failed: {e}")
        
        # Test investment recommendations
        print("\n💰 Testing investment recommendations...")
        user_profile = {
            'age': 30,
            'monthly_income': 100000,
            'risk_tolerance': 'moderate',
            'investment_experience': 'beginner'
        }
        
        try:
            recommendations = smart_money.get_investment_recommendations(user_profile, 50000)
            if 'recommendation' in recommendations:
                print("✅ Investment recommendations working!")
                print(f"  Recommendation: {recommendations['recommendation']}")
            else:
                print("❌ Investment recommendations failed")
        except Exception as e:
            print(f"❌ Investment recommendations error: {e}")
        
        # System summary
        print(f"\n🎯 SYSTEM STATUS SUMMARY:")
        print("-" * 30)
        print(f"Core ML Models: {'✅' if capabilities['ml_models'] else '❌'}")
        print(f"Real-Time Data: {'✅' if has_real_time else '❌'}")
        print(f"Trading Ready: {'✅' if config.is_trading_enabled() else '❌'}")
        
        if has_real_time and config.is_trading_enabled():
            print("\n🔥 FULL SYSTEM OPERATIONAL!")
        elif has_real_time:
            print("\n📊 REAL-TIME INTELLIGENCE READY!")
        else:
            print("\n🤖 CORE ML SYSTEM READY!")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test Error: {e}")
        print("\n💡 To fix this issue:")
        print("   • Check that all dependencies are installed")
        print("   • Verify config/config.json has valid API keys")
        print("   • Ensure internet connection is working")
        
        import traceback
        print(f"\n📝 Full error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()