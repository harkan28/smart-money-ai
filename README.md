# Smart Money AI - Complete Financial Intelligence System

> **Enhanced 4-Part ML System with Real-Time Market Intelligence & Automated Trading**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/Version-4.0.0-orange)](docs/README.md)

## 🚀 Quick Start

```python
from smart_money_ai import SmartMoneyAI

# Initialize with your API keys
smart_money = SmartMoneyAI(
    finnhub_api_key="your_finnhub_key",
    breeze_app_key="your_breeze_app_key", 
    breeze_secret_key="your_breeze_secret_key"
)

# Parse SMS transactions
result = smart_money.parse_sms("Your A/c debited for Rs.2,500.00 at Coffee House Mumbai")

# Get investment recommendations with real-time data
recommendations = smart_money.get_real_time_investment_recommendations(
    user_profile={'age': 30, 'risk_tolerance': 'moderate'},
    investment_amount=100000
)

# Monitor your portfolio in real-time
portfolio_status = smart_money.monitor_investment_portfolio(['AAPL', 'MSFT', 'GOOGL'])
```

## 📋 System Capabilities

### ✅ Core ML Models (Always Available)
- **SMS Transaction Parser** - Extract amount, merchant, date from banking SMS
- **Expense Categorization** - Auto-categorize expenses with ML
- **Savings & Budget Optimizer** - Create optimized monthly budgets
- **Investment Recommendation Engine** - AI-powered investment advice

### 🔥 Real-Time Intelligence (with Finnhub API)
- **Live Market Data** - Real-time stock quotes and market sentiment
- **Enhanced Investment Analysis** - ML + market data recommendations
- **Portfolio Monitoring** - Live tracking with alerts
- **Market News Integration** - Latest financial news analysis

### 🎯 Automated Trading (with Breeze API)
- **Account Authentication** - Secure ICICI Direct integration
- **Live Portfolio Analysis** - Real-time holdings and performance
- **Risk Management** - Automated stop-loss and position sizing
- **Order Execution** - Automated buy/sell based on AI recommendations

## 🗂️ Project Structure

```
SMART MONEY/
├── 📁 smart_money_ai/         # Main AI system
│   ├── __init__.py           # Core SmartMoneyAI class (v4.0)
│   ├── core/                 # Core ML models
│   ├── intelligence/         # Advanced analytics
│   └── ml_models/           # Machine learning models
├── 📁 config/               # Configuration management
│   ├── settings.py          # Centralized config system
│   └── config.json          # API keys and settings
├── 📁 integrations/         # External API integrations
│   ├── finnhub_market_data.py   # Real-time market data
│   └── breeze_trading.py        # Automated trading
├── 📁 examples/             # Demo scripts and examples
│   ├── real_time_market_demo.py      # Market intelligence demo
│   ├── complete_trading_automation_demo.py  # Full trading demo
│   └── simple_system_test.py             # Basic system test
├── 📁 docs/                 # Documentation
│   ├── SMART_MONEY_AI_GUIDE.md           # Complete system guide
│   └── COMPLETE_TRADING_AUTOMATION_GUIDE.md  # Trading guide
├── 📁 utils/                # Utility functions
│   └── common_utils.py      # Shared helper functions
├── 📁 data/                 # Data storage
├── 📁 tests/                # Test files
└── README.md                # This file
```

## ⚙️ Configuration Setup

### 1. API Keys Configuration

Edit `config/config.json`:

```json
{
  "api": {
    "finnhub_api_key": "your_finnhub_api_key_here",
    "breeze_app_key": "your_breeze_app_key_here",
    "breeze_secret_key": "your_breeze_secret_key_here",
    "breeze_session_token": "your_session_token_here"
  },
  "trading": {
    "max_position_size": 0.05,
    "stop_loss_percentage": 0.02,
    "take_profit_percentage": 0.06,
    "risk_tolerance": "moderate"
  }
}
```

### 2. Environment Variables (Optional)

```bash
export FINNHUB_API_KEY="your_finnhub_key"
export BREEZE_APP_KEY="your_breeze_app_key"
export BREEZE_SECRET_KEY="your_breeze_secret_key"
```

## 🎮 Examples & Demos

### Run Real-Time Market Demo
```bash
python examples/real_time_market_demo.py
```

### Run Complete Trading Automation Demo
```bash
python examples/complete_trading_automation_demo.py
```

### Run Simple System Test
```bash
python examples/simple_system_test.py
```

## 📊 Usage Examples

### 1. SMS Transaction Processing
```python
# Parse banking SMS
sms_text = "Your A/c XXXXXX1234 debited for Rs.2,500.00 on 15-JUL-23 at Coffee House Mumbai"
result = smart_money.parse_sms(sms_text)

print(f"Amount: ₹{result['amount']}")
print(f"Merchant: {result['merchant']}")
print(f"Category: {result['category']}")
print(f"Confidence: {result['category_confidence']*100:.1f}%")
```

### 2. Investment Recommendations
```python
# Get AI-powered investment advice
user_profile = {
    'age': 32,
    'monthly_income': 150000,
    'risk_tolerance': 'moderate',
    'investment_experience': 'intermediate'
}

recommendations = smart_money.get_investment_recommendations(user_profile, 100000)
print(f"Recommendation: {recommendations['recommendation']}")
```

### 3. Real-Time Market Analysis
```python
# Get live market overview
market_data = smart_money.get_market_overview()
print(f"Market Sentiment: {market_data['market_sentiment']}")

# Analyze specific stocks
analysis = smart_money.get_real_time_stock_analysis('AAPL')
print(f"AAPL Recommendation: {analysis['investment_recommendation']['action']}")
```

### 4. Portfolio Monitoring
```python
# Monitor portfolio in real-time
portfolio = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
status = smart_money.monitor_investment_portfolio(portfolio)

print(f"Portfolio Performance: {status['summary']['average_change']:.2f}%")
print(f"Positive Performers: {status['summary']['positive_performers']}")
```

## 🔧 Installation & Dependencies

### Install Required Packages
```bash
pip install -r requirements.txt
```

### For Real-Time Features
```bash
pip install -r requirements-realtime.txt
```

### For Production Use
```bash
pip install -r requirements-production.txt
```

## 🎯 System Features

### 🤖 ML Models
- **SMS Parser**: Extract transaction data with 95%+ accuracy
- **Expense Categorizer**: Auto-categorize with 9 expense categories
- **Savings Optimizer**: Create personalized budget recommendations
- **Investment Engine**: Risk-profiled investment recommendations

### 📈 Real-Time Intelligence
- **Market Data**: Live quotes via Finnhub API
- **Sentiment Analysis**: Market sentiment from news and price movements
- **Technical Indicators**: Basic technical analysis for stocks
- **News Integration**: Latest financial news with sentiment scoring

### 🔥 Trading Automation
- **ICICI Direct Integration**: Via Breeze API
- **Account Management**: Portfolio tracking and analysis
- **Risk Controls**: Automated stop-loss and position limits
- **Order Management**: Automated trade execution

## 🛡️ Security & Best Practices

- **API Key Management**: Secure storage in config files
- **Error Handling**: Comprehensive exception handling
- **Rate Limiting**: Built-in API rate limiting
- **Data Validation**: Input validation for all user data
- **Logging**: Detailed logging for debugging and monitoring

## 📚 Documentation

- [Complete System Guide](docs/SMART_MONEY_AI_GUIDE.md)
- [Trading Automation Guide](docs/COMPLETE_TRADING_AUTOMATION_GUIDE.md)
- [API Documentation](docs/API_DOCUMENTATION.md)
- [Configuration Reference](docs/CONFIGURATION.md)

## 🚨 Important Notes

### For Real-Time Features
- Requires valid Finnhub API key
- Rate limits apply (60 calls/minute for free tier)
- Internet connection required

### For Trading Features
- Requires ICICI Direct Breeze API credentials
- Only works with valid trading account
- Use paper trading for testing

### Data Privacy
- No user data is stored externally
- All processing happens locally
- API keys stored securely in config files

## 🔄 Version History

### v4.0.0 (Current)
- ✅ Reorganized project structure
- ✅ Centralized configuration system
- ✅ Modular integration architecture
- ✅ Enhanced error handling
- ✅ Streamlined imports and dependencies

### v3.0.0
- ✅ Added Breeze API trading integration
- ✅ Real-time portfolio monitoring
- ✅ Automated risk management

### v2.0.0
- ✅ Added real-time market data integration
- ✅ Enhanced investment recommendations
- ✅ Market sentiment analysis

### v1.0.0
- ✅ Core 4-part ML system
- ✅ SMS parsing and expense categorization
- ✅ Budget optimization and investment recommendations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

For support and questions:
- Check the [documentation](docs/)
- Run the system test: `python examples/simple_system_test.py`
- Review the example demos in `examples/`

---

## 🎉 Ready to Get Started?

1. **Configure your API keys** in `config/config.json`
2. **Run the system test**: `python examples/simple_system_test.py`
3. **Try the demos**: Start with `examples/real_time_market_demo.py`
4. **Build something amazing** with Smart Money AI! 🚀

*Smart Money AI - Making financial intelligence accessible to everyone.*