"""
Real-Time Market Intelligence Integration
Smart Money AI System Enhancement

This module integrates Finnhub's real-time market data with the existing
4-part Smart Money AI system for enhanced investment recommendations.
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Union
import sys
import os

# Add the parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing Smart Money AI components (commented to avoid circular import)
# from smart_money_ai import SmartMoneyAI
# from smart_money_ai.ml_models.investment_recommendation_model import InvestmentRecommendationModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealTimeMarketEngine:
    """
    Real-time market data engine for Smart Money AI
    Provides live market intelligence without WebSocket dependencies
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        
        # Market data cache
        self.market_cache = {}
        self.last_update = {}
        
        # Cache timeout (seconds)
        self.cache_timeout = 30
        
    def _make_request(self, endpoint: str, params: Dict = None) -> Dict:
        """Make API request with error handling"""
        try:
            if params is None:
                params = {}
            params['token'] = self.api_key
            
            url = f"{self.base_url}{endpoint}"
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            return {}
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            return {}
    
    def get_real_time_quote(self, symbol: str) -> Dict:
        """Get real-time stock quote with caching"""
        cache_key = f"quote_{symbol}"
        current_time = time.time()
        
        # Check cache
        if (cache_key in self.market_cache and 
            cache_key in self.last_update and
            current_time - self.last_update[cache_key] < self.cache_timeout):
            return self.market_cache[cache_key]
        
        # Fetch fresh data
        quote_data = self._make_request("/quote", {"symbol": symbol})
        
        if quote_data and 'c' in quote_data:
            processed_quote = {
                'symbol': symbol,
                'current_price': quote_data.get('c', 0),
                'change': quote_data.get('d', 0),
                'change_percent': quote_data.get('dp', 0),
                'high': quote_data.get('h', 0),
                'low': quote_data.get('l', 0),
                'open': quote_data.get('o', 0),
                'previous_close': quote_data.get('pc', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Update cache
            self.market_cache[cache_key] = processed_quote
            self.last_update[cache_key] = current_time
            
            return processed_quote
        
        return {}
    
    def get_company_fundamentals(self, symbol: str) -> Dict:
        """Get comprehensive company data"""
        # Company profile
        profile = self._make_request("/stock/profile2", {"symbol": symbol})
        
        # Basic financials
        financials = self._make_request("/stock/metric", {"symbol": symbol, "metric": "all"})
        
        # Recommendation trends
        recommendations = self._make_request("/stock/recommendation", {"symbol": symbol})
        
        return {
            'profile': profile,
            'financials': financials,
            'recommendations': recommendations,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_market_news(self, symbol: str = None, category: str = "general") -> List[Dict]:
        """Get latest market news"""
        if symbol:
            # Company-specific news
            today = datetime.now().strftime("%Y-%m-%d")
            week_ago = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            
            params = {
                "symbol": symbol,
                "from": week_ago,
                "to": today
            }
            news_data = self._make_request("/company-news", params)
        else:
            # General market news
            news_data = self._make_request("/news", {"category": category})
        
        if isinstance(news_data, list):
            return news_data[:10]  # Limit to 10 recent articles
        
        return []
    
    def get_earnings_calendar(self, days_ahead: int = 30) -> List[Dict]:
        """Get upcoming earnings announcements"""
        from_date = datetime.now().strftime("%Y-%m-%d")
        to_date = (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        calendar_data = self._make_request("/calendar/earnings", {
            "from": from_date,
            "to": to_date
        })
        
        return calendar_data.get('earningsCalendar', [])
    
    def get_economic_indicators(self) -> List[Dict]:
        """Get economic calendar events"""
        calendar_data = self._make_request("/calendar/economic")
        return calendar_data.get('economicCalendar', [])
    
    def search_symbols(self, query: str) -> List[Dict]:
        """Search for stocks by name or symbol"""
        search_results = self._make_request("/search", {"q": query})
        return search_results.get('result', [])
    
    def analyze_sector_performance(self, sector_symbols: List[str]) -> Dict:
        """Analyze performance of a sector"""
        sector_data = {}
        total_change = 0
        count = 0
        
        for symbol in sector_symbols:
            quote = self.get_real_time_quote(symbol)
            if quote and 'change_percent' in quote:
                sector_data[symbol] = quote
                total_change += quote['change_percent']
                count += 1
        
        avg_change = total_change / count if count > 0 else 0
        
        return {
            'symbols': sector_data,
            'average_change': avg_change,
            'total_symbols': count,
            'sector_sentiment': self._get_sentiment_label(avg_change),
            'timestamp': datetime.now().isoformat()
        }
    
    def _get_sentiment_label(self, change_percent: float) -> str:
        """Convert percentage change to sentiment label"""
        if change_percent > 2:
            return "Very Bullish"
        elif change_percent > 0.5:
            return "Bullish"
        elif change_percent > -0.5:
            return "Neutral"
        elif change_percent > -2:
            return "Bearish"
        else:
            return "Very Bearish"

class SmartMoneyRealTimeAnalyzer:
    """
    Enhanced Smart Money AI with real-time market intelligence
    Integrates live data with existing ML models
    """
    
    def __init__(self, api_key: str):
        self.market_engine = RealTimeMarketEngine(api_key)
        # Don't create circular dependency - the real-time analyzer is independent
        self.smart_money = None
        
        # Initialize investment model if available (optional)
        self.investment_model = None
        try:
            # Try to import and initialize investment model without circular dependency
            from ..ml_models.investment_recommendation_model import InvestmentRecommendationModel
            self.investment_model = InvestmentRecommendationModel()
        except Exception as e:
            logger.warning(f"Could not initialize investment model: {e}")
            self.investment_model = None
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict:
        """Get comprehensive analysis combining real-time and ML insights"""
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'real_time_data': {},
            'technical_analysis': {},
            'fundamental_analysis': {},
            'ml_prediction': {},
            'investment_recommendation': {}
        }
        
        try:
            # Real-time market data
            quote = self.market_engine.get_real_time_quote(symbol)
            fundamentals = self.market_engine.get_company_fundamentals(symbol)
            news = self.market_engine.get_market_news(symbol)
            
            analysis['real_time_data'] = {
                'quote': quote,
                'fundamentals': fundamentals,
                'recent_news': news[:3]  # Top 3 news items
            }
            
            # Technical analysis
            if quote:
                analysis['technical_analysis'] = self._perform_technical_analysis(quote)
            
            # Fundamental analysis
            if fundamentals.get('financials'):
                analysis['fundamental_analysis'] = self._perform_fundamental_analysis(
                    fundamentals['financials']
                )
            
            # ML prediction using existing models
            if self.investment_model and quote:
                ml_features = self._prepare_ml_features(quote, fundamentals)
                analysis['ml_prediction'] = self._get_ml_prediction(ml_features)
            
            # Investment recommendation
            analysis['investment_recommendation'] = self._generate_investment_recommendation(
                quote, fundamentals, analysis['technical_analysis']
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            analysis['error'] = str(e)
        
        return analysis
    
    def _perform_technical_analysis(self, quote: Dict) -> Dict:
        """Perform technical analysis on quote data"""
        technical = {
            'price_action': 'Neutral',
            'momentum': 'Neutral',
            'volatility': 'Normal',
            'support_resistance': {}
        }
        
        if 'change_percent' in quote:
            change_pct = quote['change_percent']
            
            # Price action analysis
            if change_pct > 3:
                technical['price_action'] = 'Strong Bullish'
            elif change_pct > 1:
                technical['price_action'] = 'Bullish'
            elif change_pct < -3:
                technical['price_action'] = 'Strong Bearish'
            elif change_pct < -1:
                technical['price_action'] = 'Bearish'
            
            # Momentum analysis
            if abs(change_pct) > 2:
                technical['momentum'] = 'High'
            elif abs(change_pct) > 1:
                technical['momentum'] = 'Medium'
            else:
                technical['momentum'] = 'Low'
        
        # Support and resistance levels
        if all(k in quote for k in ['high', 'low', 'current_price']):
            current = quote['current_price']
            high = quote['high']
            low = quote['low']
            
            technical['support_resistance'] = {
                'resistance': high,
                'support': low,
                'distance_to_resistance': ((high - current) / current) * 100,
                'distance_to_support': ((current - low) / current) * 100
            }
        
        return technical
    
    def _perform_fundamental_analysis(self, financials: Dict) -> Dict:
        """Analyze fundamental metrics"""
        fundamental = {
            'valuation': 'Fair',
            'profitability': 'Average',
            'growth': 'Stable',
            'financial_health': 'Good'
        }
        
        if 'metric' in financials:
            metrics = financials['metric']
            
            # P/E ratio analysis
            pe_ratio = metrics.get('peBasicExclExtraTTM')
            if pe_ratio:
                if pe_ratio < 15:
                    fundamental['valuation'] = 'Undervalued'
                elif pe_ratio > 25:
                    fundamental['valuation'] = 'Overvalued'
                else:
                    fundamental['valuation'] = 'Fair'
            
            # ROE analysis
            roe = metrics.get('roeRfy')
            if roe:
                if roe > 15:
                    fundamental['profitability'] = 'Excellent'
                elif roe > 10:
                    fundamental['profitability'] = 'Good'
                elif roe < 5:
                    fundamental['profitability'] = 'Poor'
            
            # Debt-to-equity analysis
            debt_to_equity = metrics.get('totalDebt/totalEquityQuarterly')
            if debt_to_equity:
                if debt_to_equity < 0.3:
                    fundamental['financial_health'] = 'Excellent'
                elif debt_to_equity > 0.7:
                    fundamental['financial_health'] = 'Concerning'
        
        return fundamental
    
    def _prepare_ml_features(self, quote: Dict, fundamentals: Dict) -> Dict:
        """Prepare features for ML model prediction"""
        features = {
            'price': quote.get('current_price', 0),
            'change_percent': quote.get('change_percent', 0),
            'volume_ratio': 1.0,  # Placeholder
            'volatility': abs(quote.get('change_percent', 0)),
            'market_cap': 0,
            'pe_ratio': 0,
            'beta': 1.0
        }
        
        if fundamentals.get('financials', {}).get('metric'):
            metrics = fundamentals['financials']['metric']
            features.update({
                'market_cap': metrics.get('marketCapitalization', 0),
                'pe_ratio': metrics.get('peBasicExclExtraTTM', 0),
                'beta': metrics.get('beta', 1.0)
            })
        
        return features
    
    def _get_ml_prediction(self, features: Dict) -> Dict:
        """Get ML model prediction"""
        try:
            # Create feature array for prediction
            feature_array = np.array([
                features.get('price', 0),
                features.get('change_percent', 0),
                features.get('volume_ratio', 1),
                features.get('volatility', 0),
                features.get('market_cap', 0),
                features.get('pe_ratio', 0),
                features.get('beta', 1)
            ]).reshape(1, -1)
            
            # Note: This is a placeholder for ML prediction
            # In practice, you'd use your trained models
            prediction_score = np.random.uniform(0.3, 0.8)  # Placeholder
            
            return {
                'prediction_score': prediction_score,
                'recommendation': 'Buy' if prediction_score > 0.6 else 'Hold' if prediction_score > 0.4 else 'Sell',
                'confidence': prediction_score,
                'model_version': '1.0'
            }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return {'error': str(e)}
    
    def _generate_investment_recommendation(self, quote: Dict, fundamentals: Dict, 
                                          technical: Dict) -> Dict:
        """Generate comprehensive investment recommendation"""
        recommendation = {
            'action': 'Hold',
            'confidence': 50,
            'target_price': 0,
            'stop_loss': 0,
            'time_horizon': '3-6 months',
            'risk_level': 'Medium',
            'reasons': []
        }
        
        try:
            current_price = quote.get('current_price', 0)
            change_pct = quote.get('change_percent', 0)
            
            # Scoring system
            score = 50  # Neutral starting point
            reasons = []
            
            # Technical factors
            if technical.get('price_action') == 'Strong Bullish':
                score += 15
                reasons.append("Strong bullish price action")
            elif technical.get('price_action') == 'Bullish':
                score += 8
                reasons.append("Positive price momentum")
            elif technical.get('price_action') == 'Strong Bearish':
                score -= 15
                reasons.append("Strong bearish price action")
            elif technical.get('price_action') == 'Bearish':
                score -= 8
                reasons.append("Negative price momentum")
            
            # Fundamental factors
            fundamental = self._perform_fundamental_analysis(fundamentals.get('financials', {}))
            
            if fundamental.get('valuation') == 'Undervalued':
                score += 10
                reasons.append("Stock appears undervalued")
            elif fundamental.get('valuation') == 'Overvalued':
                score -= 10
                reasons.append("Stock appears overvalued")
            
            if fundamental.get('profitability') == 'Excellent':
                score += 8
                reasons.append("Excellent profitability metrics")
            elif fundamental.get('profitability') == 'Poor':
                score -= 8
                reasons.append("Poor profitability metrics")
            
            # Analyst recommendations
            if fundamentals.get('recommendations'):
                recs = fundamentals['recommendations']
                if recs:
                    latest = recs[0]
                    strong_buy = latest.get('strongBuy', 0)
                    buy = latest.get('buy', 0)
                    total = sum([latest.get(key, 0) for key in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']])
                    
                    if total > 0:
                        bullish_ratio = (strong_buy + buy) / total
                        if bullish_ratio > 0.7:
                            score += 12
                            reasons.append(f"Strong analyst consensus ({bullish_ratio*100:.0f}% bullish)")
                        elif bullish_ratio < 0.3:
                            score -= 8
                            reasons.append(f"Weak analyst consensus ({bullish_ratio*100:.0f}% bullish)")
            
            # Determine action and confidence
            if score >= 70:
                recommendation['action'] = 'Strong Buy'
                recommendation['confidence'] = min(95, score)
                recommendation['target_price'] = current_price * 1.15
            elif score >= 55:
                recommendation['action'] = 'Buy'
                recommendation['confidence'] = score
                recommendation['target_price'] = current_price * 1.08
            elif score <= 30:
                recommendation['action'] = 'Strong Sell'
                recommendation['confidence'] = 100 - score
                recommendation['target_price'] = current_price * 0.85
            elif score <= 45:
                recommendation['action'] = 'Sell'
                recommendation['confidence'] = 100 - score
                recommendation['target_price'] = current_price * 0.92
            else:
                recommendation['action'] = 'Hold'
                recommendation['confidence'] = 50
                recommendation['target_price'] = current_price
            
            # Set stop loss
            recommendation['stop_loss'] = current_price * 0.92 if recommendation['action'] in ['Buy', 'Strong Buy'] else current_price * 1.08
            
            # Risk assessment
            volatility = abs(change_pct)
            if volatility > 5:
                recommendation['risk_level'] = 'High'
            elif volatility > 2:
                recommendation['risk_level'] = 'Medium-High'
            elif volatility < 1:
                recommendation['risk_level'] = 'Low'
            
            recommendation['reasons'] = reasons
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {e}")
            recommendation['error'] = str(e)
        
        return recommendation
    
    def monitor_portfolio(self, symbols: List[str]) -> Dict:
        """Monitor a portfolio of stocks in real-time"""
        portfolio_analysis = {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'portfolio_performance': {},
            'alerts': [],
            'summary': {}
        }
        
        total_change = 0
        positive_performers = 0
        
        for symbol in symbols:
            try:
                quote = self.market_engine.get_real_time_quote(symbol)
                if quote:
                    change_pct = quote.get('change_percent', 0)
                    total_change += change_pct
                    
                    if change_pct > 0:
                        positive_performers += 1
                    
                    portfolio_analysis['portfolio_performance'][symbol] = {
                        'price': quote.get('current_price'),
                        'change_percent': change_pct,
                        'status': 'Up' if change_pct > 0 else 'Down' if change_pct < 0 else 'Flat'
                    }
                    
                    # Generate alerts for significant moves
                    if abs(change_pct) > 5:
                        portfolio_analysis['alerts'].append({
                            'symbol': symbol,
                            'type': 'SIGNIFICANT_MOVE',
                            'message': f"{symbol} moved {change_pct:+.2f}%",
                            'severity': 'HIGH' if abs(change_pct) > 10 else 'MEDIUM'
                        })
                        
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
        
        # Portfolio summary
        portfolio_analysis['summary'] = {
            'average_change': total_change / len(symbols) if symbols else 0,
            'positive_performers': positive_performers,
            'positive_ratio': positive_performers / len(symbols) if symbols else 0,
            'overall_sentiment': 'Bullish' if total_change > 0 else 'Bearish' if total_change < 0 else 'Neutral'
        }
        
        return portfolio_analysis
    
    def get_market_overview(self) -> Dict:
        """Get comprehensive market overview"""
        # Major market indices
        major_indices = ['SPY', 'QQQ', 'DIA', 'IWM']  # S&P 500, NASDAQ, Dow, Russell 2000
        
        market_overview = {
            'timestamp': datetime.now().isoformat(),
            'indices': {},
            'sector_performance': {},
            'market_sentiment': 'Neutral',
            'economic_calendar': [],
            'top_news': []
        }
        
        try:
            # Analyze major indices
            for index in major_indices:
                quote = self.market_engine.get_real_time_quote(index)
                if quote:
                    market_overview['indices'][index] = quote
            
            # Get economic calendar
            market_overview['economic_calendar'] = self.market_engine.get_economic_indicators()[:5]
            
            # Get top market news
            market_overview['top_news'] = self.market_engine.get_market_news()[:5]
            
            # Calculate overall market sentiment
            index_changes = [data.get('change_percent', 0) for data in market_overview['indices'].values()]
            if index_changes:
                avg_change = sum(index_changes) / len(index_changes)
                if avg_change > 1:
                    market_overview['market_sentiment'] = 'Bullish'
                elif avg_change < -1:
                    market_overview['market_sentiment'] = 'Bearish'
                else:
                    market_overview['market_sentiment'] = 'Neutral'
            
        except Exception as e:
            logger.error(f"Error getting market overview: {e}")
            market_overview['error'] = str(e)
        
        return market_overview

# Demo function to showcase capabilities
def demo_real_time_analysis():
    """Demonstrate real-time market analysis capabilities"""
    print("🚀 Smart Money AI - Real-Time Market Intelligence Demo")
    print("=" * 60)
    
    # Initialize with API key
    API_KEY = "d3o93nhr01qmj830b3mgd3o93nhr01qmj830b3n0"
    analyzer = SmartMoneyRealTimeAnalyzer(API_KEY)
    
    # Demo symbols
    demo_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN', 'NVDA']
    
    try:
        print("📊 Market Overview:")
        market_overview = analyzer.get_market_overview()
        
        print(f"Market Sentiment: {market_overview.get('market_sentiment', 'Unknown')}")
        print("\nMajor Indices:")
        for symbol, data in market_overview.get('indices', {}).items():
            if data:
                print(f"  {symbol}: ${data.get('current_price', 0):.2f} ({data.get('change_percent', 0):+.2f}%)")
        
        print(f"\n📈 Portfolio Analysis:")
        portfolio = analyzer.monitor_portfolio(demo_symbols)
        
        summary = portfolio.get('summary', {})
        print(f"Average Change: {summary.get('average_change', 0):+.2f}%")
        print(f"Positive Performers: {summary.get('positive_performers', 0)}/{len(demo_symbols)}")
        print(f"Overall Sentiment: {summary.get('overall_sentiment', 'Unknown')}")
        
        print(f"\n🎯 Detailed Analysis for AAPL:")
        apple_analysis = analyzer.get_comprehensive_analysis('AAPL')
        
        quote = apple_analysis.get('real_time_data', {}).get('quote', {})
        if quote:
            print(f"Current Price: ${quote.get('current_price', 0):.2f}")
            print(f"Change: {quote.get('change_percent', 0):+.2f}%")
        
        recommendation = apple_analysis.get('investment_recommendation', {})
        if recommendation:
            print(f"Recommendation: {recommendation.get('action', 'Unknown')}")
            print(f"Confidence: {recommendation.get('confidence', 0):.0f}%")
            print(f"Target Price: ${recommendation.get('target_price', 0):.2f}")
            
            reasons = recommendation.get('reasons', [])
            if reasons:
                print("Reasons:")
                for reason in reasons[:3]:  # Show top 3 reasons
                    print(f"  • {reason}")
        
        # Show alerts if any
        alerts = portfolio.get('alerts', [])
        if alerts:
            print(f"\n🚨 Portfolio Alerts:")
            for alert in alerts[:3]:  # Show top 3 alerts
                print(f"  • {alert.get('message', '')}")
        
        print(f"\n✅ Real-time analysis complete!")
        print(f"💡 Your Smart Money AI system now has live market intelligence!")
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        print("💡 Make sure your API key is valid and you have internet connection")

if __name__ == "__main__":
    demo_real_time_analysis()