"""
Finnhub API Integration for Real-Time Market Data
================================================

This module provides comprehensive real-time market data integration using the Finnhub API.
It includes stock quotes, market sentiment analysis, news, and technical indicators.
"""

import requests
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json

class FinnhubRealTimeAnalyzer:
    """Real-time market data analyzer using Finnhub API"""
    
    def __init__(self, api_key: str):
        """Initialize with Finnhub API key"""
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.session = requests.Session()
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _make_request(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make rate-limited API request"""
        # Ensure rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        if params is None:
            params = {}
        
        params['token'] = self.api_key
        
        try:
            url = f"{self.base_url}/{endpoint}"
            response = self.session.get(url, params=params, timeout=10)
            self.last_request_time = time.time()
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                return {'error': 'Rate limit exceeded. Please wait before making another request.'}
            else:
                return {'error': f'API request failed with status {response.status_code}'}
                
        except requests.exceptions.RequestException as e:
            return {'error': f'Network error: {str(e)}'}
        except json.JSONDecodeError:
            return {'error': 'Invalid JSON response from API'}
    
    def get_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol"""
        result = self._make_request('quote', {'symbol': symbol})
        
        if 'error' in result:
            return result
        
        if 'c' not in result or result['c'] == 0:
            return {'error': f'No data available for symbol {symbol}'}
        
        return {
            'symbol': symbol,
            'current_price': result.get('c', 0),
            'change': result.get('d', 0),
            'change_percent': result.get('dp', 0),
            'high': result.get('h', 0),
            'low': result.get('l', 0),
            'open': result.get('o', 0),
            'previous_close': result.get('pc', 0),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get company profile information"""
        result = self._make_request('stock/profile2', {'symbol': symbol})
        
        if 'error' in result:
            return result
        
        return {
            'symbol': symbol,
            'name': result.get('name', 'Unknown'),
            'country': result.get('country', 'Unknown'),
            'currency': result.get('currency', 'USD'),
            'exchange': result.get('exchange', 'Unknown'),
            'industry': result.get('finnhubIndustry', 'Unknown'),
            'market_cap': result.get('marketCapitalization', 0),
            'shares_outstanding': result.get('shareOutstanding', 0),
            'website': result.get('weburl', '')
        }
    
    def get_basic_financials(self, symbol: str) -> Dict[str, Any]:
        """Get basic financial metrics"""
        result = self._make_request('stock/metric', {'symbol': symbol, 'metric': 'all'})
        
        if 'error' in result:
            return result
        
        metrics = result.get('metric', {})
        
        return {
            'symbol': symbol,
            'pe_ratio': metrics.get('peBasicExclExtraTTM', 0),
            'pb_ratio': metrics.get('pbQuarterly', 0),
            'dividend_yield': metrics.get('dividendYieldIndicatedAnnual', 0),
            'roe': metrics.get('roeRfy', 0),
            'roa': metrics.get('roaRfy', 0),
            'debt_to_equity': metrics.get('totalDebt/totalEquityQuarterly', 0),
            'current_ratio': metrics.get('currentRatioQuarterly', 0),
            'revenue_growth': metrics.get('revenueGrowthTTMYoy', 0),
            'earnings_growth': metrics.get('epsGrowthTTMYoy', 0)
        }
    
    def get_market_news(self, category: str = 'general', count: int = 10) -> List[Dict[str, Any]]:
        """Get latest market news"""
        result = self._make_request('news', {
            'category': category,
            'minId': 0
        })
        
        if 'error' in result:
            return [{'error': result['error']}]
        
        if not isinstance(result, list):
            return [{'error': 'Unexpected news API response format'}]
        
        # Format news items
        news_items = []
        for item in result[:count]:
            if isinstance(item, dict):
                news_items.append({
                    'headline': item.get('headline', 'No headline'),
                    'summary': item.get('summary', 'No summary'),
                    'source': item.get('source', 'Unknown'),
                    'url': item.get('url', ''),
                    'datetime': datetime.fromtimestamp(item.get('datetime', 0)).isoformat() if item.get('datetime') else '',
                    'sentiment': self._analyze_news_sentiment(item.get('headline', '') + ' ' + item.get('summary', ''))
                })
        
        return news_items
    
    def get_company_news(self, symbol: str, days_back: int = 7) -> List[Dict[str, Any]]:
        """Get company-specific news"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        result = self._make_request('company-news', {
            'symbol': symbol,
            'from': start_date.strftime('%Y-%m-%d'),
            'to': end_date.strftime('%Y-%m-%d')
        })
        
        if 'error' in result:
            return [{'error': result['error']}]
        
        if not isinstance(result, list):
            return [{'error': 'Unexpected company news API response format'}]
        
        # Format news items
        news_items = []
        for item in result[:10]:  # Limit to 10 items
            if isinstance(item, dict):
                news_items.append({
                    'headline': item.get('headline', 'No headline'),
                    'summary': item.get('summary', 'No summary'),
                    'source': item.get('source', 'Unknown'),
                    'url': item.get('url', ''),
                    'datetime': datetime.fromtimestamp(item.get('datetime', 0)).isoformat() if item.get('datetime') else '',
                    'sentiment': self._analyze_news_sentiment(item.get('headline', '') + ' ' + item.get('summary', ''))
                })
        
        return news_items
    
    def get_market_status(self, exchange: str = 'US') -> Dict[str, Any]:
        """Get market status (open/closed)"""
        result = self._make_request('stock/market-status', {'exchange': exchange})
        
        if 'error' in result:
            return result
        
        return {
            'exchange': exchange,
            'is_open': result.get('isOpen', False),
            'session': result.get('session', 'Unknown'),
            'timezone': result.get('timezone', 'Unknown'),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get basic technical analysis indicators"""
        # This is a simplified implementation
        # In a real scenario, you might use additional APIs or calculate these yourself
        
        quote = self.get_quote(symbol)
        if 'error' in quote:
            return quote
        
        current_price = quote['current_price']
        high = quote['high']
        low = quote['low']
        open_price = quote['open']
        
        # Simple technical indicators
        indicators = {
            'symbol': symbol,
            'price_position': {
                'vs_high': ((current_price - high) / high * 100) if high > 0 else 0,
                'vs_low': ((current_price - low) / low * 100) if low > 0 else 0,
                'vs_open': ((current_price - open_price) / open_price * 100) if open_price > 0 else 0
            },
            'volatility': {
                'daily_range': ((high - low) / low * 100) if low > 0 else 0,
                'price_change': quote['change_percent']
            },
            'momentum': {
                'bullish_signals': 0,
                'bearish_signals': 0,
                'overall': 'Neutral'
            }
        }
        
        # Simple momentum analysis
        if quote['change_percent'] > 2:
            indicators['momentum']['bullish_signals'] += 2
        elif quote['change_percent'] > 0:
            indicators['momentum']['bullish_signals'] += 1
        elif quote['change_percent'] < -2:
            indicators['momentum']['bearish_signals'] += 2
        elif quote['change_percent'] < 0:
            indicators['momentum']['bearish_signals'] += 1
        
        # Determine overall momentum
        bullish = indicators['momentum']['bullish_signals']
        bearish = indicators['momentum']['bearish_signals']
        
        if bullish > bearish:
            indicators['momentum']['overall'] = 'Bullish'
        elif bearish > bullish:
            indicators['momentum']['overall'] = 'Bearish'
        else:
            indicators['momentum']['overall'] = 'Neutral'
        
        return indicators
    
    def _analyze_news_sentiment(self, text: str) -> str:
        """Simple sentiment analysis for news"""
        text = text.lower()
        
        positive_words = ['gain', 'rise', 'up', 'bull', 'positive', 'growth', 'profit', 'beat', 'strong', 'good']
        negative_words = ['fall', 'drop', 'down', 'bear', 'negative', 'loss', 'miss', 'weak', 'bad', 'decline']
        
        positive_count = sum(1 for word in positive_words if word in text)
        negative_count = sum(1 for word in negative_words if word in text)
        
        if positive_count > negative_count:
            return 'Positive'
        elif negative_count > positive_count:
            return 'Negative'
        else:
            return 'Neutral'
    
    def get_market_sentiment(self, symbols: List[str] = None) -> Dict[str, Any]:
        """Analyze overall market sentiment"""
        if symbols is None:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']  # Default major stocks
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        
        for symbol in symbols[:5]:  # Limit to 5 to avoid rate limits
            quote = self.get_quote(symbol)
            if 'error' not in quote:
                change_percent = quote['change_percent']
                if change_percent > 0:
                    positive_count += 1
                elif change_percent < 0:
                    negative_count += 1
                
                sentiments.append({
                    'symbol': symbol,
                    'change_percent': change_percent,
                    'sentiment': 'Positive' if change_percent > 0 else 'Negative' if change_percent < 0 else 'Neutral'
                })
        
        # Overall market sentiment
        if positive_count > negative_count:
            overall_sentiment = 'Bullish'
        elif negative_count > positive_count:
            overall_sentiment = 'Bearish'
        else:
            overall_sentiment = 'Neutral'
        
        return {
            'overall_sentiment': overall_sentiment,
            'positive_stocks': positive_count,
            'negative_stocks': negative_count,
            'analyzed_stocks': len(sentiments),
            'individual_sentiments': sentiments,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_comprehensive_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive analysis for a symbol"""
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat()
        }
        
        # Get quote data
        quote = self.get_quote(symbol)
        if 'error' not in quote:
            analysis['quote'] = quote
        else:
            analysis['quote_error'] = quote['error']
        
        # Get company profile
        profile = self.get_company_profile(symbol)
        if 'error' not in profile:
            analysis['company_profile'] = profile
        else:
            analysis['profile_error'] = profile['error']
        
        # Get basic financials
        financials = self.get_basic_financials(symbol)
        if 'error' not in financials:
            analysis['financials'] = financials
        else:
            analysis['financials_error'] = financials['error']
        
        # Get technical indicators
        technical = self.get_technical_indicators(symbol)
        if 'error' not in technical:
            analysis['technical_indicators'] = technical
        else:
            analysis['technical_error'] = technical['error']
        
        # Get recent news
        news = self.get_company_news(symbol, days_back=3)
        if news and 'error' not in news[0]:
            analysis['recent_news'] = news[:3]  # Top 3 news items
        else:
            analysis['news_error'] = news[0]['error'] if news else 'No news available'
        
        return analysis

# Example usage
if __name__ == "__main__":
    # This would be used for testing
    api_key = "your_api_key_here"
    analyzer = FinnhubRealTimeAnalyzer(api_key)
    
    # Test quote
    quote = analyzer.get_quote('AAPL')
    print("Quote:", quote)
    
    # Test market sentiment
    sentiment = analyzer.get_market_sentiment()
    print("Market Sentiment:", sentiment)