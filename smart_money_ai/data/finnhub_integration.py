"""
Finnhub API Integration for Real-Time Market Data
Smart Money AI System - Real-Time Data Module

This module provides comprehensive real-time market data integration
using Finnhub's extensive financial API for investment recommendations.
"""

import requests
import websocket
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Optional, Union, Callable
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketData:
    """Real-time market data structure"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    high: float = 0.0
    low: float = 0.0
    open: float = 0.0
    previous_close: float = 0.0

@dataclass
class NewsItem:
    """News data structure"""
    headline: str
    summary: str
    source: str
    timestamp: datetime
    url: str
    related_symbols: List[str]

class FinnhubMarketDataStreamer:
    """
    Real-time market data streaming and analysis engine
    Integrates with Smart Money AI's investment recommendation system
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://finnhub.io/api/v1"
        self.ws_url = f"wss://ws.finnhub.io?token={api_key}"
        
        # Real-time data storage
        self.market_data = {}
        self.news_data = []
        self.economic_data = {}
        
        # WebSocket connection
        self.ws = None
        self.is_streaming = False
        self.subscribed_symbols = set()
        
        # Callbacks for real-time updates
        self.price_callbacks = []
        self.news_callbacks = []
        
    def connect_websocket(self):
        """Establish WebSocket connection for real-time data"""
        try:
            self.ws = websocket.WebSocketApp(
                self.ws_url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start WebSocket in separate thread
            ws_thread = threading.Thread(target=self.ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            self.is_streaming = True
            logger.info("WebSocket connection established")
            
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
    
    def _on_open(self, ws):
        """WebSocket connection opened"""
        logger.info("WebSocket connection opened")
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            
            if data.get('type') == 'trade':
                self._process_trade_data(data)
            elif data.get('type') == 'news':
                self._process_news_data(data)
                
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket errors"""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection close"""
        logger.info("WebSocket connection closed")
        self.is_streaming = False
    
    def _process_trade_data(self, data):
        """Process real-time trade data"""
        for trade in data.get('data', []):
            symbol = trade.get('s')
            price = trade.get('p')
            volume = trade.get('v', 0)
            timestamp = datetime.fromtimestamp(trade.get('t', 0) / 1000)
            
            if symbol in self.market_data:
                prev_price = self.market_data[symbol].price
                change = price - prev_price
                change_percent = (change / prev_price) * 100 if prev_price > 0 else 0
            else:
                change = 0
                change_percent = 0
            
            market_data = MarketData(
                symbol=symbol,
                price=price,
                change=change,
                change_percent=change_percent,
                volume=volume,
                timestamp=timestamp
            )
            
            self.market_data[symbol] = market_data
            
            # Trigger callbacks
            for callback in self.price_callbacks:
                callback(market_data)
    
    def _process_news_data(self, data):
        """Process real-time news data"""
        for news in data.get('data', []):
            news_item = NewsItem(
                headline=news.get('headline', ''),
                summary=news.get('summary', ''),
                source=news.get('source', ''),
                timestamp=datetime.fromtimestamp(news.get('datetime', 0)),
                url=news.get('url', ''),
                related_symbols=news.get('related', '').split(',') if news.get('related') else []
            )
            
            self.news_data.append(news_item)
            
            # Trigger callbacks
            for callback in self.news_callbacks:
                callback(news_item)
    
    def subscribe_to_symbol(self, symbol: str):
        """Subscribe to real-time data for a symbol"""
        if self.ws and self.is_streaming:
            subscription = {
                "type": "subscribe",
                "symbol": symbol
            }
            self.ws.send(json.dumps(subscription))
            self.subscribed_symbols.add(symbol)
            logger.info(f"Subscribed to {symbol}")
    
    def subscribe_to_news(self, symbol: str):
        """Subscribe to real-time news for a symbol"""
        if self.ws and self.is_streaming:
            subscription = {
                "type": "subscribe-news",
                "symbol": symbol
            }
            self.ws.send(json.dumps(subscription))
            logger.info(f"Subscribed to news for {symbol}")
    
    def get_quote(self, symbol: str) -> Optional[MarketData]:
        """Get current quote for a symbol"""
        try:
            url = f"{self.base_url}/quote"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if 'c' in data:
                market_data = MarketData(
                    symbol=symbol,
                    price=data['c'],
                    change=data.get('d', 0),
                    change_percent=data.get('dp', 0),
                    volume=0,
                    timestamp=datetime.now(),
                    high=data.get('h', 0),
                    low=data.get('l', 0),
                    open=data.get('o', 0),
                    previous_close=data.get('pc', 0)
                )
                
                self.market_data[symbol] = market_data
                return market_data
                
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
        
        return None
    
    def get_company_profile(self, symbol: str) -> Dict:
        """Get company profile information"""
        try:
            url = f"{self.base_url}/stock/profile2"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {e}")
            return {}
    
    def get_basic_financials(self, symbol: str) -> Dict:
        """Get basic financial metrics"""
        try:
            url = f"{self.base_url}/stock/metric"
            params = {
                'symbol': symbol,
                'metric': 'all',
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return {}
    
    def get_recommendation_trends(self, symbol: str) -> List[Dict]:
        """Get analyst recommendation trends"""
        try:
            url = f"{self.base_url}/stock/recommendation"
            params = {
                'symbol': symbol,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching recommendations for {symbol}: {e}")
            return []
    
    def get_earnings_calendar(self, from_date: str, to_date: str) -> List[Dict]:
        """Get earnings calendar"""
        try:
            url = f"{self.base_url}/calendar/earnings"
            params = {
                'from': from_date,
                'to': to_date,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            return response.json().get('earningsCalendar', [])
            
        except Exception as e:
            logger.error(f"Error fetching earnings calendar: {e}")
            return []
    
    def get_market_news(self, category: str = 'general') -> List[NewsItem]:
        """Get latest market news"""
        try:
            url = f"{self.base_url}/news"
            params = {
                'category': category,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            news_data = response.json()
            
            news_items = []
            for news in news_data:
                news_item = NewsItem(
                    headline=news.get('headline', ''),
                    summary=news.get('summary', ''),
                    source=news.get('source', ''),
                    timestamp=datetime.fromtimestamp(news.get('datetime', 0)),
                    url=news.get('url', ''),
                    related_symbols=news.get('related', '').split(',') if news.get('related') else []
                )
                news_items.append(news_item)
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching market news: {e}")
            return []
    
    def get_economic_calendar(self) -> List[Dict]:
        """Get economic calendar events"""
        try:
            url = f"{self.base_url}/calendar/economic"
            params = {
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            return response.json().get('economicCalendar', [])
            
        except Exception as e:
            logger.error(f"Error fetching economic calendar: {e}")
            return []
    
    def search_stocks(self, query: str) -> List[Dict]:
        """Search for stocks by name or symbol"""
        try:
            url = f"{self.base_url}/search"
            params = {
                'q': query,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            return response.json().get('result', [])
            
        except Exception as e:
            logger.error(f"Error searching stocks: {e}")
            return []
    
    def get_stock_candles(self, symbol: str, resolution: str = 'D', 
                         from_timestamp: int = None, to_timestamp: int = None) -> Dict:
        """Get historical stock data (OHLCV)"""
        try:
            if not from_timestamp:
                from_timestamp = int((datetime.now() - timedelta(days=365)).timestamp())
            if not to_timestamp:
                to_timestamp = int(datetime.now().timestamp())
                
            url = f"{self.base_url}/stock/candle"
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': from_timestamp,
                'to': to_timestamp,
                'token': self.api_key
            }
            
            response = requests.get(url, params=params)
            return response.json()
            
        except Exception as e:
            logger.error(f"Error fetching candles for {symbol}: {e}")
            return {}
    
    def add_price_callback(self, callback: Callable[[MarketData], None]):
        """Add callback for real-time price updates"""
        self.price_callbacks.append(callback)
    
    def add_news_callback(self, callback: Callable[[NewsItem], None]):
        """Add callback for real-time news updates"""
        self.news_callbacks.append(callback)
    
    def get_portfolio_analysis(self, symbols: List[str]) -> Dict:
        """Analyze a portfolio of stocks"""
        portfolio_data = {}
        
        for symbol in symbols:
            quote = self.get_quote(symbol)
            profile = self.get_company_profile(symbol)
            financials = self.get_basic_financials(symbol)
            recommendations = self.get_recommendation_trends(symbol)
            
            portfolio_data[symbol] = {
                'quote': quote,
                'profile': profile,
                'financials': financials,
                'recommendations': recommendations
            }
        
        return portfolio_data
    
    def disconnect(self):
        """Disconnect WebSocket and cleanup"""
        if self.ws:
            self.ws.close()
        self.is_streaming = False
        logger.info("Disconnected from market data stream")

class SmartMoneyMarketAnalyzer:
    """
    Advanced market analysis engine for Smart Money AI
    Combines real-time data with ML models for investment insights
    """
    
    def __init__(self, finnhub_streamer: FinnhubMarketDataStreamer):
        self.streamer = finnhub_streamer
        self.watchlist = []
        self.alerts = []
        
    def analyze_market_sentiment(self, symbols: List[str]) -> Dict:
        """Analyze overall market sentiment"""
        sentiment_data = {}
        
        for symbol in symbols:
            # Get recommendation trends
            recommendations = self.streamer.get_recommendation_trends(symbol)
            
            if recommendations:
                latest = recommendations[0]
                total_recs = (latest.get('strongBuy', 0) + latest.get('buy', 0) + 
                            latest.get('hold', 0) + latest.get('sell', 0) + 
                            latest.get('strongSell', 0))
                
                if total_recs > 0:
                    bullish_score = ((latest.get('strongBuy', 0) * 2 + latest.get('buy', 0)) / 
                                   total_recs) * 100
                    
                    sentiment_data[symbol] = {
                        'bullish_score': bullish_score,
                        'total_recommendations': total_recs,
                        'breakdown': latest
                    }
        
        return sentiment_data
    
    def generate_investment_signals(self, symbol: str) -> Dict:
        """Generate investment signals based on multiple factors"""
        signals = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'signals': []
        }
        
        # Get current data
        quote = self.streamer.get_quote(symbol)
        financials = self.streamer.get_basic_financials(symbol)
        recommendations = self.streamer.get_recommendation_trends(symbol)
        
        if quote:
            # Price momentum signal
            if quote.change_percent > 5:
                signals['signals'].append({
                    'type': 'STRONG_BUY',
                    'reason': f'Strong upward momentum (+{quote.change_percent:.2f}%)',
                    'strength': 0.8
                })
            elif quote.change_percent > 2:
                signals['signals'].append({
                    'type': 'BUY',
                    'reason': f'Positive momentum (+{quote.change_percent:.2f}%)',
                    'strength': 0.6
                })
            elif quote.change_percent < -5:
                signals['signals'].append({
                    'type': 'STRONG_SELL',
                    'reason': f'Sharp decline ({quote.change_percent:.2f}%)',
                    'strength': 0.8
                })
        
        # Financial metrics signals
        if financials and 'metric' in financials:
            metrics = financials['metric']
            
            # P/E ratio analysis
            pe_ratio = metrics.get('peBasicExclExtraTTM')
            if pe_ratio and pe_ratio < 15:
                signals['signals'].append({
                    'type': 'BUY',
                    'reason': f'Low P/E ratio ({pe_ratio:.2f})',
                    'strength': 0.7
                })
            
            # Beta analysis for risk assessment
            beta = metrics.get('beta')
            if beta and beta > 1.5:
                signals['signals'].append({
                    'type': 'RISK_WARNING',
                    'reason': f'High volatility (Beta: {beta:.2f})',
                    'strength': 0.6
                })
        
        # Analyst recommendations
        if recommendations:
            latest = recommendations[0]
            strong_buy = latest.get('strongBuy', 0)
            buy = latest.get('buy', 0)
            total = sum([latest.get(key, 0) for key in ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']])
            
            if total > 0:
                bullish_ratio = (strong_buy + buy) / total
                if bullish_ratio > 0.7:
                    signals['signals'].append({
                        'type': 'BUY',
                        'reason': f'Strong analyst consensus ({bullish_ratio*100:.0f}% bullish)',
                        'strength': 0.8
                    })
        
        return signals
    
    def create_watchlist_alerts(self, symbols: List[str], price_thresholds: Dict[str, Dict]):
        """Create price alerts for watchlist"""
        def price_alert_callback(market_data: MarketData):
            symbol = market_data.symbol
            if symbol in price_thresholds:
                thresholds = price_thresholds[symbol]
                
                if 'upper' in thresholds and market_data.price >= thresholds['upper']:
                    alert = {
                        'type': 'PRICE_ALERT',
                        'symbol': symbol,
                        'message': f'{symbol} reached upper threshold: ${market_data.price:.2f}',
                        'timestamp': datetime.now()
                    }
                    self.alerts.append(alert)
                    logger.info(f"ALERT: {alert['message']}")
                
                if 'lower' in thresholds and market_data.price <= thresholds['lower']:
                    alert = {
                        'type': 'PRICE_ALERT',
                        'symbol': symbol,
                        'message': f'{symbol} reached lower threshold: ${market_data.price:.2f}',
                        'timestamp': datetime.now()
                    }
                    self.alerts.append(alert)
                    logger.info(f"ALERT: {alert['message']}")
        
        # Add callback for price monitoring
        self.streamer.add_price_callback(price_alert_callback)
        
        # Subscribe to symbols
        for symbol in symbols:
            self.streamer.subscribe_to_symbol(symbol)
    
    def get_sector_analysis(self, sector_symbols: List[str]) -> Dict:
        """Analyze sector performance"""
        sector_data = {}
        total_change = 0
        count = 0
        
        for symbol in sector_symbols:
            quote = self.streamer.get_quote(symbol)
            if quote:
                sector_data[symbol] = {
                    'price': quote.price,
                    'change_percent': quote.change_percent
                }
                total_change += quote.change_percent
                count += 1
        
        avg_change = total_change / count if count > 0 else 0
        
        return {
            'symbols': sector_data,
            'average_change': avg_change,
            'sector_sentiment': 'Bullish' if avg_change > 1 else 'Bearish' if avg_change < -1 else 'Neutral'
        }

# Example usage and integration with Smart Money AI
if __name__ == "__main__":
    # Initialize with your API key
    API_KEY = "d3o93nhr01qmj830b3mgd3o93nhr01qmj830b3n0"
    
    # Create market data streamer
    streamer = FinnhubMarketDataStreamer(API_KEY)
    
    # Create analyzer
    analyzer = SmartMoneyMarketAnalyzer(streamer)
    
    # Connect to real-time data
    streamer.connect_websocket()
    
    # Test with popular stocks
    test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
    
    print("🔥 Smart Money AI - Real-Time Market Analysis Started")
    print("=" * 60)
    
    # Get current quotes
    for symbol in test_symbols:
        quote = streamer.get_quote(symbol)
        if quote:
            print(f"{symbol}: ${quote.price:.2f} ({quote.change_percent:+.2f}%)")
    
    # Analyze market sentiment
    sentiment = analyzer.analyze_market_sentiment(test_symbols)
    print(f"\nMarket Sentiment Analysis:")
    for symbol, data in sentiment.items():
        print(f"{symbol}: {data['bullish_score']:.1f}% bullish")
    
    # Generate investment signals
    print(f"\nInvestment Signals:")
    for symbol in test_symbols:
        signals = analyzer.generate_investment_signals(symbol)
        if signals['signals']:
            print(f"\n{symbol}:")
            for signal in signals['signals']:
                print(f"  • {signal['type']}: {signal['reason']}")
    
    print("\n✅ Real-time market data integration ready!")
    print("Your Smart Money AI system can now make predictions with live data!")