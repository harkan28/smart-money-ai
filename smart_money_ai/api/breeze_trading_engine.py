"""
Breeze API Trading Engine Integration
====================================

Advanced trading automation using ICICI Direct's Breeze API
Integrates with Smart Money AI for automated investment execution

Features:
- Automated order placement based on ML recommendations
- Portfolio management and monitoring
- Real-time market data integration
- Risk management and position sizing
- Trade execution with Smart Money AI insights
"""

import requests
import json
import hashlib
import hmac
import base64
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreezeAPIError(Exception):
    """Custom exception for Breeze API errors"""
    pass

class BreezeChecksumGenerator:
    """Utility class for generating Breeze API checksums"""
    
    @staticmethod
    def generate_checksum(timestamp: str, json_data: str, secret_key: str) -> str:
        """Generate SHA256 checksum for Breeze API authentication"""
        message = timestamp + json_data + secret_key
        checksum = hashlib.sha256(message.encode()).hexdigest()
        return checksum

class BreezeTradeEngine:
    """
    Core trading engine using Breeze API
    Handles all trading operations and market data retrieval
    """
    
    def __init__(self, app_key: str, secret_key: str, session_token: str = None):
        """Initialize Breeze trading engine"""
        self.app_key = app_key
        self.secret_key = secret_key
        self.session_token = session_token
        self.base_url = "https://api.icicidirect.com/breezeapi/api/v1"
        self.websocket_url = "https://livefeeds.icicidirect.com/"
        
        # Rate limiting: 100 calls per minute, 5000 per day
        self.last_request_time = 0
        self.request_count = 0
        self.daily_request_count = 0
        self.last_reset_time = datetime.now()
        
        logger.info("🚀 Breeze Trading Engine initialized")
    
    def _get_headers(self, json_data: str = "{}") -> Dict[str, str]:
        """Generate request headers with checksum"""
        timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.000Z")
        checksum = BreezeChecksumGenerator.generate_checksum(
            timestamp, json_data, self.secret_key
        )
        
        headers = {
            'Content-Type': 'application/json',
            'X-Checksum': f"token {checksum}",
            'X-Timestamp': timestamp,
            'X-AppKey': self.app_key,
        }
        
        if self.session_token:
            headers['X-SessionToken'] = self.session_token
            
        return headers
    
    def _rate_limit_check(self):
        """Ensure API rate limits are respected"""
        current_time = time.time()
        
        # Reset daily counter if new day
        if datetime.now().date() > self.last_reset_time.date():
            self.daily_request_count = 0
            self.last_reset_time = datetime.now()
        
        # Check daily limit
        if self.daily_request_count >= 5000:
            raise BreezeAPIError("Daily API limit (5000) exceeded")
        
        # Check per-minute limit
        if current_time - self.last_request_time < 0.6:  # 60s/100 = 0.6s per request
            time.sleep(0.6 - (current_time - self.last_request_time))
        
        self.last_request_time = time.time()
        self.request_count += 1
        self.daily_request_count += 1
    
    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make authenticated request to Breeze API"""
        self._rate_limit_check()
        
        url = f"{self.base_url}/{endpoint}"
        json_data = json.dumps(data) if data else "{}"
        headers = self._get_headers(json_data)
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, json=data)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, json=data)
            elif method.upper() == 'PUT':
                response = requests.put(url, headers=headers, json=data)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, headers=headers, json=data)
            else:
                raise BreezeAPIError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('Status') != 200:
                error_msg = result.get('Error', 'Unknown error')
                raise BreezeAPIError(f"API Error: {error_msg}")
            
            return result.get('Success', {})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise BreezeAPIError(f"Request failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise BreezeAPIError(f"Unexpected error: {e}")
    
    def authenticate(self, api_session: str) -> str:
        """Complete authentication and get session token"""
        try:
            data = {
                "SessionToken": api_session,
                "AppKey": self.app_key
            }
            
            # Note: Customer details API doesn't require headers
            url = f"{self.base_url}/customerdetails"
            response = requests.get(url, json=data)
            response.raise_for_status()
            
            result = response.json()
            if result.get('Status') != 200:
                raise BreezeAPIError(f"Authentication failed: {result.get('Error')}")
            
            # Extract session token from response
            # You'll need to decode this based on Breeze API documentation
            self.session_token = api_session  # Simplified for demo
            
            logger.info("✅ Breeze API authentication successful")
            return self.session_token
            
        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            raise BreezeAPIError(f"Authentication failed: {e}")
    
    def get_customer_details(self) -> Dict:
        """Get customer details and trading permissions"""
        try:
            data = {
                "SessionToken": self.session_token,
                "AppKey": self.app_key
            }
            return self._make_request('GET', 'customerdetails', data)
        except Exception as e:
            logger.error(f"Failed to get customer details: {e}")
            return {'error': str(e)}
    
    def get_portfolio_holdings(self, exchange_code: str = "NSE") -> Dict:
        """Get current portfolio holdings"""
        try:
            data = {
                "exchange_code": exchange_code,
                "from_date": "",
                "to_date": "",
                "stock_code": "",
                "portfolio_type": ""
            }
            return self._make_request('GET', 'portfolioholdings', data)
        except Exception as e:
            logger.error(f"Failed to get portfolio holdings: {e}")
            return {'error': str(e)}
    
    def get_portfolio_positions(self) -> Dict:
        """Get current portfolio positions"""
        try:
            return self._make_request('GET', 'portfoliopositions')
        except Exception as e:
            logger.error(f"Failed to get portfolio positions: {e}")
            return {'error': str(e)}
    
    def get_funds(self) -> Dict:
        """Get available funds and margins"""
        try:
            return self._make_request('GET', 'funds')
        except Exception as e:
            logger.error(f"Failed to get funds: {e}")
            return {'error': str(e)}
    
    def get_quotes(self, stock_code: str, exchange_code: str = "NSE", 
                   product_type: str = "cash", expiry_date: str = "", 
                   right: str = "others", strike_price: str = "0") -> Dict:
        """Get real-time quotes for a stock"""
        try:
            data = {
                "stock_code": stock_code,
                "exchange_code": exchange_code,
                "expiry_date": expiry_date,
                "product_type": product_type,
                "right": right,
                "strike_price": strike_price
            }
            return self._make_request('GET', 'quotes', data)
        except Exception as e:
            logger.error(f"Failed to get quotes for {stock_code}: {e}")
            return {'error': str(e)}
    
    def place_order(self, stock_code: str, exchange_code: str, product: str,
                    action: str, order_type: str, quantity: str, price: str,
                    validity: str = "day", **kwargs) -> Dict:
        """Place a trading order"""
        try:
            data = {
                "stock_code": stock_code,
                "exchange_code": exchange_code,
                "product": product,
                "action": action,
                "order_type": order_type,
                "quantity": quantity,
                "price": price,
                "validity": validity,
                **kwargs
            }
            
            result = self._make_request('POST', 'order', data)
            logger.info(f"✅ Order placed: {stock_code} {action} {quantity} @ {price}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to place order: {e}")
            return {'error': str(e)}
    
    def get_order_status(self, order_id: str, exchange_code: str) -> Dict:
        """Get status of a specific order"""
        try:
            data = {
                "exchange_code": exchange_code,
                "order_id": order_id
            }
            return self._make_request('GET', 'order', data)
        except Exception as e:
            logger.error(f"Failed to get order status: {e}")
            return {'error': str(e)}
    
    def cancel_order(self, order_id: str, exchange_code: str) -> Dict:
        """Cancel an existing order"""
        try:
            data = {
                "order_id": order_id,
                "exchange_code": exchange_code
            }
            result = self._make_request('DELETE', 'order', data)
            logger.info(f"✅ Order cancelled: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to cancel order: {e}")
            return {'error': str(e)}
    
    def modify_order(self, order_id: str, exchange_code: str, 
                     quantity: str = None, price: str = None, **kwargs) -> Dict:
        """Modify an existing order"""
        try:
            data = {
                "order_id": order_id,
                "exchange_code": exchange_code,
                **kwargs
            }
            
            if quantity:
                data["quantity"] = quantity
            if price:
                data["price"] = price
                
            result = self._make_request('PUT', 'order', data)
            logger.info(f"✅ Order modified: {order_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to modify order: {e}")
            return {'error': str(e)}
    
    def get_historical_data(self, stock_code: str, exchange_code: str,
                           from_date: str, to_date: str, interval: str = "day",
                           product_type: str = "cash") -> Dict:
        """Get historical price data"""
        try:
            data = {
                "interval": interval,
                "from_date": from_date,
                "to_date": to_date,
                "stock_code": stock_code,
                "exchange_code": exchange_code,
                "product_type": product_type
            }
            return self._make_request('GET', 'historicalcharts', data)
        except Exception as e:
            logger.error(f"Failed to get historical data: {e}")
            return {'error': str(e)}

class SmartMoneyBreezeIntegration:
    """
    Integration layer between Smart Money AI and Breeze API
    Automates trading based on ML recommendations
    """
    
    def __init__(self, app_key: str, secret_key: str, session_token: str = None):
        """Initialize Smart Money Breeze integration"""
        self.breeze_engine = BreezeTradeEngine(app_key, secret_key, session_token)
        self.risk_manager = TradingRiskManager()
        
        # Trading parameters
        self.max_position_size = 0.05  # 5% of portfolio per position
        self.stop_loss_percentage = 0.02  # 2% stop loss
        self.take_profit_percentage = 0.06  # 6% take profit
        
        logger.info("🤖 Smart Money Breeze Integration initialized")
    
    def authenticate_with_session(self, api_session: str) -> bool:
        """Authenticate with Breeze API"""
        try:
            self.breeze_engine.authenticate(api_session)
            customer_details = self.breeze_engine.get_customer_details()
            
            if 'error' not in customer_details:
                logger.info("✅ Breeze API authentication successful")
                return True
            else:
                logger.error(f"❌ Authentication failed: {customer_details['error']}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False
    
    def get_portfolio_analysis(self) -> Dict:
        """Get comprehensive portfolio analysis"""
        try:
            holdings = self.breeze_engine.get_portfolio_holdings()
            positions = self.breeze_engine.get_portfolio_positions()
            funds = self.breeze_engine.get_funds()
            
            if any('error' in data for data in [holdings, positions, funds]):
                return {'error': 'Failed to fetch portfolio data'}
            
            # Analyze portfolio
            total_value = 0
            portfolio_stocks = []
            
            for holding in holdings:
                stock_code = holding.get('stock_code', '')
                quantity = float(holding.get('quantity', 0))
                current_price = float(holding.get('current_market_price', 0))
                
                if quantity > 0 and current_price > 0:
                    position_value = quantity * current_price
                    total_value += position_value
                    
                    portfolio_stocks.append({
                        'stock_code': stock_code,
                        'quantity': quantity,
                        'current_price': current_price,
                        'position_value': position_value,
                        'weight': 0  # Will calculate after total
                    })
            
            # Calculate weights
            for stock in portfolio_stocks:
                stock['weight'] = stock['position_value'] / total_value if total_value > 0 else 0
            
            analysis = {
                'total_portfolio_value': total_value,
                'available_funds': funds.get('total_bank_balance', 0),
                'portfolio_stocks': portfolio_stocks,
                'diversification_score': self._calculate_diversification_score(portfolio_stocks),
                'risk_metrics': self._calculate_portfolio_risk(portfolio_stocks)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Portfolio analysis failed: {e}")
            return {'error': str(e)}
    
    def execute_smart_money_recommendation(self, recommendation: Dict, 
                                          investment_amount: float = None) -> Dict:
        """Execute trading based on Smart Money AI recommendation"""
        try:
            if 'error' in recommendation:
                return {'error': 'Invalid recommendation provided'}
            
            action = recommendation.get('action', '').lower()
            symbols = recommendation.get('symbols', [])
            confidence = recommendation.get('confidence', 0)
            
            if confidence < 60:  # Minimum confidence threshold
                return {'error': f'Confidence too low: {confidence}%'}
            
            results = []
            
            for symbol_data in symbols:
                symbol = symbol_data.get('symbol', '')
                recommended_allocation = symbol_data.get('allocation', 0)
                
                if not symbol or recommended_allocation <= 0:
                    continue
                
                # Get current market data
                quote_data = self.breeze_engine.get_quotes(symbol)
                if 'error' in quote_data:
                    results.append({'symbol': symbol, 'error': quote_data['error']})
                    continue
                
                current_price = float(quote_data[0].get('ltp', 0))
                if current_price <= 0:
                    results.append({'symbol': symbol, 'error': 'Invalid price'})
                    continue
                
                # Calculate position size
                position_amount = investment_amount * (recommended_allocation / 100) if investment_amount else 10000
                quantity = int(position_amount / current_price)
                
                if quantity <= 0:
                    results.append({'symbol': symbol, 'error': 'Calculated quantity is zero'})
                    continue
                
                # Risk management checks
                risk_check = self.risk_manager.validate_trade({
                    'symbol': symbol,
                    'action': action,
                    'quantity': quantity,
                    'price': current_price,
                    'position_amount': position_amount
                })
                
                if not risk_check['approved']:
                    results.append({
                        'symbol': symbol, 
                        'error': f"Risk check failed: {risk_check['reason']}"
                    })
                    continue
                
                # Place order
                if action in ['buy', 'sell']:
                    order_result = self.breeze_engine.place_order(
                        stock_code=symbol,
                        exchange_code="NSE",
                        product="cash",
                        action=action,
                        order_type="market",
                        quantity=str(quantity),
                        price=str(current_price),
                        validity="day"
                    )
                    
                    if 'error' not in order_result:
                        results.append({
                            'symbol': symbol,
                            'action': action,
                            'quantity': quantity,
                            'price': current_price,
                            'order_id': order_result.get('order_id', ''),
                            'status': 'Order placed successfully'
                        })
                    else:
                        results.append({
                            'symbol': symbol,
                            'error': order_result['error']
                        })
            
            return {
                'execution_summary': {
                    'total_symbols': len(symbols),
                    'successful_orders': len([r for r in results if 'error' not in r]),
                    'failed_orders': len([r for r in results if 'error' in r]),
                    'recommendation_confidence': confidence
                },
                'order_results': results
            }
            
        except Exception as e:
            logger.error(f"Failed to execute recommendation: {e}")
            return {'error': str(e)}
    
    def monitor_positions_with_alerts(self) -> Dict:
        """Monitor positions and generate alerts based on Smart Money AI analysis"""
        try:
            positions = self.breeze_engine.get_portfolio_positions()
            if 'error' in positions:
                return positions
            
            alerts = []
            position_updates = []
            
            for position in positions:
                symbol = position.get('stock_code', '')
                quantity = float(position.get('quantity', 0))
                avg_price = float(position.get('average_price', 0))
                current_price = float(position.get('ltp', 0))
                
                if quantity == 0 or avg_price == 0 or current_price == 0:
                    continue
                
                # Calculate P&L
                pnl_percentage = ((current_price - avg_price) / avg_price) * 100
                pnl_amount = (current_price - avg_price) * quantity
                
                # Generate alerts based on thresholds
                if pnl_percentage <= -self.stop_loss_percentage * 100:
                    alerts.append({
                        'type': 'STOP_LOSS_TRIGGERED',
                        'symbol': symbol,
                        'pnl_percentage': pnl_percentage,
                        'recommendation': f'Consider selling {symbol} - Stop loss triggered'
                    })
                
                elif pnl_percentage >= self.take_profit_percentage * 100:
                    alerts.append({
                        'type': 'TAKE_PROFIT',
                        'symbol': symbol,
                        'pnl_percentage': pnl_percentage,
                        'recommendation': f'Consider booking profits for {symbol}'
                    })
                
                position_updates.append({
                    'symbol': symbol,
                    'quantity': quantity,
                    'avg_price': avg_price,
                    'current_price': current_price,
                    'pnl_percentage': pnl_percentage,
                    'pnl_amount': pnl_amount
                })
            
            return {
                'positions': position_updates,
                'alerts': alerts,
                'monitoring_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Position monitoring failed: {e}")
            return {'error': str(e)}
    
    def _calculate_diversification_score(self, portfolio_stocks: List[Dict]) -> float:
        """Calculate portfolio diversification score (0-100)"""
        if not portfolio_stocks:
            return 0
        
        # Simple diversification score based on number of stocks and weight distribution
        num_stocks = len(portfolio_stocks)
        weights = [stock['weight'] for stock in portfolio_stocks]
        
        # Herfindahl Index (lower is more diversified)
        hhi = sum(w**2 for w in weights)
        
        # Convert to score (0-100, higher is better)
        diversification_score = max(0, 100 - (hhi * 100))
        
        return round(diversification_score, 2)
    
    def _calculate_portfolio_risk(self, portfolio_stocks: List[Dict]) -> Dict:
        """Calculate basic portfolio risk metrics"""
        if not portfolio_stocks:
            return {'concentration_risk': 0, 'largest_position': 0}
        
        weights = [stock['weight'] for stock in portfolio_stocks]
        largest_position = max(weights) if weights else 0
        concentration_risk = 100 if largest_position > 0.2 else 50 if largest_position > 0.1 else 25
        
        return {
            'concentration_risk': concentration_risk,
            'largest_position': largest_position * 100,
            'risk_level': 'High' if concentration_risk > 75 else 'Medium' if concentration_risk > 40 else 'Low'
        }

class TradingRiskManager:
    """Risk management system for automated trading"""
    
    def __init__(self):
        self.max_single_position = 0.1  # 10% max per position
        self.max_daily_trades = 50
        self.max_daily_loss = 0.05  # 5% max daily loss
        
        self.daily_trades = 0
        self.daily_pnl = 0
        self.last_reset = datetime.now().date()
    
    def validate_trade(self, trade_data: Dict) -> Dict:
        """Validate if trade meets risk management criteria"""
        # Reset daily counters if new day
        if datetime.now().date() > self.last_reset:
            self.daily_trades = 0
            self.daily_pnl = 0
            self.last_reset = datetime.now().date()
        
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return {'approved': False, 'reason': 'Daily trade limit exceeded'}
        
        # Check position size
        position_amount = trade_data.get('position_amount', 0)
        portfolio_value = 100000  # This should come from actual portfolio value
        
        if position_amount / portfolio_value > self.max_single_position:
            return {'approved': False, 'reason': 'Position size too large'}
        
        # Check daily loss limit
        if self.daily_pnl < -self.max_daily_loss * portfolio_value:
            return {'approved': False, 'reason': 'Daily loss limit reached'}
        
        return {'approved': True, 'reason': 'Trade approved'}
    
    def update_trade_metrics(self, pnl: float):
        """Update daily trading metrics"""
        self.daily_trades += 1
        self.daily_pnl += pnl

# Example usage and testing functions
def create_breeze_demo():
    """Create a demo of Breeze API integration"""
    print("🚀 Breeze API Integration Demo")
    print("=" * 50)
    
    # Demo credentials (replace with actual)
    APP_KEY = "your_app_key"
    SECRET_KEY = "your_secret_key"
    API_SESSION = "your_api_session"
    
    try:
        # Initialize integration
        smart_breeze = SmartMoneyBreezeIntegration(APP_KEY, SECRET_KEY)
        
        # Authenticate
        if smart_breeze.authenticate_with_session(API_SESSION):
            print("✅ Authentication successful")
            
            # Get portfolio analysis
            portfolio = smart_breeze.get_portfolio_analysis()
            if 'error' not in portfolio:
                print(f"💰 Portfolio Value: ₹{portfolio['total_portfolio_value']:,.2f}")
                print(f"📊 Diversification Score: {portfolio['diversification_score']}/100")
            
            # Example Smart Money recommendation
            sample_recommendation = {
                'action': 'buy',
                'confidence': 75,
                'symbols': [
                    {'symbol': 'RELIANCE', 'allocation': 30},
                    {'symbol': 'TCS', 'allocation': 25},
                    {'symbol': 'HDFCBANK', 'allocation': 20}
                ]
            }
            
            # Execute recommendation (demo mode)
            print("\n🤖 Executing Smart Money AI recommendation...")
            execution_result = smart_breeze.execute_smart_money_recommendation(
                sample_recommendation, 
                investment_amount=100000
            )
            
            if 'error' not in execution_result:
                summary = execution_result['execution_summary']
                print(f"✅ Orders placed: {summary['successful_orders']}")
                print(f"❌ Failed orders: {summary['failed_orders']}")
            
        else:
            print("❌ Authentication failed")
            
    except Exception as e:
        print(f"❌ Demo error: {e}")

if __name__ == "__main__":
    create_breeze_demo()