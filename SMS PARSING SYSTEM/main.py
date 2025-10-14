#!/usr/bin/env python3
"""
Smart Money - Complete Integration System
SMS Parsing + ML Categorization + Real-time Processing
"""

import asyncio
import json
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'budgeting_ml_model', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'sms_parser'))

from core_parser import SMSParser, Transaction
from dataclasses import asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SmartMoneyIntegration:
    """Main integration class for Smart Money system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.sms_parser = SMSParser()
        self.ml_endpoint = None
        self.processed_transactions = []
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file"""
        # Default configuration
        default_config = {
            'ml_endpoint': 'http://localhost:8000/categorize',
            'database': {
                'type': 'json',
                'file': 'transactions.json'
            },
            'notifications': {
                'enabled': True,
                'webhook_url': None
            },
            'processing': {
                'batch_size': 100,
                'retry_attempts': 3
            }
        }
        
        # Try to load from file
        if Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    import yaml
                    user_config = yaml.safe_load(f)
                    default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        return default_config
    
    async def process_sms(self, sms_text: str, sender_id: str) -> Optional[Dict]:
        """
        Process a single SMS message
        
        Args:
            sms_text: SMS message content
            sender_id: SMS sender ID
            
        Returns:
            Processed transaction with ML categorization
        """
        try:
            # Parse SMS
            logger.info(f"Parsing SMS from {sender_id}")
            transaction = self.sms_parser.parse_sms(sms_text, sender_id)
            
            if not transaction:
                logger.warning("SMS parsing failed")
                return None
            
            # Convert to dict for ML processing
            transaction_dict = asdict(transaction)
            transaction_dict['timestamp'] = transaction.timestamp.isoformat()
            
            # Get ML categorization
            logger.info("Getting ML categorization")
            ml_result = await self._get_ml_categorization(transaction_dict)
            
            if ml_result:
                transaction_dict.update(ml_result)
            
            # Store transaction
            await self._store_transaction(transaction_dict)
            
            # Send notifications
            await self._send_notification(transaction_dict)
            
            logger.info("SMS processed successfully")
            return transaction_dict
            
        except Exception as e:
            logger.error(f"Error processing SMS: {e}")
            return None
    
    async def process_sms_batch(self, sms_batch: List[Dict]) -> List[Dict]:
        """
        Process multiple SMS messages
        
        Args:
            sms_batch: List of SMS data {'text': str, 'sender': str}
            
        Returns:
            List of processed transactions
        """
        results = []
        
        for sms_data in sms_batch:
            result = await self.process_sms(
                sms_data['text'], 
                sms_data['sender']
            )
            if result:
                results.append(result)
        
        logger.info(f"Processed {len(results)}/{len(sms_batch)} SMS messages")
        return results
    
    async def _get_ml_categorization(self, transaction: Dict) -> Optional[Dict]:
        """Get ML categorization for transaction"""
        try:
            # If no ML endpoint configured, use local model
            if not self.config.get('ml_endpoint'):
                return self._local_categorization(transaction)
            
            # Call ML API
            response = requests.post(
                self.config['ml_endpoint'],
                json={
                    'merchant': transaction['merchant'],
                    'description': transaction.get('description', ''),
                    'amount': transaction['amount']
                },
                timeout=5
            )
            
            if response.status_code == 200:
                ml_result = response.json()
                return {
                    'predicted_category': ml_result.get('category'),
                    'confidence': ml_result.get('confidence'),
                    'alternatives': ml_result.get('alternatives', [])
                }
            else:
                logger.warning(f"ML API error: {response.status_code}")
                return self._fallback_categorization(transaction)
                
        except Exception as e:
            logger.error(f"ML categorization error: {e}")
            return self._fallback_categorization(transaction)
    
    def _local_categorization(self, transaction: Dict) -> Dict:
        """Simple local categorization rules"""
        merchant = transaction['merchant'].lower()
        
        # Simple keyword-based categorization
        categories = {
            'FOOD_DINING': ['zomato', 'swiggy', 'dominos', 'pizza', 'restaurant', 'cafe', 'food'],
            'TRANSPORTATION': ['uber', 'ola', 'petrol', 'fuel', 'metro', 'taxi', 'bus'],
            'SHOPPING': ['amazon', 'flipkart', 'myntra', 'shop', 'store', 'mall'],
            'ENTERTAINMENT': ['netflix', 'spotify', 'movie', 'cinema', 'game'],
            'UTILITIES': ['electricity', 'phone', 'internet', 'gas', 'water'],
            'HEALTHCARE': ['hospital', 'pharmacy', 'medical', 'doctor'],
            'EDUCATION': ['school', 'college', 'course', 'book'],
            'INVESTMENT': ['mutual', 'sip', 'insurance', 'fd'],
            'MISCELLANEOUS': []
        }
        
        for category, keywords in categories.items():
            if any(keyword in merchant for keyword in keywords):
                return {
                    'predicted_category': category,
                    'confidence': 0.8,
                    'alternatives': []
                }
        
        return {
            'predicted_category': 'MISCELLANEOUS',
            'confidence': 0.5,
            'alternatives': []
        }
    
    def _fallback_categorization(self, transaction: Dict) -> Dict:
        """Fallback categorization when ML fails"""
        return self._local_categorization(transaction)
    
    async def _store_transaction(self, transaction: Dict) -> None:
        """Store transaction in database"""
        try:
            db_config = self.config.get('database', {})
            
            if db_config.get('type') == 'json':
                # Store in JSON file
                db_file = db_config.get('file', 'transactions.json')
                
                # Load existing data
                transactions = []
                if Path(db_file).exists():
                    with open(db_file, 'r') as f:
                        transactions = json.load(f)
                
                # Add new transaction
                transactions.append(transaction)
                
                # Save back to file
                with open(db_file, 'w') as f:
                    json.dump(transactions, f, indent=2, default=str)
                
                logger.info(f"Transaction stored in {db_file}")
            
            # Add to in-memory list
            self.processed_transactions.append(transaction)
            
        except Exception as e:
            logger.error(f"Error storing transaction: {e}")
    
    async def _send_notification(self, transaction: Dict) -> None:
        """Send notification for processed transaction"""
        try:
            if not self.config.get('notifications', {}).get('enabled'):
                return
            
            webhook_url = self.config.get('notifications', {}).get('webhook_url')
            if webhook_url:
                # Send webhook notification
                notification = {
                    'type': 'transaction_processed',
                    'data': transaction,
                    'timestamp': datetime.now().isoformat()
                }
                
                response = requests.post(webhook_url, json=notification, timeout=5)
                if response.status_code == 200:
                    logger.info("Notification sent successfully")
                else:
                    logger.warning(f"Notification failed: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Error sending notification: {e}")
    
    def get_spending_summary(self, days: int = 30) -> Dict:
        """Get spending summary for last N days"""
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_transactions = [
            tx for tx in self.processed_transactions
            if datetime.fromisoformat(tx['timestamp']) >= cutoff_date
        ]
        
        # Calculate summary
        total_spent = sum(tx['amount'] for tx in recent_transactions if tx['transaction_type'] == 'debit')
        total_received = sum(tx['amount'] for tx in recent_transactions if tx['transaction_type'] == 'credit')
        
        # Category-wise spending
        category_spending = {}
        for tx in recent_transactions:
            if tx['transaction_type'] == 'debit':
                category = tx.get('predicted_category', 'MISCELLANEOUS')
                category_spending[category] = category_spending.get(category, 0) + tx['amount']
        
        return {
            'period_days': days,
            'total_transactions': len(recent_transactions),
            'total_spent': total_spent,
            'total_received': total_received,
            'net_flow': total_received - total_spent,
            'category_spending': category_spending,
            'top_merchants': self._get_top_merchants(recent_transactions)
        }
    
    def _get_top_merchants(self, transactions: List[Dict], limit: int = 5) -> List[Dict]:
        """Get top merchants by spending"""
        merchant_spending = {}
        
        for tx in transactions:
            if tx['transaction_type'] == 'debit':
                merchant = tx['merchant']
                merchant_spending[merchant] = merchant_spending.get(merchant, 0) + tx['amount']
        
        # Sort by spending
        sorted_merchants = sorted(
            merchant_spending.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [
            {'merchant': merchant, 'amount': amount}
            for merchant, amount in sorted_merchants[:limit]
        ]


class SMSProcessingServer:
    """Server for processing SMS in real-time"""
    
    def __init__(self, port: int = 8080):
        self.port = port
        self.integration = SmartMoneyIntegration()
        
    async def start_server(self):
        """Start the SMS processing server"""
        from aiohttp import web, web_runner
        
        app = web.Application()
        
        # Routes
        app.router.add_post('/process-sms', self.handle_sms)
        app.router.add_post('/process-batch', self.handle_batch)
        app.router.add_get('/summary', self.handle_summary)
        app.router.add_get('/health', self.handle_health)
        
        # Start server
        runner = web_runner.AppRunner(app)
        await runner.setup()
        
        site = web_runner.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        
        logger.info(f"SMS Processing Server started on port {self.port}")
        
        # Keep running
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Server stopping...")
            await runner.cleanup()
    
    async def handle_sms(self, request):
        """Handle single SMS processing"""
        from aiohttp import web
        
        try:
            data = await request.json()
            sms_text = data.get('sms_text')
            sender_id = data.get('sender_id')
            
            if not sms_text or not sender_id:
                return web.json_response(
                    {'error': 'Missing sms_text or sender_id'}, 
                    status=400
                )
            
            result = await self.integration.process_sms(sms_text, sender_id)
            
            if result:
                return web.json_response({'success': True, 'transaction': result})
            else:
                return web.json_response({'success': False, 'error': 'Processing failed'})
                
        except Exception as e:
            logger.error(f"Error handling SMS: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_batch(self, request):
        """Handle batch SMS processing"""
        from aiohttp import web
        
        try:
            data = await request.json()
            sms_batch = data.get('sms_batch', [])
            
            results = await self.integration.process_sms_batch(sms_batch)
            
            return web.json_response({
                'success': True,
                'processed': len(results),
                'total': len(sms_batch),
                'transactions': results
            })
            
        except Exception as e:
            logger.error(f"Error handling batch: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_summary(self, request):
        """Handle spending summary request"""
        from aiohttp import web
        
        try:
            days = int(request.query.get('days', 30))
            summary = self.integration.get_spending_summary(days)
            return web.json_response(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return web.json_response({'error': str(e)}, status=500)
    
    async def handle_health(self, request):
        """Health check endpoint"""
        from aiohttp import web
        
        return web.json_response({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'processed_transactions': len(self.integration.processed_transactions)
        })


def demo_integration():
    """Demo the complete integration system"""
    print("🎯 SMART MONEY - COMPLETE INTEGRATION DEMO")
    print("=" * 60)
    
    # Initialize integration
    integration = SmartMoneyIntegration()
    
    # Sample SMS messages
    sample_sms = [
        {
            'text': "HDFC Bank: Rs.450.00 debited from A/c **1234 on 13-Oct-23 at ZOMATO MUMBAI(UPI). Avbl Bal: Rs.12,345.67",
            'sender': "HDFCBK"
        },
        {
            'text': "SBI: Rs 850 debited from A/c **5678 on 13Oct23 for UPI/SWIGGY/mumbai@oksbi. Ref# 123456789. Avl Bal Rs 45,678",
            'sender': "SBIINB"
        },
        {
            'text': "ICICI Bank A/c **3456 debited with Rs.675 on 13-Oct-23 UPI-ZOMATO. Balance: Rs.23,456.78",
            'sender': "ICICIB"
        },
        {
            'text': "PhonePe: You paid Rs.1,250 to SWIGGY UPI ID: swiggy@ybl on 13-Oct-23",
            'sender': "PHONEPE"
        },
    ]
    
    async def run_demo():
        print("\n📱 Processing Sample SMS Messages...")
        
        for i, sms_data in enumerate(sample_sms, 1):
            print(f"\n🔄 Processing SMS {i}/{len(sample_sms)}")
            print(f"   📧 From: {sms_data['sender']}")
            print(f"   💬 Text: {sms_data['text'][:60]}...")
            
            result = await integration.process_sms(
                sms_data['text'], 
                sms_data['sender']
            )
            
            if result:
                print(f"   ✅ Success:")
                print(f"      💰 Amount: ₹{result['amount']}")
                print(f"      🏪 Merchant: {result['merchant']}")
                print(f"      🏦 Bank: {result['bank_name']}")
                print(f"      📊 Category: {result.get('predicted_category', 'Unknown')}")
                print(f"      🎯 Confidence: {result.get('confidence', 0):.1%}")
            else:
                print("   ❌ Processing failed")
        
        # Generate summary
        print(f"\n📊 SPENDING SUMMARY")
        print("-" * 40)
        summary = integration.get_spending_summary(30)
        print(f"Total Transactions: {summary['total_transactions']}")
        print(f"Total Spent: ₹{summary['total_spent']:,.2f}")
        print(f"Total Received: ₹{summary['total_received']:,.2f}")
        print(f"Net Flow: ₹{summary['net_flow']:,.2f}")
        
        print(f"\n📈 Category-wise Spending:")
        for category, amount in summary['category_spending'].items():
            print(f"   {category}: ₹{amount:,.2f}")
        
        print(f"\n🏪 Top Merchants:")
        for merchant_data in summary['top_merchants']:
            print(f"   {merchant_data['merchant']}: ₹{merchant_data['amount']:,.2f}")
    
    # Run demo
    asyncio.run(run_demo())
    
    print(f"\n🎉 INTEGRATION DEMO COMPLETED!")
    print(f"🚀 Ready for production deployment!")


async def main():
    """Main function"""
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            demo_integration()
        elif sys.argv[1] == 'server':
            port = int(sys.argv[2]) if len(sys.argv) > 2 else 8080
            server = SMSProcessingServer(port)
            await server.start_server()
        else:
            print("Usage: python main.py [demo|server] [port]")
    else:
        demo_integration()


if __name__ == "__main__":
    asyncio.run(main())