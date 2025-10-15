"""
Smart Money AI - Core SMS Parser
Handles SMS transaction parsing for Indian banks
"""

import re
import json
from datetime import datetime

class SMSParser:
    """Parse SMS transactions from Indian banks"""
    
    def __init__(self):
        """Initialize SMS parser with bank patterns"""
        self.bank_patterns = self._load_bank_patterns()
    
    def _load_bank_patterns(self):
        """Load bank-specific patterns"""
        return {
            'default': {
                'amount_pattern': r'(?:rs\.?|inr|₹)\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                'merchant_pattern': r'(?:at|to|@)\s+([a-zA-Z0-9\s\-]+)',
                'balance_pattern': r'(?:bal|balance|avbl)[\s:]*(?:rs\.?|₹)?\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
            }
        }
    
    def parse_transaction(self, sms_text):
        """Parse SMS transaction text"""
        
        if not sms_text:
            return {'status': 'error', 'message': 'Empty SMS text'}
        
        sms_lower = sms_text.lower()
        
        # Extract amount
        amount_match = re.search(self.bank_patterns['default']['amount_pattern'], sms_lower)
        amount = 0.0
        if amount_match:
            amount_str = amount_match.group(1).replace(',', '')
            amount = float(amount_str)
        
        # Extract merchant
        merchant_match = re.search(self.bank_patterns['default']['merchant_pattern'], sms_lower)
        merchant = merchant_match.group(1).strip() if merchant_match else 'Unknown'
        
        # Determine transaction type
        if any(word in sms_lower for word in ['debited', 'spent', 'paid', 'purchase']):
            transaction_type = 'debit'
        elif any(word in sms_lower for word in ['credited', 'received', 'deposit']):
            transaction_type = 'credit'
        else:
            transaction_type = 'unknown'
        
        # Basic categorization
        category = self._categorize_transaction(merchant, sms_text)
        
        return {
            'status': 'success',
            'amount': amount,
            'merchant': merchant,
            'type': transaction_type,
            'category': category,
            'timestamp': datetime.now().isoformat(),
            'raw_text': sms_text
        }
    
    def _categorize_transaction(self, merchant, sms_text):
        """Basic transaction categorization"""
        
        merchant_lower = merchant.lower()
        sms_lower = sms_text.lower()
        
        # Category keywords
        categories = {
            'food': ['restaurant', 'cafe', 'food', 'zomato', 'swiggy', 'bigbasket'],
            'transport': ['uber', 'ola', 'metro', 'bus', 'taxi', 'fuel', 'petrol'],
            'shopping': ['amazon', 'flipkart', 'mall', 'store', 'shop'],
            'utilities': ['electricity', 'gas', 'water', 'internet', 'mobile'],
            'entertainment': ['movie', 'theater', 'netflix', 'spotify', 'game'],
            'health': ['hospital', 'pharmacy', 'medical', 'doctor', 'clinic'],
            'education': ['school', 'college', 'course', 'book', 'education']
        }
        
        for category, keywords in categories.items():
            if any(keyword in merchant_lower or keyword in sms_lower for keyword in keywords):
                return category
        
        return 'others'