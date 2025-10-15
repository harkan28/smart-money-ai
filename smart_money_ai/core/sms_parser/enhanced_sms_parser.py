#!/usr/bin/env python3
"""
Enhanced SMS Transaction Parser
===============================

Advanced SMS parsing with configuration-based patterns, OTP detection,
and improved accuracy for Indian banking transactions.
Supports 60+ banks and UPI providers with intelligent fallback mechanisms.
"""

import re
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Transaction:
    """Enhanced transaction data structure"""
    amount: float
    merchant: str
    account_number: str
    transaction_id: str
    timestamp: datetime
    transaction_type: str  # debit/credit
    bank_name: str
    balance: Optional[float] = None
    upi_id: Optional[str] = None
    card_number: Optional[str] = None
    reference_number: Optional[str] = None
    confidence_score: float = 0.0
    raw_message: str = ""
    sender: str = ""
    category_hint: Optional[str] = None


@dataclass
class OTPInfo:
    """OTP information structure"""
    type: str  # transaction_otp, login_otp
    otp: str
    amount: Optional[float] = None
    raw_text: str = ""
    sender: str = ""
    timestamp: datetime = None


class BankPatternManager:
    """Enhanced pattern management for different banks with configuration file support"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self.patterns = {}
        self.upi_patterns = {}
        self.otp_patterns = {}
        self.fallback_patterns = {}
        self._load_patterns()
    
    def _get_default_config_path(self) -> str:
        """Get default config path relative to this file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(os.path.dirname(os.path.dirname(current_dir)), 'config', 'bank_patterns.json')
    
    def _load_patterns(self):
        """Load patterns from configuration file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    
                self.patterns = config.get('banks', {})
                self.upi_patterns = config.get('upi_providers', {})
                self.otp_patterns = config.get('otp_patterns', {})
                self.fallback_patterns = config.get('fallback_patterns', {})
                
                logger.info(f"Loaded patterns for {len(self.patterns)} banks and {len(self.upi_patterns)} UPI providers")
            else:
                logger.warning(f"Config file not found at {self.config_path}, using default patterns")
                self._load_default_patterns()
        except Exception as e:
            logger.error(f"Error loading patterns from config: {e}")
            self._load_default_patterns()
    
    def _load_default_patterns(self):
        """Load basic default patterns as fallback"""
        self.patterns = {
            'HDFC': {
                'identifiers': ['HDFCBK', 'HDFC', 'HDFCBANK'],
                'debit_patterns': [
                    r'HDFC Bank: Rs\.?(\d+\.?\d*) debited from A/c \*\*(\d+) on ([\d\-]+) at (.+?)\(UPI\)',
                    r'HDFC Bank: Rs\.?(\d+\.?\d*) debited from A/c \*\*(\d+) on ([\d\-]+) for (.+?) Avbl Bal'
                ],
                'credit_patterns': [
                    r'HDFC Bank: Rs\.?(\d+\.?\d*) credited to A/c \*\*(\d+) on ([\d\-]+) from (.+)'
                ]
            },
            'SBI': {
                'identifiers': ['SBIINB', 'SBICRD', 'SBI', 'SBIUPI'],
                'debit_patterns': [
                    r'SBI: Rs (\d+\.?\d*) debited from A/c \*\*(\d+) on (\w+) for UPI/(.+?)/(.+?)@.+? Ref# (\w+)'
                ],
                'credit_patterns': [
                    r'SBI: Rs (\d+\.?\d*) credited to A/c \*\*(\d+) on (\w+) from (.+)'
                ]
            }
        }
        
        self.fallback_patterns = {
            'generic_debit': [
                r'Rs\.?\s*(\d+\.?\d*)\s*debited',
                r'Amount\s*Rs\.?\s*(\d+\.?\d*)\s*debited'
            ],
            'generic_credit': [
                r'Rs\.?\s*(\d+\.?\d*)\s*credited',
                r'Amount\s*Rs\.?\s*(\d+\.?\d*)\s*credited'
            ]
        }
    
    def reload_patterns(self):
        """Reload patterns from configuration file"""
        self._load_patterns()
    
    def add_custom_pattern(self, bank: str, transaction_type: str, pattern: str):
        """Add custom pattern for a specific bank"""
        if bank not in self.patterns:
            self.patterns[bank] = {'identifiers': [bank.upper()], 'debit_patterns': [], 'credit_patterns': []}
        
        pattern_key = f'{transaction_type}_patterns'
        if pattern_key not in self.patterns[bank]:
            self.patterns[bank][pattern_key] = []
        
        self.patterns[bank][pattern_key].append(pattern)
        logger.info(f"Added custom {transaction_type} pattern for {bank}")
    
    def identify_bank(self, sender: str, message: str) -> Optional[str]:
        """Identify bank from sender ID or message content"""
        sender_upper = sender.upper()
        message_upper = message.upper()
        
        # Check each bank's identifiers
        for bank, config in self.patterns.items():
            for identifier in config.get('identifiers', []):
                if identifier in sender_upper or identifier in message_upper:
                    return bank
        
        # Check UPI providers
        for provider, config in self.upi_patterns.items():
            for identifier in config.get('identifiers', []):
                if identifier in sender_upper or identifier in message_upper:
                    return 'UPI'
        
        return None
    
    def get_patterns(self, bank: str, transaction_type: str) -> List[str]:
        """Get patterns for specific bank and transaction type"""
        if bank == 'UPI':
            patterns = []
            for provider, config in self.upi_patterns.items():
                patterns.extend(config.get('debit_patterns', []))
            return patterns
        
        if bank in self.patterns:
            return self.patterns[bank].get(f'{transaction_type}_patterns', [])
        
        # Return fallback patterns
        fallback_key = f'generic_{transaction_type}'
        return self.fallback_patterns.get(fallback_key, [])
    
    def detect_otp(self, message: str) -> Optional[OTPInfo]:
        """Detect OTP from message"""
        for otp_type, patterns in self.otp_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    groups = match.groups()
                    if len(groups) >= 1:
                        otp_info = OTPInfo(
                            type=otp_type,
                            otp=groups[0] if otp_type.endswith('otp') else groups[1],
                            amount=float(groups[1]) if otp_type.endswith('otp') and len(groups) > 1 else (float(groups[0]) if not otp_type.endswith('otp') else None),
                            raw_text=message,
                            timestamp=datetime.now()
                        )
                        return otp_info
        return None


class EnhancedSMSParser:
    """Enhanced SMS transaction parser with improved accuracy and features"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.pattern_manager = BankPatternManager(config_path)
        self.transaction_cache = {}
        self.parsing_stats = {
            'total_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'otp_detected': 0,
            'by_bank': {}
        }
    
    def parse_sms(self, sender: str, message: str, timestamp: Optional[datetime] = None) -> Optional[Transaction]:
        """Parse SMS message into transaction object with enhanced accuracy"""
        self.parsing_stats['total_processed'] += 1
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Clean message
        cleaned_message = self._clean_message(message)
        
        # Check if it's an OTP
        otp_info = self.pattern_manager.detect_otp(cleaned_message)
        if otp_info:
            self.parsing_stats['otp_detected'] += 1
            logger.info(f"OTP detected: {otp_info.type}")
            return None  # OTPs are not transactions
        
        # Identify bank
        bank = self.pattern_manager.identify_bank(sender, cleaned_message)
        if not bank:
            bank = 'UNKNOWN'
        
        # Track by bank
        if bank not in self.parsing_stats['by_bank']:
            self.parsing_stats['by_bank'][bank] = {'success': 0, 'failure': 0}
        
        # Determine transaction type
        transaction_type = self._determine_transaction_type(cleaned_message)
        
        # Get patterns for this bank and transaction type
        patterns = self.pattern_manager.get_patterns(bank, transaction_type)
        
        # Try to parse with bank-specific patterns
        transaction = self._parse_with_patterns(
            patterns, cleaned_message, bank, transaction_type, sender, timestamp
        )
        
        if not transaction:
            # Try fallback parsing
            transaction = self._fallback_parse(
                cleaned_message, bank, transaction_type, sender, timestamp
            )
        
        if transaction:
            transaction.raw_message = message
            transaction.sender = sender
            transaction.confidence_score = self._calculate_confidence(transaction, cleaned_message)
            
            self.parsing_stats['successful_parses'] += 1
            self.parsing_stats['by_bank'][bank]['success'] += 1
            
            logger.info(f"Successfully parsed transaction: {transaction.amount} {transaction_type} from {bank}")
            return transaction
        else:
            self.parsing_stats['failed_parses'] += 1
            self.parsing_stats['by_bank'][bank]['failure'] += 1
            logger.warning(f"Failed to parse message from {sender}: {message[:100]}...")
            return None
    
    def _clean_message(self, message: str) -> str:
        """Clean and normalize SMS message"""
        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', message.strip())
        
        # Normalize currency symbols
        cleaned = re.sub(r'INR\.?', 'Rs.', cleaned)
        cleaned = re.sub(r'₹\.?', 'Rs.', cleaned)
        
        # Normalize account number patterns
        cleaned = re.sub(r'A/C\s*NO\.?\s*', 'A/c **', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'ACCOUNT\s*NO\.?\s*', 'A/c **', cleaned, flags=re.IGNORECASE)
        
        return cleaned
    
    def _determine_transaction_type(self, message: str) -> str:
        """Determine if transaction is debit or credit"""
        message_lower = message.lower()
        
        debit_keywords = ['debited', 'withdrawn', 'spent', 'paid', 'purchase', 'debit']
        credit_keywords = ['credited', 'received', 'deposited', 'credit', 'refund']
        
        debit_score = sum(1 for keyword in debit_keywords if keyword in message_lower)
        credit_score = sum(1 for keyword in credit_keywords if keyword in message_lower)
        
        return 'debit' if debit_score >= credit_score else 'credit'
    
    def _parse_with_patterns(self, patterns: List[str], message: str, bank: str, 
                           transaction_type: str, sender: str, timestamp: datetime) -> Optional[Transaction]:
        """Parse message using specific patterns"""
        for pattern in patterns:
            try:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    return self._extract_transaction_data(match, bank, transaction_type, timestamp)
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern}, error: {e}")
                continue
        
        return None
    
    def _extract_transaction_data(self, match: re.Match, bank: str, 
                                transaction_type: str, timestamp: datetime) -> Transaction:
        """Extract transaction data from regex match"""
        groups = match.groups()
        
        # Default values
        amount = 0.0
        merchant = "Unknown"
        account_number = ""
        transaction_id = ""
        
        # Extract amount (usually first group)
        if len(groups) > 0:
            amount_str = groups[0].replace(',', '')
            try:
                amount = float(amount_str)
            except ValueError:
                amount = 0.0
        
        # Extract account number (usually second group)
        if len(groups) > 1:
            account_number = groups[1]
        
        # Extract merchant/description (varies by pattern)
        if len(groups) > 3:
            merchant = groups[3]
        elif len(groups) > 2:
            merchant = groups[2]
        
        # Extract transaction ID (often last group)
        if len(groups) > 4:
            transaction_id = groups[-1]
        
        # Clean merchant name
        merchant = self._clean_merchant_name(merchant)
        
        return Transaction(
            amount=amount,
            merchant=merchant,
            account_number=account_number,
            transaction_id=transaction_id or f"{bank}_{timestamp.strftime('%Y%m%d%H%M%S')}",
            timestamp=timestamp,
            transaction_type=transaction_type,
            bank_name=bank
        )
    
    def _fallback_parse(self, message: str, bank: str, transaction_type: str, 
                       sender: str, timestamp: datetime) -> Optional[Transaction]:
        """Fallback parsing using generic patterns"""
        patterns = self.pattern_manager.get_patterns('UNKNOWN', transaction_type)
        
        for pattern in patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                amount_str = match.group(1).replace(',', '')
                try:
                    amount = float(amount_str)
                    
                    # Extract merchant from context
                    merchant = self._extract_merchant_fallback(message)
                    
                    return Transaction(
                        amount=amount,
                        merchant=merchant,
                        account_number="UNKNOWN",
                        transaction_id=f"FALLBACK_{timestamp.strftime('%Y%m%d%H%M%S')}",
                        timestamp=timestamp,
                        transaction_type=transaction_type,
                        bank_name=bank,
                        confidence_score=0.5
                    )
                except ValueError:
                    continue
        
        return None
    
    def _extract_merchant_fallback(self, message: str) -> str:
        """Extract merchant name using fallback methods"""
        # Look for common merchant patterns
        merchant_patterns = self.pattern_manager.fallback_patterns.get('merchant_keywords', [])
        
        for pattern in merchant_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return self._clean_merchant_name(match.group(1))
        
        # If no pattern matches, try to extract from UPI patterns
        upi_match = re.search(r'UPI[/-](.+?)[/-]', message, re.IGNORECASE)
        if upi_match:
            return self._clean_merchant_name(upi_match.group(1))
        
        return "Unknown Merchant"
    
    def _clean_merchant_name(self, merchant: str) -> str:
        """Clean and normalize merchant name"""
        if not merchant:
            return "Unknown"
        
        # Remove common suffixes and prefixes
        merchant = re.sub(r'\(UPI\).*$', '', merchant)
        merchant = re.sub(r'Avbl.*$', '', merchant, flags=re.IGNORECASE)
        merchant = re.sub(r'Balance.*$', '', merchant, flags=re.IGNORECASE)
        merchant = re.sub(r'Ref.*$', '', merchant, flags=re.IGNORECASE)
        
        # Clean extra spaces and special characters
        merchant = re.sub(r'[^\w\s.-]', '', merchant)
        merchant = re.sub(r'\s+', ' ', merchant).strip()
        
        return merchant[:50] if len(merchant) > 50 else merchant
    
    def _calculate_confidence(self, transaction: Transaction, message: str) -> float:
        """Calculate confidence score for parsed transaction"""
        score = 0.0
        
        # Amount extraction confidence
        if transaction.amount > 0:
            score += 0.3
        
        # Bank identification confidence
        if transaction.bank_name != 'UNKNOWN':
            score += 0.2
        
        # Merchant extraction confidence
        if transaction.merchant != "Unknown" and transaction.merchant != "Unknown Merchant":
            score += 0.2
        
        # Account number confidence
        if transaction.account_number and transaction.account_number != "UNKNOWN":
            score += 0.1
        
        # Transaction ID confidence
        if transaction.transaction_id and not transaction.transaction_id.startswith('FALLBACK'):
            score += 0.1
        
        # Message completeness
        if len(message) > 50:  # Detailed messages usually more accurate
            score += 0.1
        
        return min(score, 1.0)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        total = self.parsing_stats['total_processed']
        success_rate = (self.parsing_stats['successful_parses'] / total * 100) if total > 0 else 0
        
        return {
            'total_processed': total,
            'successful_parses': self.parsing_stats['successful_parses'],
            'failed_parses': self.parsing_stats['failed_parses'],
            'success_rate': round(success_rate, 2),
            'otp_detected': self.parsing_stats['otp_detected'],
            'by_bank': self.parsing_stats['by_bank']
        }
    
    def reset_statistics(self):
        """Reset parsing statistics"""
        self.parsing_stats = {
            'total_processed': 0,
            'successful_parses': 0,
            'failed_parses': 0,
            'otp_detected': 0,
            'by_bank': {}
        }
    
    def export_transaction(self, transaction: Transaction) -> Dict[str, Any]:
        """Export transaction to dictionary format"""
        return asdict(transaction)
    
    def batch_parse(self, messages: List[Tuple[str, str, Optional[datetime]]]) -> List[Optional[Transaction]]:
        """Parse multiple SMS messages in batch"""
        results = []
        for sender, message, timestamp in messages:
            transaction = self.parse_sms(sender, message, timestamp)
            results.append(transaction)
        
        return results


# Convenience function for backward compatibility
def parse_sms(sender: str, message: str, timestamp: Optional[datetime] = None) -> Optional[Transaction]:
    """Parse a single SMS message"""
    parser = EnhancedSMSParser()
    return parser.parse_sms(sender, message, timestamp)


if __name__ == "__main__":
    # Test the enhanced parser
    parser = EnhancedSMSParser()
    
    # Test messages
    test_messages = [
        ("HDFCBK", "HDFC Bank: Rs.500.00 debited from A/c **1234 on 15-01-2024 at Amazon(UPI) Avbl Bal Rs.5000.00"),
        ("SBIUPI", "SBI: Rs 250 debited from A/c **5678 on 15Jan for UPI/PAYTM/user@paytm Ref# 123456"),
        ("PHONEPE", "PhonePe: You paid Rs.100 to Swiggy UPI ID: swiggy@ybl on 15-01-2024")
    ]
    
    for sender, message in test_messages:
        transaction = parser.parse_sms(sender, message)
        if transaction:
            print(f"✅ Parsed: {transaction.amount} {transaction.transaction_type} to {transaction.merchant}")
        else:
            print(f"❌ Failed to parse message from {sender}")
    
    print(f"\nStatistics: {parser.get_statistics()}")