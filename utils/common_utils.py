"""
Common Utility Functions for Smart Money AI
==========================================

Collection of utility functions used across the Smart Money AI system.
"""

import re
import json
import hashlib
import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from pathlib import Path

def sanitize_amount(amount_str: str) -> float:
    """
    Clean and convert amount string to float
    
    Args:
        amount_str: String containing amount (e.g., "Rs.2,500.00", "₹1,23,456")
    
    Returns:
        float: Cleaned amount value
    """
    if not amount_str:
        return 0.0
    
    # Remove currency symbols and common prefixes
    cleaned = re.sub(r'[Rs.₹,\s]', '', str(amount_str))
    
    # Extract numeric value
    match = re.search(r'[\d.]+', cleaned)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return 0.0
    
    return 0.0

def extract_date_from_text(text: str) -> Optional[str]:
    """
    Extract date from text in various formats
    
    Args:
        text: Text containing date
    
    Returns:
        str: ISO format date string or None
    """
    # Common date patterns
    patterns = [
        r'(\d{1,2})-([A-Za-z]{3})-(\d{2,4})',  # 15-JUL-23
        r'(\d{1,2})/(\d{1,2})/(\d{2,4})',      # 15/07/23
        r'(\d{4})-(\d{1,2})-(\d{1,2})',        # 2023-07-15
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                if '-' in match.group() and len(match.group(2)) == 3:
                    # Format: 15-JUL-23
                    day, month_str, year = match.groups()
                    month_map = {
                        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
                        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
                    }
                    month = month_map.get(month_str.upper(), 1)
                    year = int(year)
                    if year < 100:
                        year += 2000
                    
                    date = datetime.date(year, month, int(day))
                    return date.isoformat()
                    
                elif '/' in match.group():
                    # Format: 15/07/23
                    day, month, year = map(int, match.groups())
                    if year < 100:
                        year += 2000
                    
                    date = datetime.date(year, month, day)
                    return date.isoformat()
                    
                elif len(match.groups()) == 3 and match.group(1).isdigit() and len(match.group(1)) == 4:
                    # Format: 2023-07-15
                    year, month, day = map(int, match.groups())
                    date = datetime.date(year, month, day)
                    return date.isoformat()
                    
            except (ValueError, TypeError):
                continue
    
    return None

def calculate_portfolio_metrics(portfolio_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate portfolio performance metrics
    
    Args:
        portfolio_data: List of portfolio holdings with price and change data
    
    Returns:
        dict: Portfolio metrics
    """
    if not portfolio_data:
        return {
            'total_value': 0,
            'total_change': 0,
            'total_change_percent': 0,
            'positive_performers': 0,
            'negative_performers': 0,
            'average_change': 0
        }
    
    total_value = 0
    total_change = 0
    positive_count = 0
    negative_count = 0
    changes = []
    
    for holding in portfolio_data:
        price = holding.get('price', 0)
        change_percent = holding.get('change_percent', 0)
        quantity = holding.get('quantity', 1)
        
        holding_value = price * quantity
        total_value += holding_value
        
        holding_change = holding_value * (change_percent / 100)
        total_change += holding_change
        
        if change_percent > 0:
            positive_count += 1
        elif change_percent < 0:
            negative_count += 1
            
        changes.append(change_percent)
    
    total_change_percent = (total_change / total_value * 100) if total_value > 0 else 0
    average_change = np.mean(changes) if changes else 0
    
    return {
        'total_value': total_value,
        'total_change': total_change,
        'total_change_percent': total_change_percent,
        'positive_performers': positive_count,
        'negative_performers': negative_count,
        'average_change': average_change,
        'total_holdings': len(portfolio_data)
    }

def calculate_risk_metrics(prices: List[float], returns: List[float] = None) -> Dict[str, Any]:
    """
    Calculate risk metrics for investment analysis
    
    Args:
        prices: List of historical prices
        returns: List of returns (optional, calculated if not provided)
    
    Returns:
        dict: Risk metrics
    """
    if len(prices) < 2:
        return {
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'beta': 0
        }
    
    if returns is None:
        returns = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
    
    if not returns:
        return {
            'volatility': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'beta': 0
        }
    
    # Calculate volatility (standard deviation of returns)
    volatility = np.std(returns) * np.sqrt(252)  # Annualized
    
    # Calculate Sharpe ratio (assuming 3% risk-free rate)
    risk_free_rate = 0.03
    mean_return = np.mean(returns) * 252  # Annualized
    sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod([1 + r for r in returns])
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)
    
    return {
        'volatility': volatility,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': abs(max_drawdown),
        'mean_return': mean_return,
        'total_returns': len(returns)
    }

def format_currency(amount: float, currency: str = "INR") -> str:
    """
    Format amount as currency string
    
    Args:
        amount: Amount to format
        currency: Currency code (INR, USD, etc.)
    
    Returns:
        str: Formatted currency string
    """
    if currency == "INR":
        return f"₹{amount:,.2f}"
    elif currency == "USD":
        return f"${amount:,.2f}"
    else:
        return f"{amount:,.2f} {currency}"

def generate_transaction_id(text: str) -> str:
    """
    Generate unique transaction ID from SMS text
    
    Args:
        text: SMS text
    
    Returns:
        str: Unique transaction ID
    """
    # Create hash of the text
    hash_object = hashlib.md5(text.encode())
    return hash_object.hexdigest()[:12]

def categorize_expense(merchant: str, amount: float) -> Dict[str, Any]:
    """
    Basic expense categorization based on merchant name and amount
    
    Args:
        merchant: Merchant name
        amount: Transaction amount
    
    Returns:
        dict: Category prediction with confidence
    """
    merchant_lower = merchant.lower()
    
    # Define category keywords
    categories = {
        'food_dining': ['restaurant', 'cafe', 'coffee', 'pizza', 'burger', 'food', 'zomato', 'swiggy'],
        'transportation': ['uber', 'ola', 'taxi', 'metro', 'bus', 'petrol', 'fuel', 'parking'],
        'entertainment': ['movie', 'cinema', 'theater', 'spotify', 'netflix', 'game'],
        'utilities': ['electricity', 'gas', 'water', 'internet', 'mobile', 'phone'],
        'shopping': ['amazon', 'flipkart', 'mall', 'store', 'shop', 'myntra', 'ajio'],
        'healthcare': ['hospital', 'pharmacy', 'doctor', 'medical', 'apollo', 'clinic'],
        'education': ['school', 'college', 'university', 'course', 'book', 'tuition']
    }
    
    # Score each category
    category_scores = {}
    for category, keywords in categories.items():
        score = sum(1 for keyword in keywords if keyword in merchant_lower)
        if score > 0:
            category_scores[category] = score
    
    # Determine best category
    if category_scores:
        best_category = max(category_scores, key=category_scores.get)
        confidence = min(category_scores[best_category] / 3, 1.0)  # Max confidence 1.0
        
        return {
            'category': best_category,
            'confidence': confidence,
            'all_scores': category_scores
        }
    else:
        return {
            'category': 'others',
            'confidence': 0.3,
            'all_scores': {}
        }

def validate_api_response(response: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate API response has required fields
    
    Args:
        response: API response dictionary
        required_fields: List of required field names
    
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(response, dict):
        return False
    
    if 'error' in response:
        return False
    
    return all(field in response for field in required_fields)

def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert value to float with default fallback
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        float: Converted value or default
    """
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int_conversion(value: Any, default: int = 0) -> int:
    """
    Safely convert value to integer with default fallback
    
    Args:
        value: Value to convert
        default: Default value if conversion fails
    
    Returns:
        int: Converted value or default
    """
    try:
        if value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

def create_model_directory(model_path: str) -> bool:
    """
    Create directory for model storage if it doesn't exist
    
    Args:
        model_path: Path to model directory
    
    Returns:
        bool: True if created or exists, False if failed
    """
    try:
        Path(model_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception:
        return False

def load_json_file(file_path: str, default: Any = None) -> Any:
    """
    Safely load JSON file with default fallback
    
    Args:
        file_path: Path to JSON file
        default: Default value if loading fails
    
    Returns:
        Any: Loaded data or default
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return default if default is not None else {}

def save_json_file(data: Any, file_path: str) -> bool:
    """
    Safely save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save file
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create directory if it doesn't exist
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False

def get_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.datetime.now().isoformat()

def format_percentage(value: float) -> str:
    """Format percentage with appropriate sign and precision"""
    if value > 0:
        return f"+{value:.2f}%"
    else:
        return f"{value:.2f}%"

def truncate_text(text: str, max_length: int = 100) -> str:
    """Truncate text to specified length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length-3] + "..."

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass