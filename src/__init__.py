"""
Smart Money AI - Intelligent Financial Management System
=========================================================

A comprehensive AI-powered financial management platform that combines:
- SMS transaction parsing
- Machine learning expense categorization  
- Investment recommendation engine
- Behavioral finance analysis
- Predictive analytics
- Advanced sentiment analysis

Version: 2.0.0
Author: Smart Money AI Team
License: MIT
"""

__version__ = "2.0.0"
__author__ = "Smart Money AI Team"
__email__ = "contact@smartmoneyai.com"

from .core.smart_money_ai import SmartMoneyAI
from .parsers.sms_parser import SMSParser
from .ml_models.expense_categorizer import ExpenseCategorizer
from .investment.investment_engine import InvestmentEngine
from .analytics.behavioral_analyzer import BehavioralAnalyzer
from .analytics.predictive_analytics import PredictiveAnalytics

__all__ = [
    "SmartMoneyAI",
    "SMSParser", 
    "ExpenseCategorizer",
    "InvestmentEngine",
    "BehavioralAnalyzer",
    "PredictiveAnalytics"
]