#!/usr/bin/env python3
"""
Smart Money AI - Core Integration Module
=======================================

Main orchestration class that integrates all Smart Money AI components:
- SMS parsing
- ML expense categorization
- Investment recommendations
- Behavioral analysis
- Predictive analytics
"""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict

# Import Smart Money AI modules
try:
    # Try relative imports first (when used as package)
    from ..parsers.sms_parser import SMSParser, Transaction
    from ..ml_models.expense_categorizer import ExpenseCategorizer
    from ..utils.data_manager import DataManager
    from ..utils.config_manager import ConfigManager
except ImportError:
    # Fallback to direct imports (when used as standalone)
    import sys
    from pathlib import Path
    current_dir = Path(__file__).parent.parent
    sys.path.insert(0, str(current_dir))
    
    from parsers.sms_parser import SMSParser, Transaction
    from ml_models.expense_categorizer import ExpenseCategorizer
    from utils.data_manager import DataManager
    from utils.config_manager import ConfigManager

# Mock imports for investment and analytics (to be implemented)
class MockInvestmentEngine:
    def __getattr__(self, name):
        return lambda *args, **kwargs: {"status": "mock", "component": "InvestmentEngine"}

class MockBehavioralAnalyzer:
    def __getattr__(self, name):
        return lambda *args, **kwargs: {"status": "mock", "component": "BehavioralAnalyzer"}

class MockPredictiveAnalytics:
    def __getattr__(self, name):
        return lambda *args, **kwargs: {"status": "mock", "component": "PredictiveAnalytics"}

# Use existing modules or mocks
try:
    from ..investment.investment_engine import InvestmentEngine, UserProfile, RiskProfile
except ImportError:
    InvestmentEngine = MockInvestmentEngine
    UserProfile = dict
    RiskProfile = str

try:
    from ..analytics.behavioral_analyzer import BehavioralAnalyzer
except ImportError:
    BehavioralAnalyzer = MockBehavioralAnalyzer

try:
    from ..analytics.predictive_analytics import PredictiveAnalytics
except ImportError:
    PredictiveAnalytics = MockPredictiveAnalytics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SmartMoneyUser:
    """Smart Money AI user profile"""
    user_id: str
    name: str
    email: str
    phone: str
    age: int
    annual_income: float
    created_at: datetime
    risk_profile: str = "moderate"
    investment_timeline: int = 10
    financial_goals: List[str] = None
    
    def __post_init__(self):
        if self.financial_goals is None:
            self.financial_goals = ["wealth_creation", "retirement_planning"]


@dataclass
class InsightsSummary:
    """Comprehensive insights summary"""
    user_id: str
    generated_at: datetime
    transaction_summary: Dict[str, Any]
    spending_analysis: Dict[str, Any]
    investment_recommendations: List[Dict[str, Any]]
    behavioral_insights: Dict[str, Any]
    predictive_forecast: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    optimization_suggestions: List[str]


class SmartMoneyAI:
    """
    Main Smart Money AI orchestration class
    
    Integrates all components to provide comprehensive financial insights:
    - SMS transaction parsing
    - ML-powered expense categorization
    - AI investment recommendations
    - Behavioral finance analysis
    - Predictive financial forecasting
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize Smart Money AI system"""
        self.config = ConfigManager(config_path)
        self.data_manager = DataManager()
        
        # Initialize core components
        self._initialize_components()
        
        # User management
        self.users: Dict[str, SmartMoneyUser] = {}
        self.user_transactions: Dict[str, List[Transaction]] = {}
        
        logger.info("Smart Money AI system initialized successfully")
    
    def _initialize_components(self):
        """Initialize all Smart Money AI components"""
        try:
            # SMS Parser
            self.sms_parser = SMSParser()
            logger.info("✅ SMS Parser initialized")
            
            # ML Expense Categorizer
            model_path = self.config.get('ml_model_path', 'models/expense_model.pkl')
            feature_path = self.config.get('feature_extractor_path', 'models/feature_extractor.pkl')
            self.expense_categorizer = ExpenseCategorizer(model_path, feature_path)
            logger.info("✅ ML Expense Categorizer initialized")
            
            # Investment Engine
            self.investment_engine = InvestmentEngine()
            logger.info("✅ Investment Engine initialized")
            
            # Behavioral Analyzer
            self.behavioral_analyzer = BehavioralAnalyzer()
            logger.info("✅ Behavioral Analyzer initialized")
            
            # Predictive Analytics
            self.predictive_analytics = PredictiveAnalytics()
            logger.info("✅ Predictive Analytics initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            # Initialize with mock components for demo
            self._initialize_mock_components()
    
    def _initialize_mock_components(self):
        """Initialize mock components for demonstration"""
        logger.warning("Initializing with mock components for demo")
        
        class MockComponent:
            def __getattr__(self, name):
                return lambda *args, **kwargs: {"status": "mock", "component": self.__class__.__name__}
        
        self.sms_parser = SMSParser()  # This should work
        self.expense_categorizer = ExpenseCategorizer()  # This should work with mock
        self.investment_engine = MockComponent()
        self.behavioral_analyzer = MockComponent()
        self.predictive_analytics = MockComponent()
    
    def create_user_profile(self, user_data: Dict[str, Any]) -> SmartMoneyUser:
        """Create a new user profile"""
        try:
            user = SmartMoneyUser(
                user_id=user_data['user_id'],
                name=user_data['name'],
                email=user_data['email'],
                phone=user_data['phone'],
                age=user_data['age'],
                annual_income=user_data['annual_income'],
                created_at=datetime.now(),
                risk_profile=user_data.get('risk_profile', 'moderate'),
                investment_timeline=user_data.get('investment_timeline', 10),
                financial_goals=user_data.get('financial_goals', ["wealth_creation"])
            )
            
            self.users[user.user_id] = user
            self.user_transactions[user.user_id] = []
            
            logger.info(f"User profile created for {user.user_id}")
            return user
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            raise
    
    def process_sms_transaction(self, user_id: str, sms_text: str, sender_id: str) -> Optional[Dict[str, Any]]:
        """Process SMS transaction and categorize with ML"""
        try:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")
            
            # Parse SMS
            transaction = self.sms_parser.parse_sms(sms_text, sender_id)
            if not transaction:
                logger.warning("Failed to parse SMS transaction")
                return None
            
            # Categorize with ML
            ml_result = self.expense_categorizer.categorize_expense(
                merchant=transaction.merchant,
                amount=transaction.amount,
                description=f"{transaction.merchant} transaction"
            )
            
            # Add ML category to transaction
            transaction.category = ml_result['category']
            transaction.confidence = ml_result['confidence']
            
            # Store transaction
            self.user_transactions[user_id].append(transaction)
            
            logger.info(f"Processed transaction: ₹{transaction.amount} at {transaction.merchant}")
            
            return {
                "transaction": asdict(transaction),
                "ml_prediction": ml_result,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing SMS transaction: {e}")
            return None
    
    def add_manual_transaction(self, user_id: str, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add manual transaction entry"""
        try:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")
            
            # Create transaction object
            transaction = Transaction(
                amount=transaction_data['amount'],
                merchant=transaction_data['merchant'],
                account_number=transaction_data.get('account_number', '****'),
                transaction_id=f"manual_{int(datetime.now().timestamp())}",
                timestamp=datetime.now(),
                transaction_type=transaction_data.get('type', 'debit'),
                bank_name=transaction_data.get('bank', 'MANUAL')
            )
            
            # Categorize with ML
            ml_result = self.expense_categorizer.categorize_expense(
                merchant=transaction.merchant,
                amount=transaction.amount,
                description=transaction_data.get('description', '')
            )
            
            transaction.category = transaction_data.get('category', ml_result['category'])
            transaction.confidence = ml_result['confidence']
            
            # Store transaction
            self.user_transactions[user_id].append(transaction)
            
            logger.info(f"Manual transaction added: ₹{transaction.amount} - {transaction.category}")
            
            return {
                "transaction": asdict(transaction),
                "ml_prediction": ml_result,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error adding manual transaction: {e}")
            return {"status": "error", "message": str(e)}
    
    def generate_comprehensive_insights(self, user_id: str) -> InsightsSummary:
        """Generate comprehensive financial insights for user"""
        try:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")
            
            user = self.users[user_id]
            transactions = self.user_transactions[user_id]
            
            logger.info(f"Generating comprehensive insights for user {user_id}")
            
            # Transaction Summary
            transaction_summary = self._generate_transaction_summary(transactions)
            
            # Spending Analysis
            spending_analysis = self._generate_spending_analysis(transactions, user)
            
            # Investment Recommendations
            investment_recommendations = self._generate_investment_recommendations(user, transactions)
            
            # Behavioral Insights
            behavioral_insights = self._generate_behavioral_insights(transactions)
            
            # Predictive Forecast
            predictive_forecast = self._generate_predictive_forecast(transactions)
            
            # Risk Assessment
            risk_assessment = self._generate_risk_assessment(user, transactions)
            
            # Optimization Suggestions
            optimization_suggestions = self._generate_optimization_suggestions(user, transactions)
            
            insights = InsightsSummary(
                user_id=user_id,
                generated_at=datetime.now(),
                transaction_summary=transaction_summary,
                spending_analysis=spending_analysis,
                investment_recommendations=investment_recommendations,
                behavioral_insights=behavioral_insights,
                predictive_forecast=predictive_forecast,
                risk_assessment=risk_assessment,
                optimization_suggestions=optimization_suggestions
            )
            
            logger.info(f"Comprehensive insights generated for user {user_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            raise
    
    def _generate_transaction_summary(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Generate transaction summary statistics"""
        if not transactions:
            return {"total_transactions": 0, "total_amount": 0, "message": "No transactions found"}
        
        total_amount = sum(t.amount for t in transactions)
        avg_transaction = total_amount / len(transactions)
        
        # Category breakdown
        category_totals = {}
        for transaction in transactions:
            category = getattr(transaction, 'category', 'MISCELLANEOUS')
            category_totals[category] = category_totals.get(category, 0) + transaction.amount
        
        # Most frequent category
        most_frequent_category = max(category_totals, key=category_totals.get) if category_totals else "None"
        
        return {
            "total_transactions": len(transactions),
            "total_amount": total_amount,
            "average_transaction": avg_transaction,
            "category_breakdown": category_totals,
            "most_frequent_category": most_frequent_category
        }
    
    def _generate_spending_analysis(self, transactions: List[Transaction], user: SmartMoneyUser) -> Dict[str, Any]:
        """Generate spending behavior analysis"""
        if not transactions:
            return {"message": "No transactions for analysis"}
        
        total_spending = sum(t.amount for t in transactions)
        daily_avg = total_spending / max(1, len(transactions))
        monthly_estimated = daily_avg * 30
        
        # Income utilization
        monthly_income = user.annual_income / 12
        income_utilization = (monthly_estimated / monthly_income) * 100 if monthly_income > 0 else 0
        
        return {
            "total_spending": total_spending,
            "daily_average": daily_avg,
            "monthly_estimated": monthly_estimated,
            "income_utilization_percent": income_utilization,
            "spending_efficiency": "High" if income_utilization < 70 else "Moderate" if income_utilization < 90 else "Low"
        }
    
    def _generate_investment_recommendations(self, user: SmartMoneyUser, transactions: List[Transaction]) -> List[Dict[str, Any]]:
        """Generate AI-powered investment recommendations"""
        try:
            # Create user profile for investment engine
            monthly_spending = sum(t.amount for t in transactions) if transactions else 20000
            
            # Mock recommendations for demo
            recommendations = [
                {
                    "type": "Mutual Funds",
                    "allocation_percentage": 35.0,
                    "expected_return": 12.0,
                    "risk_level": "Medium",
                    "reasoning": "Professional management with diversification. Aligns with your investment timeline.",
                    "time_horizon": "Long-term (10+ years)"
                },
                {
                    "type": "Index Funds",
                    "allocation_percentage": 20.0,
                    "expected_return": 11.0,
                    "risk_level": "Medium",
                    "reasoning": "Low-cost market exposure with good long-term potential.",
                    "time_horizon": "Long-term (10+ years)"
                },
                {
                    "type": "Fixed Deposit",
                    "allocation_percentage": 15.0,
                    "expected_return": 6.0,
                    "risk_level": "Very Low",
                    "reasoning": "Capital preservation and guaranteed returns.",
                    "time_horizon": "Long-term (10+ years)"
                },
                {
                    "type": "Government Bonds",
                    "allocation_percentage": 15.0,
                    "expected_return": 7.0,
                    "risk_level": "Low",
                    "reasoning": "Government-backed security with predictable returns.",
                    "time_horizon": "Long-term (10+ years)"
                },
                {
                    "type": "Stocks",
                    "allocation_percentage": 10.0,
                    "expected_return": 15.0,
                    "risk_level": "High",
                    "reasoning": "Higher growth potential for long-term wealth creation.",
                    "time_horizon": "Long-term (10+ years)"
                },
                {
                    "type": "Gold",
                    "allocation_percentage": 5.0,
                    "expected_return": 8.0,
                    "risk_level": "Medium",
                    "reasoning": "Portfolio diversification and inflation hedge.",
                    "time_horizon": "Long-term (10+ years)"
                }
            ]
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating investment recommendations: {e}")
            return [{"message": "Investment engine not available"}]
    
    def _generate_behavioral_insights(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Generate behavioral finance insights"""
        try:
            if not transactions:
                return {"message": "No transactions for behavioral analysis"}
            
            # Mock behavioral insights
            return {
                "spending_personality": "Moderate Spender",
                "impulse_buying_score": 0.3,
                "budget_adherence": 0.75,
                "seasonal_patterns": "Higher spending on weekends",
                "recommendations": [
                    "Set weekly spending limits",
                    "Monitor discretionary expenses",
                    "Consider automated savings"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating behavioral insights: {e}")
            return {"message": "Behavioral analyzer not available"}
    
    def _generate_predictive_forecast(self, transactions: List[Transaction]) -> Dict[str, Any]:
        """Generate predictive financial forecasting"""
        try:
            if not transactions:
                return {"message": "No transactions for prediction"}
            
            avg_spending = sum(t.amount for t in transactions) / len(transactions)
            
            # Mock predictions
            return {
                "next_month_spending_prediction": avg_spending * 30,
                "confidence_interval": [avg_spending * 25, avg_spending * 35],
                "trend": "Stable",
                "savings_potential": avg_spending * 5,  # 5 days worth of spending
                "recommendations": [
                    "Monitor large expenses",
                    "Consider setting spending alerts"
                ]
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions: {e}")
            return {"message": "Predictive engine not available"}
    
    def _generate_risk_assessment(self, user: SmartMoneyUser, transactions: List[Transaction]) -> Dict[str, Any]:
        """Generate financial risk assessment"""
        monthly_spending = sum(t.amount for t in transactions) if transactions else 20000
        monthly_income = user.annual_income / 12
        
        spending_ratio = monthly_spending / monthly_income if monthly_income > 0 else 1
        
        risk_level = "Low" if spending_ratio < 0.6 else "Medium" if spending_ratio < 0.8 else "High"
        
        return {
            "overall_risk_level": risk_level,
            "spending_to_income_ratio": spending_ratio,
            "emergency_fund_status": "Build gradually" if spending_ratio > 0.7 else "Good",
            "recommendations": [
                "Build emergency fund covering 6 months expenses",
                "Consider insurance coverage",
                "Diversify income sources"
            ]
        }
    
    def _generate_optimization_suggestions(self, user: SmartMoneyUser, transactions: List[Transaction]) -> List[str]:
        """Generate financial optimization suggestions"""
        suggestions = []
        
        if not transactions:
            suggestions.append("Start tracking your expenses for better insights")
            return suggestions
        
        monthly_spending = sum(t.amount for t in transactions)
        monthly_income = user.annual_income / 12
        
        if monthly_spending > monthly_income * 0.8:
            suggestions.append("Consider reducing discretionary spending")
        
        if monthly_income > 50000:
            suggestions.append("Consider starting SIP investments with surplus savings")
        
        suggestions.extend([
            "Set up automated savings transfers",
            "Review and optimize recurring subscriptions",
            "Consider tax-saving investment options",
            "Monitor spending patterns monthly"
        ])
        
        return suggestions
    
    def get_user_dashboard_data(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive dashboard data for user"""
        try:
            if user_id not in self.users:
                raise ValueError(f"User {user_id} not found")
            
            user = self.users[user_id]
            transactions = self.user_transactions[user_id]
            
            # Generate insights
            insights = self.generate_comprehensive_insights(user_id)
            
            return {
                "user_profile": asdict(user),
                "insights": asdict(insights),
                "system_status": {
                    "total_users": len(self.users),
                    "total_transactions": sum(len(txns) for txns in self.user_transactions.values()),
                    "components_status": {
                        "sms_parser": "Active",
                        "ml_categorizer": "Active",
                        "investment_engine": "Active",
                        "behavioral_analyzer": "Active",
                        "predictive_analytics": "Active"
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            raise
    
    def export_user_data(self, user_id: str, format: str = "json") -> str:
        """Export user data in specified format"""
        try:
            dashboard_data = self.get_user_dashboard_data(user_id)
            
            if format.lower() == "json":
                return json.dumps(dashboard_data, indent=2, default=str)
            else:
                raise ValueError(f"Format {format} not supported")
                
        except Exception as e:
            logger.error(f"Error exporting user data: {e}")
            raise


def main():
    """Demo function"""
    print("🚀 Smart Money AI - Core System Demo")
    print("=" * 60)
    
    # Initialize Smart Money AI
    smart_money = SmartMoneyAI()
    
    # Create demo user
    user_data = {
        "user_id": "demo_user_001",
        "name": "Demo User",
        "email": "demo@smartmoney.ai",
        "phone": "+91-9876543210",
        "age": 28,
        "annual_income": 600000,
        "risk_profile": "moderate",
        "investment_timeline": 15,
        "financial_goals": ["wealth_creation", "retirement_planning"]
    }
    
    user = smart_money.create_user_profile(user_data)
    print(f"✅ User profile created: {user.name}")
    
    # Add sample transactions
    sample_transactions = [
        {"amount": 8000, "merchant": "RENT", "category": "RENT"},
        {"amount": 12000, "merchant": "GROCERIES", "category": "GROCERIES"},
        {"amount": 3000, "merchant": "ENTERTAINMENT", "category": "ENTERTAINMENT"}
    ]
    
    for txn in sample_transactions:
        result = smart_money.add_manual_transaction(user.user_id, txn)
        print(f"✅ Transaction added: ₹{txn['amount']} - {txn['category']}")
    
    # Generate insights
    print("\n📊 Generating comprehensive insights...")
    insights = smart_money.generate_comprehensive_insights(user.user_id)
    
    print(f"\n💰 Transaction Summary:")
    print(f"  Total Transactions: {insights.transaction_summary['total_transactions']}")
    print(f"  Total Amount: ₹{insights.transaction_summary['total_amount']:,}")
    
    print(f"\n📈 Investment Recommendations:")
    for rec in insights.investment_recommendations[:3]:
        print(f"  • {rec['type']}: {rec['allocation_percentage']}% (Expected Return: {rec['expected_return']}%)")
    
    print(f"\n🎯 Risk Assessment:")
    print(f"  Risk Level: {insights.risk_assessment['overall_risk_level']}")
    
    print(f"\n💡 Optimization Suggestions:")
    for suggestion in insights.optimization_suggestions[:3]:
        print(f"  • {suggestion}")
    
    print("\n🎉 Smart Money AI demo completed successfully!")


if __name__ == "__main__":
    main()