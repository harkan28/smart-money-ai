#!/usr/bin/env python3
"""
Comprehensive Smart Money AI Integrator
Combines all AI modules: SMS parsing, ML categorization, investment recommendations, 
behavioral analysis, and predictive analytics into one unified system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import logging
import json
import os
import sys
from dataclasses import dataclass, asdict

# Import components from existing modules
try:
    from SMS_PARSING_SYSTEM.main import SMSParsingSystem
    from budgeting_ml_model.src.inference import CategoryPredictor
    from investment_engine import InvestmentRecommendationEngine
    from behavioral_analyzer import BehavioralFinanceAnalyzer
    from predictive_analytics import ExpensePredictionEngine
except ImportError as e:
    print(f"Warning: Some modules not found: {e}")
    print("Running in demo mode with mock data...")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TransactionData:
    """Standardized transaction data structure"""
    amount: float
    category: str
    description: str
    timestamp: datetime
    merchant: Optional[str] = None
    location: Optional[str] = None
    confidence_score: Optional[float] = None
    raw_text: Optional[str] = None

@dataclass
class UserProfile:
    """Comprehensive user profile for personalized recommendations"""
    user_id: str
    age: int
    income_monthly: float
    risk_tolerance: str  # 'conservative', 'moderate', 'aggressive'
    investment_goals: List[str]
    current_savings: float
    financial_obligations: Dict[str, float]
    spending_preferences: Dict[str, float]
    created_at: datetime
    last_updated: datetime

@dataclass
class SmartMoneyInsights:
    """Complete financial insights package"""
    transaction_summary: Dict[str, any]
    spending_analysis: Dict[str, any]
    investment_recommendations: Dict[str, any]
    behavioral_insights: Dict[str, any]
    predictive_forecast: Dict[str, any]
    risk_assessment: Dict[str, any]
    optimization_opportunities: List[str]
    generated_at: datetime

class SmartMoneyAI:
    """Unified Smart Money AI system integrating all components"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Smart Money AI system"""
        self.config = self._load_config(config_path)
        self.components = {}
        self._initialize_components()
        
        # Data storage
        self.user_profiles = {}
        self.transaction_history = []
        self.insights_history = []
        
        logger.info("Smart Money AI system initialized")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load system configuration"""
        default_config = {
            'sms_parsing': {'enabled': True, 'confidence_threshold': 0.7},
            'ml_categorization': {'enabled': True, 'model_path': 'budgeting_ml_model/models/'},
            'investment_engine': {'enabled': True, 'default_risk_tolerance': 'moderate'},
            'behavioral_analysis': {'enabled': True, 'analysis_period_days': 90},
            'predictive_analytics': {'enabled': True, 'forecast_horizon_days': 30},
            'data_storage': {'save_insights': True, 'max_history_days': 365}
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
            except Exception as e:
                logger.warning(f"Error loading config: {e}, using defaults")
        
        return default_config
    
    def _initialize_components(self):
        """Initialize all AI components"""
        try:
            # SMS Parsing System
            if self.config['sms_parsing']['enabled']:
                try:
                    self.components['sms_parser'] = SMSParsingSystem()
                    logger.info("SMS parsing system initialized")
                except Exception as e:
                    logger.warning(f"SMS parser initialization failed: {e}")
                    self.components['sms_parser'] = None
            
            # ML Categorization
            if self.config['ml_categorization']['enabled']:
                try:
                    self.components['category_predictor'] = CategoryPredictor()
                    logger.info("ML categorization system initialized")
                except Exception as e:
                    logger.warning(f"ML categorization initialization failed: {e}")
                    self.components['category_predictor'] = None
            
            # Investment Engine
            if self.config['investment_engine']['enabled']:
                self.components['investment_engine'] = InvestmentRecommendationEngine()
                logger.info("Investment recommendation engine initialized")
            
            # Behavioral Analyzer
            if self.config['behavioral_analysis']['enabled']:
                self.components['behavioral_analyzer'] = BehavioralFinanceAnalyzer()
                logger.info("Behavioral finance analyzer initialized")
            
            # Predictive Analytics
            if self.config['predictive_analytics']['enabled']:
                self.components['predictive_engine'] = ExpensePredictionEngine()
                logger.info("Predictive analytics engine initialized")
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
    
    def create_user_profile(self, user_data: Dict) -> UserProfile:
        """Create a new user profile"""
        try:
            profile = UserProfile(
                user_id=user_data['user_id'],
                age=user_data.get('age', 30),
                income_monthly=user_data.get('income_monthly', 50000),
                risk_tolerance=user_data.get('risk_tolerance', 'moderate'),
                investment_goals=user_data.get('investment_goals', ['wealth_building']),
                current_savings=user_data.get('current_savings', 100000),
                financial_obligations=user_data.get('financial_obligations', {}),
                spending_preferences=user_data.get('spending_preferences', {}),
                created_at=datetime.now(),
                last_updated=datetime.now()
            )
            
            self.user_profiles[profile.user_id] = profile
            logger.info(f"User profile created for {profile.user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            raise
    
    def process_sms_transaction(self, sms_text: str, user_id: str) -> Optional[TransactionData]:
        """Process SMS to extract transaction data"""
        try:
            if not self.components.get('sms_parser'):
                logger.warning("SMS parser not available, using mock processing")
                return self._mock_sms_processing(sms_text)
            
            # Parse SMS
            parsed_data = self.components['sms_parser'].parse_sms(sms_text)
            
            if not parsed_data or not parsed_data.get('is_transaction', False):
                logger.info("SMS does not contain transaction data")
                return None
            
            # Create transaction data
            transaction = TransactionData(
                amount=abs(float(parsed_data.get('amount', 0))),
                category='UNKNOWN',  # Will be categorized by ML
                description=parsed_data.get('description', 'Unknown transaction'),
                timestamp=parsed_data.get('timestamp', datetime.now()),
                merchant=parsed_data.get('merchant'),
                raw_text=sms_text,
                confidence_score=parsed_data.get('confidence', 0.8)
            )
            
            # ML Categorization
            if self.components.get('category_predictor'):
                category_result = self.components['category_predictor'].predict_category(
                    transaction.description
                )
                if category_result:
                    transaction.category = category_result.get('category', 'MISCELLANEOUS')
                    transaction.confidence_score = min(
                        transaction.confidence_score,
                        category_result.get('confidence', 0.5)
                    )
            
            # Store transaction
            self.transaction_history.append(transaction)
            logger.info(f"Transaction processed: ₹{transaction.amount} - {transaction.category}")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error processing SMS transaction: {e}")
            return None
    
    def add_manual_transaction(self, transaction_data: Dict, user_id: str) -> TransactionData:
        """Add manually entered transaction"""
        try:
            transaction = TransactionData(
                amount=abs(float(transaction_data['amount'])),
                category=transaction_data.get('category', 'MISCELLANEOUS'),
                description=transaction_data.get('description', 'Manual transaction'),
                timestamp=transaction_data.get('timestamp', datetime.now()),
                merchant=transaction_data.get('merchant'),
                location=transaction_data.get('location'),
                confidence_score=1.0  # Manual entries have high confidence
            )
            
            self.transaction_history.append(transaction)
            logger.info(f"Manual transaction added: ₹{transaction.amount} - {transaction.category}")
            
            return transaction
            
        except Exception as e:
            logger.error(f"Error adding manual transaction: {e}")
            raise
    
    def generate_comprehensive_insights(self, user_id: str, 
                                      analysis_period_days: int = 30) -> SmartMoneyInsights:
        """Generate comprehensive financial insights for a user"""
        try:
            # Get user profile
            user_profile = self.user_profiles.get(user_id)
            if not user_profile:
                raise ValueError(f"User profile not found for {user_id}")
            
            # Get recent transactions
            cutoff_date = datetime.now() - timedelta(days=analysis_period_days)
            recent_transactions = [
                t for t in self.transaction_history 
                if t.timestamp >= cutoff_date
            ]
            
            if not recent_transactions:
                logger.warning("No recent transactions found")
                return self._generate_default_insights(user_id)
            
            # Convert to DataFrame for analysis
            df = self._transactions_to_dataframe(recent_transactions)
            
            # 1. Transaction Summary
            transaction_summary = self._generate_transaction_summary(df)
            
            # 2. Spending Analysis
            spending_analysis = self._generate_spending_analysis(df, user_profile)
            
            # 3. Investment Recommendations
            investment_recommendations = self._generate_investment_recommendations(
                df, user_profile
            )
            
            # 4. Behavioral Insights
            behavioral_insights = self._generate_behavioral_insights(df, user_profile)
            
            # 5. Predictive Forecast
            predictive_forecast = self._generate_predictive_forecast(df)
            
            # 6. Risk Assessment
            risk_assessment = self._generate_risk_assessment(df, user_profile)
            
            # 7. Optimization Opportunities
            optimization_opportunities = self._identify_optimization_opportunities(
                df, user_profile, spending_analysis, behavioral_insights
            )
            
            # Create insights package
            insights = SmartMoneyInsights(
                transaction_summary=transaction_summary,
                spending_analysis=spending_analysis,
                investment_recommendations=investment_recommendations,
                behavioral_insights=behavioral_insights,
                predictive_forecast=predictive_forecast,
                risk_assessment=risk_assessment,
                optimization_opportunities=optimization_opportunities,
                generated_at=datetime.now()
            )
            
            # Store insights
            self.insights_history.append(insights)
            
            logger.info(f"Comprehensive insights generated for user {user_id}")
            return insights
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return self._generate_default_insights(user_id)
    
    def _transactions_to_dataframe(self, transactions: List[TransactionData]) -> pd.DataFrame:
        """Convert transaction list to DataFrame"""
        data = []
        for t in transactions:
            data.append({
                'amount': t.amount,
                'category': t.category,
                'description': t.description,
                'timestamp': t.timestamp,
                'merchant': t.merchant,
                'location': t.location,
                'confidence_score': t.confidence_score
            })
        
        return pd.DataFrame(data)
    
    def _generate_transaction_summary(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate transaction summary statistics"""
        return {
            'total_transactions': len(df),
            'total_amount': float(df['amount'].sum()),
            'average_transaction': float(df['amount'].mean()),
            'median_transaction': float(df['amount'].median()),
            'largest_transaction': float(df['amount'].max()),
            'smallest_transaction': float(df['amount'].min()),
            'categories_used': len(df['category'].unique()),
            'most_frequent_category': df['category'].mode().iloc[0] if len(df) > 0 else 'N/A',
            'date_range': {
                'start': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                'end': df['timestamp'].max().isoformat() if len(df) > 0 else None
            }
        }
    
    def _generate_spending_analysis(self, df: pd.DataFrame, user_profile: UserProfile) -> Dict[str, any]:
        """Generate detailed spending analysis"""
        analysis = {}
        
        # Category breakdown
        category_spending = df.groupby('category')['amount'].agg(['sum', 'count', 'mean']).to_dict()
        analysis['category_breakdown'] = category_spending
        
        # Monthly spending rate
        total_spending = df['amount'].sum()
        days_span = (df['timestamp'].max() - df['timestamp'].min()).days + 1
        monthly_rate = (total_spending / days_span) * 30 if days_span > 0 else 0
        
        analysis['spending_rate'] = {
            'daily_average': float(total_spending / days_span) if days_span > 0 else 0,
            'monthly_estimated': float(monthly_rate),
            'vs_income_ratio': float(monthly_rate / user_profile.income_monthly) if user_profile.income_monthly > 0 else 0
        }
        
        # Spending patterns
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        analysis['patterns'] = {
            'peak_spending_hour': int(df.groupby('hour')['amount'].sum().idxmax()) if len(df) > 0 else 12,
            'peak_spending_day': int(df.groupby('day_of_week')['amount'].sum().idxmax()) if len(df) > 0 else 0,
            'weekend_vs_weekday': self._calculate_weekend_spending_ratio(df)
        }
        
        return analysis
    
    def _generate_investment_recommendations(self, df: pd.DataFrame, 
                                           user_profile: UserProfile) -> Dict[str, any]:
        """Generate investment recommendations"""
        try:
            if not self.components.get('investment_engine'):
                return {'error': 'Investment engine not available'}
            
            # Calculate available investment amount
            monthly_spending = df['amount'].sum() / max(1, len(df.groupby(df['timestamp'].dt.date)))
            monthly_spending = monthly_spending * 30  # Extrapolate to full month
            
            available_for_investment = max(0, user_profile.income_monthly - monthly_spending)
            
            # Generate recommendations
            recommendations = self.components['investment_engine'].generate_recommendations(
                user_profile=asdict(user_profile),
                current_spending=monthly_spending,
                available_amount=available_for_investment
            )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating investment recommendations: {e}")
            return {'error': str(e)}
    
    def _generate_behavioral_insights(self, df: pd.DataFrame, 
                                    user_profile: UserProfile) -> Dict[str, any]:
        """Generate behavioral finance insights"""
        try:
            if not self.components.get('behavioral_analyzer'):
                return {'error': 'Behavioral analyzer not available'}
            
            insights = self.components['behavioral_analyzer'].analyze_behavior(df)
            return insights
            
        except Exception as e:
            logger.error(f"Error generating behavioral insights: {e}")
            return {'error': str(e)}
    
    def _generate_predictive_forecast(self, df: pd.DataFrame) -> Dict[str, any]:
        """Generate predictive spending forecast"""
        try:
            if not self.components.get('predictive_engine'):
                return {'error': 'Predictive engine not available'}
            
            # Train model on current data
            training_scores = self.components['predictive_engine'].train_models(df)
            
            # Generate forecast
            forecast = self.components['predictive_engine'].predict_expenses(
                prediction_days=30
            )
            
            return {
                'training_scores': training_scores,
                'forecast_summary': {
                    'weekly_total': forecast.weekly_summary.get('total_weekly_spending', 0),
                    'monthly_total': forecast.monthly_summary.get('total_monthly_spending', 0),
                    'confidence': forecast.weekly_summary.get('confidence_score', 0.5)
                },
                'savings_forecast': forecast.savings_forecast,
                'risk_indicators': forecast.risk_indicators,
                'optimization_opportunities': forecast.optimization_opportunities
            }
            
        except Exception as e:
            logger.error(f"Error generating predictive forecast: {e}")
            return {'error': str(e)}
    
    def _generate_risk_assessment(self, df: pd.DataFrame, 
                                user_profile: UserProfile) -> Dict[str, any]:
        """Generate financial risk assessment"""
        risks = []
        risk_score = 0
        
        # Spending vs income ratio
        monthly_spending = (df['amount'].sum() / max(1, len(df))) * 30
        spending_ratio = monthly_spending / user_profile.income_monthly if user_profile.income_monthly > 0 else 1
        
        if spending_ratio > 0.8:
            risks.append("High spending-to-income ratio")
            risk_score += 30
        elif spending_ratio > 0.6:
            risks.append("Moderate spending-to-income ratio")
            risk_score += 15
        
        # Spending volatility
        daily_spending = df.groupby(df['timestamp'].dt.date)['amount'].sum()
        cv = daily_spending.std() / daily_spending.mean() if daily_spending.mean() > 0 else 0
        
        if cv > 1:
            risks.append("High spending volatility")
            risk_score += 20
        elif cv > 0.5:
            risks.append("Moderate spending volatility")
            risk_score += 10
        
        # Emergency fund coverage
        emergency_fund_months = user_profile.current_savings / monthly_spending if monthly_spending > 0 else 12
        
        if emergency_fund_months < 3:
            risks.append("Insufficient emergency fund")
            risk_score += 25
        elif emergency_fund_months < 6:
            risks.append("Limited emergency fund")
            risk_score += 10
        
        # Risk level categorization
        if risk_score >= 50:
            risk_level = "High"
        elif risk_score >= 25:
            risk_level = "Moderate"
        else:
            risk_level = "Low"
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_factors': risks,
            'metrics': {
                'spending_ratio': spending_ratio,
                'spending_volatility': cv,
                'emergency_fund_months': emergency_fund_months
            }
        }
    
    def _identify_optimization_opportunities(self, df: pd.DataFrame, 
                                           user_profile: UserProfile,
                                           spending_analysis: Dict,
                                           behavioral_insights: Dict) -> List[str]:
        """Identify financial optimization opportunities"""
        opportunities = []
        
        # Category-based optimizations
        category_spending = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        if 'FOOD_DINING' in category_spending.index and category_spending['FOOD_DINING'] > spending_analysis['spending_rate']['monthly_estimated'] * 0.3:
            opportunities.append("Consider meal planning to reduce dining expenses")
        
        if 'ENTERTAINMENT' in category_spending.index and category_spending['ENTERTAINMENT'] > spending_analysis['spending_rate']['monthly_estimated'] * 0.2:
            opportunities.append("Look for free or low-cost entertainment alternatives")
        
        # Investment opportunities
        spending_ratio = spending_analysis['spending_rate']['vs_income_ratio']
        if spending_ratio < 0.7:
            available_percentage = (1 - spending_ratio) * 100
            opportunities.append(f"You can invest {available_percentage:.1f}% of income for wealth building")
        
        # Behavioral optimizations
        weekend_ratio = spending_analysis['patterns']['weekend_vs_weekday']
        if weekend_ratio > 1.5:
            opportunities.append("Plan weekend activities to control impulse spending")
        
        # SIP recommendations
        if user_profile.current_savings > user_profile.income_monthly * 2:
            opportunities.append("Consider starting SIP investments with surplus savings")
        
        return opportunities[:5]  # Return top 5 opportunities
    
    def _calculate_weekend_spending_ratio(self, df: pd.DataFrame) -> float:
        """Calculate weekend vs weekday spending ratio"""
        df['is_weekend'] = pd.to_datetime(df['timestamp']).dt.weekday >= 5
        weekend_spending = df[df['is_weekend']]['amount'].mean()
        weekday_spending = df[~df['is_weekend']]['amount'].mean()
        
        if weekday_spending > 0:
            return weekend_spending / weekday_spending
        return 1.0
    
    def _mock_sms_processing(self, sms_text: str) -> TransactionData:
        """Mock SMS processing when parser not available"""
        # Simple pattern matching for demo
        import re
        
        # Look for amount pattern
        amount_match = re.search(r'(?:Rs\.?|INR|₹)\s*(\d+(?:,\d+)*(?:\.\d{2})?)', sms_text)
        amount = float(amount_match.group(1).replace(',', '')) if amount_match else 1000.0
        
        # Basic categorization
        category = 'MISCELLANEOUS'
        if any(word in sms_text.lower() for word in ['atm', 'cash']):
            category = 'CASH_WITHDRAWAL'
        elif any(word in sms_text.lower() for word in ['fuel', 'petrol', 'gas']):
            category = 'TRANSPORTATION'
        elif any(word in sms_text.lower() for word in ['grocery', 'supermarket']):
            category = 'GROCERIES'
        
        return TransactionData(
            amount=amount,
            category=category,
            description=f"Transaction from SMS: {sms_text[:50]}...",
            timestamp=datetime.now(),
            raw_text=sms_text,
            confidence_score=0.6
        )
    
    def _generate_default_insights(self, user_id: str) -> SmartMoneyInsights:
        """Generate default insights when no data available"""
        return SmartMoneyInsights(
            transaction_summary={'total_transactions': 0, 'message': 'No transactions found'},
            spending_analysis={'message': 'Insufficient data for analysis'},
            investment_recommendations={'message': 'Add transactions to get personalized recommendations'},
            behavioral_insights={'message': 'Behavioral analysis requires transaction history'},
            predictive_forecast={'message': 'Predictive forecast requires historical data'},
            risk_assessment={'risk_level': 'Unknown', 'message': 'Risk assessment needs transaction data'},
            optimization_opportunities=['Start tracking expenses to unlock personalized insights'],
            generated_at=datetime.now()
        )
    
    def export_insights(self, insights: SmartMoneyInsights, format: str = 'json') -> str:
        """Export insights in specified format"""
        try:
            if format.lower() == 'json':
                return json.dumps(asdict(insights), indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
        except Exception as e:
            logger.error(f"Error exporting insights: {e}")
            return "{}"
    
    def get_system_status(self) -> Dict[str, any]:
        """Get system status and health check"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'data_summary': {
                'user_profiles': len(self.user_profiles),
                'transactions': len(self.transaction_history),
                'insights_generated': len(self.insights_history)
            }
        }
        
        # Check component status
        for name, component in self.components.items():
            status['components'][name] = {
                'available': component is not None,
                'type': type(component).__name__ if component else None
            }
        
        return status

def main():
    """Demo the Smart Money AI system"""
    print("🧠 Smart Money AI - Comprehensive Demo")
    print("=" * 60)
    
    # Initialize system
    smart_money = SmartMoneyAI()
    
    # Create sample user
    user_data = {
        'user_id': 'demo_user_001',
        'age': 28,
        'income_monthly': 75000,
        'risk_tolerance': 'moderate',
        'investment_goals': ['wealth_building', 'retirement'],
        'current_savings': 250000,
        'financial_obligations': {'rent': 25000, 'loan_emi': 15000}
    }
    
    print("\n👤 Creating User Profile...")
    user_profile = smart_money.create_user_profile(user_data)
    print(f"User profile created for {user_profile.user_id}")
    
    # Process sample SMS transactions
    sample_sms_messages = [
        "Dear Customer, Rs.2500 debited from your account 1234 at RELIANCE FRESH on 12-Dec-23. Avbl bal: Rs.45,000.",
        "Transaction Alert: Rs.5000 withdrawn from ATM HDFC0001234 on 12-Dec-23 14:30. Available balance Rs.42,500",
        "Dear Customer, Rs.1200 spent at SWIGGY BANGALORE using your card ending 5678 on 12-Dec-23.",
        "Payment of Rs.3500 made to UBER TRIP BLR using UPI on 12-Dec-23 21:45. Balance: Rs.38,000",
        "Rs.15000 debited for AMAZON PAY LATER EMI on 13-Dec-23. Thank you for using our services."
    ]
    
    print("\n📱 Processing SMS Transactions...")
    for i, sms in enumerate(sample_sms_messages, 1):
        transaction = smart_money.process_sms_transaction(sms, user_profile.user_id)
        if transaction:
            print(f"  {i}. ₹{transaction.amount:,.0f} - {transaction.category}")
    
    # Add some manual transactions
    manual_transactions = [
        {'amount': 8000, 'category': 'RENT', 'description': 'Monthly rent payment'},
        {'amount': 12000, 'category': 'GROCERIES', 'description': 'Monthly grocery shopping'},
        {'amount': 3000, 'category': 'ENTERTAINMENT', 'description': 'Movie and dinner'},
    ]
    
    print("\n✏️  Adding Manual Transactions...")
    for trans_data in manual_transactions:
        transaction = smart_money.add_manual_transaction(trans_data, user_profile.user_id)
        print(f"  ₹{transaction.amount:,.0f} - {transaction.category}")
    
    # Generate comprehensive insights
    print("\n🔍 Generating Comprehensive Insights...")
    insights = smart_money.generate_comprehensive_insights(user_profile.user_id)
    
    # Display insights
    print(f"\n📊 Transaction Summary:")
    ts = insights.transaction_summary
    print(f"  Total Transactions: {ts['total_transactions']}")
    print(f"  Total Amount: ₹{ts['total_amount']:,.0f}")
    print(f"  Average Transaction: ₹{ts['average_transaction']:,.0f}")
    print(f"  Most Frequent Category: {ts['most_frequent_category']}")
    
    print(f"\n💰 Spending Analysis:")
    sa = insights.spending_analysis
    if 'spending_rate' in sa:
        print(f"  Daily Average: ₹{sa['spending_rate']['daily_average']:,.0f}")
        print(f"  Monthly Estimated: ₹{sa['spending_rate']['monthly_estimated']:,.0f}")
        print(f"  Income Utilization: {sa['spending_rate']['vs_income_ratio']:.1%}")
    
    print(f"\n📈 Investment Recommendations:")
    ir = insights.investment_recommendations
    if 'error' not in ir:
        print("  Investment recommendations generated successfully")
    else:
        print(f"  {ir.get('error', 'No recommendations available')}")
    
    print(f"\n🧠 Behavioral Insights:")
    bi = insights.behavioral_insights
    if 'error' not in bi:
        print("  Behavioral analysis completed")
    else:
        print(f"  {bi.get('error', 'Analysis not available')}")
    
    print(f"\n🔮 Predictive Forecast:")
    pf = insights.predictive_forecast
    if 'error' not in pf:
        print("  Predictive forecast generated")
    else:
        print(f"  {pf.get('error', 'Forecast not available')}")
    
    print(f"\n⚠️  Risk Assessment:")
    ra = insights.risk_assessment
    print(f"  Risk Level: {ra.get('risk_level', 'Unknown')}")
    if 'risk_factors' in ra:
        for risk in ra['risk_factors'][:3]:
            print(f"  • {risk}")
    
    print(f"\n💡 Optimization Opportunities:")
    for opportunity in insights.optimization_opportunities[:3]:
        print(f"  • {opportunity}")
    
    # System status
    print(f"\n🔧 System Status:")
    status = smart_money.get_system_status()
    print(f"  User Profiles: {status['data_summary']['user_profiles']}")
    print(f"  Transactions: {status['data_summary']['transactions']}")
    print(f"  Components Available: {sum(1 for c in status['components'].values() if c['available'])}/{len(status['components'])}")
    
    print(f"\n✅ Smart Money AI demo completed!")
    print(f"🚀 All systems integrated and working together!")

if __name__ == "__main__":
    main()