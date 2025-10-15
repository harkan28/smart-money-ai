#!/usr/bin/env python3
"""
Investment Recommendation Engine - Core Implementation
Advanced AI-powered financial insights and investment recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskProfile(Enum):
    """User risk profile classifications"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class InvestmentType(Enum):
    """Available investment instruments"""
    FIXED_DEPOSIT = "fixed_deposit"
    GOVERNMENT_BONDS = "government_bonds"
    MUTUAL_FUNDS = "mutual_funds"
    INDEX_FUNDS = "index_funds"
    STOCKS = "stocks"
    GOLD = "gold"
    REAL_ESTATE = "real_estate"
    CRYPTOCURRENCY = "cryptocurrency"

@dataclass
class UserProfile:
    """User financial profile for investment recommendations"""
    user_id: str
    age: int
    annual_income: float
    monthly_expenses: float
    current_savings: float
    financial_goals: List[str]
    investment_timeline: int  # in years
    risk_tolerance: RiskProfile
    investment_experience: str  # "beginner", "intermediate", "advanced"
    
@dataclass
class SpendingPattern:
    """User spending behavior analysis"""
    monthly_spending: float
    spending_volatility: float
    impulse_spending_score: float
    category_distribution: Dict[str, float]
    temporal_patterns: Dict[str, float]
    savings_rate: float

@dataclass
class InvestmentRecommendation:
    """Investment recommendation with detailed analysis"""
    investment_type: InvestmentType
    allocation_percentage: float
    expected_return: float
    risk_level: str
    reasoning: str
    time_horizon: str
    minimum_investment: float
    liquidity: str
    tax_implications: str

class SpendingBehaviorAnalyzer:
    """Analyzes spending behavior patterns for investment insights"""
    
    def __init__(self):
        self.spending_personas = {
            "conservative_spender": {"volatility": 0.1, "impulse_score": 0.2},
            "moderate_spender": {"volatility": 0.3, "impulse_score": 0.5},
            "impulsive_spender": {"volatility": 0.6, "impulse_score": 0.8}
        }
    
    def analyze_spending_patterns(self, transactions: List[Dict]) -> SpendingPattern:
        """Analyze transaction data to extract spending patterns"""
        try:
            if not transactions:
                logger.warning("No transactions provided for analysis")
                return self._default_spending_pattern()
            
            df = pd.DataFrame(transactions)
            
            # Calculate basic spending metrics
            monthly_spending = self._calculate_monthly_spending(df)
            spending_volatility = self._calculate_spending_volatility(df)
            impulse_score = self._calculate_impulse_score(df)
            category_dist = self._analyze_category_distribution(df)
            temporal_patterns = self._analyze_temporal_patterns(df)
            savings_rate = self._estimate_savings_rate(df)
            
            return SpendingPattern(
                monthly_spending=monthly_spending,
                spending_volatility=spending_volatility,
                impulse_spending_score=impulse_score,
                category_distribution=category_dist,
                temporal_patterns=temporal_patterns,
                savings_rate=savings_rate
            )
            
        except Exception as e:
            logger.error(f"Error analyzing spending patterns: {e}")
            return self._default_spending_pattern()
    
    def _calculate_monthly_spending(self, df: pd.DataFrame) -> float:
        """Calculate average monthly spending"""
        if 'amount' not in df.columns:
            return 0.0
        
        df['date'] = pd.to_datetime(df.get('timestamp', datetime.now()))
        monthly_totals = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
        return float(monthly_totals.mean())
    
    def _calculate_spending_volatility(self, df: pd.DataFrame) -> float:
        """Calculate spending volatility as coefficient of variation"""
        if 'amount' not in df.columns:
            return 0.0
        
        daily_spending = df.groupby(df['date'].dt.date)['amount'].sum()
        if len(daily_spending) < 2:
            return 0.0
        
        return float(daily_spending.std() / daily_spending.mean())
    
    def _calculate_impulse_score(self, df: pd.DataFrame) -> float:
        """Calculate impulse spending score based on transaction patterns"""
        if len(df) < 2:
            return 0.0
        
        # Factors indicating impulse spending
        late_night_transactions = len(df[df['date'].dt.hour >= 22]) / len(df)
        high_amount_transactions = len(df[df['amount'] > df['amount'].quantile(0.8)]) / len(df)
        
        # Quick successive transactions (within 1 hour)
        df_sorted = df.sort_values('date')
        time_diffs = df_sorted['date'].diff().dt.total_seconds() / 3600  # hours
        quick_transactions = len(time_diffs[time_diffs < 1]) / len(df)
        
        impulse_score = (late_night_transactions * 0.3 + 
                        high_amount_transactions * 0.4 + 
                        quick_transactions * 0.3)
        
        return min(1.0, impulse_score)
    
    def _analyze_category_distribution(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze spending distribution across categories"""
        if 'category' not in df.columns:
            return {"MISCELLANEOUS": 1.0}
        
        category_totals = df.groupby('category')['amount'].sum()
        total_spending = category_totals.sum()
        
        if total_spending == 0:
            return {"MISCELLANEOUS": 1.0}
        
        return (category_totals / total_spending).to_dict()
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze temporal spending patterns"""
        patterns = {
            "weekend_spending_ratio": 0.0,
            "evening_spending_ratio": 0.0,
            "month_end_spike": 0.0
        }
        
        if len(df) == 0:
            return patterns
        
        # Weekend spending
        weekend_spending = df[df['date'].dt.weekday >= 5]['amount'].sum()
        total_spending = df['amount'].sum()
        if total_spending > 0:
            patterns["weekend_spending_ratio"] = weekend_spending / total_spending
        
        # Evening spending (after 6 PM)
        evening_spending = df[df['date'].dt.hour >= 18]['amount'].sum()
        if total_spending > 0:
            patterns["evening_spending_ratio"] = evening_spending / total_spending
        
        # Month-end spending spike
        month_end_spending = df[df['date'].dt.day >= 25]['amount'].sum()
        if total_spending > 0:
            patterns["month_end_spike"] = month_end_spending / total_spending
        
        return patterns
    
    def _estimate_savings_rate(self, df: pd.DataFrame) -> float:
        """Estimate savings rate based on spending patterns"""
        # This is a simplified estimation - in practice, you'd need income data
        # For now, assume average income based on spending patterns
        monthly_spending = self._calculate_monthly_spending(df)
        
        # Estimate income as 1.3x spending (assuming 23% savings rate baseline)
        estimated_income = monthly_spending * 1.3
        estimated_savings = estimated_income - monthly_spending
        
        if estimated_income > 0:
            return estimated_savings / estimated_income
        return 0.0
    
    def _default_spending_pattern(self) -> SpendingPattern:
        """Return default spending pattern when analysis fails"""
        return SpendingPattern(
            monthly_spending=25000.0,
            spending_volatility=0.3,
            impulse_spending_score=0.4,
            category_distribution={"MISCELLANEOUS": 1.0},
            temporal_patterns={"weekend_spending_ratio": 0.3, "evening_spending_ratio": 0.4, "month_end_spike": 0.2},
            savings_rate=0.2
        )

class RiskProfiler:
    """Determines user risk profile based on multiple factors"""
    
    def determine_risk_profile(self, user_profile: UserProfile, spending_pattern: SpendingPattern) -> RiskProfile:
        """Determine risk profile based on user characteristics and behavior"""
        try:
            risk_factors = self._calculate_risk_factors(user_profile, spending_pattern)
            risk_score = self._aggregate_risk_score(risk_factors)
            
            # Convert risk score to profile
            if risk_score <= 0.3:
                return RiskProfile.CONSERVATIVE
            elif risk_score <= 0.7:
                return RiskProfile.MODERATE
            else:
                return RiskProfile.AGGRESSIVE
                
        except Exception as e:
            logger.error(f"Error determining risk profile: {e}")
            return RiskProfile.MODERATE  # Default to moderate
    
    def _calculate_risk_factors(self, user_profile: UserProfile, spending_pattern: SpendingPattern) -> Dict[str, float]:
        """Calculate individual risk factors"""
        factors = {}
        
        # Age factor (younger = higher risk tolerance)
        factors["age"] = max(0, (65 - user_profile.age) / 40)  # Normalize to 0-1
        
        # Income stability (higher income = higher risk tolerance)
        factors["income"] = min(1.0, user_profile.annual_income / 1000000)  # Normalize to 0-1
        
        # Investment timeline (longer = higher risk tolerance)
        factors["timeline"] = min(1.0, user_profile.investment_timeline / 30)  # Normalize to 0-1
        
        # Spending stability (lower volatility = higher risk tolerance)
        factors["spending_stability"] = max(0, 1 - spending_pattern.spending_volatility)
        
        # Savings rate (higher savings = higher risk tolerance)
        factors["savings_rate"] = min(1.0, spending_pattern.savings_rate * 2)
        
        # Experience factor
        experience_mapping = {"beginner": 0.2, "intermediate": 0.6, "advanced": 1.0}
        factors["experience"] = experience_mapping.get(user_profile.investment_experience, 0.4)
        
        return factors
    
    def _aggregate_risk_score(self, risk_factors: Dict[str, float]) -> float:
        """Aggregate risk factors into single score"""
        weights = {
            "age": 0.25,
            "income": 0.20,
            "timeline": 0.20,
            "spending_stability": 0.15,
            "savings_rate": 0.10,
            "experience": 0.10
        }
        
        weighted_score = sum(risk_factors.get(factor, 0) * weight 
                           for factor, weight in weights.items())
        
        return min(1.0, max(0.0, weighted_score))

class InvestmentRecommendationEngine:
    """Core engine for generating personalized investment recommendations"""
    
    def __init__(self):
        self.investment_data = self._initialize_investment_data()
        self.market_conditions = self._get_current_market_conditions()
    
    def generate_recommendations(self, user_profile: UserProfile, 
                               spending_pattern: SpendingPattern) -> List[InvestmentRecommendation]:
        """Generate personalized investment recommendations"""
        try:
            # Determine risk profile
            risk_profiler = RiskProfiler()
            risk_profile = risk_profiler.determine_risk_profile(user_profile, spending_pattern)
            
            # Calculate available investment amount
            available_amount = self._calculate_available_investment_amount(user_profile, spending_pattern)
            
            # Generate portfolio allocation
            portfolio = self._generate_portfolio_allocation(risk_profile, user_profile, available_amount)
            
            # Create detailed recommendations
            recommendations = self._create_detailed_recommendations(portfolio, user_profile, risk_profile)
            
            logger.info(f"Generated {len(recommendations)} investment recommendations for user {user_profile.user_id}")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._default_recommendations()
    
    def _initialize_investment_data(self) -> Dict:
        """Initialize investment instrument data"""
        return {
            InvestmentType.FIXED_DEPOSIT: {
                "expected_return": 0.06,
                "risk_level": "Very Low",
                "liquidity": "Low",
                "minimum_investment": 1000,
                "tax_implications": "Interest taxed as income"
            },
            InvestmentType.GOVERNMENT_BONDS: {
                "expected_return": 0.07,
                "risk_level": "Low",
                "liquidity": "Medium",
                "minimum_investment": 1000,
                "tax_implications": "Interest taxed as income"
            },
            InvestmentType.MUTUAL_FUNDS: {
                "expected_return": 0.12,
                "risk_level": "Medium",
                "liquidity": "High",
                "minimum_investment": 500,
                "tax_implications": "LTCG tax applicable"
            },
            InvestmentType.INDEX_FUNDS: {
                "expected_return": 0.11,
                "risk_level": "Medium",
                "liquidity": "High",
                "minimum_investment": 500,
                "tax_implications": "LTCG tax applicable"
            },
            InvestmentType.STOCKS: {
                "expected_return": 0.15,
                "risk_level": "High",
                "liquidity": "High",
                "minimum_investment": 100,
                "tax_implications": "STCG/LTCG tax applicable"
            },
            InvestmentType.GOLD: {
                "expected_return": 0.08,
                "risk_level": "Medium",
                "liquidity": "Medium",
                "minimum_investment": 1000,
                "tax_implications": "LTCG tax applicable"
            }
        }
    
    def _get_current_market_conditions(self) -> Dict:
        """Get current market conditions (simplified)"""
        return {
            "market_sentiment": "neutral",
            "inflation_rate": 0.06,
            "interest_rate": 0.065,
            "volatility_index": 0.3
        }
    
    def _calculate_available_investment_amount(self, user_profile: UserProfile, 
                                             spending_pattern: SpendingPattern) -> float:
        """Calculate amount available for investment"""
        monthly_surplus = user_profile.annual_income / 12 - spending_pattern.monthly_spending
        
        # Ensure emergency fund (6 months expenses)
        emergency_fund_needed = spending_pattern.monthly_spending * 6
        available_savings = max(0, user_profile.current_savings - emergency_fund_needed)
        
        # Monthly investment capacity (80% of surplus to account for unexpected expenses)
        monthly_investment = max(0, monthly_surplus * 0.8)
        
        return {
            "lump_sum_available": available_savings,
            "monthly_investment_capacity": monthly_investment,
            "total_annual_capacity": available_savings + (monthly_investment * 12)
        }
    
    def _generate_portfolio_allocation(self, risk_profile: RiskProfile, 
                                     user_profile: UserProfile, 
                                     available_amount: Dict) -> Dict[InvestmentType, float]:
        """Generate portfolio allocation based on risk profile"""
        allocation_templates = {
            RiskProfile.CONSERVATIVE: {
                InvestmentType.FIXED_DEPOSIT: 0.30,
                InvestmentType.GOVERNMENT_BONDS: 0.25,
                InvestmentType.MUTUAL_FUNDS: 0.25,
                InvestmentType.INDEX_FUNDS: 0.15,
                InvestmentType.GOLD: 0.05
            },
            RiskProfile.MODERATE: {
                InvestmentType.FIXED_DEPOSIT: 0.15,
                InvestmentType.GOVERNMENT_BONDS: 0.15,
                InvestmentType.MUTUAL_FUNDS: 0.35,
                InvestmentType.INDEX_FUNDS: 0.20,
                InvestmentType.STOCKS: 0.10,
                InvestmentType.GOLD: 0.05
            },
            RiskProfile.AGGRESSIVE: {
                InvestmentType.MUTUAL_FUNDS: 0.30,
                InvestmentType.INDEX_FUNDS: 0.25,
                InvestmentType.STOCKS: 0.30,
                InvestmentType.GOVERNMENT_BONDS: 0.10,
                InvestmentType.GOLD: 0.05
            }
        }
        
        base_allocation = allocation_templates[risk_profile]
        
        # Adjust based on age and timeline
        age_adjustment = self._adjust_for_age(base_allocation, user_profile.age)
        timeline_adjustment = self._adjust_for_timeline(age_adjustment, user_profile.investment_timeline)
        
        return timeline_adjustment
    
    def _adjust_for_age(self, allocation: Dict, age: int) -> Dict[InvestmentType, float]:
        """Adjust allocation based on age"""
        adjusted = allocation.copy()
        
        # Rule of thumb: (100 - age)% in equity
        equity_target = max(0.3, min(0.8, (100 - age) / 100))
        
        # Calculate current equity allocation
        equity_types = [InvestmentType.MUTUAL_FUNDS, InvestmentType.INDEX_FUNDS, InvestmentType.STOCKS]
        current_equity = sum(adjusted.get(inv_type, 0) for inv_type in equity_types)
        
        # Adjust if needed
        if current_equity > equity_target:
            # Reduce equity, increase fixed income
            excess = current_equity - equity_target
            for inv_type in equity_types:
                if inv_type in adjusted:
                    adjusted[inv_type] *= (equity_target / current_equity)
            
            # Increase fixed income proportionally
            fixed_income_types = [InvestmentType.FIXED_DEPOSIT, InvestmentType.GOVERNMENT_BONDS]
            for inv_type in fixed_income_types:
                if inv_type in adjusted:
                    adjusted[inv_type] += excess / len(fixed_income_types)
        
        return adjusted
    
    def _adjust_for_timeline(self, allocation: Dict, timeline: int) -> Dict[InvestmentType, float]:
        """Adjust allocation based on investment timeline"""
        if timeline < 3:
            # Short term - reduce equity, increase liquid instruments
            for inv_type in [InvestmentType.STOCKS, InvestmentType.MUTUAL_FUNDS]:
                if inv_type in allocation:
                    allocation[inv_type] *= 0.5
            
            # Increase liquid investments
            allocation[InvestmentType.FIXED_DEPOSIT] = allocation.get(InvestmentType.FIXED_DEPOSIT, 0) + 0.3
        
        elif timeline > 20:
            # Very long term - can take more risk
            for inv_type in [InvestmentType.STOCKS, InvestmentType.MUTUAL_FUNDS]:
                if inv_type in allocation:
                    allocation[inv_type] *= 1.2
            
            # Reduce fixed income
            for inv_type in [InvestmentType.FIXED_DEPOSIT, InvestmentType.GOVERNMENT_BONDS]:
                if inv_type in allocation:
                    allocation[inv_type] *= 0.8
        
        # Normalize to ensure sum equals 1
        total = sum(allocation.values())
        if total > 0:
            allocation = {k: v / total for k, v in allocation.items()}
        
        return allocation
    
    def _create_detailed_recommendations(self, portfolio: Dict[InvestmentType, float], 
                                       user_profile: UserProfile, 
                                       risk_profile: RiskProfile) -> List[InvestmentRecommendation]:
        """Create detailed investment recommendations"""
        recommendations = []
        
        for investment_type, allocation in portfolio.items():
            if allocation > 0.01:  # Only include allocations > 1%
                investment_info = self.investment_data[investment_type]
                
                recommendation = InvestmentRecommendation(
                    investment_type=investment_type,
                    allocation_percentage=allocation * 100,
                    expected_return=investment_info["expected_return"],
                    risk_level=investment_info["risk_level"],
                    reasoning=self._generate_reasoning(investment_type, allocation, user_profile, risk_profile),
                    time_horizon=self._determine_time_horizon(investment_type, user_profile.investment_timeline),
                    minimum_investment=investment_info["minimum_investment"],
                    liquidity=investment_info["liquidity"],
                    tax_implications=investment_info["tax_implications"]
                )
                
                recommendations.append(recommendation)
        
        return sorted(recommendations, key=lambda x: x.allocation_percentage, reverse=True)
    
    def _generate_reasoning(self, investment_type: InvestmentType, allocation: float, 
                          user_profile: UserProfile, risk_profile: RiskProfile) -> str:
        """Generate reasoning for investment recommendation"""
        reasonings = {
            InvestmentType.FIXED_DEPOSIT: f"Recommended for capital preservation and guaranteed returns. Suitable for your {risk_profile.value} risk profile.",
            InvestmentType.GOVERNMENT_BONDS: f"Government-backed security with predictable returns. Good for {user_profile.investment_timeline}-year investment horizon.",
            InvestmentType.MUTUAL_FUNDS: f"Professional management with diversification. Aligns with your investment timeline of {user_profile.investment_timeline} years.",
            InvestmentType.INDEX_FUNDS: f"Low-cost market exposure with good long-term potential. Suitable for {risk_profile.value} investors.",
            InvestmentType.STOCKS: f"Higher growth potential for long-term wealth creation. Appropriate given your {risk_profile.value} risk tolerance.",
            InvestmentType.GOLD: f"Portfolio diversification and inflation hedge. Recommended as {allocation:.1%} of portfolio."
        }
        
        return reasonings.get(investment_type, "Recommended based on your financial profile and goals.")
    
    def _determine_time_horizon(self, investment_type: InvestmentType, timeline: int) -> str:
        """Determine appropriate time horizon for investment"""
        if timeline < 3:
            return "Short-term (1-3 years)"
        elif timeline < 10:
            return "Medium-term (3-10 years)"
        else:
            return "Long-term (10+ years)"
    
    def _default_recommendations(self) -> List[InvestmentRecommendation]:
        """Provide default recommendations when generation fails"""
        return [
            InvestmentRecommendation(
                investment_type=InvestmentType.MUTUAL_FUNDS,
                allocation_percentage=50.0,
                expected_return=0.12,
                risk_level="Medium",
                reasoning="Diversified growth potential with professional management",
                time_horizon="Medium-term (3-10 years)",
                minimum_investment=500,
                liquidity="High",
                tax_implications="LTCG tax applicable"
            ),
            InvestmentRecommendation(
                investment_type=InvestmentType.FIXED_DEPOSIT,
                allocation_percentage=30.0,
                expected_return=0.06,
                risk_level="Very Low",
                reasoning="Capital preservation and guaranteed returns",
                time_horizon="Short-term (1-3 years)",
                minimum_investment=1000,
                liquidity="Low",
                tax_implications="Interest taxed as income"
            ),
            InvestmentRecommendation(
                investment_type=InvestmentType.INDEX_FUNDS,
                allocation_percentage=20.0,
                expected_return=0.11,
                risk_level="Medium",
                reasoning="Low-cost market exposure with diversification",
                time_horizon="Long-term (10+ years)",
                minimum_investment=500,
                liquidity="High",
                tax_implications="LTCG tax applicable"
            )
        ]

def main():
    """Demo the investment recommendation system"""
    print("🎯 AI-Powered Investment Recommendation Engine")
    print("=" * 60)
    
    # Sample user profile
    user_profile = UserProfile(
        user_id="user_123",
        age=28,
        annual_income=800000,  # 8 LPA
        monthly_expenses=45000,
        current_savings=200000,
        financial_goals=["retirement", "house_purchase"],
        investment_timeline=15,
        risk_tolerance=RiskProfile.MODERATE,
        investment_experience="intermediate"
    )
    
    # Sample transaction data
    sample_transactions = [
        {"amount": 2500, "category": "FOOD_DINING", "timestamp": "2025-10-01T19:30:00"},
        {"amount": 450, "category": "TRANSPORTATION", "timestamp": "2025-10-02T08:15:00"},
        {"amount": 1200, "category": "ENTERTAINMENT", "timestamp": "2025-10-03T21:45:00"},
        {"amount": 3500, "category": "SHOPPING", "timestamp": "2025-10-04T14:20:00"},
        {"amount": 800, "category": "UTILITIES", "timestamp": "2025-10-05T10:30:00"}
    ]
    
    # Analyze spending behavior
    analyzer = SpendingBehaviorAnalyzer()
    spending_pattern = analyzer.analyze_spending_patterns(sample_transactions)
    
    print(f"\n📊 Spending Analysis:")
    print(f"Monthly Spending: ₹{spending_pattern.monthly_spending:,.0f}")
    print(f"Spending Volatility: {spending_pattern.spending_volatility:.2f}")
    print(f"Impulse Score: {spending_pattern.impulse_spending_score:.2f}")
    print(f"Savings Rate: {spending_pattern.savings_rate:.1%}")
    
    # Generate investment recommendations
    engine = InvestmentRecommendationEngine()
    recommendations = engine.generate_recommendations(user_profile, spending_pattern)
    
    print(f"\n💰 Investment Recommendations:")
    print(f"Risk Profile: {user_profile.risk_tolerance.value.title()}")
    print("-" * 60)
    
    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. {rec.investment_type.value.replace('_', ' ').title()}")
        print(f"   Allocation: {rec.allocation_percentage:.1f}%")
        print(f"   Expected Return: {rec.expected_return:.1%}")
        print(f"   Risk Level: {rec.risk_level}")
        print(f"   Time Horizon: {rec.time_horizon}")
        print(f"   Reasoning: {rec.reasoning}")
    
    print(f"\n✅ Investment recommendation system demo completed!")

if __name__ == "__main__":
    main()