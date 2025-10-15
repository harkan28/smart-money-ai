#!/usr/bin/env python3
"""
Investment Recommendations Engine - Backend Core
===============================================

Backend investment recommendation system with:
- Risk profiling based on spending patterns
- Personalized investment suggestions (FD, Mutual Funds, SIPs)
- Goal-based investment planning
- Market data integration
- Portfolio optimization
"""

import os
import json
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class RiskProfile(Enum):
    """Investment risk profiles"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class InvestmentType(Enum):
    """Types of investment instruments"""
    FIXED_DEPOSIT = "fixed_deposit"
    MUTUAL_FUND_EQUITY = "mutual_fund_equity"
    MUTUAL_FUND_DEBT = "mutual_fund_debt"
    MUTUAL_FUND_HYBRID = "mutual_fund_hybrid"
    SIP_EQUITY = "sip_equity"
    SIP_DEBT = "sip_debt"
    LIQUID_FUND = "liquid_fund"
    GOLD_ETF = "gold_etf"
    PPF = "ppf"
    ELSS = "elss"


class InvestmentGoal(Enum):
    """Investment goals"""
    EMERGENCY_FUND = "emergency_fund"
    SHORT_TERM_SAVINGS = "short_term_savings"
    LONG_TERM_WEALTH = "long_term_wealth"
    RETIREMENT = "retirement"
    CHILD_EDUCATION = "child_education"
    HOME_PURCHASE = "home_purchase"
    TAX_SAVING = "tax_saving"


@dataclass
class InvestmentRecommendation:
    """Individual investment recommendation"""
    investment_type: InvestmentType
    suggested_amount: float
    expected_return: float
    risk_level: RiskProfile
    investment_horizon: str
    rationale: str
    fund_names: List[str]
    allocation_percentage: float
    liquidity: str
    tax_benefit: bool
    minimum_investment: float


@dataclass
class UserRiskProfile:
    """User's risk assessment profile"""
    risk_score: int  # 1-10
    risk_category: RiskProfile
    factors: Dict[str, Any]
    investment_horizon: str
    monthly_surplus: float
    existing_investments: float
    financial_goals: List[InvestmentGoal]
    created_at: datetime


class RiskProfileAnalyzer:
    """Analyze user financial behavior to determine risk profile"""
    
    def __init__(self):
        self.risk_factors = {
            'age_weight': 0.2,
            'income_stability_weight': 0.25,
            'spending_pattern_weight': 0.2,
            'savings_rate_weight': 0.15,
            'investment_experience_weight': 0.2
        }
    
    def analyze_risk_profile(self, user_data: Dict[str, Any], 
                           transaction_history: List[Dict]) -> UserRiskProfile:
        """Comprehensive risk profile analysis"""
        try:
            # Analyze spending behavior
            spending_analysis = self._analyze_spending_behavior(transaction_history)
            
            # Calculate risk factors
            risk_factors = {
                'age_factor': self._calculate_age_factor(user_data.get('age', 30)),
                'income_stability': self._assess_income_stability(transaction_history),
                'spending_volatility': spending_analysis['volatility_score'],
                'savings_discipline': spending_analysis['savings_discipline'],
                'financial_cushion': self._calculate_financial_cushion(user_data, transaction_history)
            }
            
            # Calculate overall risk score
            risk_score = self._calculate_risk_score(risk_factors)
            
            # Determine risk category
            risk_category = self._determine_risk_category(risk_score)
            
            # Calculate monthly surplus
            monthly_surplus = self._calculate_monthly_surplus(user_data, transaction_history)
            
            # Infer investment goals
            investment_goals = self._infer_investment_goals(user_data, spending_analysis)
            
            return UserRiskProfile(
                risk_score=risk_score,
                risk_category=risk_category,
                factors=risk_factors,
                investment_horizon=self._determine_investment_horizon(user_data.get('age', 30)),
                monthly_surplus=monthly_surplus,
                existing_investments=user_data.get('existing_investments', 0),
                financial_goals=investment_goals,
                created_at=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error in risk profile analysis: {e}")
            return self._get_default_risk_profile()
    
    def _analyze_spending_behavior(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze spending patterns for risk assessment"""
        if not transactions:
            return {'volatility_score': 0.5, 'savings_discipline': 0.5}
        
        try:
            # Calculate monthly spending
            monthly_spending = {}
            total_income = 0
            
            for txn in transactions:
                try:
                    timestamp = datetime.fromisoformat(txn.get('timestamp', '2024-01-01')).replace(tzinfo=None)
                    amount = float(txn.get('amount', 0))
                    txn_type = txn.get('type', 'debit')
                    
                    month_key = f"{timestamp.year}-{timestamp.month:02d}"
                    
                    if txn_type == 'credit':
                        total_income += amount
                    else:
                        monthly_spending[month_key] = monthly_spending.get(month_key, 0) + amount
                        
                except Exception:
                    continue
            
            # Calculate spending metrics
            spending_amounts = list(monthly_spending.values())
            if spending_amounts:
                mean_spending = sum(spending_amounts) / len(spending_amounts)
                variance = sum((x - mean_spending) ** 2 for x in spending_amounts) / len(spending_amounts)
                volatility_score = min(1.0, (variance ** 0.5) / mean_spending) if mean_spending > 0 else 0.5
            else:
                volatility_score = 0.5
                mean_spending = 0
            
            # Calculate savings discipline (rough estimate)
            avg_monthly_spending = mean_spending
            estimated_monthly_income = total_income / max(1, len(set(monthly_spending.keys())))
            savings_rate = max(0, (estimated_monthly_income - avg_monthly_spending) / estimated_monthly_income) if estimated_monthly_income > 0 else 0
            savings_discipline = min(1.0, savings_rate)
            
            return {
                'volatility_score': volatility_score,
                'savings_discipline': savings_discipline,
                'avg_monthly_spending': avg_monthly_spending,
                'savings_rate': savings_rate
            }
            
        except Exception as e:
            logger.error(f"Error analyzing spending behavior: {e}")
            return {'volatility_score': 0.5, 'savings_discipline': 0.5}
    
    def _calculate_age_factor(self, age: int) -> float:
        """Calculate age-based risk factor (0-1, higher = more risk tolerance)"""
        if age < 25:
            return 0.9  # High risk tolerance
        elif age < 35:
            return 0.8
        elif age < 45:
            return 0.6
        elif age < 55:
            return 0.4
        else:
            return 0.2  # Conservative for older investors
    
    def _assess_income_stability(self, transactions: List[Dict]) -> float:
        """Assess income stability from transaction patterns"""
        try:
            # Look for regular income patterns
            monthly_credits = {}
            
            for txn in transactions:
                if txn.get('type') == 'credit':
                    timestamp = datetime.fromisoformat(txn.get('timestamp', '2024-01-01')).replace(tzinfo=None)
                    amount = float(txn.get('amount', 0))
                    month_key = f"{timestamp.year}-{timestamp.month:02d}"
                    
                    monthly_credits[month_key] = monthly_credits.get(month_key, 0) + amount
            
            if len(monthly_credits) < 2:
                return 0.5  # Default if insufficient data
            
            # Calculate coefficient of variation for income stability
            amounts = list(monthly_credits.values())
            mean_income = sum(amounts) / len(amounts)
            variance = sum((x - mean_income) ** 2 for x in amounts) / len(amounts)
            cv = (variance ** 0.5) / mean_income if mean_income > 0 else 1
            
            # Convert to stability score (lower CV = higher stability)
            stability_score = max(0, min(1, 1 - cv))
            return stability_score
            
        except Exception as e:
            logger.error(f"Error assessing income stability: {e}")
            return 0.5
    
    def _calculate_financial_cushion(self, user_data: Dict, transactions: List[Dict]) -> float:
        """Calculate financial cushion score"""
        try:
            # Estimate based on savings account balance and emergency fund
            current_balance = user_data.get('account_balance', 0)
            monthly_expenses = self._estimate_monthly_expenses(transactions)
            
            # Emergency fund ratio (3-6 months expenses is ideal)
            emergency_fund_ratio = current_balance / (monthly_expenses * 3) if monthly_expenses > 0 else 0
            cushion_score = min(1.0, emergency_fund_ratio / 2)  # Normalize to 0-1
            
            return cushion_score
            
        except Exception as e:
            logger.error(f"Error calculating financial cushion: {e}")
            return 0.3
    
    def _estimate_monthly_expenses(self, transactions: List[Dict]) -> float:
        """Estimate average monthly expenses"""
        try:
            monthly_expenses = {}
            
            for txn in transactions:
                if txn.get('type', 'debit') == 'debit':
                    timestamp = datetime.fromisoformat(txn.get('timestamp', '2024-01-01')).replace(tzinfo=None)
                    amount = float(txn.get('amount', 0))
                    month_key = f"{timestamp.year}-{timestamp.month:02d}"
                    
                    monthly_expenses[month_key] = monthly_expenses.get(month_key, 0) + amount
            
            if monthly_expenses:
                return sum(monthly_expenses.values()) / len(monthly_expenses)
            else:
                return 25000  # Default estimate
                
        except Exception as e:
            logger.error(f"Error estimating monthly expenses: {e}")
            return 25000
    
    def _calculate_risk_score(self, factors: Dict[str, float]) -> int:
        """Calculate overall risk score (1-10)"""
        try:
            weighted_score = (
                factors.get('age_factor', 0.5) * self.risk_factors['age_weight'] +
                factors.get('income_stability', 0.5) * self.risk_factors['income_stability_weight'] +
                (1 - factors.get('spending_volatility', 0.5)) * self.risk_factors['spending_pattern_weight'] +
                factors.get('savings_discipline', 0.5) * self.risk_factors['savings_rate_weight'] +
                factors.get('financial_cushion', 0.3) * self.risk_factors['investment_experience_weight']
            )
            
            # Convert to 1-10 scale
            risk_score = max(1, min(10, int(weighted_score * 10)))
            return risk_score
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return 5
    
    def _determine_risk_category(self, risk_score: int) -> RiskProfile:
        """Determine risk category from score"""
        if risk_score <= 3:
            return RiskProfile.CONSERVATIVE
        elif risk_score <= 6:
            return RiskProfile.MODERATE
        else:
            return RiskProfile.AGGRESSIVE
    
    def _determine_investment_horizon(self, age: int) -> str:
        """Determine investment horizon based on age"""
        if age < 30:
            return "long_term"  # 20+ years
        elif age < 45:
            return "medium_term"  # 10-20 years
        else:
            return "short_term"  # 5-10 years
    
    def _calculate_monthly_surplus(self, user_data: Dict, transactions: List[Dict]) -> float:
        """Calculate monthly investment surplus"""
        try:
            # Estimate from transaction analysis
            spending_analysis = self._analyze_spending_behavior(transactions)
            estimated_income = user_data.get('monthly_income', spending_analysis.get('avg_monthly_spending', 25000) * 1.3)
            avg_expenses = spending_analysis.get('avg_monthly_spending', 25000)
            
            monthly_surplus = max(0, estimated_income - avg_expenses)
            return monthly_surplus * 0.7  # Recommend investing 70% of surplus
            
        except Exception as e:
            logger.error(f"Error calculating monthly surplus: {e}")
            return 5000  # Default
    
    def _infer_investment_goals(self, user_data: Dict, spending_analysis: Dict) -> List[InvestmentGoal]:
        """Infer likely investment goals from user profile"""
        goals = []
        
        age = user_data.get('age', 30)
        savings_rate = spending_analysis.get('savings_rate', 0.1)
        
        # Everyone needs emergency fund
        goals.append(InvestmentGoal.EMERGENCY_FUND)
        
        # Age-based goals
        if age < 35:
            goals.extend([InvestmentGoal.LONG_TERM_WEALTH, InvestmentGoal.TAX_SAVING])
        elif age < 50:
            goals.extend([InvestmentGoal.RETIREMENT, InvestmentGoal.CHILD_EDUCATION])
        else:
            goals.extend([InvestmentGoal.RETIREMENT, InvestmentGoal.SHORT_TERM_SAVINGS])
        
        # High savers might consider home purchase
        if savings_rate > 0.2:
            goals.append(InvestmentGoal.HOME_PURCHASE)
        
        return goals[:4]  # Limit to 4 primary goals
    
    def _get_default_risk_profile(self) -> UserRiskProfile:
        """Return default risk profile"""
        return UserRiskProfile(
            risk_score=5,
            risk_category=RiskProfile.MODERATE,
            factors={'age_factor': 0.6, 'income_stability': 0.5, 'savings_discipline': 0.5},
            investment_horizon="medium_term",
            monthly_surplus=5000,
            existing_investments=0,
            financial_goals=[InvestmentGoal.EMERGENCY_FUND, InvestmentGoal.LONG_TERM_WEALTH],
            created_at=datetime.now()
        )


class InvestmentRecommendationEngine:
    """Generate personalized investment recommendations"""
    
    def __init__(self):
        self.fund_database = self._load_fund_database()
        self.current_rates = self._load_current_rates()
    
    def generate_recommendations(self, risk_profile: UserRiskProfile) -> List[InvestmentRecommendation]:
        """Generate comprehensive investment recommendations"""
        try:
            recommendations = []
            monthly_surplus = risk_profile.monthly_surplus
            
            # Emergency fund (always first priority)
            if InvestmentGoal.EMERGENCY_FUND in risk_profile.financial_goals:
                emergency_rec = self._recommend_emergency_fund(risk_profile, monthly_surplus * 0.3)
                recommendations.append(emergency_rec)
            
            # Tax saving recommendations
            if InvestmentGoal.TAX_SAVING in risk_profile.financial_goals:
                tax_rec = self._recommend_tax_saving(risk_profile, monthly_surplus * 0.2)
                recommendations.append(tax_rec)
            
            # Core portfolio recommendations based on risk profile
            core_recommendations = self._generate_core_portfolio(risk_profile, monthly_surplus * 0.5)
            recommendations.extend(core_recommendations)
            
            # Goal-specific recommendations
            goal_recommendations = self._generate_goal_based_recommendations(risk_profile)
            recommendations.extend(goal_recommendations)
            
            # Optimize allocation
            recommendations = self._optimize_allocation(recommendations, monthly_surplus)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return self._get_default_recommendations(risk_profile)
    
    def _recommend_emergency_fund(self, risk_profile: UserRiskProfile, amount: float) -> InvestmentRecommendation:
        """Recommend emergency fund options"""
        return InvestmentRecommendation(
            investment_type=InvestmentType.LIQUID_FUND,
            suggested_amount=amount,
            expected_return=4.5,  # % per annum
            risk_level=RiskProfile.CONSERVATIVE,
            investment_horizon="immediate",
            rationale="Emergency funds should be highly liquid with capital protection",
            fund_names=["Axis Liquid Fund", "HDFC Liquid Fund", "ICICI Liquid Fund"],
            allocation_percentage=30.0,
            liquidity="immediate",
            tax_benefit=False,
            minimum_investment=1000
        )
    
    def _recommend_tax_saving(self, risk_profile: UserRiskProfile, amount: float) -> InvestmentRecommendation:
        """Recommend tax saving options"""
        return InvestmentRecommendation(
            investment_type=InvestmentType.ELSS,
            suggested_amount=min(amount, 12500),  # Max 1.5L per year = ~12.5k per month
            expected_return=12.0 if risk_profile.risk_category == RiskProfile.AGGRESSIVE else 10.0,
            risk_level=risk_profile.risk_category,
            investment_horizon="3+ years",
            rationale="ELSS provides tax deduction under 80C with potential for good returns",
            fund_names=["Axis Long Term Equity Fund", "Mirae Asset Tax Saver Fund", "Invesco India Tax Plan"],
            allocation_percentage=20.0,
            liquidity="3 years lock-in",
            tax_benefit=True,
            minimum_investment=500
        )
    
    def _generate_core_portfolio(self, risk_profile: UserRiskProfile, amount: float) -> List[InvestmentRecommendation]:
        """Generate core portfolio based on risk profile"""
        recommendations = []
        
        if risk_profile.risk_category == RiskProfile.CONSERVATIVE:
            # Conservative: 70% debt, 30% equity
            debt_rec = InvestmentRecommendation(
                investment_type=InvestmentType.MUTUAL_FUND_DEBT,
                suggested_amount=amount * 0.7,
                expected_return=7.0,
                risk_level=RiskProfile.CONSERVATIVE,
                investment_horizon="1-3 years",
                rationale="Debt funds provide stable returns with low volatility",
                fund_names=["HDFC Short Term Debt Fund", "ICICI Corporate Bond Fund"],
                allocation_percentage=35.0,
                liquidity="1-7 days",
                tax_benefit=False,
                minimum_investment=1000
            )
            
            equity_rec = InvestmentRecommendation(
                investment_type=InvestmentType.SIP_EQUITY,
                suggested_amount=amount * 0.3,
                expected_return=10.0,
                risk_level=RiskProfile.MODERATE,
                investment_horizon="5+ years",
                rationale="Small equity allocation for long-term wealth creation",
                fund_names=["HDFC Index Fund Nifty 50", "Axis Bluechip Fund"],
                allocation_percentage=15.0,
                liquidity="1-3 days",
                tax_benefit=False,
                minimum_investment=500
            )
            
            recommendations.extend([debt_rec, equity_rec])
            
        elif risk_profile.risk_category == RiskProfile.MODERATE:
            # Moderate: 50% equity, 30% debt, 20% hybrid
            equity_rec = InvestmentRecommendation(
                investment_type=InvestmentType.SIP_EQUITY,
                suggested_amount=amount * 0.5,
                expected_return=12.0,
                risk_level=RiskProfile.MODERATE,
                investment_horizon="5+ years",
                rationale="Balanced equity exposure for wealth creation",
                fund_names=["Mirae Asset Large Cap Fund", "Axis Midcap Fund", "HDFC Small Cap Fund"],
                allocation_percentage=25.0,
                liquidity="1-3 days",
                tax_benefit=False,
                minimum_investment=500
            )
            
            debt_rec = InvestmentRecommendation(
                investment_type=InvestmentType.MUTUAL_FUND_DEBT,
                suggested_amount=amount * 0.3,
                expected_return=7.5,
                risk_level=RiskProfile.CONSERVATIVE,
                investment_horizon="2-5 years",
                rationale="Debt component for stability and regular income",
                fund_names=["ICICI Dynamic Bond Fund", "HDFC Medium Term Debt Fund"],
                allocation_percentage=15.0,
                liquidity="1-7 days",
                tax_benefit=False,
                minimum_investment=1000
            )
            
            hybrid_rec = InvestmentRecommendation(
                investment_type=InvestmentType.MUTUAL_FUND_HYBRID,
                suggested_amount=amount * 0.2,
                expected_return=9.5,
                risk_level=RiskProfile.MODERATE,
                investment_horizon="3-7 years",
                rationale="Hybrid funds provide balanced exposure with professional management",
                fund_names=["HDFC Hybrid Equity Fund", "ICICI Balanced Advantage Fund"],
                allocation_percentage=10.0,
                liquidity="1-3 days",
                tax_benefit=False,
                minimum_investment=1000
            )
            
            recommendations.extend([equity_rec, debt_rec, hybrid_rec])
            
        else:  # AGGRESSIVE
            # Aggressive: 80% equity, 20% debt
            equity_rec = InvestmentRecommendation(
                investment_type=InvestmentType.SIP_EQUITY,
                suggested_amount=amount * 0.8,
                expected_return=14.0,
                risk_level=RiskProfile.AGGRESSIVE,
                investment_horizon="7+ years",
                rationale="High equity allocation for maximum wealth creation potential",
                fund_names=["Mirae Asset Emerging Bluechip Fund", "Axis Small Cap Fund", "Parag Parikh Flexi Cap Fund"],
                allocation_percentage=40.0,
                liquidity="1-3 days",
                tax_benefit=False,
                minimum_investment=500
            )
            
            debt_rec = InvestmentRecommendation(
                investment_type=InvestmentType.MUTUAL_FUND_DEBT,
                suggested_amount=amount * 0.2,
                expected_return=7.0,
                risk_level=RiskProfile.CONSERVATIVE,
                investment_horizon="1-3 years",
                rationale="Small debt allocation for portfolio stability",
                fund_names=["HDFC Ultra Short Term Fund"],
                allocation_percentage=10.0,
                liquidity="1-2 days",
                tax_benefit=False,
                minimum_investment=1000
            )
            
            recommendations.extend([equity_rec, debt_rec])
        
        return recommendations
    
    def _generate_goal_based_recommendations(self, risk_profile: UserRiskProfile) -> List[InvestmentRecommendation]:
        """Generate recommendations for specific goals"""
        recommendations = []
        
        for goal in risk_profile.financial_goals:
            if goal == InvestmentGoal.RETIREMENT and risk_profile.monthly_surplus > 3000:
                ppf_rec = InvestmentRecommendation(
                    investment_type=InvestmentType.PPF,
                    suggested_amount=min(12500, risk_profile.monthly_surplus * 0.1),  # Max 1.5L per year
                    expected_return=7.1,  # Current PPF rate
                    risk_level=RiskProfile.CONSERVATIVE,
                    investment_horizon="15 years",
                    rationale="PPF offers tax-free returns and is ideal for retirement planning",
                    fund_names=["Public Provident Fund"],
                    allocation_percentage=5.0,
                    liquidity="15 years lock-in",
                    tax_benefit=True,
                    minimum_investment=500
                )
                recommendations.append(ppf_rec)
                
            elif goal == InvestmentGoal.HOME_PURCHASE:
                # FD for home down payment
                fd_rec = InvestmentRecommendation(
                    investment_type=InvestmentType.FIXED_DEPOSIT,
                    suggested_amount=risk_profile.monthly_surplus * 0.1,
                    expected_return=6.5,
                    risk_level=RiskProfile.CONSERVATIVE,
                    investment_horizon="2-5 years",
                    rationale="Fixed deposits provide guaranteed returns for home purchase goals",
                    fund_names=["HDFC Bank FD", "SBI Bank FD", "ICICI Bank FD"],
                    allocation_percentage=5.0,
                    liquidity="Premature withdrawal allowed",
                    tax_benefit=False,
                    minimum_investment=10000
                )
                recommendations.append(fd_rec)
        
        return recommendations
    
    def _optimize_allocation(self, recommendations: List[InvestmentRecommendation], 
                           total_amount: float) -> List[InvestmentRecommendation]:
        """Optimize allocation percentages to match available amount"""
        if not recommendations:
            return recommendations
        
        try:
            total_suggested = sum(rec.suggested_amount for rec in recommendations)
            
            if total_suggested <= total_amount:
                return recommendations
            
            # Scale down proportionally if over budget
            scale_factor = total_amount / total_suggested
            
            for rec in recommendations:
                rec.suggested_amount *= scale_factor
                rec.allocation_percentage *= scale_factor
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error optimizing allocation: {e}")
            return recommendations
    
    def _load_fund_database(self) -> Dict[str, Any]:
        """Load fund database (placeholder for real data)"""
        return {
            "equity_funds": [
                {"name": "HDFC Top 100 Fund", "category": "large_cap", "returns_1y": 15.2, "expense_ratio": 1.05},
                {"name": "Mirae Asset Large Cap Fund", "category": "large_cap", "returns_1y": 14.8, "expense_ratio": 1.25},
                {"name": "Axis Midcap Fund", "category": "mid_cap", "returns_1y": 18.5, "expense_ratio": 1.35},
            ],
            "debt_funds": [
                {"name": "HDFC Short Term Debt Fund", "category": "short_term", "returns_1y": 7.2, "expense_ratio": 0.45},
                {"name": "ICICI Corporate Bond Fund", "category": "corporate", "returns_1y": 7.8, "expense_ratio": 0.55},
            ]
        }
    
    def _load_current_rates(self) -> Dict[str, float]:
        """Load current market rates"""
        return {
            "fd_rates": {"sbi": 6.5, "hdfc": 6.75, "icici": 6.5},
            "ppf_rate": 7.1,
            "savings_rate": 3.0,
            "inflation": 5.5
        }
    
    def _get_default_recommendations(self, risk_profile: UserRiskProfile) -> List[InvestmentRecommendation]:
        """Return default recommendations if generation fails"""
        return [
            InvestmentRecommendation(
                investment_type=InvestmentType.SIP_EQUITY,
                suggested_amount=3000,
                expected_return=12.0,
                risk_level=RiskProfile.MODERATE,
                investment_horizon="5+ years",
                rationale="Diversified equity SIP for long-term wealth creation",
                fund_names=["HDFC Index Fund Nifty 50"],
                allocation_percentage=60.0,
                liquidity="1-3 days",
                tax_benefit=False,
                minimum_investment=500
            ),
            InvestmentRecommendation(
                investment_type=InvestmentType.LIQUID_FUND,
                suggested_amount=2000,
                expected_return=4.5,
                risk_level=RiskProfile.CONSERVATIVE,
                investment_horizon="immediate",
                rationale="Emergency fund with high liquidity",
                fund_names=["HDFC Liquid Fund"],
                allocation_percentage=40.0,
                liquidity="immediate",
                tax_benefit=False,
                minimum_investment=1000
            )
        ]


class InvestmentPlatformIntegrator:
    """Integration with investment platforms"""
    
    def __init__(self):
        self.supported_platforms = {
            "zerodha": {"api_available": True, "commission": 0, "features": ["mutual_funds", "stocks"]},
            "groww": {"api_available": False, "commission": 0, "features": ["mutual_funds", "stocks", "fd"]},
            "paytm_money": {"api_available": False, "commission": 0, "features": ["mutual_funds"]},
            "kuvera": {"api_available": False, "commission": 0, "features": ["mutual_funds"]}
        }
    
    def get_platform_recommendations(self, investment_recommendations: List[InvestmentRecommendation]) -> Dict[str, Any]:
        """Recommend platforms for specific investments"""
        platform_mapping = {}
        
        for rec in investment_recommendations:
            suitable_platforms = []
            
            if rec.investment_type in [InvestmentType.SIP_EQUITY, InvestmentType.MUTUAL_FUND_EQUITY]:
                suitable_platforms = ["groww", "zerodha", "paytm_money", "kuvera"]
            elif rec.investment_type == InvestmentType.FIXED_DEPOSIT:
                suitable_platforms = ["groww", "banks"]
            elif rec.investment_type == InvestmentType.LIQUID_FUND:
                suitable_platforms = ["groww", "zerodha", "kuvera"]
            
            platform_mapping[rec.investment_type.value] = {
                "recommended_platforms": suitable_platforms,
                "investment_links": self._generate_investment_links(rec, suitable_platforms)
            }
        
        return {
            "platform_recommendations": platform_mapping,
            "setup_instructions": self._generate_setup_instructions(),
            "comparison": self._compare_platforms()
        }
    
    def _generate_investment_links(self, recommendation: InvestmentRecommendation, platforms: List[str]) -> Dict[str, str]:
        """Generate investment links for platforms"""
        links = {}
        
        for platform in platforms:
            if platform == "groww":
                links[platform] = f"https://groww.in/mutual-funds/{recommendation.fund_names[0].lower().replace(' ', '-')}"
            elif platform == "zerodha":
                links[platform] = f"https://zerodha.com/mutual-funds/{recommendation.fund_names[0].lower().replace(' ', '-')}"
            elif platform == "paytm_money":
                links[platform] = f"https://www.paytmmoney.com/mutual-funds/{recommendation.fund_names[0].lower().replace(' ', '-')}"
            elif platform == "kuvera":
                links[platform] = f"https://kuvera.in/mutual-funds/{recommendation.fund_names[0].lower().replace(' ', '-')}"
        
        return links
    
    def _generate_setup_instructions(self) -> Dict[str, List[str]]:
        """Generate platform setup instructions"""
        return {
            "groww": [
                "1. Download Groww app or visit groww.in",
                "2. Complete KYC with Aadhaar and PAN",
                "3. Add bank account for auto-debit",
                "4. Start SIP with minimum ₹500"
            ],
            "zerodha": [
                "1. Open Zerodha Kite account",
                "2. Complete KYC process",
                "3. Fund your account",
                "4. Access Coin for mutual fund investments"
            ],
            "general": [
                "1. Keep PAN card and Aadhaar ready",
                "2. Have bank account details available",
                "3. Complete video KYC if required",
                "4. Set up auto-debit for SIPs"
            ]
        }
    
    def _compare_platforms(self) -> Dict[str, Any]:
        """Compare investment platforms"""
        return {
            "commission_free": ["groww", "zerodha", "paytm_money", "kuvera"],
            "best_for_beginners": ["groww", "paytm_money"],
            "advanced_features": ["zerodha"],
            "mobile_first": ["groww", "paytm_money"],
            "recommendation": "Groww is recommended for beginners due to ease of use and zero commission"
        }


class InvestmentRecommendationSystem:
    """Main investment recommendation system"""
    
    def __init__(self, database_path: str = "data/investments.db"):
        self.database_path = database_path
        self.risk_analyzer = RiskProfileAnalyzer()
        self.recommendation_engine = InvestmentRecommendationEngine()
        self.platform_integrator = InvestmentPlatformIntegrator()
        self.init_database()
    
    def init_database(self):
        """Initialize investment database"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # User risk profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS risk_profiles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        risk_score INTEGER NOT NULL,
                        risk_category TEXT NOT NULL,
                        monthly_surplus REAL NOT NULL,
                        investment_horizon TEXT NOT NULL,
                        factors TEXT NOT NULL,
                        financial_goals TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                ''')
                
                # Investment recommendations table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS investment_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        investment_type TEXT NOT NULL,
                        suggested_amount REAL NOT NULL,
                        expected_return REAL NOT NULL,
                        allocation_percentage REAL NOT NULL,
                        fund_names TEXT NOT NULL,
                        rationale TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                ''')
                
                conn.commit()
                logger.info("Investment database initialized")
                
        except Exception as e:
            logger.error(f"Error initializing investment database: {e}")
    
    def get_investment_recommendations(self, user_data: Dict[str, Any], 
                                     transaction_history: List[Dict],
                                     user_id: str = "default") -> Dict[str, Any]:
        """Get comprehensive investment recommendations"""
        try:
            # Analyze risk profile
            risk_profile = self.risk_analyzer.analyze_risk_profile(user_data, transaction_history)
            
            # Generate recommendations
            recommendations = self.recommendation_engine.generate_recommendations(risk_profile)
            
            # Get platform recommendations
            platform_info = self.platform_integrator.get_platform_recommendations(recommendations)
            
            # Save to database
            self._save_recommendations(user_id, risk_profile, recommendations)
            
            # Prepare response
            response = {
                'user_profile': {
                    'risk_score': risk_profile.risk_score,
                    'risk_category': risk_profile.risk_category.value,
                    'investment_horizon': risk_profile.investment_horizon,
                    'monthly_surplus': risk_profile.monthly_surplus,
                    'financial_goals': [goal.value for goal in risk_profile.financial_goals]
                },
                'recommendations': [asdict(rec) for rec in recommendations],
                'portfolio_summary': self._create_portfolio_summary(recommendations),
                'platform_guidance': platform_info,
                'next_steps': self._generate_next_steps(recommendations),
                'disclaimer': self._get_investment_disclaimer(),
                'generated_at': datetime.now().isoformat()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting investment recommendations: {e}")
            return {'error': str(e)}
    
    def _save_recommendations(self, user_id: str, risk_profile: UserRiskProfile, 
                            recommendations: List[InvestmentRecommendation]):
        """Save recommendations to database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Save risk profile
                cursor.execute('''
                    INSERT OR REPLACE INTO risk_profiles 
                    (user_id, risk_score, risk_category, monthly_surplus, investment_horizon, 
                     factors, financial_goals, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    user_id, risk_profile.risk_score, risk_profile.risk_category.value,
                    risk_profile.monthly_surplus, risk_profile.investment_horizon,
                    json.dumps(risk_profile.factors), 
                    json.dumps([goal.value for goal in risk_profile.financial_goals]),
                    risk_profile.created_at.isoformat(), datetime.now().isoformat()
                ))
                
                # Clear old recommendations
                cursor.execute('DELETE FROM investment_recommendations WHERE user_id = ?', (user_id,))
                
                # Save new recommendations
                for rec in recommendations:
                    cursor.execute('''
                        INSERT INTO investment_recommendations 
                        (user_id, investment_type, suggested_amount, expected_return, 
                         allocation_percentage, fund_names, rationale, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id, rec.investment_type.value, rec.suggested_amount,
                        rec.expected_return, rec.allocation_percentage,
                        json.dumps(rec.fund_names), rec.rationale, datetime.now().isoformat()
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving recommendations: {e}")
    
    def _create_portfolio_summary(self, recommendations: List[InvestmentRecommendation]) -> Dict[str, Any]:
        """Create portfolio summary"""
        total_amount = sum(rec.suggested_amount for rec in recommendations)
        expected_weighted_return = sum(rec.expected_return * rec.suggested_amount for rec in recommendations) / total_amount if total_amount > 0 else 0
        
        asset_allocation = {}
        for rec in recommendations:
            if rec.risk_level == RiskProfile.CONSERVATIVE:
                asset_allocation['debt'] = asset_allocation.get('debt', 0) + rec.allocation_percentage
            elif rec.risk_level == RiskProfile.AGGRESSIVE:
                asset_allocation['equity'] = asset_allocation.get('equity', 0) + rec.allocation_percentage
            else:
                asset_allocation['hybrid'] = asset_allocation.get('hybrid', 0) + rec.allocation_percentage
        
        return {
            'total_monthly_investment': total_amount,
            'expected_annual_return': expected_weighted_return,
            'asset_allocation': asset_allocation,
            'number_of_investments': len(recommendations),
            'tax_saving_amount': sum(rec.suggested_amount for rec in recommendations if rec.tax_benefit),
            'high_liquidity_amount': sum(rec.suggested_amount for rec in recommendations if rec.liquidity in ['immediate', '1-3 days'])
        }
    
    def _generate_next_steps(self, recommendations: List[InvestmentRecommendation]) -> List[str]:
        """Generate actionable next steps"""
        steps = [
            "1. Complete KYC on your preferred investment platform",
            "2. Link your bank account for auto-debit",
            "3. Start with emergency fund (liquid funds) first",
            "4. Set up SIPs for equity funds with auto-debit",
            "5. Review and rebalance portfolio quarterly"
        ]
        
        # Add specific steps based on recommendations
        if any(rec.tax_benefit for rec in recommendations):
            steps.append("6. Prioritize tax-saving investments before March 31st")
        
        if any(rec.investment_type == InvestmentType.PPF for rec in recommendations):
            steps.append("6. Open PPF account at bank or post office")
        
        return steps
    
    def _get_investment_disclaimer(self) -> str:
        """Get investment disclaimer"""
        return ("Mutual fund investments are subject to market risks. "
                "Please read all scheme related documents carefully. "
                "Past performance is not indicative of future returns. "
                "Consider consulting a financial advisor for personalized advice.")


def main():
    """Demo the investment recommendation system"""
    print("💼 Investment Recommendation System - Backend Demo")
    print("=" * 70)
    
    # Initialize system
    investment_system = InvestmentRecommendationSystem()
    
    # Sample user data
    user_data = {
        "age": 28,
        "monthly_income": 75000,
        "account_balance": 150000,
        "existing_investments": 50000
    }
    
    # Sample transaction history
    transaction_history = [
        {"amount": 75000, "type": "credit", "category": "SALARY", "timestamp": "2025-01-01T00:00:00"},
        {"amount": 15000, "type": "debit", "category": "RENT", "timestamp": "2025-01-02T00:00:00"},
        {"amount": 8000, "type": "debit", "category": "GROCERIES", "timestamp": "2025-01-03T00:00:00"},
        {"amount": 3000, "type": "debit", "category": "FOOD_DINING", "timestamp": "2025-01-04T00:00:00"},
        {"amount": 2000, "type": "debit", "category": "TRANSPORTATION", "timestamp": "2025-01-05T00:00:00"},
        {"amount": 5000, "type": "debit", "category": "UTILITIES", "timestamp": "2025-01-06T00:00:00"},
        {"amount": 10000, "type": "debit", "category": "SHOPPING", "timestamp": "2025-01-07T00:00:00"},
        {"amount": 2000, "type": "debit", "category": "ENTERTAINMENT", "timestamp": "2025-01-08T00:00:00"},
    ]
    
    print("\n🎯 Generating Investment Recommendations...")
    print("-" * 60)
    
    # Get recommendations
    result = investment_system.get_investment_recommendations(
        user_data, transaction_history, "demo_user"
    )
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    # Display user profile
    profile = result['user_profile']
    print(f"\n👤 User Investment Profile:")
    print(f"   Risk Score: {profile['risk_score']}/10")
    print(f"   Risk Category: {profile['risk_category'].upper()}")
    print(f"   Investment Horizon: {profile['investment_horizon']}")
    print(f"   Monthly Surplus: ₹{profile['monthly_surplus']:,.0f}")
    print(f"   Financial Goals: {', '.join(profile['financial_goals'])}")
    
    # Display recommendations
    print(f"\n💰 Investment Recommendations:")
    print("-" * 60)
    
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"\n{i}. {rec['investment_type'].value.replace('_', ' ').title()}")
        print(f"   Amount: ₹{rec['suggested_amount']:,.0f}/month ({rec['allocation_percentage']:.1f}%)")
        print(f"   Expected Return: {rec['expected_return']:.1f}% p.a.")
        print(f"   Risk Level: {rec['risk_level'].value.title()}")
        print(f"   Horizon: {rec['investment_horizon']}")
        print(f"   Funds: {', '.join(rec['fund_names'][:2])}")
        print(f"   Rationale: {rec['rationale']}")
        if rec['tax_benefit']:
            print(f"   💡 Tax Benefit: ₹80C deduction available")
    
    # Portfolio summary
    summary = result['portfolio_summary']
    print(f"\n📊 Portfolio Summary:")
    print("-" * 30)
    print(f"Total Monthly Investment: ₹{summary['total_monthly_investment']:,.0f}")
    print(f"Expected Annual Return: {summary['expected_annual_return']:.1f}%")
    print(f"Number of Investments: {summary['number_of_investments']}")
    print(f"Tax Saving Amount: ₹{summary['tax_saving_amount']:,.0f}")
    
    # Next steps
    print(f"\n📝 Next Steps:")
    print("-" * 20)
    for step in result['next_steps'][:5]:
        print(f"   {step}")
    
    # Platform recommendations
    platforms = result['platform_guidance']['comparison']
    print(f"\n🔗 Recommended Platform: {platforms['recommendation']}")
    
    print(f"\n⚠️ Disclaimer: {result['disclaimer']}")
    print(f"\n✅ Investment recommendation system demo completed!")


if __name__ == "__main__":
    main()