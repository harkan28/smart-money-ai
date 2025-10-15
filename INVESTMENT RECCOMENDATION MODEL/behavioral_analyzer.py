#!/usr/bin/env python3
"""
Behavioral Finance Analysis Module
Advanced spending behavior analysis and financial psychology insights
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BehavioralInsight:
    """Behavioral finance insight with scoring and recommendations"""
    insight_type: str
    score: float  # 0-1 scale
    description: str
    recommendations: List[str]
    impact_level: str  # "low", "medium", "high"
    financial_impact: float  # estimated monthly impact in currency

@dataclass
class SpendingTrigger:
    """Identified spending trigger with analysis"""
    trigger_type: str
    trigger_conditions: Dict
    average_amount: float
    frequency: int
    emotional_score: float
    mitigation_strategies: List[str]

class BehavioralFinanceAnalyzer:
    """Advanced behavioral finance analysis for spending patterns"""
    
    def __init__(self):
        self.spending_triggers = []
        self.behavioral_patterns = {}
        self.emotional_states = {}
    
    def analyze_behavioral_patterns(self, transactions: List[Dict]) -> Dict[str, BehavioralInsight]:
        """Comprehensive behavioral finance analysis"""
        try:
            if not transactions:
                logger.warning("No transactions provided for behavioral analysis")
                return self._default_behavioral_insights()
            
            df = self._prepare_transaction_data(transactions)
            
            insights = {}
            
            # 1. Emotional Spending Analysis
            insights["emotional_spending"] = self._analyze_emotional_spending(df)
            
            # 2. Temporal Spending Patterns
            insights["temporal_patterns"] = self._analyze_temporal_patterns(df)
            
            # 3. Social Spending Influence
            insights["social_influence"] = self._analyze_social_spending(df)
            
            # 4. Impulse Buying Behavior
            insights["impulse_buying"] = self._analyze_impulse_buying(df)
            
            # 5. Subscription and Recurring Payment Analysis
            insights["subscription_analysis"] = self._analyze_subscriptions(df)
            
            # 6. Spending Escalation Patterns
            insights["spending_escalation"] = self._analyze_spending_escalation(df)
            
            # 7. Category-based Behavioral Patterns
            insights["category_behavior"] = self._analyze_category_behavior(df)
            
            # 8. Financial Stress Indicators
            insights["financial_stress"] = self._analyze_financial_stress(df)
            
            logger.info(f"Completed behavioral analysis with {len(insights)} insights")
            return insights
            
        except Exception as e:
            logger.error(f"Error in behavioral analysis: {e}")
            return self._default_behavioral_insights()
    
    def _prepare_transaction_data(self, transactions: List[Dict]) -> pd.DataFrame:
        """Prepare and enrich transaction data for analysis"""
        df = pd.DataFrame(transactions)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'])
        else:
            df['datetime'] = pd.to_datetime('today')
        
        # Extract temporal features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6])
        df['is_late_night'] = df['hour'].isin([22, 23, 0, 1, 2])
        df['is_early_morning'] = df['hour'].isin([6, 7, 8, 9])
        df['month'] = df['datetime'].dt.month
        df['day_of_month'] = df['datetime'].dt.day
        
        # Calculate transaction intervals
        df = df.sort_values('datetime')
        df['time_since_last'] = df['datetime'].diff().dt.total_seconds() / 3600  # hours
        
        # Add amount categories
        df['amount_category'] = pd.cut(df['amount'], 
                                     bins=[0, 100, 500, 2000, float('inf')], 
                                     labels=['micro', 'small', 'medium', 'large'])
        
        return df
    
    def _analyze_emotional_spending(self, df: pd.DataFrame) -> BehavioralInsight:
        """Analyze emotional spending patterns"""
        emotional_indicators = []
        
        # Late night transactions (often emotional)
        late_night_ratio = df['is_late_night'].mean()
        emotional_indicators.append(late_night_ratio * 0.3)
        
        # Quick successive transactions
        quick_transactions = (df['time_since_last'] < 1).mean()  # within 1 hour
        emotional_indicators.append(quick_transactions * 0.25)
        
        # High amount transactions in emotional categories
        emotional_categories = ['ENTERTAINMENT', 'SHOPPING', 'FOOD_DINING']
        emotional_spending = df[df['category'].isin(emotional_categories)]['amount'].sum()
        total_spending = df['amount'].sum()
        emotional_ratio = emotional_spending / total_spending if total_spending > 0 else 0
        emotional_indicators.append(emotional_ratio * 0.25)
        
        # Weekend spending spikes
        weekend_spending_ratio = df[df['is_weekend']]['amount'].sum() / total_spending if total_spending > 0 else 0
        weekend_excess = max(0, weekend_spending_ratio - 0.3)  # Above normal weekend spending
        emotional_indicators.append(weekend_excess * 0.2)
        
        emotional_score = sum(emotional_indicators)
        
        # Generate recommendations
        recommendations = []
        if emotional_score > 0.6:
            recommendations.extend([
                "Consider implementing a 24-hour cooling-off period for purchases over ₹1000",
                "Set up automatic savings transfers before weekend to reduce available spending money",
                "Use mindfulness apps or techniques before making unplanned purchases"
            ])
        elif emotional_score > 0.3:
            recommendations.extend([
                "Track your mood before making purchases to identify patterns",
                "Set weekly limits for entertainment and discretionary spending"
            ])
        else:
            recommendations.append("Good emotional spending control maintained")
        
        # Estimate financial impact
        emotional_transactions = df[
            (df['is_late_night']) | 
            (df['time_since_last'] < 1) | 
            (df['category'].isin(emotional_categories) & (df['amount'] > df['amount'].quantile(0.8)))
        ]
        monthly_emotional_spending = emotional_transactions['amount'].sum() * (30 / len(df)) if len(df) > 0 else 0
        
        return BehavioralInsight(
            insight_type="emotional_spending",
            score=emotional_score,
            description=f"Emotional spending score: {emotional_score:.1%}. "
                       f"Late-night transactions: {late_night_ratio:.1%}, "
                       f"Quick successive purchases: {quick_transactions:.1%}",
            recommendations=recommendations,
            impact_level="high" if emotional_score > 0.6 else "medium" if emotional_score > 0.3 else "low",
            financial_impact=monthly_emotional_spending
        )
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> BehavioralInsight:
        """Analyze time-based spending patterns"""
        patterns = {}
        
        # Peak spending hours
        hourly_spending = df.groupby('hour')['amount'].sum()
        peak_hour = hourly_spending.idxmax() if len(hourly_spending) > 0 else 12
        
        # Day of week patterns
        dow_spending = df.groupby('day_of_week')['amount'].sum()
        peak_day = dow_spending.idxmax() if len(dow_spending) > 0 else 1
        
        # Month-end spending pattern
        month_end_spending = df[df['day_of_month'] >= 25]['amount'].sum()
        month_start_spending = df[df['day_of_month'] <= 5]['amount'].sum()
        month_end_ratio = month_end_spending / (month_end_spending + month_start_spending) if (month_end_spending + month_start_spending) > 0 else 0.5
        
        # Weekend vs weekday analysis
        weekend_avg = df[df['is_weekend']]['amount'].mean() if df['is_weekend'].any() else 0
        weekday_avg = df[~df['is_weekend']]['amount'].mean() if (~df['is_weekend']).any() else 0
        weekend_ratio = weekend_avg / weekday_avg if weekday_avg > 0 else 1
        
        # Calculate temporal score (higher = more irregular patterns)
        temporal_score = 0
        if weekend_ratio > 1.5:  # 50% higher weekend spending
            temporal_score += 0.3
        if month_end_ratio > 0.7:  # Heavy month-end spending
            temporal_score += 0.3
        if peak_hour in [22, 23, 0, 1]:  # Late night peak
            temporal_score += 0.2
        
        # Pattern variability
        daily_spending = df.groupby(df['datetime'].dt.date)['amount'].sum()
        spending_cv = daily_spending.std() / daily_spending.mean() if daily_spending.mean() > 0 else 0
        if spending_cv > 1:  # High variability
            temporal_score += 0.2
        
        temporal_score = min(1.0, temporal_score)
        
        recommendations = []
        if month_end_ratio > 0.7:
            recommendations.append("Implement automatic monthly budgeting to avoid month-end overspending")
        if weekend_ratio > 1.5:
            recommendations.append("Set specific weekend spending limits to control leisure expenses")
        if peak_hour in [22, 23, 0, 1]:
            recommendations.append("Avoid late-night online shopping when decision-making may be impaired")
        if spending_cv > 1:
            recommendations.append("Create daily spending limits to reduce spending volatility")
        
        if not recommendations:
            recommendations.append("Good temporal spending patterns maintained")
        
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        return BehavioralInsight(
            insight_type="temporal_patterns",
            score=temporal_score,
            description=f"Peak spending: {peak_hour:02d}:00 hours on {days_of_week[peak_day]}. "
                       f"Weekend spending {weekend_ratio:.1f}x weekday average. "
                       f"Month-end concentration: {month_end_ratio:.1%}",
            recommendations=recommendations,
            impact_level="medium" if temporal_score > 0.5 else "low",
            financial_impact=abs(weekend_avg - weekday_avg) * 8 if weekend_avg > weekday_avg else 0  # 8 weekend days per month
        )
    
    def _analyze_social_spending(self, df: pd.DataFrame) -> BehavioralInsight:
        """Analyze social influence on spending"""
        social_indicators = []
        
        # Group spending (restaurants, entertainment during social hours)
        social_hours = df['hour'].isin([18, 19, 20, 21, 22])  # Evening social hours
        social_categories = ['FOOD_DINING', 'ENTERTAINMENT']
        
        social_spending = df[
            (df['category'].isin(social_categories)) & 
            (social_hours) & 
            (df['is_weekend'] | df['hour'].isin([18, 19, 20, 21]))
        ]
        
        social_amount = social_spending['amount'].sum()
        total_amount = df['amount'].sum()
        social_ratio = social_amount / total_amount if total_amount > 0 else 0
        
        # Large group transactions (higher amounts in social categories)
        social_large_transactions = social_spending[
            social_spending['amount'] > social_spending['amount'].quantile(0.7)
        ]
        
        # Frequency of social spending
        social_frequency = len(social_spending) / len(df) if len(df) > 0 else 0
        
        social_score = min(1.0, social_ratio * 2 + social_frequency * 0.5)
        
        recommendations = []
        if social_score > 0.6:
            recommendations.extend([
                "Set a monthly budget specifically for social activities",
                "Suggest alternative low-cost social activities to friends",
                "Use apps to split bills and track group expenses"
            ])
        elif social_score > 0.3:
            recommendations.extend([
                "Plan social spending in advance to avoid overspending",
                "Alternate between expensive and budget-friendly social activities"
            ])
        else:
            recommendations.append("Social spending is well-controlled")
        
        return BehavioralInsight(
            insight_type="social_influence",
            score=social_score,
            description=f"Social spending: {social_ratio:.1%} of total spending. "
                       f"Social activity frequency: {social_frequency:.1%} of transactions",
            recommendations=recommendations,
            impact_level="high" if social_score > 0.6 else "medium" if social_score > 0.3 else "low",
            financial_impact=social_amount * (30 / len(df)) if len(df) > 0 else 0
        )
    
    def _analyze_impulse_buying(self, df: pd.DataFrame) -> BehavioralInsight:
        """Analyze impulse buying patterns"""
        impulse_indicators = []
        
        # Quick successive transactions
        quick_transactions = df[df['time_since_last'] < 0.5]  # within 30 minutes
        quick_ratio = len(quick_transactions) / len(df) if len(df) > 0 else 0
        impulse_indicators.append(quick_ratio * 0.4)
        
        # Unusual amount transactions
        amount_threshold = df['amount'].quantile(0.85)
        large_transactions = df[df['amount'] > amount_threshold]
        large_ratio = len(large_transactions) / len(df) if len(df) > 0 else 0
        impulse_indicators.append(large_ratio * 0.3)
        
        # Shopping category analysis
        shopping_transactions = df[df['category'] == 'SHOPPING']
        shopping_ratio = len(shopping_transactions) / len(df) if len(df) > 0 else 0
        impulse_indicators.append(shopping_ratio * 0.3)
        
        impulse_score = sum(impulse_indicators)
        
        # Identify specific impulse patterns
        impulse_patterns = []
        if quick_ratio > 0.1:
            impulse_patterns.append("Multiple purchases in short time windows")
        if large_ratio > 0.15:
            impulse_patterns.append("Frequent high-value transactions")
        if shopping_ratio > 0.3:
            impulse_patterns.append("High frequency of shopping transactions")
        
        recommendations = []
        if impulse_score > 0.5:
            recommendations.extend([
                "Implement a mandatory 24-hour waiting period for purchases over ₹2000",
                "Remove saved payment methods from shopping apps",
                "Create a wish list and review monthly instead of buying immediately"
            ])
        elif impulse_score > 0.25:
            recommendations.extend([
                "Use the 'sleep on it' rule for non-essential purchases",
                "Set up purchase notifications to increase awareness"
            ])
        else:
            recommendations.append("Good impulse control maintained")
        
        # Calculate financial impact
        impulse_amount = quick_transactions['amount'].sum() + large_transactions['amount'].sum()
        monthly_impulse_impact = impulse_amount * (30 / len(df)) if len(df) > 0 else 0
        
        return BehavioralInsight(
            insight_type="impulse_buying",
            score=impulse_score,
            description=f"Impulse buying score: {impulse_score:.1%}. "
                       f"Quick purchases: {quick_ratio:.1%}, Large transactions: {large_ratio:.1%}",
            recommendations=recommendations,
            impact_level="high" if impulse_score > 0.5 else "medium" if impulse_score > 0.25 else "low",
            financial_impact=monthly_impulse_impact
        )
    
    def _analyze_subscriptions(self, df: pd.DataFrame) -> BehavioralInsight:
        """Analyze subscription and recurring payment patterns"""
        # Identify potential subscriptions (regular amounts and intervals)
        subscription_candidates = []
        
        # Group by merchant and amount to find recurring patterns
        merchant_amounts = df.groupby(['merchant', 'amount']).size().reset_index(name='frequency')
        potential_subscriptions = merchant_amounts[merchant_amounts['frequency'] >= 2]
        
        total_subscription_amount = 0
        subscription_count = 0
        
        for _, row in potential_subscriptions.iterrows():
            merchant = row['merchant']
            amount = row['amount']
            frequency = row['frequency']
            
            # Check if transactions are regular (monthly-like intervals)
            merchant_transactions = df[
                (df['merchant'] == merchant) & 
                (df['amount'] == amount)
            ].sort_values('datetime')
            
            if len(merchant_transactions) >= 2:
                intervals = merchant_transactions['datetime'].diff().dt.days.dropna()
                avg_interval = intervals.mean()
                
                # Consider as subscription if interval is roughly monthly (20-40 days)
                if 20 <= avg_interval <= 40:
                    subscription_candidates.append({
                        'merchant': merchant,
                        'amount': amount,
                        'frequency': frequency,
                        'avg_interval': avg_interval
                    })
                    total_subscription_amount += amount
                    subscription_count += 1
        
        # Calculate subscription metrics
        total_spending = df['amount'].sum()
        subscription_ratio = total_subscription_amount / total_spending if total_spending > 0 else 0
        
        # Subscription optimization score (higher = more optimization needed)
        subscription_score = min(1.0, subscription_ratio * 2 + subscription_count * 0.1)
        
        recommendations = []
        if subscription_score > 0.4:
            recommendations.extend([
                "Review all subscriptions monthly to identify unused services",
                f"Cancel unused subscriptions - potential savings: ₹{total_subscription_amount * 0.3:.0f}/month",
                "Use subscription tracking apps to monitor recurring payments"
            ])
        elif subscription_score > 0.2:
            recommendations.extend([
                "Quarterly review of subscription services for value assessment",
                "Consider annual payments for frequently used services (often cheaper)"
            ])
        else:
            recommendations.append("Subscription spending appears optimized")
        
        subscription_description = f"Identified {subscription_count} potential subscriptions totaling ₹{total_subscription_amount:.0f}/month"
        
        return BehavioralInsight(
            insight_type="subscription_analysis",
            score=subscription_score,
            description=subscription_description,
            recommendations=recommendations,
            impact_level="medium" if subscription_score > 0.3 else "low",
            financial_impact=total_subscription_amount * 0.2  # Potential 20% savings
        )
    
    def _analyze_spending_escalation(self, df: pd.DataFrame) -> BehavioralInsight:
        """Analyze spending escalation patterns"""
        if len(df) < 10:  # Need sufficient data
            return BehavioralInsight(
                insight_type="spending_escalation",
                score=0.0,
                description="Insufficient data for escalation analysis",
                recommendations=["Collect more transaction data for analysis"],
                impact_level="low",
                financial_impact=0.0
            )
        
        # Sort by date and analyze spending trends
        df_sorted = df.sort_values('datetime')
        
        # Weekly spending analysis
        df_sorted['week'] = df_sorted['datetime'].dt.isocalendar().week
        weekly_spending = df_sorted.groupby('week')['amount'].sum()
        
        # Calculate trend
        weeks = np.arange(len(weekly_spending))
        if len(weeks) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(weeks, weekly_spending.values)
            
            # Normalize slope by average spending
            avg_weekly_spending = weekly_spending.mean()
            escalation_rate = slope / avg_weekly_spending if avg_weekly_spending > 0 else 0
            
            # Score escalation (higher = more concerning)
            escalation_score = min(1.0, max(0, escalation_rate * 10))  # Scale appropriately
        else:
            escalation_score = 0.0
            escalation_rate = 0.0
        
        # Category-specific escalation
        category_escalations = []
        for category in df['category'].unique():
            cat_data = df_sorted[df_sorted['category'] == category]
            if len(cat_data) >= 5:
                cat_weekly = cat_data.groupby('week')['amount'].sum()
                if len(cat_weekly) > 2:
                    cat_weeks = np.arange(len(cat_weekly))
                    cat_slope, _, _, _, _ = stats.linregress(cat_weeks, cat_weekly.values)
                    cat_avg = cat_weekly.mean()
                    cat_rate = cat_slope / cat_avg if cat_avg > 0 else 0
                    if cat_rate > 0.1:  # 10% weekly increase
                        category_escalations.append((category, cat_rate))
        
        recommendations = []
        if escalation_score > 0.6:
            recommendations.extend([
                f"Spending is increasing at {escalation_rate:.1%} per week - immediate budget review needed",
                "Implement strict monthly spending limits",
                "Identify and address the root cause of spending increases"
            ])
        elif escalation_score > 0.3:
            recommendations.extend([
                "Monitor spending trends closely",
                "Set alerts for unusual spending increases"
            ])
        else:
            recommendations.append("Spending patterns are stable or improving")
        
        if category_escalations:
            for category, rate in category_escalations[:3]:  # Top 3
                recommendations.append(f"Monitor {category} spending - increasing at {rate:.1%}/week")
        
        # Estimate future impact
        future_impact = slope * 4 if escalation_score > 0.3 else 0  # 4 weeks ahead
        
        return BehavioralInsight(
            insight_type="spending_escalation",
            score=escalation_score,
            description=f"Spending trend: {escalation_rate:+.1%} per week. "
                       f"{'Increasing' if escalation_rate > 0 else 'Stable/Decreasing'} pattern detected",
            recommendations=recommendations,
            impact_level="high" if escalation_score > 0.6 else "medium" if escalation_score > 0.3 else "low",
            financial_impact=abs(future_impact)
        )
    
    def _analyze_category_behavior(self, df: pd.DataFrame) -> BehavioralInsight:
        """Analyze category-specific behavioral patterns"""
        category_insights = {}
        
        # Category spending distribution
        category_spending = df.groupby('category')['amount'].agg(['sum', 'count', 'std']).fillna(0)
        total_spending = df['amount'].sum()
        
        behavioral_flags = []
        
        for category in category_spending.index:
            cat_amount = category_spending.loc[category, 'sum']
            cat_count = category_spending.loc[category, 'count']
            cat_std = category_spending.loc[category, 'std']
            
            cat_percentage = cat_amount / total_spending if total_spending > 0 else 0
            avg_transaction = cat_amount / cat_count if cat_count > 0 else 0
            
            # Category-specific behavioral analysis
            if category == 'FOOD_DINING' and cat_percentage > 0.3:
                behavioral_flags.append(f"High food spending: {cat_percentage:.1%} of total budget")
            elif category == 'ENTERTAINMENT' and cat_percentage > 0.2:
                behavioral_flags.append(f"High entertainment spending: {cat_percentage:.1%}")
            elif category == 'SHOPPING' and cat_std > avg_transaction * 2:
                behavioral_flags.append(f"Inconsistent shopping patterns (high variability)")
            elif category == 'TRANSPORTATION' and cat_percentage > 0.25:
                behavioral_flags.append(f"High transportation costs: {cat_percentage:.1%}")
        
        # Calculate overall category behavior score
        behavior_score = len(behavioral_flags) * 0.2  # 0.2 per flag
        behavior_score = min(1.0, behavior_score)
        
        recommendations = []
        if 'FOOD_DINING' in [flag.split(':')[0].split()[-1] for flag in behavioral_flags]:
            recommendations.append("Consider meal planning and cooking at home more often")
        if 'ENTERTAINMENT' in [flag.split(':')[0].split()[-1] for flag in behavioral_flags]:
            recommendations.append("Explore free or low-cost entertainment alternatives")
        if 'SHOPPING' in [flag.split()[1] for flag in behavioral_flags if 'shopping' in flag.lower()]:
            recommendations.append("Create a shopping budget and stick to a list")
        if 'TRANSPORTATION' in [flag.split(':')[0].split()[-1] for flag in behavioral_flags]:
            recommendations.append("Consider public transport or carpooling options")
        
        if not recommendations:
            recommendations.append("Category spending patterns appear balanced")
        
        # Estimate potential savings
        potential_savings = 0
        for category in ['FOOD_DINING', 'ENTERTAINMENT', 'SHOPPING']:
            if category in category_spending.index:
                cat_amount = category_spending.loc[category, 'sum']
                cat_percentage = cat_amount / total_spending if total_spending > 0 else 0
                if cat_percentage > 0.25:  # Above 25% threshold
                    potential_savings += cat_amount * 0.2  # 20% reduction potential
        
        return BehavioralInsight(
            insight_type="category_behavior",
            score=behavior_score,
            description=f"Category analysis flags: {len(behavioral_flags)} potential optimization areas. "
                       + ". ".join(behavioral_flags[:2]),
            recommendations=recommendations,
            impact_level="medium" if behavior_score > 0.4 else "low",
            financial_impact=potential_savings * (30 / len(df)) if len(df) > 0 else 0
        )
    
    def _analyze_financial_stress(self, df: pd.DataFrame) -> BehavioralInsight:
        """Analyze indicators of financial stress"""
        stress_indicators = []
        
        # High frequency of small transactions (cash flow issues)
        small_transactions = df[df['amount'] < 100]
        small_ratio = len(small_transactions) / len(df) if len(df) > 0 else 0
        if small_ratio > 0.6:
            stress_indicators.append("High frequency of small transactions")
        
        # Late night transactions (stress spending)
        late_night_ratio = df['is_late_night'].mean()
        if late_night_ratio > 0.2:
            stress_indicators.append("Frequent late-night transactions")
        
        # Spending volatility
        daily_spending = df.groupby(df['datetime'].dt.date)['amount'].sum()
        if len(daily_spending) > 1:
            spending_cv = daily_spending.std() / daily_spending.mean()
            if spending_cv > 1.5:
                stress_indicators.append("High spending volatility")
        
        # Month-end concentration (cash flow issues)
        month_end_spending = df[df['day_of_month'] >= 25]['amount'].sum()
        total_spending = df['amount'].sum()
        month_end_ratio = month_end_spending / total_spending if total_spending > 0 else 0
        if month_end_ratio > 0.5:
            stress_indicators.append("Heavy month-end spending concentration")
        
        # Calculate stress score
        stress_score = len(stress_indicators) * 0.25
        stress_score = min(1.0, stress_score)
        
        recommendations = []
        if stress_score > 0.5:
            recommendations.extend([
                "Consider creating an emergency fund to reduce financial stress",
                "Implement a zero-based budgeting approach",
                "Seek financial counseling if stress is affecting spending decisions"
            ])
        elif stress_score > 0.25:
            recommendations.extend([
                "Monitor spending patterns for stress-related changes",
                "Create a monthly spending plan to improve cash flow"
            ])
        else:
            recommendations.append("Financial stress indicators appear minimal")
        
        return BehavioralInsight(
            insight_type="financial_stress",
            score=stress_score,
            description=f"Financial stress indicators: {len(stress_indicators)} detected. " 
                       + ". ".join(stress_indicators[:2]),
            recommendations=recommendations,
            impact_level="high" if stress_score > 0.5 else "medium" if stress_score > 0.25 else "low",
            financial_impact=0.0  # Qualitative insight
        )
    
    def _default_behavioral_insights(self) -> Dict[str, BehavioralInsight]:
        """Provide default insights when analysis fails"""
        return {
            "emotional_spending": BehavioralInsight(
                insight_type="emotional_spending",
                score=0.3,
                description="Default analysis - collect more data for personalized insights",
                recommendations=["Track spending patterns for more detailed analysis"],
                impact_level="low",
                financial_impact=0.0
            )
        }

def main():
    """Demo the behavioral finance analyzer"""
    print("🧠 Behavioral Finance Analysis Demo")
    print("=" * 50)
    
    # Sample transaction data with behavioral patterns
    sample_transactions = [
        # Emotional spending pattern
        {"amount": 2500, "category": "SHOPPING", "merchant": "Amazon", "timestamp": "2025-10-01T23:30:00"},
        {"amount": 800, "category": "SHOPPING", "merchant": "Amazon", "timestamp": "2025-10-01T23:35:00"},
        
        # Social spending
        {"amount": 3500, "category": "FOOD_DINING", "merchant": "Restaurant", "timestamp": "2025-10-05T20:30:00"},
        {"amount": 1200, "category": "ENTERTAINMENT", "merchant": "Movies", "timestamp": "2025-10-05T21:00:00"},
        
        # Regular patterns
        {"amount": 450, "category": "TRANSPORTATION", "merchant": "Uber", "timestamp": "2025-10-02T08:15:00"},
        {"amount": 1200, "category": "UTILITIES", "merchant": "Electricity", "timestamp": "2025-10-03T14:20:00"},
        
        # Subscription pattern
        {"amount": 199, "category": "ENTERTAINMENT", "merchant": "Netflix", "timestamp": "2025-09-15T10:00:00"},
        {"amount": 199, "category": "ENTERTAINMENT", "merchant": "Netflix", "timestamp": "2025-10-15T10:00:00"},
        
        # More varied transactions
        {"amount": 800, "category": "FOOD_DINING", "merchant": "Zomato", "timestamp": "2025-10-08T21:45:00"},
        {"amount": 300, "category": "TRANSPORTATION", "merchant": "Metro", "timestamp": "2025-10-09T09:00:00"},
    ]
    
    # Run behavioral analysis
    analyzer = BehavioralFinanceAnalyzer()
    insights = analyzer.analyze_behavioral_patterns(sample_transactions)
    
    print(f"\n📊 Behavioral Analysis Results:")
    print("-" * 50)
    
    for insight_type, insight in insights.items():
        print(f"\n🎯 {insight_type.replace('_', ' ').title()}:")
        print(f"   Score: {insight.score:.1%} ({insight.impact_level} impact)")
        print(f"   Description: {insight.description}")
        print(f"   Financial Impact: ₹{insight.financial_impact:.0f}/month")
        print(f"   Recommendations:")
        for rec in insight.recommendations[:2]:  # Show top 2 recommendations
            print(f"     • {rec}")
    
    print(f"\n✅ Behavioral finance analysis completed!")

if __name__ == "__main__":
    main()