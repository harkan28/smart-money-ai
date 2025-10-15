#!/usr/bin/env python3
"""
Smart Budgeting System - Backend Core
=====================================

Backend budgeting system with:
- Automatic budget creation from historical data
- Real-time spending alerts and notifications
- Budget optimization recommendations
- Spending pattern analysis
- Goal tracking and achievement monitoring
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


class BudgetPeriod(Enum):
    """Budget period types"""
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class BudgetItem:
    """Individual budget item"""
    category: str
    allocated_amount: float
    spent_amount: float
    period: BudgetPeriod
    start_date: datetime
    end_date: datetime
    created_at: datetime
    updated_at: datetime
    
    @property
    def remaining_amount(self) -> float:
        return max(0, self.allocated_amount - self.spent_amount)
    
    @property
    def utilization_percentage(self) -> float:
        if self.allocated_amount <= 0:
            return 0.0
        return min(100.0, (self.spent_amount / self.allocated_amount) * 100)
    
    @property
    def is_exceeded(self) -> bool:
        return self.spent_amount > self.allocated_amount
    
    @property
    def days_remaining(self) -> int:
        today = datetime.now().date()
        end_date = self.end_date.date() if isinstance(self.end_date, datetime) else self.end_date
        return max(0, (end_date - today).days)


@dataclass
class BudgetAlert:
    """Budget alert notification"""
    id: str
    category: str
    severity: AlertSeverity
    message: str
    threshold_percentage: float
    current_percentage: float
    amount_exceeded: float
    created_at: datetime
    is_read: bool = False


@dataclass
class SpendingGoal:
    """Financial spending goal"""
    id: str
    name: str
    target_amount: float
    current_amount: float
    target_date: datetime
    category: Optional[str] = None
    is_active: bool = True
    created_at: datetime = None
    
    @property
    def progress_percentage(self) -> float:
        if self.target_amount <= 0:
            return 0.0
        return min(100.0, (self.current_amount / self.target_amount) * 100)
    
    @property
    def days_remaining(self) -> int:
        today = datetime.now().date()
        target_date = self.target_date.date() if isinstance(self.target_date, datetime) else self.target_date
        return max(0, (target_date - today).days)


class BudgetAnalyticsEngine:
    """Analyze spending patterns for budget optimization"""
    
    def __init__(self):
        self.historical_analysis_period = 90  # days
    
    def analyze_historical_spending(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze historical spending patterns for budget recommendations"""
        if not transactions:
            return self._get_default_spending_analysis()
        
        try:
            # Convert to analyze last 3 months
            cutoff_date = datetime.now() - timedelta(days=self.historical_analysis_period)
            recent_transactions = [
                txn for txn in transactions
                if datetime.fromisoformat(txn.get('timestamp', '2024-01-01')).replace(tzinfo=None) >= cutoff_date
            ]
            
            if not recent_transactions:
                return self._get_default_spending_analysis()
            
            # Analyze by category
            category_analysis = self._analyze_by_category(recent_transactions)
            
            # Temporal analysis
            temporal_patterns = self._analyze_temporal_patterns(recent_transactions)
            
            # Spending trends
            trend_analysis = self._analyze_spending_trends(recent_transactions)
            
            # Budget recommendations
            budget_recommendations = self._generate_budget_recommendations(
                category_analysis, temporal_patterns, trend_analysis
            )
            
            return {
                'analysis_period_days': self.historical_analysis_period,
                'total_transactions': len(recent_transactions),
                'category_analysis': category_analysis,
                'temporal_patterns': temporal_patterns,
                'trend_analysis': trend_analysis,
                'budget_recommendations': budget_recommendations,
                'analysis_date': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in historical spending analysis: {e}")
            return self._get_default_spending_analysis()
    
    def _analyze_by_category(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze spending by category"""
        category_totals = {}
        category_counts = {}
        category_transactions = {}
        
        for txn in transactions:
            category = txn.get('category', 'MISCELLANEOUS')
            amount = float(txn.get('amount', 0))
            
            # Skip zero amounts
            if amount <= 0:
                continue
                
            if category not in category_totals:
                category_totals[category] = 0
                category_counts[category] = 0
                category_transactions[category] = []
            
            category_totals[category] += amount
            category_counts[category] += 1
            category_transactions[category].append(txn)
        
        # Calculate statistics
        total_spending = sum(category_totals.values())
        
        category_stats = {}
        for category in category_totals:
            amounts = [float(txn.get('amount', 0)) for txn in category_transactions[category]]
            
            category_stats[category] = {
                'total_amount': category_totals[category],
                'transaction_count': category_counts[category],
                'average_amount': category_totals[category] / category_counts[category],
                'percentage_of_total': (category_totals[category] / total_spending * 100) if total_spending > 0 else 0,
                'min_amount': min(amounts) if amounts else 0,
                'max_amount': max(amounts) if amounts else 0,
                'monthly_average': category_totals[category] / 3,  # 3 months
                'frequency_score': category_counts[category] / len(transactions) * 100
            }
        
        return {
            'total_spending': total_spending,
            'categories': category_stats,
            'top_categories': sorted(category_stats.items(), key=lambda x: x[1]['total_amount'], reverse=True)[:5]
        }
    
    def _analyze_temporal_patterns(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze spending patterns over time"""
        daily_spending = {}
        weekly_spending = {}
        monthly_spending = {}
        
        for txn in transactions:
            try:
                timestamp = datetime.fromisoformat(txn.get('timestamp', '2024-01-01')).replace(tzinfo=None)
                amount = float(txn.get('amount', 0))
                
                # Daily
                date_key = timestamp.date().isoformat()
                daily_spending[date_key] = daily_spending.get(date_key, 0) + amount
                
                # Weekly
                week_key = f"{timestamp.year}-W{timestamp.isocalendar()[1]}"
                weekly_spending[week_key] = weekly_spending.get(week_key, 0) + amount
                
                # Monthly
                month_key = f"{timestamp.year}-{timestamp.month:02d}"
                monthly_spending[month_key] = monthly_spending.get(month_key, 0) + amount
                
            except Exception as e:
                logger.error(f"Error parsing transaction timestamp: {e}")
                continue
        
        # Calculate averages and patterns
        daily_amounts = list(daily_spending.values())
        weekly_amounts = list(weekly_spending.values())
        monthly_amounts = list(monthly_spending.values())
        
        return {
            'daily_average': sum(daily_amounts) / len(daily_amounts) if daily_amounts else 0,
            'weekly_average': sum(weekly_amounts) / len(weekly_amounts) if weekly_amounts else 0,
            'monthly_average': sum(monthly_amounts) / len(monthly_amounts) if monthly_amounts else 0,
            'spending_consistency': self._calculate_consistency_score(daily_amounts),
            'peak_spending_days': self._identify_peak_spending_days(daily_spending),
            'spending_distribution': {
                'daily_range': {'min': min(daily_amounts) if daily_amounts else 0, 'max': max(daily_amounts) if daily_amounts else 0},
                'weekly_range': {'min': min(weekly_amounts) if weekly_amounts else 0, 'max': max(weekly_amounts) if weekly_amounts else 0},
                'monthly_range': {'min': min(monthly_amounts) if monthly_amounts else 0, 'max': max(monthly_amounts) if monthly_amounts else 0}
            }
        }
    
    def _analyze_spending_trends(self, transactions: List[Dict]) -> Dict[str, Any]:
        """Analyze spending trends and growth patterns"""
        try:
            # Sort transactions by date
            sorted_transactions = sorted(
                transactions,
                key=lambda x: datetime.fromisoformat(x.get('timestamp', '2024-01-01')).replace(tzinfo=None)
            )
            
            # Split into periods for trend analysis
            total_days = self.historical_analysis_period
            period_size = total_days // 3  # 3 periods
            
            periods = [[], [], []]
            cutoff_date = datetime.now() - timedelta(days=total_days)
            
            for txn in sorted_transactions:
                txn_date = datetime.fromisoformat(txn.get('timestamp', '2024-01-01')).replace(tzinfo=None)
                days_ago = (datetime.now() - txn_date).days
                
                if days_ago <= period_size:
                    periods[2].append(txn)  # Recent period
                elif days_ago <= period_size * 2:
                    periods[1].append(txn)  # Middle period
                else:
                    periods[0].append(txn)  # Oldest period
            
            # Calculate spending for each period
            period_totals = [
                sum(float(txn.get('amount', 0)) for txn in period)
                for period in periods
            ]
            
            # Calculate trends
            if len(period_totals) >= 2 and period_totals[0] > 0:
                recent_vs_middle = ((period_totals[2] - period_totals[1]) / period_totals[1] * 100) if period_totals[1] > 0 else 0
                middle_vs_oldest = ((period_totals[1] - period_totals[0]) / period_totals[0] * 100) if period_totals[0] > 0 else 0
                overall_trend = ((period_totals[2] - period_totals[0]) / period_totals[0] * 100) if period_totals[0] > 0 else 0
            else:
                recent_vs_middle = middle_vs_oldest = overall_trend = 0
            
            return {
                'period_spending': {
                    'oldest_period': period_totals[0],
                    'middle_period': period_totals[1],
                    'recent_period': period_totals[2]
                },
                'trend_percentages': {
                    'recent_vs_middle': recent_vs_middle,
                    'middle_vs_oldest': middle_vs_oldest,
                    'overall_trend': overall_trend
                },
                'trend_direction': 'increasing' if overall_trend > 5 else 'decreasing' if overall_trend < -5 else 'stable',
                'volatility_score': self._calculate_volatility_score(period_totals)
            }
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            return {'trend_direction': 'stable', 'overall_trend': 0, 'volatility_score': 0.5}
    
    def _generate_budget_recommendations(self, category_analysis: Dict, 
                                       temporal_patterns: Dict, 
                                       trend_analysis: Dict) -> Dict[str, Any]:
        """Generate intelligent budget recommendations"""
        recommendations = {}
        
        try:
            total_monthly_spending = category_analysis.get('total_spending', 0) / 3  # 3 months average
            
            # Category-wise budget recommendations
            category_budgets = {}
            categories = category_analysis.get('categories', {})
            
            for category, stats in categories.items():
                monthly_avg = stats.get('monthly_average', 0)
                trend_factor = 1.0
                
                # Adjust for trends
                if trend_analysis.get('trend_direction') == 'increasing':
                    trend_factor = 1.1  # 10% increase buffer
                elif trend_analysis.get('trend_direction') == 'decreasing':
                    trend_factor = 0.95  # 5% optimization
                
                # Adjust for volatility
                volatility = trend_analysis.get('volatility_score', 0.5)
                volatility_buffer = 1 + (volatility * 0.2)  # Up to 20% buffer for high volatility
                
                recommended_budget = monthly_avg * trend_factor * volatility_buffer
                
                category_budgets[category] = {
                    'recommended_monthly_budget': round(recommended_budget, 2),
                    'historical_average': round(monthly_avg, 2),
                    'trend_adjustment': f"{(trend_factor - 1) * 100:+.1f}%",
                    'volatility_buffer': f"{(volatility_buffer - 1) * 100:.1f}%",
                    'priority': self._calculate_category_priority(category, stats)
                }
            
            # Overall budget recommendations
            total_recommended = sum(item['recommended_monthly_budget'] for item in category_budgets.values())
            
            recommendations = {
                'total_monthly_budget': round(total_recommended, 2),
                'current_monthly_average': round(total_monthly_spending, 2),
                'adjustment_needed': round(total_recommended - total_monthly_spending, 2),
                'category_budgets': category_budgets,
                'optimization_tips': self._generate_optimization_tips(category_analysis, trend_analysis),
                'confidence_score': self._calculate_recommendation_confidence(category_analysis, temporal_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error generating budget recommendations: {e}")
            recommendations = {'total_monthly_budget': 15000, 'confidence_score': 0.5}
        
        return recommendations
    
    def _calculate_consistency_score(self, amounts: List[float]) -> float:
        """Calculate spending consistency score (0-1)"""
        if len(amounts) < 2:
            return 0.5
        
        try:
            mean_amount = sum(amounts) / len(amounts)
            variance = sum((x - mean_amount) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** 0.5
            
            # Coefficient of variation (lower is more consistent)
            cv = std_dev / mean_amount if mean_amount > 0 else 1
            
            # Convert to consistency score (0-1, higher is better)
            consistency_score = max(0, min(1, 1 - cv))
            return consistency_score
            
        except:
            return 0.5
    
    def _identify_peak_spending_days(self, daily_spending: Dict) -> List[str]:
        """Identify days with highest spending"""
        if not daily_spending:
            return []
        
        sorted_days = sorted(daily_spending.items(), key=lambda x: x[1], reverse=True)
        return [day for day, amount in sorted_days[:5]]  # Top 5 spending days
    
    def _calculate_volatility_score(self, amounts: List[float]) -> float:
        """Calculate volatility score (0-1)"""
        if len(amounts) < 2:
            return 0.5
        
        try:
            mean_amount = sum(amounts) / len(amounts)
            variance = sum((x - mean_amount) ** 2 for x in amounts) / len(amounts)
            std_dev = variance ** 0.5
            
            # Normalized volatility
            volatility = std_dev / mean_amount if mean_amount > 0 else 0.5
            return min(1.0, volatility)
            
        except:
            return 0.5
    
    def _calculate_category_priority(self, category: str, stats: Dict) -> str:
        """Calculate category priority for budgeting"""
        essential_categories = ['RENT', 'UTILITIES', 'GROCERIES', 'HEALTHCARE', 'TRANSPORTATION']
        
        if category in essential_categories:
            return 'high'
        elif stats.get('frequency_score', 0) > 10:  # Frequent transactions
            return 'medium'
        else:
            return 'low'
    
    def _generate_optimization_tips(self, category_analysis: Dict, trend_analysis: Dict) -> List[str]:
        """Generate budget optimization tips"""
        tips = []
        
        categories = category_analysis.get('categories', {})
        top_categories = category_analysis.get('top_categories', [])
        
        # High spending category tips
        if top_categories:
            top_category = top_categories[0][0]
            top_amount = top_categories[0][1]['total_amount']
            tips.append(f"Your highest spending is in {top_category} (₹{top_amount:.0f}). Consider reviewing these expenses.")
        
        # Trend-based tips
        trend_direction = trend_analysis.get('trend_direction', 'stable')
        if trend_direction == 'increasing':
            tips.append("Your spending is trending upward. Consider setting stricter limits on discretionary categories.")
        elif trend_direction == 'decreasing':
            tips.append("Great! Your spending is trending downward. You might be able to reallocate some budget to savings.")
        
        # Category-specific tips
        for category, stats in categories.items():
            if stats.get('percentage_of_total', 0) > 30:
                tips.append(f"{category} accounts for {stats['percentage_of_total']:.1f}% of spending. Look for optimization opportunities.")
        
        # Generic tips if none generated
        if not tips:
            tips = [
                "Track daily expenses to identify spending patterns",
                "Set up automatic alerts when approaching budget limits",
                "Review and adjust budgets monthly based on actual spending"
            ]
        
        return tips[:5]  # Limit to 5 tips
    
    def _calculate_recommendation_confidence(self, category_analysis: Dict, temporal_patterns: Dict) -> float:
        """Calculate confidence in budget recommendations"""
        factors = []
        
        # Data volume factor
        total_transactions = category_analysis.get('total_spending', 0)
        if total_transactions > 0:
            data_factor = min(1.0, total_transactions / 50000)  # Normalize to reasonable spending
            factors.append(data_factor)
        
        # Consistency factor
        consistency = temporal_patterns.get('spending_consistency', 0.5)
        factors.append(consistency)
        
        # Category coverage factor
        categories_count = len(category_analysis.get('categories', {}))
        coverage_factor = min(1.0, categories_count / 10)  # Good coverage with 10+ categories
        factors.append(coverage_factor)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _get_default_spending_analysis(self) -> Dict[str, Any]:
        """Return default analysis when no data available"""
        default_categories = {
            'FOOD_DINING': {'monthly_average': 3000, 'total_amount': 9000, 'percentage_of_total': 25},
            'TRANSPORTATION': {'monthly_average': 2000, 'total_amount': 6000, 'percentage_of_total': 17},
            'GROCERIES': {'monthly_average': 2500, 'total_amount': 7500, 'percentage_of_total': 21},
            'UTILITIES': {'monthly_average': 1500, 'total_amount': 4500, 'percentage_of_total': 13},
            'ENTERTAINMENT': {'monthly_average': 1000, 'total_amount': 3000, 'percentage_of_total': 8},
            'MISCELLANEOUS': {'monthly_average': 1000, 'total_amount': 3000, 'percentage_of_total': 8}
        }
        
        default_budget_recommendations = {}
        for category, stats in default_categories.items():
            default_budget_recommendations[category] = {
                'recommended_monthly_budget': stats['monthly_average'] * 1.1,  # 10% buffer
                'historical_average': stats['monthly_average'],
                'trend_adjustment': '+10.0%',
                'volatility_buffer': '10.0%',
                'priority': 'high' if category in ['FOOD_DINING', 'UTILITIES', 'GROCERIES'] else 'medium'
            }
        
        return {
            'analysis_period_days': self.historical_analysis_period,
            'total_transactions': 0,
            'category_analysis': {
                'total_spending': 36000,  # 3 months total
                'categories': default_categories, 
                'top_categories': list(default_categories.items())
            },
            'temporal_patterns': {'daily_average': 400, 'monthly_average': 12000, 'spending_consistency': 0.5},
            'trend_analysis': {'trend_direction': 'stable', 'overall_trend': 0},
            'budget_recommendations': {
                'total_monthly_budget': 13200,  # Sum of recommended budgets
                'category_budgets': default_budget_recommendations,
                'confidence_score': 0.3,
                'optimization_tips': [
                    "Start tracking expenses to get personalized recommendations",
                    "Review and adjust budgets monthly based on actual spending",
                    "Set up automatic alerts when approaching budget limits"
                ]
            }
        }


class SmartBudgetingSystem:
    """Main smart budgeting system backend"""
    
    def __init__(self, database_path: str = "data/budgets.db"):
        self.database_path = database_path
        self.analytics_engine = BudgetAnalyticsEngine()
        self.alert_thresholds = {
            'warning': 75.0,   # 75% of budget used
            'critical': 90.0   # 90% of budget used
        }
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for budget storage"""
        try:
            os.makedirs(os.path.dirname(self.database_path), exist_ok=True)
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Budget items table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS budget_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        category TEXT NOT NULL,
                        allocated_amount REAL NOT NULL,
                        spent_amount REAL DEFAULT 0,
                        period TEXT NOT NULL,
                        start_date TEXT NOT NULL,
                        end_date TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        user_id TEXT DEFAULT 'default'
                    )
                ''')
                
                # Budget alerts table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS budget_alerts (
                        id TEXT PRIMARY KEY,
                        category TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        threshold_percentage REAL NOT NULL,
                        current_percentage REAL NOT NULL,
                        amount_exceeded REAL NOT NULL,
                        created_at TEXT NOT NULL,
                        is_read INTEGER DEFAULT 0,
                        user_id TEXT DEFAULT 'default'
                    )
                ''')
                
                # Spending goals table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS spending_goals (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        target_amount REAL NOT NULL,
                        current_amount REAL DEFAULT 0,
                        target_date TEXT NOT NULL,
                        category TEXT,
                        is_active INTEGER DEFAULT 1,
                        created_at TEXT NOT NULL,
                        user_id TEXT DEFAULT 'default'
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
    
    def create_budget_from_history(self, user_transaction_history: List[Dict], 
                                 period: BudgetPeriod = BudgetPeriod.MONTHLY,
                                 user_id: str = "default") -> Dict[str, Any]:
        """Create intelligent budget based on historical spending"""
        try:
            # Analyze historical spending
            analysis = self.analytics_engine.analyze_historical_spending(user_transaction_history)
            recommendations = analysis.get('budget_recommendations', {})
            
            # Calculate budget period dates
            start_date = datetime.now().replace(day=1)  # Start of current month
            if period == BudgetPeriod.MONTHLY:
                end_date = (start_date + timedelta(days=32)).replace(day=1) - timedelta(days=1)
            elif period == BudgetPeriod.WEEKLY:
                start_date = datetime.now() - timedelta(days=datetime.now().weekday())
                end_date = start_date + timedelta(days=6)
            elif period == BudgetPeriod.QUARTERLY:
                month = start_date.month
                quarter_start_month = ((month - 1) // 3) * 3 + 1
                start_date = start_date.replace(month=quarter_start_month)
                end_date = (start_date + timedelta(days=95)).replace(day=1) - timedelta(days=1)
            else:  # YEARLY
                start_date = start_date.replace(month=1)
                end_date = start_date.replace(year=start_date.year + 1) - timedelta(days=1)
            
            # Create budget items
            created_budgets = []
            category_budgets = recommendations.get('category_budgets', {})
            
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Clear existing budgets for the period
                cursor.execute('''
                    DELETE FROM budget_items 
                    WHERE user_id = ? AND period = ? AND start_date = ?
                ''', (user_id, period.value, start_date.isoformat()))
                
                # Create new budget items
                for category, budget_info in category_budgets.items():
                    allocated_amount = budget_info['recommended_monthly_budget']
                    
                    # Adjust for different periods
                    if period == BudgetPeriod.WEEKLY:
                        allocated_amount = allocated_amount / 4.33  # Average weeks per month
                    elif period == BudgetPeriod.QUARTERLY:
                        allocated_amount = allocated_amount * 3
                    elif period == BudgetPeriod.YEARLY:
                        allocated_amount = allocated_amount * 12
                    
                    budget_item = BudgetItem(
                        category=category,
                        allocated_amount=allocated_amount,
                        spent_amount=0.0,
                        period=period,
                        start_date=start_date,
                        end_date=end_date,
                        created_at=datetime.now(),
                        updated_at=datetime.now()
                    )
                    
                    cursor.execute('''
                        INSERT INTO budget_items 
                        (category, allocated_amount, spent_amount, period, start_date, end_date, created_at, updated_at, user_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        budget_item.category, budget_item.allocated_amount, budget_item.spent_amount,
                        budget_item.period.value, budget_item.start_date.isoformat(), budget_item.end_date.isoformat(),
                        budget_item.created_at.isoformat(), budget_item.updated_at.isoformat(), user_id
                    ))
                    
                    created_budgets.append(budget_item)
                
                conn.commit()
            
            logger.info(f"Created {len(created_budgets)} budget items for {period.value} period")
            
            return {
                'success': True,
                'period': period.value,
                'budgets_created': len(created_budgets),
                'total_budget': sum(b.allocated_amount for b in created_budgets),
                'budget_items': [asdict(budget) for budget in created_budgets],
                'analysis_confidence': recommendations.get('confidence_score', 0.5),
                'optimization_tips': recommendations.get('optimization_tips', [])
            }
            
        except Exception as e:
            logger.error(f"Error creating budget from history: {e}")
            return {'success': False, 'error': str(e)}
    
    def update_budget_spending(self, category: str, amount: float, 
                             user_id: str = "default", 
                             period: BudgetPeriod = BudgetPeriod.MONTHLY) -> Dict[str, Any]:
        """Update budget spending and check for alerts"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get current budget for category
                cursor.execute('''
                    SELECT * FROM budget_items 
                    WHERE category = ? AND user_id = ? AND period = ?
                    AND start_date <= ? AND end_date >= ?
                    ORDER BY created_at DESC LIMIT 1
                ''', (category, user_id, period.value, datetime.now().isoformat(), datetime.now().isoformat()))
                
                row = cursor.fetchone()
                if not row:
                    logger.warning(f"No budget found for category {category}")
                    return {'success': False, 'error': 'Budget not found'}
                
                # Update spending
                new_spent_amount = row[3] + amount  # spent_amount + new amount
                
                cursor.execute('''
                    UPDATE budget_items 
                    SET spent_amount = ?, updated_at = ?
                    WHERE id = ?
                ''', (new_spent_amount, datetime.now().isoformat(), row[0]))
                
                conn.commit()
                
                # Check for alerts
                budget_item = BudgetItem(
                    category=row[1],
                    allocated_amount=row[2],
                    spent_amount=new_spent_amount,
                    period=BudgetPeriod(row[4]),
                    start_date=datetime.fromisoformat(row[5]),
                    end_date=datetime.fromisoformat(row[6]),
                    created_at=datetime.fromisoformat(row[7]),
                    updated_at=datetime.now()
                )
                
                alerts = self._check_budget_alerts(budget_item, user_id)
                
                return {
                    'success': True,
                    'category': category,
                    'new_spent_amount': new_spent_amount,
                    'allocated_amount': budget_item.allocated_amount,
                    'remaining_amount': budget_item.remaining_amount,
                    'utilization_percentage': budget_item.utilization_percentage,
                    'is_exceeded': budget_item.is_exceeded,
                    'alerts_generated': len(alerts),
                    'alerts': [asdict(alert) for alert in alerts]
                }
                
        except Exception as e:
            logger.error(f"Error updating budget spending: {e}")
            return {'success': False, 'error': str(e)}
    
    def _check_budget_alerts(self, budget_item: BudgetItem, user_id: str) -> List[BudgetAlert]:
        """Check if budget alerts should be generated"""
        alerts = []
        
        try:
            utilization = budget_item.utilization_percentage
            
            # Check thresholds
            if utilization >= self.alert_thresholds['critical']:
                severity = AlertSeverity.CRITICAL
                if budget_item.is_exceeded:
                    message = f"Budget EXCEEDED for {budget_item.category}! Over by ₹{budget_item.spent_amount - budget_item.allocated_amount:.2f}"
                else:
                    message = f"Critical: {budget_item.category} budget at {utilization:.1f}% (₹{budget_item.remaining_amount:.2f} remaining)"
            elif utilization >= self.alert_thresholds['warning']:
                severity = AlertSeverity.WARNING
                message = f"Warning: {budget_item.category} budget at {utilization:.1f}% (₹{budget_item.remaining_amount:.2f} remaining)"
            else:
                return []  # No alert needed
            
            alert = BudgetAlert(
                id=f"{budget_item.category}_{int(datetime.now().timestamp())}",
                category=budget_item.category,
                severity=severity,
                message=message,
                threshold_percentage=utilization,
                current_percentage=utilization,
                amount_exceeded=max(0, budget_item.spent_amount - budget_item.allocated_amount),
                created_at=datetime.now()
            )
            
            # Save alert to database
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO budget_alerts 
                    (id, category, severity, message, threshold_percentage, current_percentage, amount_exceeded, created_at, user_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id, alert.category, alert.severity.value, alert.message,
                    alert.threshold_percentage, alert.current_percentage, alert.amount_exceeded,
                    alert.created_at.isoformat(), user_id
                ))
                conn.commit()
            
            alerts.append(alert)
            
        except Exception as e:
            logger.error(f"Error checking budget alerts: {e}")
        
        return alerts
    
    def get_budget_status(self, user_id: str = "default", 
                         period: BudgetPeriod = BudgetPeriod.MONTHLY) -> Dict[str, Any]:
        """Get comprehensive budget status"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get current budgets
                cursor.execute('''
                    SELECT * FROM budget_items 
                    WHERE user_id = ? AND period = ?
                    AND start_date <= ? AND end_date >= ?
                    ORDER BY category
                ''', (user_id, period.value, datetime.now().isoformat(), datetime.now().isoformat()))
                
                budget_rows = cursor.fetchall()
                
                # Get recent alerts
                cursor.execute('''
                    SELECT * FROM budget_alerts 
                    WHERE user_id = ? AND created_at >= ?
                    ORDER BY created_at DESC LIMIT 10
                ''', (user_id, (datetime.now() - timedelta(days=7)).isoformat()))
                
                alert_rows = cursor.fetchall()
            
            # Process budget items
            budget_items = []
            total_allocated = 0
            total_spent = 0
            
            for row in budget_rows:
                budget_item = BudgetItem(
                    category=row[1],
                    allocated_amount=row[2],
                    spent_amount=row[3],
                    period=BudgetPeriod(row[4]),
                    start_date=datetime.fromisoformat(row[5]),
                    end_date=datetime.fromisoformat(row[6]),
                    created_at=datetime.fromisoformat(row[7]),
                    updated_at=datetime.fromisoformat(row[8])
                )
                
                budget_items.append(budget_item)
                total_allocated += budget_item.allocated_amount
                total_spent += budget_item.spent_amount
            
            # Process alerts
            recent_alerts = []
            for row in alert_rows:
                alert = BudgetAlert(
                    id=row[0],
                    category=row[1],
                    severity=AlertSeverity(row[2]),
                    message=row[3],
                    threshold_percentage=row[4],
                    current_percentage=row[5],
                    amount_exceeded=row[6],
                    created_at=datetime.fromisoformat(row[7]),
                    is_read=bool(row[8])
                )
                recent_alerts.append(alert)
            
            # Calculate overall status
            overall_utilization = (total_spent / total_allocated * 100) if total_allocated > 0 else 0
            budget_health = 'good' if overall_utilization < 75 else 'warning' if overall_utilization < 90 else 'critical'
            
            return {
                'period': period.value,
                'budget_summary': {
                    'total_allocated': total_allocated,
                    'total_spent': total_spent,
                    'total_remaining': total_allocated - total_spent,
                    'overall_utilization': overall_utilization,
                    'budget_health': budget_health
                },
                'budget_items': [asdict(item) for item in budget_items],
                'recent_alerts': [asdict(alert) for alert in recent_alerts],
                'categories_over_budget': [item.category for item in budget_items if item.is_exceeded],
                'status_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting budget status: {e}")
            return {'error': str(e)}
    
    def get_spending_insights(self, user_transaction_history: List[Dict], 
                            user_id: str = "default") -> Dict[str, Any]:
        """Get comprehensive spending insights and recommendations"""
        try:
            # Historical analysis
            analysis = self.analytics_engine.analyze_historical_spending(user_transaction_history)
            
            # Current budget status
            budget_status = self.get_budget_status(user_id)
            
            # Generate insights
            insights = {
                'spending_analysis': analysis,
                'current_budget_status': budget_status,
                'recommendations': self._generate_spending_recommendations(analysis, budget_status),
                'key_metrics': self._calculate_key_metrics(analysis, budget_status),
                'insights_generated_at': datetime.now().isoformat()
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating spending insights: {e}")
            return {'error': str(e)}
    
    def _generate_spending_recommendations(self, analysis: Dict, budget_status: Dict) -> List[str]:
        """Generate personalized spending recommendations"""
        recommendations = []
        
        try:
            # Budget-based recommendations
            budget_summary = budget_status.get('budget_summary', {})
            utilization = budget_summary.get('overall_utilization', 0)
            
            if utilization > 90:
                recommendations.append("🚨 You're using over 90% of your budget. Consider reducing discretionary spending.")
            elif utilization > 75:
                recommendations.append("⚠️ You've used 75% of your budget. Monitor spending closely for the rest of the period.")
            elif utilization < 50:
                recommendations.append("✅ Great! You're well within budget. Consider allocating extra to savings or investments.")
            
            # Category-specific recommendations
            over_budget_categories = budget_status.get('categories_over_budget', [])
            if over_budget_categories:
                recommendations.append(f"📊 Categories over budget: {', '.join(over_budget_categories)}. Review these expenses immediately.")
            
            # Trend-based recommendations
            trend_direction = analysis.get('trend_analysis', {}).get('trend_direction', 'stable')
            if trend_direction == 'increasing':
                recommendations.append("📈 Your spending is trending upward. Set stricter daily spending limits.")
            elif trend_direction == 'decreasing':
                recommendations.append("📉 Your spending is decreasing. Good job! Consider increasing your savings rate.")
            
            # Optimization tips from analysis
            optimization_tips = analysis.get('budget_recommendations', {}).get('optimization_tips', [])
            recommendations.extend(optimization_tips[:3])  # Add top 3 tips
            
            # Default recommendations if none generated
            if not recommendations:
                recommendations = [
                    "📱 Set up daily spending notifications to stay on track",
                    "💡 Review your largest expense categories monthly",
                    "🎯 Set specific savings goals for better financial discipline"
                ]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            recommendations = ["Unable to generate personalized recommendations at this time."]
        
        return recommendations[:10]  # Limit to 10 recommendations
    
    def _calculate_key_metrics(self, analysis: Dict, budget_status: Dict) -> Dict[str, Any]:
        """Calculate key financial metrics"""
        try:
            spending_analysis = analysis.get('category_analysis', {})
            temporal_patterns = analysis.get('temporal_patterns', {})
            budget_summary = budget_status.get('budget_summary', {})
            
            return {
                'daily_average_spending': temporal_patterns.get('daily_average', 0),
                'monthly_average_spending': temporal_patterns.get('monthly_average', 0),
                'spending_consistency_score': temporal_patterns.get('spending_consistency', 0.5),
                'budget_utilization_rate': budget_summary.get('overall_utilization', 0),
                'top_spending_category': spending_analysis.get('top_categories', [{}])[0].get(0, 'Unknown') if spending_analysis.get('top_categories') else 'Unknown',
                'savings_rate': max(0, (budget_summary.get('total_allocated', 0) - budget_summary.get('total_spent', 0)) / budget_summary.get('total_allocated', 1) * 100) if budget_summary.get('total_allocated', 0) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating key metrics: {e}")
            return {}


def main():
    """Demo the smart budgeting system"""
    print("💰 Smart Budgeting System - Backend Demo")
    print("=" * 60)
    
    # Initialize system
    budget_system = SmartBudgetingSystem()
    
    # Sample transaction history
    transaction_history = [
        {"category": "FOOD_DINING", "amount": 450, "timestamp": "2025-01-10T19:30:00"},
        {"category": "TRANSPORTATION", "amount": 350, "timestamp": "2025-01-10T09:15:00"},
        {"category": "SUBSCRIPTION", "amount": 799, "timestamp": "2025-01-01T12:00:00"},
        {"category": "SHOPPING", "amount": 2500, "timestamp": "2025-01-08T14:30:00"},
        {"category": "HEALTHCARE", "amount": 850, "timestamp": "2025-01-09T16:45:00"},
        {"category": "GROCERIES", "amount": 1200, "timestamp": "2025-01-12T10:30:00"},
        {"category": "UTILITIES", "amount": 2100, "timestamp": "2025-01-05T15:00:00"},
        {"category": "FOOD_DINING", "amount": 680, "timestamp": "2025-01-14T20:15:00"},
        {"category": "TRANSPORTATION", "amount": 280, "timestamp": "2025-01-15T08:45:00"},
        {"category": "ENTERTAINMENT", "amount": 1500, "timestamp": "2025-01-13T16:20:00"},
    ]
    
    print("\n📊 Creating Budget from Historical Data...")
    print("-" * 50)
    
    # Create budget
    budget_result = budget_system.create_budget_from_history(
        transaction_history, 
        BudgetPeriod.MONTHLY,
        "demo_user"
    )
    
    if budget_result.get('success'):
        print(f"✅ Budget created successfully!")
        print(f"   Total Monthly Budget: ₹{budget_result['total_budget']:.2f}")
        print(f"   Budget Items Created: {budget_result['budgets_created']}")
        print(f"   Confidence Score: {budget_result['analysis_confidence']:.2f}")
    
    print("\n💸 Simulating Spending Updates...")
    print("-" * 50)
    
    # Simulate some spending
    spending_updates = [
        {"category": "FOOD_DINING", "amount": 500},
        {"category": "TRANSPORTATION", "amount": 400},
        {"category": "SHOPPING", "amount": 3000},  # This should trigger alerts
    ]
    
    for update in spending_updates:
        result = budget_system.update_budget_spending(
            update["category"], 
            update["amount"], 
            "demo_user"
        )
        
        if result.get('success'):
            print(f"💰 {update['category']}: +₹{update['amount']}")
            print(f"   Utilization: {result['utilization_percentage']:.1f}%")
            print(f"   Remaining: ₹{result['remaining_amount']:.2f}")
            
            if result['alerts_generated'] > 0:
                print(f"   🚨 Alerts: {result['alerts_generated']}")
    
    print("\n📈 Budget Status Overview:")
    print("-" * 50)
    
    # Get budget status
    status = budget_system.get_budget_status("demo_user")
    
    if 'budget_summary' in status:
        summary = status['budget_summary']
        print(f"Total Allocated: ₹{summary['total_allocated']:.2f}")
        print(f"Total Spent: ₹{summary['total_spent']:.2f}")
        print(f"Overall Utilization: {summary['overall_utilization']:.1f}%")
        print(f"Budget Health: {summary['budget_health'].upper()}")
        
        if status.get('categories_over_budget'):
            print(f"⚠️ Over Budget: {', '.join(status['categories_over_budget'])}")
    
    print("\n🎯 Spending Insights:")
    print("-" * 50)
    
    # Get insights
    insights = budget_system.get_spending_insights(transaction_history, "demo_user")
    
    if 'recommendations' in insights:
        print("Recommendations:")
        for i, rec in enumerate(insights['recommendations'][:5], 1):
            print(f"   {i}. {rec}")
    
    if 'key_metrics' in insights:
        metrics = insights['key_metrics']
        print(f"\nKey Metrics:")
        print(f"   Daily Average: ₹{metrics.get('daily_average_spending', 0):.2f}")
        print(f"   Top Category: {metrics.get('top_spending_category', 'Unknown')}")
        print(f"   Savings Rate: {metrics.get('savings_rate', 0):.1f}%")
    
    print(f"\n✅ Smart budgeting system demo completed!")


if __name__ == "__main__":
    main()