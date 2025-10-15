#!/usr/bin/env python3
"""
Smart Budgeting System
======================

Intelligent budgeting features with:
- Smart budget creation based on spending patterns
- Real-time alerts and notifications
- Spending insights and recommendations
- Goal tracking and progress monitoring
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AlertType(Enum):
    """Types of budget alerts"""
    APPROACHING_LIMIT = "approaching_limit"
    EXCEEDED_LIMIT = "exceeded_limit"
    UNUSUAL_SPENDING = "unusual_spending"
    GOAL_ACHIEVED = "goal_achieved"
    MONTHLY_SUMMARY = "monthly_summary"


class BudgetPeriod(Enum):
    """Budget period types"""
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"


@dataclass
class BudgetCategory:
    """Budget category configuration"""
    name: str
    limit: float
    spent: float = 0.0
    period: BudgetPeriod = BudgetPeriod.MONTHLY
    alert_threshold: float = 0.8  # Alert at 80% of limit
    is_active: bool = True
    created_date: datetime = None
    last_reset: datetime = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()
        if self.last_reset is None:
            self.last_reset = datetime.now()
    
    @property
    def remaining(self) -> float:
        """Remaining budget amount"""
        return max(0, self.limit - self.spent)
    
    @property
    def utilization_percentage(self) -> float:
        """Budget utilization as percentage"""
        return (self.spent / self.limit * 100) if self.limit > 0 else 0
    
    @property
    def is_exceeded(self) -> bool:
        """Check if budget is exceeded"""
        return self.spent > self.limit
    
    @property
    def is_approaching_limit(self) -> bool:
        """Check if approaching alert threshold"""
        return self.utilization_percentage >= (self.alert_threshold * 100)


@dataclass
class SpendingAlert:
    """Spending alert notification"""
    alert_type: AlertType
    category: str
    message: str
    amount: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = None
    is_read: bool = False
    priority: str = "medium"  # low, medium, high
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class SavingsGoal:
    """Savings goal tracking"""
    name: str
    target_amount: float
    current_amount: float = 0.0
    target_date: Optional[datetime] = None
    category: str = "General"
    is_active: bool = True
    created_date: datetime = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()
    
    @property
    def progress_percentage(self) -> float:
        """Progress towards goal as percentage"""
        return (self.current_amount / self.target_amount * 100) if self.target_amount > 0 else 0
    
    @property
    def remaining_amount(self) -> float:
        """Remaining amount to reach goal"""
        return max(0, self.target_amount - self.current_amount)
    
    @property
    def days_remaining(self) -> Optional[int]:
        """Days remaining to reach target date"""
        if self.target_date:
            delta = self.target_date - datetime.now()
            return max(0, delta.days)
        return None
    
    @property
    def monthly_savings_needed(self) -> Optional[float]:
        """Monthly savings needed to reach goal"""
        if self.target_date and self.days_remaining:
            months_remaining = max(1, self.days_remaining / 30)
            return self.remaining_amount / months_remaining
        return None


class SpendingAnalyzer:
    """Analyzes spending patterns for budget recommendations"""
    
    def __init__(self):
        self.spending_patterns = {}
        self.seasonal_adjustments = {}
    
    def analyze_spending_history(self, transactions: List[Dict[str, Any]], 
                               months_to_analyze: int = 6) -> Dict[str, Any]:
        """Analyze spending history to extract patterns"""
        if not transactions:
            return {}
        
        # Filter transactions to analysis period
        cutoff_date = datetime.now() - timedelta(days=months_to_analyze * 30)
        recent_transactions = [
            txn for txn in transactions 
            if txn.get('timestamp', datetime.now()) >= cutoff_date
            and txn.get('transaction_type') == 'debit'
        ]
        
        if not recent_transactions:
            return {}
        
        # Group by category and month
        category_spending = defaultdict(list)
        monthly_totals = defaultdict(float)
        
        for txn in recent_transactions:
            category = txn.get('category', 'Other')
            amount = txn.get('amount', 0)
            timestamp = txn.get('timestamp', datetime.now())
            month_key = timestamp.strftime('%Y-%m')
            
            category_spending[category].append({
                'amount': amount,
                'month': month_key,
                'timestamp': timestamp
            })
            monthly_totals[month_key] += amount
        
        # Calculate statistics for each category
        analysis = {}
        for category, spending_data in category_spending.items():
            amounts = [item['amount'] for item in spending_data]
            
            # Monthly aggregation
            monthly_category_spending = defaultdict(float)
            for item in spending_data:
                monthly_category_spending[item['month']] += item['amount']
            
            monthly_amounts = list(monthly_category_spending.values())
            
            analysis[category] = {
                'total_spent': sum(amounts),
                'transaction_count': len(amounts),
                'average_transaction': sum(amounts) / len(amounts) if amounts else 0,
                'monthly_average': sum(monthly_amounts) / len(monthly_amounts) if monthly_amounts else 0,
                'monthly_median': sorted(monthly_amounts)[len(monthly_amounts)//2] if monthly_amounts else 0,
                'trend': self._calculate_trend(monthly_amounts),
                'volatility': self._calculate_volatility(monthly_amounts),
                'seasonal_factor': self._detect_seasonality(spending_data)
            }
        
        return analysis
    
    def _calculate_trend(self, monthly_amounts: List[float]) -> str:
        """Calculate spending trend"""
        if len(monthly_amounts) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        recent_avg = sum(monthly_amounts[-3:]) / min(3, len(monthly_amounts))
        earlier_avg = sum(monthly_amounts[:3]) / min(3, len(monthly_amounts))
        
        if recent_avg > earlier_avg * 1.1:
            return "increasing"
        elif recent_avg < earlier_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def _calculate_volatility(self, monthly_amounts: List[float]) -> float:
        """Calculate spending volatility (coefficient of variation)"""
        if len(monthly_amounts) < 2:
            return 0.0
        
        mean_amount = sum(monthly_amounts) / len(monthly_amounts)
        if mean_amount == 0:
            return 0.0
        
        variance = sum((x - mean_amount) ** 2 for x in monthly_amounts) / len(monthly_amounts)
        std_dev = variance ** 0.5
        
        return std_dev / mean_amount
    
    def _detect_seasonality(self, spending_data: List[Dict]) -> Dict[str, float]:
        """Detect seasonal spending patterns"""
        if len(spending_data) < 12:  # Need at least a year of data
            return {}
        
        monthly_spending = defaultdict(list)
        for item in spending_data:
            month = item['timestamp'].month
            monthly_spending[month].append(item['amount'])
        
        # Calculate average spending per month
        seasonal_factors = {}
        overall_monthly_avg = sum(
            sum(amounts) / len(amounts) for amounts in monthly_spending.values()
        ) / len(monthly_spending)
        
        for month, amounts in monthly_spending.items():
            if amounts:
                month_avg = sum(amounts) / len(amounts)
                seasonal_factors[month] = month_avg / overall_monthly_avg if overall_monthly_avg > 0 else 1.0
        
        return seasonal_factors
    
    def recommend_budgets(self, spending_analysis: Dict[str, Any], 
                         income: Optional[float] = None) -> Dict[str, float]:
        """Recommend budget limits based on spending analysis"""
        recommendations = {}
        
        for category, analysis in spending_analysis.items():
            monthly_avg = analysis['monthly_average']
            trend = analysis['trend']
            volatility = analysis['volatility']
            
            # Base recommendation on monthly average
            base_budget = monthly_avg
            
            # Adjust for trend
            if trend == "increasing":
                base_budget *= 1.2  # Allow 20% increase
            elif trend == "decreasing":
                base_budget *= 0.9  # Encourage continued reduction
            
            # Adjust for volatility (higher volatility needs higher buffer)
            volatility_buffer = 1 + (volatility * 0.5)
            base_budget *= volatility_buffer
            
            # Round to reasonable amounts
            if base_budget < 1000:
                recommended_budget = round(base_budget / 100) * 100
            elif base_budget < 10000:
                recommended_budget = round(base_budget / 500) * 500
            else:
                recommended_budget = round(base_budget / 1000) * 1000
            
            recommendations[category] = max(recommended_budget, 100)  # Minimum budget
        
        # If income is provided, ensure total doesn't exceed 80% of income
        if income and income > 0:
            total_recommended = sum(recommendations.values())
            max_budget = income * 0.8
            
            if total_recommended > max_budget:
                # Scale down all recommendations proportionally
                scale_factor = max_budget / total_recommended
                recommendations = {
                    category: budget * scale_factor 
                    for category, budget in recommendations.items()
                }
        
        return recommendations


class SmartBudgetManager:
    """Smart budgeting system with alerts and recommendations"""
    
    def __init__(self, data_file: Optional[str] = None):
        self.data_file = data_file or "data/budget_data.json"
        self.budgets: Dict[str, BudgetCategory] = {}
        self.alerts: List[SpendingAlert] = []
        self.savings_goals: Dict[str, SavingsGoal] = {}
        self.spending_analyzer = SpendingAnalyzer()
        self.notification_settings = {
            'email_alerts': True,
            'push_notifications': True,
            'alert_frequency': 'immediate',  # immediate, daily, weekly
            'quiet_hours': (22, 8)  # No alerts between 10 PM and 8 AM
        }
        
        self._load_data()
    
    def _load_data(self):
        """Load budget data from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                
                # Load budgets
                for name, budget_data in data.get('budgets', {}).items():
                    budget_data['period'] = BudgetPeriod(budget_data.get('period', 'monthly'))
                    budget_data['created_date'] = datetime.fromisoformat(budget_data['created_date'])
                    budget_data['last_reset'] = datetime.fromisoformat(budget_data['last_reset'])
                    self.budgets[name] = BudgetCategory(**budget_data)
                
                # Load savings goals
                for name, goal_data in data.get('savings_goals', {}).items():
                    if goal_data.get('target_date'):
                        goal_data['target_date'] = datetime.fromisoformat(goal_data['target_date'])
                    goal_data['created_date'] = datetime.fromisoformat(goal_data['created_date'])
                    self.savings_goals[name] = SavingsGoal(**goal_data)
                
                # Load alerts
                for alert_data in data.get('alerts', []):
                    alert_data['alert_type'] = AlertType(alert_data['alert_type'])
                    alert_data['timestamp'] = datetime.fromisoformat(alert_data['timestamp'])
                    self.alerts.append(SpendingAlert(**alert_data))
                
                # Load notification settings
                self.notification_settings.update(data.get('notification_settings', {}))
                
                logger.info(f"Loaded budget data: {len(self.budgets)} budgets, {len(self.savings_goals)} goals")
                
        except Exception as e:
            logger.warning(f"Could not load budget data: {e}")
    
    def _save_data(self):
        """Save budget data to file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
            
            data = {
                'budgets': {
                    name: {
                        **asdict(budget),
                        'period': budget.period.value,
                        'created_date': budget.created_date.isoformat(),
                        'last_reset': budget.last_reset.isoformat()
                    }
                    for name, budget in self.budgets.items()
                },
                'savings_goals': {
                    name: {
                        **asdict(goal),
                        'target_date': goal.target_date.isoformat() if goal.target_date else None,
                        'created_date': goal.created_date.isoformat()
                    }
                    for name, goal in self.savings_goals.items()
                },
                'alerts': [
                    {
                        **asdict(alert),
                        'alert_type': alert.alert_type.value,
                        'timestamp': alert.timestamp.isoformat()
                    }
                    for alert in self.alerts[-100:]  # Keep only last 100 alerts
                ],
                'notification_settings': self.notification_settings
            }
            
            with open(self.data_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save budget data: {e}")
    
    def create_smart_budgets(self, transactions: List[Dict[str, Any]], 
                           income: Optional[float] = None) -> Dict[str, Any]:
        """Create smart budgets based on spending analysis"""
        # Analyze spending patterns
        spending_analysis = self.spending_analyzer.analyze_spending_history(transactions)
        
        if not spending_analysis:
            return {'error': 'Insufficient transaction data for analysis'}
        
        # Get budget recommendations
        recommendations = self.spending_analyzer.recommend_budgets(spending_analysis, income)
        
        # Create or update budgets
        created_budgets = {}
        for category, recommended_amount in recommendations.items():
            if category not in self.budgets:
                # Create new budget
                self.budgets[category] = BudgetCategory(
                    name=category,
                    limit=recommended_amount,
                    period=BudgetPeriod.MONTHLY
                )
                created_budgets[category] = {
                    'limit': recommended_amount,
                    'status': 'created',
                    'basis': spending_analysis[category]
                }
            else:
                # Update existing budget if recommendation is significantly different
                current_limit = self.budgets[category].limit
                if abs(recommended_amount - current_limit) / current_limit > 0.2:  # 20% difference
                    created_budgets[category] = {
                        'old_limit': current_limit,
                        'new_limit': recommended_amount,
                        'status': 'updated',
                        'basis': spending_analysis[category]
                    }
                    self.budgets[category].limit = recommended_amount
        
        self._save_data()
        
        return {
            'budgets_created': len(created_budgets),
            'total_budget': sum(recommendations.values()),
            'budget_details': created_budgets,
            'spending_analysis': spending_analysis
        }
    
    def add_budget(self, category: str, limit: float, period: BudgetPeriod = BudgetPeriod.MONTHLY,
                  alert_threshold: float = 0.8) -> bool:
        """Add a new budget category"""
        try:
            self.budgets[category] = BudgetCategory(
                name=category,
                limit=limit,
                period=period,
                alert_threshold=alert_threshold
            )
            self._save_data()
            logger.info(f"Added budget for {category}: ₹{limit}")
            return True
        except Exception as e:
            logger.error(f"Error adding budget: {e}")
            return False
    
    def update_budget_spending(self, category: str, amount: float) -> List[SpendingAlert]:
        """Update spending for a budget category and check for alerts"""
        new_alerts = []
        
        if category not in self.budgets:
            # Create default budget if category doesn't exist
            self.budgets[category] = BudgetCategory(
                name=category,
                limit=amount * 3,  # Set limit to 3x first transaction
                period=BudgetPeriod.MONTHLY
            )
        
        budget = self.budgets[category]
        old_spent = budget.spent
        budget.spent += amount
        
        # Check for alerts
        if not budget.is_exceeded and old_spent <= budget.limit * budget.alert_threshold and budget.is_approaching_limit:
            # Approaching limit alert
            alert = SpendingAlert(
                alert_type=AlertType.APPROACHING_LIMIT,
                category=category,
                message=f"You've spent ₹{budget.spent:.0f} of ₹{budget.limit:.0f} in {category} ({budget.utilization_percentage:.1f}%)",
                amount=budget.spent,
                threshold=budget.limit * budget.alert_threshold,
                priority="medium"
            )
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        if budget.is_exceeded and old_spent <= budget.limit:
            # Exceeded limit alert
            alert = SpendingAlert(
                alert_type=AlertType.EXCEEDED_LIMIT,
                category=category,
                message=f"Budget exceeded! You've spent ₹{budget.spent:.0f}, which is ₹{budget.spent - budget.limit:.0f} over your ₹{budget.limit:.0f} budget for {category}",
                amount=budget.spent,
                threshold=budget.limit,
                priority="high"
            )
            new_alerts.append(alert)
            self.alerts.append(alert)
        
        self._save_data()
        return new_alerts
    
    def add_savings_goal(self, name: str, target_amount: float, 
                        target_date: Optional[datetime] = None, category: str = "General") -> bool:
        """Add a new savings goal"""
        try:
            self.savings_goals[name] = SavingsGoal(
                name=name,
                target_amount=target_amount,
                target_date=target_date,
                category=category
            )
            self._save_data()
            logger.info(f"Added savings goal: {name} - ₹{target_amount}")
            return True
        except Exception as e:
            logger.error(f"Error adding savings goal: {e}")
            return False
    
    def update_savings_progress(self, goal_name: str, amount: float) -> Optional[SpendingAlert]:
        """Update progress towards a savings goal"""
        if goal_name not in self.savings_goals:
            return None
        
        goal = self.savings_goals[goal_name]
        old_amount = goal.current_amount
        goal.current_amount += amount
        
        # Check if goal is achieved
        if goal.current_amount >= goal.target_amount and old_amount < goal.target_amount:
            alert = SpendingAlert(
                alert_type=AlertType.GOAL_ACHIEVED,
                category=goal.category,
                message=f"Congratulations! You've achieved your savings goal '{goal.name}' of ₹{goal.target_amount:.0f}!",
                amount=goal.current_amount,
                threshold=goal.target_amount,
                priority="high"
            )
            self.alerts.append(alert)
            self._save_data()
            return alert
        
        self._save_data()
        return None
    
    def get_budget_summary(self) -> Dict[str, Any]:
        """Get comprehensive budget summary"""
        total_limit = sum(budget.limit for budget in self.budgets.values() if budget.is_active)
        total_spent = sum(budget.spent for budget in self.budgets.values() if budget.is_active)
        
        categories_summary = {}
        for name, budget in self.budgets.items():
            if budget.is_active:
                categories_summary[name] = {
                    'limit': budget.limit,
                    'spent': budget.spent,
                    'remaining': budget.remaining,
                    'utilization': budget.utilization_percentage,
                    'status': 'exceeded' if budget.is_exceeded else 'approaching' if budget.is_approaching_limit else 'ok'
                }
        
        return {
            'total_budget': total_limit,
            'total_spent': total_spent,
            'total_remaining': total_limit - total_spent,
            'overall_utilization': (total_spent / total_limit * 100) if total_limit > 0 else 0,
            'categories': categories_summary,
            'active_budgets': len([b for b in self.budgets.values() if b.is_active]),
            'exceeded_budgets': len([b for b in self.budgets.values() if b.is_active and b.is_exceeded])
        }
    
    def get_savings_summary(self) -> Dict[str, Any]:
        """Get savings goals summary"""
        total_target = sum(goal.target_amount for goal in self.savings_goals.values() if goal.is_active)
        total_saved = sum(goal.current_amount for goal in self.savings_goals.values() if goal.is_active)
        
        goals_summary = {}
        for name, goal in self.savings_goals.items():
            if goal.is_active:
                goals_summary[name] = {
                    'target': goal.target_amount,
                    'current': goal.current_amount,
                    'remaining': goal.remaining_amount,
                    'progress': goal.progress_percentage,
                    'days_remaining': goal.days_remaining,
                    'monthly_needed': goal.monthly_savings_needed
                }
        
        return {
            'total_target': total_target,
            'total_saved': total_saved,
            'overall_progress': (total_saved / total_target * 100) if total_target > 0 else 0,
            'goals': goals_summary,
            'active_goals': len([g for g in self.savings_goals.values() if g.is_active])
        }
    
    def get_unread_alerts(self) -> List[SpendingAlert]:
        """Get unread alerts"""
        return [alert for alert in self.alerts if not alert.is_read]
    
    def mark_alert_read(self, alert_index: int) -> bool:
        """Mark an alert as read"""
        try:
            if 0 <= alert_index < len(self.alerts):
                self.alerts[alert_index].is_read = True
                self._save_data()
                return True
        except Exception as e:
            logger.error(f"Error marking alert as read: {e}")
        return False
    
    def reset_monthly_budgets(self):
        """Reset monthly budgets (should be called at start of each month)"""
        current_month = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        for budget in self.budgets.values():
            if budget.period == BudgetPeriod.MONTHLY and budget.last_reset < current_month:
                budget.spent = 0.0
                budget.last_reset = current_month
        
        self._save_data()
        logger.info("Reset monthly budgets")
    
    def get_spending_insights(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get spending insights and recommendations"""
        insights = {}
        
        # Analyze recent spending (last 30 days)
        recent_cutoff = datetime.now() - timedelta(days=30)
        recent_transactions = [
            txn for txn in transactions 
            if txn.get('timestamp', datetime.now()) >= recent_cutoff
            and txn.get('transaction_type') == 'debit'
        ]
        
        if recent_transactions:
            # Category spending analysis
            category_spending = defaultdict(float)
            for txn in recent_transactions:
                category = txn.get('category', 'Other')
                category_spending[category] += txn.get('amount', 0)
            
            # Top spending categories
            top_categories = sorted(category_spending.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Average daily spending
            total_spent = sum(category_spending.values())
            avg_daily = total_spent / 30
            
            insights = {
                'total_spent_30_days': total_spent,
                'average_daily_spending': avg_daily,
                'top_categories': top_categories,
                'transaction_count': len(recent_transactions),
                'average_transaction': total_spent / len(recent_transactions) if recent_transactions else 0
            }
            
            # Compare with budgets
            budget_comparison = {}
            for category, spent in category_spending.items():
                if category in self.budgets:
                    budget = self.budgets[category]
                    # Project monthly spending
                    projected_monthly = spent * (30 / 30)  # Already 30 days
                    budget_comparison[category] = {
                        'spent': spent,
                        'budget': budget.limit,
                        'projected_monthly': projected_monthly,
                        'will_exceed': projected_monthly > budget.limit
                    }
            
            insights['budget_comparison'] = budget_comparison
        
        return insights


# Convenience functions
def create_budget_from_transactions(transactions: List[Dict[str, Any]], 
                                  income: Optional[float] = None) -> Dict[str, Any]:
    """Create budgets from transaction history"""
    manager = SmartBudgetManager()
    return manager.create_smart_budgets(transactions, income)


def process_transaction_for_budget(transaction: Dict[str, Any]) -> List[SpendingAlert]:
    """Process a transaction for budget tracking"""
    manager = SmartBudgetManager()
    category = transaction.get('category', 'Other')
    amount = transaction.get('amount', 0)
    
    if transaction.get('transaction_type') == 'debit':
        return manager.update_budget_spending(category, amount)
    return []


if __name__ == "__main__":
    # Test the budgeting system
    manager = SmartBudgetManager()
    
    # Test budget creation
    manager.add_budget("Food & Dining", 5000, BudgetPeriod.MONTHLY)
    manager.add_budget("Transportation", 3000, BudgetPeriod.MONTHLY)
    
    # Test spending update
    alerts = manager.update_budget_spending("Food & Dining", 1500)
    print(f"Alerts generated: {len(alerts)}")
    
    # Test savings goal
    manager.add_savings_goal("Vacation", 50000, datetime(2024, 12, 31))
    
    # Get summaries
    budget_summary = manager.get_budget_summary()
    savings_summary = manager.get_savings_summary()
    
    print(f"Budget Summary: {budget_summary}")
    print(f"Savings Summary: {savings_summary}")