"""
Smart Money AI - Budget Creator
Create intelligent budgets based on user profile and transaction history
"""

import json
from datetime import datetime

class BudgetCreator:
    """Create AI-powered budgets with demographic insights"""
    
    def __init__(self):
        """Initialize budget creator"""
        self.category_templates = self._load_category_templates()
    
    def _load_category_templates(self):
        """Load budget category templates"""
        return {
            'essential': {
                'groceries': {'percentage': 15, 'priority': 'high'},
                'utilities': {'percentage': 10, 'priority': 'high'},
                'rent': {'percentage': 30, 'priority': 'high'},
                'transport': {'percentage': 8, 'priority': 'medium'}
            },
            'lifestyle': {
                'entertainment': {'percentage': 5, 'priority': 'low'},
                'eating_out': {'percentage': 8, 'priority': 'medium'},
                'shopping': {'percentage': 10, 'priority': 'low'},
                'health': {'percentage': 5, 'priority': 'medium'}
            },
            'financial': {
                'savings': {'percentage': 20, 'priority': 'high'},
                'investments': {'percentage': 15, 'priority': 'high'},
                'emergency_fund': {'percentage': 5, 'priority': 'high'}
            }
        }
    
    def create_smart_budget(self, user_profile, transaction_history=None):
        """Create intelligent budget based on user profile"""
        
        income = user_profile.get('income', 50000)
        age = user_profile.get('age', 30)
        dependents = user_profile.get('dependents', 0)
        city_tier = user_profile.get('city_tier', 'Tier_2')
        
        # Adjust percentages based on profile
        budget = {}
        total_percentage = 0
        
        # Essential expenses
        for category, template in self.category_templates['essential'].items():
            percentage = template['percentage']
            
            # Adjust for city tier
            if city_tier == 'Tier_1':
                percentage *= 1.2  # Higher costs in metro cities
            elif city_tier == 'Tier_3':
                percentage *= 0.8  # Lower costs in smaller cities
            
            # Adjust for dependents
            if dependents > 0:
                if category in ['groceries', 'utilities']:
                    percentage *= (1 + dependents * 0.3)
            
            budget[category] = {
                'amount': (income * percentage / 100),
                'percentage': percentage,
                'priority': template['priority']
            }
            total_percentage += percentage
        
        # Lifestyle expenses
        for category, template in self.category_templates['lifestyle'].items():
            percentage = template['percentage']
            
            # Adjust for age
            if age < 30:
                if category in ['entertainment', 'eating_out']:
                    percentage *= 1.3  # Young people spend more on lifestyle
            elif age > 40:
                if category == 'health':
                    percentage *= 1.5  # Older people spend more on health
            
            budget[category] = {
                'amount': (income * percentage / 100),
                'percentage': percentage,
                'priority': template['priority']
            }
            total_percentage += percentage
        
        # Financial goals
        remaining_percentage = max(0, 100 - total_percentage)
        financial_percentage = min(remaining_percentage, 40)  # Max 40% for financial goals
        
        for category, template in self.category_templates['financial'].items():
            percentage = template['percentage']
            
            # Adjust financial allocation based on age
            if age < 30:
                if category == 'investments':
                    percentage *= 1.2  # More aggressive investing when young
            elif age > 45:
                if category == 'emergency_fund':
                    percentage *= 1.5  # More emergency savings when older
            
            # Scale to available percentage
            scaled_percentage = (percentage / 40) * financial_percentage
            
            budget[category] = {
                'amount': (income * scaled_percentage / 100),
                'percentage': scaled_percentage,
                'priority': template['priority']
            }
        
        # Add metadata
        budget_metadata = {
            'created_date': datetime.now().isoformat(),
            'user_profile': user_profile,
            'total_income': income,
            'total_allocated': sum(item['amount'] for item in budget.values()),
            'allocation_percentage': sum(item['percentage'] for item in budget.values())
        }
        
        return {
            'status': 'success',
            'budget': budget,
            'metadata': budget_metadata,
            'recommendations': self._generate_budget_recommendations(budget, user_profile)
        }
    
    def _generate_budget_recommendations(self, budget, user_profile):
        """Generate budget recommendations"""
        
        recommendations = []
        age = user_profile.get('age', 30)
        income = user_profile.get('income', 50000)
        
        # High priority recommendations
        investment_amount = budget.get('investments', {}).get('amount', 0)
        if investment_amount < income * 0.15:
            recommendations.append("Consider increasing investments to at least 15% of income for long-term wealth creation")
        
        emergency_fund = budget.get('emergency_fund', {}).get('amount', 0)
        if emergency_fund < income * 0.05:
            recommendations.append("Build emergency fund to cover 6 months of expenses")
        
        # Age-specific recommendations
        if age < 30:
            recommendations.append("Great time to take higher investment risks for better long-term returns")
        elif age > 40:
            recommendations.append("Focus on balancing growth investments with stable income sources")
        
        return recommendations