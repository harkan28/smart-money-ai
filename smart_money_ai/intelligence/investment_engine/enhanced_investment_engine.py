
import sqlite3
import json
import pandas as pd

class EnhancedInvestmentEngine:
    """Enhanced investment recommendation engine with behavioral insights"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            # Auto-detect database path relative to current module
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'investment_behavioral_data.db')
        else:
            self.db_path = db_path
    
    def get_behavioral_risk_profile(self, user_profile):
        """Determine behavioral risk profile based on user characteristics"""
        age = user_profile.get('age', 30)
        income = user_profile.get('income', 50000)
        investment_goal = user_profile.get('goal', 'wealth_generation')
        
        # Base risk assessment
        if age < 30 and income > 500000:
            base_risk = 'aggressive'
        elif age < 40 and income > 300000:
            base_risk = 'moderate'
        else:
            base_risk = 'conservative'
        
        # Adjust based on goal
        goal_risk_adjustment = {
            'wealth_generation': 'aggressive',
            'retirement': 'moderate',
            'tax_saving': 'moderate',
            'education': 'moderate',
            'emergency_fund': 'conservative'
        }
        
        goal_risk = goal_risk_adjustment.get(investment_goal.lower(), 'moderate')
        
        # Final risk profile (weighted average)
        risk_hierarchy = {'conservative': 1, 'moderate': 2, 'aggressive': 3}
        avg_risk_score = (risk_hierarchy[base_risk] + risk_hierarchy[goal_risk]) / 2
        
        if avg_risk_score <= 1.5:
            return 'conservative'
        elif avg_risk_score <= 2.5:
            return 'moderate'
        else:
            return 'aggressive'
    
    def get_investment_recommendations(self, user_profile):
        """Get comprehensive investment recommendations"""
        
        # Get user characteristics
        age = user_profile.get('age', 30)
        income = user_profile.get('income', 50000)
        investment_amount = user_profile.get('monthly_investment', 2000)
        goal = user_profile.get('goal', 'wealth_generation')
        duration = user_profile.get('duration_years', 10)
        
        # Determine risk profile
        risk_profile = self.get_behavioral_risk_profile(user_profile)
        
        # Get investment amount category
        if investment_amount <= 1000:
            amount_category = 'micro_investor'
        elif investment_amount <= 5000:
            amount_category = 'regular_investor'
        else:
            amount_category = 'substantial_investor'
        
        # Load recommendation mappings
        conn = sqlite3.connect(self.db_path)
        mappings_data = conn.execute(
            "SELECT mapping_type, mapping_data FROM recommendation_mappings"
        ).fetchall()
        conn.close()
        
        mappings = {}
        for mapping_type, mapping_data in mappings_data:
            mappings[mapping_type] = json.loads(mapping_data)
        
        # Generate recommendations
        recommendations = {
            'risk_profile': risk_profile,
            'investment_category': amount_category,
            'primary_recommendations': {},
            'portfolio_allocation': {},
            'specific_instruments': [],
            'expected_returns': '',
            'investment_strategy': '',
            'goal_specific_advice': {},
            'projected_wealth': 0
        }
        
        # Risk-based recommendations
        if risk_profile in mappings.get('risk_based_recommendations', {}):
            risk_rec = mappings['risk_based_recommendations'][risk_profile]
            recommendations['primary_recommendations'] = risk_rec
            recommendations['portfolio_allocation'] = risk_rec.get('allocation', {})
            recommendations['expected_returns'] = risk_rec.get('expected_returns', '')
            recommendations['specific_instruments'] = risk_rec.get('primary_instruments', [])
        
        # Goal-based enhancements
        goal_key = goal.lower().replace(' ', '_')
        if goal_key in mappings.get('goal_based_recommendations', {}):
            goal_rec = mappings['goal_based_recommendations'][goal_key]
            recommendations['goal_specific_advice'] = goal_rec
            
            # Update allocation based on age for retirement planning
            if goal_key == 'retirement' and 'allocation_by_age' in goal_rec:
                age_bracket = self._get_age_bracket(age)
                if age_bracket in goal_rec['allocation_by_age']:
                    recommendations['portfolio_allocation'] = goal_rec['allocation_by_age'][age_bracket]
        
        # Amount-based adjustments
        if amount_category in mappings.get('amount_based_recommendations', {}):
            amount_rec = mappings['amount_based_recommendations'][amount_category]
            recommendations['amount_specific_advice'] = amount_rec
        
        # Calculate projected wealth
        annual_investment = investment_amount * 12
        expected_return = self._parse_return_rate(recommendations['expected_returns'])
        recommendations['projected_wealth'] = self._calculate_future_value(
            annual_investment, expected_return, duration
        )
        
        # Generate strategy
        recommendations['investment_strategy'] = self._generate_strategy(
            risk_profile, goal, duration, investment_amount
        )
        
        return recommendations
    
    def _get_age_bracket(self, age):
        """Get age bracket for recommendations"""
        if age <= 30:
            return '20-30'
        elif age <= 40:
            return '30-40'
        elif age <= 50:
            return '40-50'
        else:
            return '50+'
    
    def _parse_return_rate(self, return_range):
        """Parse expected return rate from string"""
        if not return_range or '%' not in return_range:
            return 0.10  # Default 10%
        
        # Extract numbers from "8-12%" format
        numbers = [float(x) for x in return_range.replace('%', '').split('-')]
        return (sum(numbers) / len(numbers)) / 100
    
    def _calculate_future_value(self, annual_payment, rate, years):
        """Calculate future value of SIP"""
        if rate == 0:
            return annual_payment * years
        
        # SIP future value formula
        monthly_rate = rate / 12
        months = years * 12
        fv = annual_payment * (((1 + monthly_rate) ** months - 1) / monthly_rate) * (1 + monthly_rate)
        return fv / 12  # Convert back to annual terms
    
    def _generate_strategy(self, risk_profile, goal, duration, amount):
        """Generate investment strategy description"""
        strategies = {
            'conservative': f"Focus on capital preservation with steady returns. With ₹{amount:,}/month for {duration} years",
            'moderate': f"Balanced approach mixing growth and stability. Your ₹{amount:,}/month investment",
            'aggressive': f"Growth-focused strategy for wealth creation. Investing ₹{amount:,}/month consistently"
        }
        
        base_strategy = strategies.get(risk_profile, "Systematic investment approach")
        
        if duration < 5:
            base_strategy += " over this short-term horizon suggests focusing on liquid and debt instruments."
        elif duration < 10:
            base_strategy += " over this medium-term period allows for balanced equity-debt allocation."
        else:
            base_strategy += " over this long-term horizon enables maximum equity exposure for wealth creation."
        
        return base_strategy

# Example usage
if __name__ == "__main__":
    engine = EnhancedInvestmentEngine()
    
    user_profile = {
        'age': 28,
        'income': 600000,
        'monthly_investment': 5000,
        'goal': 'wealth_generation',
        'duration_years': 15
    }
    
    recommendations = engine.get_investment_recommendations(user_profile)
    print(json.dumps(recommendations, indent=2))
