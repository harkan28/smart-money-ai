
import sqlite3
import json

class SpendingComparator:
    """Compare user spending with demographic benchmarks"""
    
    def __init__(self, db_path=None):
        if db_path is None:
            # Auto-detect database path relative to current module
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.db_path = os.path.join(current_dir, '..', '..', 'data', 'raw', 'demographic_benchmarks.db')
        else:
            self.db_path = db_path
    
    def get_user_segment(self, age, income, city_tier, dependents=0):
        """Determine user's demographic segment"""
        age_group = self._get_age_group(age)
        income_group = self._get_income_group(income)
        family_size = self._get_family_size(dependents)
        
        return f"{age_group}_{income_group}_{city_tier}"
    
    def _get_age_group(self, age):
        if age <= 25: return "18-25"
        elif age <= 35: return "26-35"
        elif age <= 45: return "36-45"
        elif age <= 55: return "46-55"
        else: return "55+"
    
    def _get_income_group(self, income):
        if income <= 25000: return "Low"
        elif income <= 50000: return "Medium"
        elif income <= 100000: return "High"
        else: return "Very High"
    
    def _get_family_size(self, dependents):
        if dependents == 0: return "Single"
        elif dependents <= 2: return "Small"
        elif dependents <= 4: return "Medium"
        else: return "Large"
    
    def compare_spending(self, user_profile, user_expenses):
        """Compare user spending with similar demographic"""
        segment = self.get_user_segment(
            user_profile['age'], 
            user_profile['income'], 
            user_profile.get('city_tier', 'Tier_2'),
            user_profile.get('dependents', 0)
        )
        
        # Get benchmark data
        conn = sqlite3.connect(self.db_path)
        result = conn.execute(
            "SELECT benchmarks FROM demographic_benchmarks WHERE segment = ?", 
            (segment,)
        ).fetchone()
        conn.close()
        
        if not result:
            return {"status": "no_benchmark", "message": "No similar users found"}
        
        benchmarks = json.loads(result[0])
        comparisons = {}
        
        # Compare each expense category
        for category, user_amount in user_expenses.items():
            if category in benchmarks['expense_breakdown']:
                benchmark = benchmarks['expense_breakdown'][category]
                comparison = {
                    'user_amount': user_amount,
                    'benchmark_amount': benchmark['avg_amount'],
                    'difference_amount': user_amount - benchmark['avg_amount'],
                    'difference_percentage': ((user_amount - benchmark['avg_amount']) / benchmark['avg_amount'] * 100) if benchmark['avg_amount'] > 0 else 0,
                    'percentile': self._calculate_percentile(user_amount, benchmark),
                    'recommendation': self._get_recommendation(user_amount, benchmark)
                }
                comparisons[category] = comparison
        
        return {
            'status': 'success',
            'segment': segment,
            'sample_size': benchmarks['count'],
            'comparisons': comparisons,
            'overall_assessment': self._get_overall_assessment(comparisons)
        }
    
    def _calculate_percentile(self, amount, benchmark):
        """Estimate percentile based on mean and std"""
        mean = benchmark['avg_amount']
        std = benchmark['std']
        if std == 0:
            return 50
        z_score = (amount - mean) / std
        # Rough percentile estimation
        return max(0, min(100, 50 + z_score * 20))
    
    def _get_recommendation(self, amount, benchmark):
        """Generate spending recommendation"""
        avg = benchmark['avg_amount']
        if amount > avg * 1.5:
            return "Consider reducing spending in this category"
        elif amount > avg * 1.2:
            return "Spending is above average, monitor closely"
        elif amount < avg * 0.8:
            return "Good control on spending in this category"
        else:
            return "Spending is within normal range"
    
    def _get_overall_assessment(self, comparisons):
        """Generate overall spending assessment"""
        above_avg_count = sum(1 for comp in comparisons.values() 
                             if comp['difference_percentage'] > 20)
        total_categories = len(comparisons)
        
        if above_avg_count > total_categories * 0.6:
            return "Consider reviewing your budget - spending above average in multiple categories"
        elif above_avg_count > total_categories * 0.3:
            return "Generally good spending habits with some areas for improvement"
        else:
            return "Excellent spending control compared to similar users"

# Example usage
if __name__ == "__main__":
    comparator = SpendingComparator()
    
    user_profile = {
        'age': 30,
        'income': 50000,
        'city_tier': 'Tier_1',
        'dependents': 1
    }
    
    user_expenses = {
        'groceries': 6000,
        'transport': 3000,
        'eating_out': 2000,
        'entertainment': 1500
    }
    
    result = comparator.compare_spending(user_profile, user_expenses)
    print(json.dumps(result, indent=2))
