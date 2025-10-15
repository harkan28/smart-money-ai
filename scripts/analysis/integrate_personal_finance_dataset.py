#!/usr/bin/env python3
"""
Smart Money AI - Personal Finance Dataset Integration
Integrate the Indian personal finance dataset for enhanced budgeting and analytics
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import numpy as np
import sqlite3
import json
import os
from datetime import datetime
from pathlib import Path

class PersonalFinanceDatasetIntegrator:
    """Integrate personal finance dataset into Smart Money AI"""
    
    def __init__(self):
        self.db_path = "data/demographic_benchmarks.db"
        self.dataset_name = "shriyashjagtap/indian-personal-finance-and-spending-habits"
        self.data_file = "data.csv"
        
    def load_dataset(self):
        """Load the personal finance dataset"""
        print("📥 Loading Personal Finance Dataset...")
        
        try:
            # Download dataset first
            path = kagglehub.dataset_download(self.dataset_name)
            dataset_path = os.path.join(path, self.data_file)
            
            # Load using pandas
            df = pd.read_csv(dataset_path)
            
            print(f"✅ Dataset loaded: {len(df)} records with {len(df.columns)} features")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return None
    
    def create_demographic_segments(self, df):
        """Create demographic segments for benchmarking"""
        print("🎯 Creating Demographic Segments...")
        
        # Define age groups
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 25, 35, 45, 55, 100], 
                                labels=['18-25', '26-35', '36-45', '46-55', '55+'])
        
        # Define income groups
        df['Income_Group'] = pd.cut(df['Income'], 
                                  bins=[0, 25000, 50000, 100000, float('inf')],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Define family size groups
        df['Family_Size'] = pd.cut(df['Dependents'], 
                                 bins=[-1, 0, 2, 4, float('inf')],
                                 labels=['Single', 'Small', 'Medium', 'Large'])
        
        # Create composite segments
        df['Segment'] = (df['Age_Group'].astype(str) + "_" + 
                        df['Income_Group'].astype(str) + "_" + 
                        df['City_Tier'].astype(str))
        
        print(f"✅ Created {df['Segment'].nunique()} demographic segments")
        return df
    
    def calculate_benchmarks(self, df):
        """Calculate spending benchmarks by demographic segments"""
        print("📊 Calculating Spending Benchmarks...")
        
        expense_categories = [
            'Rent', 'Groceries', 'Transport', 'Eating_Out', 
            'Entertainment', 'Utilities', 'Healthcare', 'Education'
        ]
        
        benchmarks = {}
        
        # Overall benchmarks
        benchmarks['overall'] = {
            'total_expenses': df[expense_categories].sum(axis=1).describe().to_dict(),
            'savings_rate': df['Desired_Savings_Percentage'].describe().to_dict(),
            'disposable_income': df['Disposable_Income'].describe().to_dict()
        }
        
        # By demographic segments
        for segment in df['Segment'].unique():
            if pd.isna(segment):
                continue
                
            segment_data = df[df['Segment'] == segment]
            
            if len(segment_data) < 10:  # Skip small segments
                continue
            
            benchmarks[segment] = {
                'count': len(segment_data),
                'avg_income': float(segment_data['Income'].mean()),
                'avg_age': float(segment_data['Age'].mean()),
                'avg_dependents': float(segment_data['Dependents'].mean()),
                'expense_breakdown': {},
                'savings_metrics': {
                    'desired_percentage': float(segment_data['Desired_Savings_Percentage'].mean()),
                    'actual_savings': float(segment_data['Desired_Savings'].mean()),
                    'disposable_income': float(segment_data['Disposable_Income'].mean())
                }
            }
            
            # Calculate expense percentages
            total_expenses = segment_data[expense_categories].sum(axis=1).mean()
            for category in expense_categories:
                avg_amount = float(segment_data[category].mean())
                percentage = (avg_amount / total_expenses * 100) if total_expenses > 0 else 0
                
                benchmarks[segment]['expense_breakdown'][category.lower()] = {
                    'avg_amount': avg_amount,
                    'percentage_of_total': percentage,
                    'min': float(segment_data[category].min()),
                    'max': float(segment_data[category].max()),
                    'std': float(segment_data[category].std())
                }
        
        print(f"✅ Calculated benchmarks for {len(benchmarks)} segments")
        return benchmarks
    
    def save_to_database(self, df, benchmarks):
        """Save data to SQLite database"""
        print("💾 Saving to Database...")
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Save raw data
            df.to_sql('personal_finance_data', conn, if_exists='replace', index=False)
            
            # Save benchmarks as JSON
            benchmarks_df = pd.DataFrame([
                {'segment': segment, 'benchmarks': json.dumps(data)}
                for segment, data in benchmarks.items()
            ])
            benchmarks_df.to_sql('demographic_benchmarks', conn, if_exists='replace', index=False)
            
            # Create optimized views
            conn.execute("""
                CREATE VIEW IF NOT EXISTS spending_patterns AS
                SELECT 
                    Age_Group,
                    Income_Group,
                    City_Tier,
                    AVG(Groceries) as avg_groceries,
                    AVG(Transport) as avg_transport,
                    AVG(Eating_Out) as avg_eating_out,
                    AVG(Entertainment) as avg_entertainment,
                    AVG(Utilities) as avg_utilities,
                    AVG(Desired_Savings_Percentage) as avg_savings_rate,
                    COUNT(*) as sample_size
                FROM personal_finance_data
                GROUP BY Age_Group, Income_Group, City_Tier
                HAVING COUNT(*) >= 10
            """)
            
            conn.commit()
            print(f"✅ Data saved to {self.db_path}")
            
        finally:
            conn.close()
    
    def create_budget_templates(self, benchmarks):
        """Create budget templates based on demographic data"""
        print("📋 Creating Budget Templates...")
        
        templates = {}
        
        for segment, data in benchmarks.items():
            if segment == 'overall':
                continue
                
            if 'expense_breakdown' not in data:
                continue
            
            # Create budget template
            template = {
                'segment': segment,
                'sample_size': data['count'],
                'recommended_allocations': {},
                'savings_target': data['savings_metrics']['desired_percentage'],
                'typical_income': data['avg_income']
            }
            
            # Calculate recommended budget allocations
            for category, metrics in data['expense_breakdown'].items():
                template['recommended_allocations'][category] = {
                    'percentage': round(metrics['percentage_of_total'], 1),
                    'typical_amount': round(metrics['avg_amount'], 0),
                    'range': {
                        'min': round(metrics['min'], 0),
                        'max': round(metrics['max'], 0)
                    }
                }
            
            templates[segment] = template
        
        # Save templates
        with open('data/budget_templates.json', 'w') as f:
            json.dump(templates, f, indent=2)
        
        print(f"✅ Created {len(templates)} budget templates")
        return templates
    
    def create_comparison_engine(self):
        """Create spending comparison functionality"""
        print("⚖️ Creating Comparison Engine...")
        
        comparison_code = '''
import sqlite3
import json

class SpendingComparator:
    """Compare user spending with demographic benchmarks"""
    
    def __init__(self, db_path="data/demographic_benchmarks.db"):
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
'''
        
        with open('src/analytics/spending_comparator.py', 'w') as f:
            f.write(comparison_code)
        
        print("✅ Comparison engine created at src/analytics/spending_comparator.py")
    
    def run_integration(self):
        """Run the complete integration process"""
        print("🚀 Starting Personal Finance Dataset Integration")
        print("=" * 60)
        
        # Load dataset
        df = self.load_dataset()
        if df is None:
            return False
        
        # Create segments
        df = self.create_demographic_segments(df)
        
        # Calculate benchmarks
        benchmarks = self.calculate_benchmarks(df)
        
        # Save to database
        self.save_to_database(df, benchmarks)
        
        # Create budget templates
        templates = self.create_budget_templates(benchmarks)
        
        # Create comparison engine
        self.create_comparison_engine()
        
        print("\n✅ Integration Complete!")
        print("📊 Features Added to Smart Money AI:")
        print("   • Demographic-based budget templates")
        print("   • Spending pattern comparisons")
        print("   • Personalized financial insights")
        print("   • Benchmark-based recommendations")
        
        return True

if __name__ == "__main__":
    integrator = PersonalFinanceDatasetIntegrator()
    success = integrator.run_integration()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Test the spending comparator with sample data")
        print("2. Integrate with existing Smart Money AI budgeting system")
        print("3. Add demographic insights to investment recommendations")
        print("4. Create user-facing comparison features")
    else:
        print("\n❌ Integration failed. Check error messages above.")