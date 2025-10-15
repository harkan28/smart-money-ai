#!/usr/bin/env python3
"""
Smart Money AI - Investment Survey Dataset Integration
Enhance the investment recommendation engine with behavioral insights
"""

import kagglehub
import pandas as pd
import numpy as np
import sqlite3
import json
import os
from datetime import datetime

class InvestmentSurveyIntegrator:
    """Integrate investment survey dataset into Smart Money AI"""
    
    def __init__(self):
        self.db_path = "data/investment_behavioral_data.db"
        self.dataset_name = "sudarsan27/investment-survey-dataset"
        self.data_file = "investment_survey.csv"
        
    def load_dataset(self):
        """Load the investment survey dataset"""
        print("📥 Loading Investment Survey Dataset...")
        
        try:
            # Download dataset
            path = kagglehub.dataset_download(self.dataset_name)
            dataset_path = os.path.join(path, self.data_file)
            
            # Load using pandas
            df = pd.read_csv(dataset_path)
            
            # Clean the data
            df = self.clean_data(df)
            
            print(f"✅ Dataset loaded and cleaned: {len(df)} records with {len(df.columns)} features")
            return df
            
        except Exception as e:
            print(f"❌ Error loading dataset: {str(e)}")
            return None
    
    def clean_data(self, df):
        """Clean and prepare the investment survey data"""
        print("🧹 Cleaning Dataset...")
        
        # Remove unnamed columns
        df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col], errors='ignore')
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Handle the duration column name variations
        duration_col = None
        for col in df.columns:
            if 'Duration' in col and 'save' in col:
                duration_col = col
                break
        
        if duration_col:
            df = df.rename(columns={duration_col: 'Duration_to_save_years'})
        
        # Convert Investment_per_month to numeric (handle mixed types)
        if 'Investment_per_month' in df.columns:
            df['Investment_per_month'] = pd.to_numeric(df['Investment_per_month'], errors='coerce')
        
        # Convert Duration_to_save_years to numeric
        if 'Duration_to_save_years' in df.columns:
            df['Duration_to_save_years'] = pd.to_numeric(df['Duration_to_save_years'], errors='coerce')
        
        # Create risk categories based on investment mode
        risk_mapping = {
            'Banking - RD, FD': 'Conservative',
            'Gold / Any other Materialistic investment': 'Conservative',
            'Chit fund': 'Conservative',
            'Mutual Funds': 'Moderate',
            'Real estate, Bonds': 'Moderate',
            'Stocks - Intraday, long term': 'Aggressive',
            'Crypto currency': 'Aggressive',
            'Marketing ': 'Moderate',
            'Not prepared ': 'Conservative'
        }
        
        df['Risk_Category'] = df['Mode_of_investment'].map(risk_mapping)
        
        # Create age groups
        df['Age_Group'] = pd.cut(df['Age'], 
                                bins=[0, 25, 35, 45, 100], 
                                labels=['18-25', '26-35', '36-45', '45+'])
        
        # Create income groups
        df['Income_Group'] = pd.cut(df['Annual_income'], 
                                  bins=[0, 300000, 600000, 1000000, float('inf')],
                                  labels=['Low', 'Medium', 'High', 'Very High'])
        
        print(f"✅ Data cleaned successfully")
        return df
    
    def create_behavioral_profiles(self, df):
        """Create behavioral investment profiles"""
        print("🎯 Creating Behavioral Investment Profiles...")
        
        profiles = {}
        
        # Risk-based profiles
        for risk_category in df['Risk_Category'].unique():
            if pd.isna(risk_category):
                continue
                
            risk_data = df[df['Risk_Category'] == risk_category]
            
            profiles[f'risk_{risk_category.lower()}'] = {
                'sample_size': len(risk_data),
                'avg_age': float(risk_data['Age'].mean()),
                'avg_income': float(risk_data['Annual_income'].mean()),
                'avg_investment_amount': float(risk_data['Investment_per_month'].mean()),
                'avg_duration': float(risk_data['Duration_to_save_years'].mean()),
                'top_goals': risk_data['Goal_for_investment'].value_counts().head(3).to_dict(),
                'motivation_sources': risk_data['Motivation_cause'].value_counts().head(3).to_dict(),
                'preferred_resources': risk_data['Resources_used'].value_counts().head(3).to_dict(),
                'investment_modes': risk_data['Mode_of_investment'].value_counts().to_dict()
            }
        
        # Goal-based profiles
        for goal in df['Goal_for_investment'].unique():
            if pd.isna(goal):
                continue
                
            goal_data = df[df['Goal_for_investment'] == goal]
            goal_key = goal.lower().replace(' ', '_').replace('/', '_')[:30]
            
            profiles[f'goal_{goal_key}'] = {
                'sample_size': len(goal_data),
                'avg_age': float(goal_data['Age'].mean()),
                'avg_income': float(goal_data['Annual_income'].mean()),
                'avg_investment_amount': float(goal_data['Investment_per_month'].mean()),
                'avg_duration': float(goal_data['Duration_to_save_years'].mean()),
                'risk_distribution': goal_data['Risk_Category'].value_counts().to_dict(),
                'preferred_modes': goal_data['Mode_of_investment'].value_counts().head(3).to_dict()
            }
        
        print(f"✅ Created {len(profiles)} behavioral profiles")
        return profiles
    
    def create_investment_recommendations_mapping(self, profiles):
        """Create investment recommendation mappings based on profiles"""
        print("💰 Creating Investment Recommendation Mappings...")
        
        mappings = {
            'risk_based_recommendations': {
                'conservative': {
                    'primary_instruments': ['Fixed Deposits', 'PPF', 'NSC', 'Debt Mutual Funds'],
                    'allocation': {'debt': 80, 'equity': 20},
                    'expected_returns': '6-8%',
                    'suitable_for': 'Capital preservation, steady income'
                },
                'moderate': {
                    'primary_instruments': ['Balanced Mutual Funds', 'ELSS', 'Large Cap Funds', 'Gold ETF'],
                    'allocation': {'debt': 50, 'equity': 50},
                    'expected_returns': '8-12%',
                    'suitable_for': 'Balanced growth with moderate risk'
                },
                'aggressive': {
                    'primary_instruments': ['Small Cap Funds', 'Mid Cap Funds', 'Direct Equity', 'Sector Funds'],
                    'allocation': {'debt': 20, 'equity': 80},
                    'expected_returns': '12-15%',
                    'suitable_for': 'Wealth creation, long-term growth'
                }
            },
            
            'goal_based_recommendations': {
                'retirement': {
                    'instruments': ['ELSS', 'PPF', 'NPS', 'Large Cap Funds'],
                    'strategy': 'Long-term systematic investment',
                    'allocation_by_age': {
                        '20-30': {'equity': 80, 'debt': 20},
                        '30-40': {'equity': 70, 'debt': 30},
                        '40-50': {'equity': 60, 'debt': 40},
                        '50+': {'equity': 40, 'debt': 60}
                    }
                },
                'wealth_generation': {
                    'instruments': ['Equity Mutual Funds', 'Direct Stocks', 'SIP'],
                    'strategy': 'Growth-focused long-term investment',
                    'min_duration': 7,
                    'recommended_allocation': {'equity': 80, 'debt': 20}
                },
                'education': {
                    'instruments': ['Child Education Plans', 'Balanced Funds', 'PPF'],
                    'strategy': 'Goal-based systematic planning',
                    'typical_duration': 15,
                    'recommended_allocation': {'equity': 60, 'debt': 40}
                },
                'tax_saving': {
                    'instruments': ['ELSS', 'PPF', 'NSC', 'Tax Saver FD'],
                    'strategy': '80C optimization with growth',
                    'max_limit': 150000,
                    'recommended_allocation': {'elss': 60, 'ppf': 40}
                }
            },
            
            'amount_based_recommendations': {
                'micro_investor': {  # ₹200-1000
                    'suitable_instruments': ['SIP Mutual Funds', 'Micro Investment Apps'],
                    'min_amount': 500,
                    'recommended_funds': ['Index Funds', 'Large Cap Funds'],
                    'advice': 'Start small, stay consistent'
                },
                'regular_investor': {  # ₹1000-5000
                    'suitable_instruments': ['Diversified Mutual Funds', 'ELSS', 'Gold ETF'],
                    'portfolio_approach': 'Balanced diversification',
                    'recommended_allocation': {'equity': 60, 'debt': 30, 'gold': 10}
                },
                'substantial_investor': {  # ₹5000+
                    'suitable_instruments': ['Direct Equity', 'Premium Funds', 'Portfolio PMS'],
                    'portfolio_approach': 'Advanced diversification',
                    'additional_options': ['Real Estate', 'International Funds']
                }
            }
        }
        
        return mappings
    
    def save_to_database(self, df, profiles, mappings):
        """Save enhanced investment data to database"""
        print("💾 Saving Investment Data to Database...")
        
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Connect to database
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Save raw survey data
            df.to_sql('investment_survey_data', conn, if_exists='replace', index=False)
            
            # Save behavioral profiles
            profiles_df = pd.DataFrame([
                {'profile_type': profile_type, 'profile_data': json.dumps(data)}
                for profile_type, data in profiles.items()
            ])
            profiles_df.to_sql('behavioral_profiles', conn, if_exists='replace', index=False)
            
            # Save recommendation mappings
            mappings_df = pd.DataFrame([
                {'mapping_type': mapping_type, 'mapping_data': json.dumps(data)}
                for mapping_type, data in mappings.items()
            ])
            mappings_df.to_sql('recommendation_mappings', conn, if_exists='replace', index=False)
            
            # Create analysis views
            conn.execute("""
                CREATE VIEW IF NOT EXISTS risk_behavior_analysis AS
                SELECT 
                    Risk_Category,
                    Age_Group,
                    Income_Group,
                    AVG(Investment_per_month) as avg_investment,
                    AVG(Duration_to_save_years) as avg_duration,
                    COUNT(*) as sample_size
                FROM investment_survey_data
                WHERE Risk_Category IS NOT NULL
                GROUP BY Risk_Category, Age_Group, Income_Group
            """)
            
            conn.execute("""
                CREATE VIEW IF NOT EXISTS goal_investment_patterns AS
                SELECT 
                    Goal_for_investment,
                    Risk_Category,
                    AVG(Investment_per_month) as avg_investment,
                    AVG(Age) as avg_age,
                    COUNT(*) as sample_size
                FROM investment_survey_data
                WHERE Goal_for_investment IS NOT NULL AND Risk_Category IS NOT NULL
                GROUP BY Goal_for_investment, Risk_Category
            """)
            
            conn.commit()
            print(f"✅ Investment data saved to {self.db_path}")
            
        finally:
            conn.close()
    
    def create_enhanced_investment_engine(self):
        """Create enhanced investment recommendation engine"""
        print("⚡ Creating Enhanced Investment Engine...")
        
        engine_code = '''
import sqlite3
import json
import pandas as pd

class EnhancedInvestmentEngine:
    """Enhanced investment recommendation engine with behavioral insights"""
    
    def __init__(self, db_path="data/investment_behavioral_data.db"):
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
'''
        
        with open('src/investment/enhanced_investment_engine.py', 'w') as f:
            f.write(engine_code)
        
        print("✅ Enhanced investment engine created at src/investment/enhanced_investment_engine.py")
    
    def run_integration(self):
        """Run the complete integration process"""
        print("🚀 Starting Investment Survey Dataset Integration")
        print("=" * 60)
        
        # Load dataset
        df = self.load_dataset()
        if df is None:
            return False
        
        # Create behavioral profiles
        profiles = self.create_behavioral_profiles(df)
        
        # Create recommendation mappings
        mappings = self.create_investment_recommendations_mapping(profiles)
        
        # Save to database
        self.save_to_database(df, profiles, mappings)
        
        # Create enhanced engine
        self.create_enhanced_investment_engine()
        
        print("\n✅ Integration Complete!")
        print("🎯 Enhanced Investment Features Added:")
        print("   • Behavioral risk profiling")
        print("   • Goal-based investment recommendations")
        print("   • Amount-appropriate investment strategies")
        print("   • Personalized portfolio allocation")
        print("   • Projected wealth calculations")
        
        return True

if __name__ == "__main__":
    integrator = InvestmentSurveyIntegrator()
    success = integrator.run_integration()
    
    if success:
        print("\n🎯 Next Steps:")
        print("1. Test the enhanced investment engine")
        print("2. Integrate with existing Smart Money AI system")
        print("3. Add behavioral insights to user recommendations")
        print("4. Create investment preference learning system")
    else:
        print("\n❌ Integration failed. Check error messages above.")