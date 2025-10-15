#!/usr/bin/env python3
"""
Analyze Investment Habits Dataset
Test if jocelyndumlao/the-role-of-habits-in-investing-behaviors can enhance Smart Money AI
"""

import kagglehub
import pandas as pd
import os
import json
from pathlib import Path

def analyze_investment_habits_dataset():
    """Download and analyze the investment habits dataset"""
    
    print("🔍 ANALYZING NEW DATASET: jocelyndumlao/the-role-of-habits-in-investing-behaviors")
    print("=" * 75)
    
    try:
        # Download the dataset
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("jocelyndumlao/the-role-of-habits-in-investing-behaviors")
        print(f"✅ Dataset downloaded to: {path}")
        
        # List all files in the dataset
        dataset_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                dataset_files.append(file_path)
        
        print(f"\n📁 Dataset contains {len(dataset_files)} files:")
        for file_path in dataset_files:
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"   • {os.path.basename(file_path)} ({file_size:.1f} KB)")
        
        # Analyze each file
        analysis_results = {}
        
        for file_path in dataset_files:
            if file_path.endswith('.csv'):
                print(f"\n📊 ANALYZING: {os.path.basename(file_path)}")
                print("-" * 50)
                
                try:
                    # Load the CSV
                    df = pd.read_csv(file_path)
                    
                    file_analysis = {
                        'filename': os.path.basename(file_path),
                        'shape': df.shape,
                        'columns': list(df.columns),
                        'sample_data': df.head(5).to_dict('records'),
                        'data_types': df.dtypes.astype(str).to_dict(),
                        'missing_values': df.isnull().sum().to_dict(),
                        'unique_values': {col: int(df[col].nunique()) for col in df.columns}
                    }
                    
                    # Basic info
                    print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    print(f"📝 Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                    
                    # Show sample data
                    print(f"\n🔍 Sample Data (first 3 rows):")
                    for i, row in df.head(3).iterrows():
                        print(f"   Row {i+1}: {dict(list(row.items())[:3])}{'...' if len(row) > 3 else ''}")
                    
                    # Data quality assessment
                    missing_count = df.isnull().sum().sum()
                    print(f"\n📈 Data Quality:")
                    print(f"   • Missing Values: {missing_count}")
                    print(f"   • Complete Records: {df.shape[0] - missing_count}")
                    print(f"   • Data Completeness: {((df.shape[0] * df.shape[1] - missing_count) / (df.shape[0] * df.shape[1]) * 100):.1f}%")
                    
                    # Investment and behavioral relevance analysis
                    investment_keywords = [
                        'invest', 'investment', 'portfolio', 'stock', 'equity', 'bond',
                        'mutual', 'fund', 'asset', 'allocation', 'risk', 'return',
                        'habit', 'behavior', 'behaviour', 'attitude', 'decision',
                        'financial', 'money', 'wealth', 'savings', 'goal',
                        'strategy', 'preference', 'psychology', 'emotion'
                    ]
                    
                    relevance_score = 0
                    relevant_columns = []
                    behavioral_indicators = []
                    
                    # Check column names for investment/behavioral terms
                    for col in df.columns:
                        col_lower = col.lower()
                        for keyword in investment_keywords:
                            if keyword in col_lower:
                                relevance_score += 15
                                relevant_columns.append(col)
                                if keyword in ['habit', 'behavior', 'behaviour', 'attitude', 'psychology']:
                                    behavioral_indicators.append(col)
                                break
                    
                    # Check data content for investment/behavioral terms
                    text_columns = df.select_dtypes(include=['object']).columns
                    content_matches = 0
                    for col in text_columns[:3]:  # Check first 3 text columns
                        if not df[col].empty:
                            sample_text = ' '.join(df[col].astype(str).head(50).values).lower()
                            keyword_matches = sum(1 for keyword in investment_keywords if keyword in sample_text)
                            content_matches += keyword_matches
                            relevance_score += keyword_matches * 3
                    
                    # Bonus for having numerical data (could be survey responses, scores, etc.)
                    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
                    if len(numerical_cols) > 3:
                        relevance_score += 10
                    
                    # Cap relevance score at 100
                    relevance_score = min(relevance_score, 100)
                    
                    file_analysis.update({
                        'relevance_score': relevance_score,
                        'relevant_columns': relevant_columns,
                        'behavioral_indicators': behavioral_indicators,
                        'numerical_columns': list(numerical_cols),
                        'content_keyword_matches': content_matches
                    })
                    
                    print(f"\n🎯 Smart Money AI Relevance: {relevance_score}/100")
                    if relevant_columns:
                        print(f"💰 Investment-Related Columns: {', '.join(relevant_columns[:5])}")
                    if behavioral_indicators:
                        print(f"🧠 Behavioral Indicators: {', '.join(behavioral_indicators[:3])}")
                    if numerical_cols.any():
                        print(f"📊 Numerical Data Columns: {len(numerical_cols)} (potential survey responses/scores)")
                    
                    analysis_results[os.path.basename(file_path)] = file_analysis
                    
                except Exception as e:
                    print(f"❌ Error analyzing {file_path}: {e}")
                    analysis_results[os.path.basename(file_path)] = {
                        'error': str(e),
                        'relevance_score': 0
                    }
        
        # Overall dataset assessment
        print(f"\n🏆 OVERALL DATASET ASSESSMENT")
        print("=" * 50)
        
        total_records = sum(result.get('shape', [0])[0] for result in analysis_results.values() if 'shape' in result)
        avg_relevance = sum(result.get('relevance_score', 0) for result in analysis_results.values()) / len(analysis_results) if analysis_results else 0
        
        print(f"📊 Total Records: {total_records:,}")
        print(f"📁 Files Analyzed: {len(analysis_results)}")
        print(f"🎯 Average Relevance Score: {avg_relevance:.1f}/100")
        
        # Assess Smart Money AI integration potential
        integration_assessment = assess_investment_habits_integration(analysis_results, total_records, avg_relevance)
        
        print(f"\n🚀 INTEGRATION POTENTIAL: {integration_assessment['level']}")
        print(f"💡 {integration_assessment['recommendation']}")
        
        if integration_assessment['level'] in ['HIGH', 'VERY HIGH']:
            print(f"\n✅ RECOMMENDED INTEGRATION STEPS:")
            for i, step in enumerate(integration_assessment['next_steps'], 1):
                print(f"   {i}. {step}")
        elif integration_assessment['level'] == 'MEDIUM':
            print(f"\n🤔 POTENTIAL INTEGRATION STEPS:")
            for i, step in enumerate(integration_assessment['next_steps'], 1):
                print(f"   {i}. {step}")
        
        # Compare with existing datasets
        print(f"\n📊 COMPARISON WITH EXISTING SMART MONEY AI DATASETS:")
        print(f"   Current Dataset 1: Personal Finance (20,000 records, 60/100 relevance) ✅ Integrated")
        print(f"   Current Dataset 2: Investment Survey (100 records, 75/100 relevance) ✅ Integrated")
        print(f"   New Dataset: Investment Habits ({total_records:,} records, {avg_relevance:.1f}/100 relevance)")
        
        # Save analysis results
        save_investment_habits_analysis(analysis_results, integration_assessment, total_records, avg_relevance)
        
        return analysis_results, integration_assessment
        
    except Exception as e:
        print(f"❌ Error downloading or analyzing dataset: {e}")
        return None, None

def assess_investment_habits_integration(analysis_results, total_records, avg_relevance):
    """Assess integration potential for investment habits dataset"""
    
    # Count behavioral indicators across all files
    total_behavioral_indicators = sum(
        len(result.get('behavioral_indicators', [])) 
        for result in analysis_results.values() 
        if 'behavioral_indicators' in result
    )
    
    # Count investment-related columns
    total_investment_columns = sum(
        len(result.get('relevant_columns', [])) 
        for result in analysis_results.values() 
        if 'relevant_columns' in result
    )
    
    integration_assessment = {
        'level': 'LOW',
        'recommendation': '',
        'next_steps': [],
        'behavioral_score': total_behavioral_indicators,
        'investment_score': total_investment_columns
    }
    
    # Assessment criteria for investment behavior datasets
    if avg_relevance >= 80 and total_records >= 500 and total_behavioral_indicators >= 3:
        integration_assessment['level'] = 'VERY HIGH'
        integration_assessment['recommendation'] = (
            f"Excellent integration potential! {total_records:,} records with {avg_relevance:.1f}/100 relevance "
            f"and {total_behavioral_indicators} behavioral indicators. This dataset could significantly enhance "
            "Smart Money AI's behavioral investment profiling capabilities."
        )
        integration_assessment['next_steps'] = [
            "Deep dive analysis of behavioral patterns and investment habits",
            "Map behavioral indicators to existing investment risk profiles", 
            "Design enhanced behavioral profiling algorithms",
            "Create habit-based investment recommendation engine",
            "Integrate with existing 100 investment behavioral profiles",
            "Implement comprehensive testing and validation"
        ]
    
    elif avg_relevance >= 60 and total_records >= 200 and (total_behavioral_indicators >= 2 or total_investment_columns >= 5):
        integration_assessment['level'] = 'HIGH'
        integration_assessment['recommendation'] = (
            f"Strong integration potential! {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            f"Contains valuable behavioral insights ({total_behavioral_indicators} indicators) that could enhance "
            "Smart Money AI's investment recommendation accuracy."
        )
        integration_assessment['next_steps'] = [
            "Analyze behavioral patterns and correlations with investment outcomes",
            "Map habits data to existing investment profiles",
            "Enhance behavioral risk assessment algorithms",
            "Test integration with current investment engine",
            "Validate improved recommendation accuracy"
        ]
    
    elif avg_relevance >= 45 and total_records >= 100:
        integration_assessment['level'] = 'MEDIUM'
        integration_assessment['recommendation'] = (
            f"Moderate integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "Could provide supplementary insights for investment behavioral analysis."
        )
        integration_assessment['next_steps'] = [
            "Focus on highest-value behavioral indicators",
            "Consider as supplementary data to enhance existing profiles",
            "Analyze specific habit patterns relevant to investment decisions"
        ]
    
    else:
        integration_assessment['level'] = 'LOW'
        integration_assessment['recommendation'] = (
            f"Limited integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "May not provide sufficient value for Smart Money AI enhancement."
        )
        integration_assessment['next_steps'] = [
            "Focus on current high-performing datasets",
            "Look for datasets with stronger behavioral investment focus"
        ]
    
    return integration_assessment

def save_investment_habits_analysis(analysis_results, integration_assessment, total_records, avg_relevance):
    """Save investment habits analysis results"""
    
    results = {
        'dataset_name': 'jocelyndumlao/the-role-of-habits-in-investing-behaviors',
        'analysis_date': '2025-10-15',
        'total_records': total_records,
        'average_relevance': avg_relevance,
        'file_analyses': analysis_results,
        'integration_assessment': integration_assessment,
        'comparison_with_existing': {
            'current_personal_finance': {
                'name': 'shriyashjagtap/indian-personal-finance-and-spending-habits',
                'records': 20000,
                'relevance': 60,
                'status': 'integrated',
                'focus': 'demographic spending analysis'
            },
            'current_investment_survey': {
                'name': 'sudarsan27/investment-survey-dataset', 
                'records': 100,
                'relevance': 75,
                'status': 'integrated',
                'focus': 'behavioral investment profiling'
            },
            'rejected_dataset': {
                'name': 'ramyapintchy/personal-finance-data',
                'records': 1500,
                'relevance': 42,
                'status': 'rejected',
                'reason': 'inferior to existing data'
            }
        },
        'recommendation_summary': {
            'should_integrate': integration_assessment['level'] in ['HIGH', 'VERY HIGH'],
            'focus_areas': ['behavioral patterns', 'investment habits', 'decision-making psychology'],
            'expected_enhancement': 'Improved behavioral investment profiling accuracy'
        }
    }
    
    # Save to organized data directory
    os.makedirs('smart_money_ai/data/raw', exist_ok=True)
    
    with open('smart_money_ai/data/raw/investment_habits_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: smart_money_ai/data/raw/investment_habits_analysis.json")

if __name__ == "__main__":
    print("🎯 Smart Money AI - Investment Habits Dataset Analyzer")
    print("Testing: jocelyndumlao/the-role-of-habits-in-investing-behaviors")
    print("Focus: Behavioral investment patterns and habits analysis")
    print()
    
    analysis, integration = analyze_investment_habits_dataset()
    
    if analysis:
        print(f"\n🎉 Analysis complete! Check smart_money_ai/data/raw/investment_habits_analysis.json for detailed results")
        
        # Show final recommendation
        if integration:
            level = integration['level']
            if level in ['HIGH', 'VERY HIGH']:
                print(f"\n🚀 RECOMMENDATION: PROCEED WITH INTEGRATION")
                print(f"💡 This dataset can significantly enhance Smart Money AI's behavioral intelligence!")
            elif level == 'MEDIUM':
                print(f"\n🤔 RECOMMENDATION: CONSIDER SELECTIVE INTEGRATION")
                print(f"💡 Focus on specific behavioral indicators that add value")
            else:
                print(f"\n❌ RECOMMENDATION: SKIP INTEGRATION") 
                print(f"💡 Current datasets provide superior intelligence")
    else:
        print(f"\n❌ Analysis failed. Please check the dataset name and try again.")