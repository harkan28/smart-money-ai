#!/usr/bin/env python3
"""
Analyze Investment Survey Dataset
Test the dataset: sudarsan27/investment-survey-dataset
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import json
import os
from datetime import datetime

def analyze_investment_survey_dataset():
    """Analyze the investment survey dataset"""
    
    print("🔍 Analyzing Investment Survey Dataset")
    print("=" * 60)
    print("Dataset: sudarsan27/investment-survey-dataset")
    print("Expected: Investment behavior and preferences data")
    print()
    
    try:
        # First, let's download and see what files are available
        print("📥 Downloading dataset to explore structure...")
        path = kagglehub.dataset_download("sudarsan27/investment-survey-dataset")
        print(f"✅ Dataset downloaded to: {path}")
        
        # List all files in the dataset
        print(f"\n📁 Dataset Structure:")
        print("-" * 40)
        
        dataset_files = []
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, path)
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                dataset_files.append({
                    'filename': rel_path,
                    'size_mb': round(file_size, 2),
                    'extension': os.path.splitext(file)[1]
                })
                print(f"   📄 {rel_path} ({file_size:.2f} MB)")
        
        # Analyze each CSV file
        print(f"\n🔍 Analyzing Dataset Files:")
        print("=" * 50)
        
        analysis_results = {}
        
        for file_info in dataset_files:
            if file_info['extension'].lower() in ['.csv', '.xlsx']:
                filename = file_info['filename']
                print(f"\n📊 Loading: {filename}")
                print("-" * 30)
                
                try:
                    # Load using pandas
                    dataset_path = os.path.join(path, filename)
                    if file_info['extension'].lower() == '.csv':
                        df = pd.read_csv(dataset_path)
                    else:
                        df = pd.read_excel(dataset_path)
                    
                    # Basic analysis
                    rows, cols = df.shape
                    print(f"   📏 Dimensions: {rows} rows × {cols} columns")
                    print(f"   📋 Columns: {list(df.columns)}")
                    
                    # Show sample data
                    print(f"   🔍 Sample Data (first 3 rows):")
                    sample_data = df.head(3)
                    print(sample_data.to_string(index=False))
                    
                    # Data types
                    print(f"   📝 Data Types:")
                    for col, dtype in df.dtypes.items():
                        print(f"      {col}: {dtype}")
                    
                    # Check for missing values
                    missing_counts = df.isnull().sum()
                    if missing_counts.sum() > 0:
                        print(f"   ⚠️  Missing Values:")
                        for col, count in missing_counts.items():
                            if count > 0:
                                print(f"      {col}: {count} ({count/len(df)*100:.1f}%)")
                    
                    # Store detailed analysis
                    analysis_results[filename] = {
                        'rows': rows,
                        'columns': cols,
                        'column_names': list(df.columns),
                        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'sample_data': df.head(5).to_dict('records'),
                        'missing_values': missing_counts.to_dict(),
                        'file_size_mb': file_info['size_mb']
                    }
                    
                    # Statistical analysis for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        print(f"   📈 Numeric Column Statistics:")
                        for col in numeric_cols:
                            stats = df[col].describe()
                            print(f"      {col}: Mean={stats['mean']:.2f}, Range=[{stats['min']:.2f}, {stats['max']:.2f}]")
                    
                    # Check for investment-related keywords
                    investment_keywords = [
                        'investment', 'invest', 'portfolio', 'risk', 'return', 'fund',
                        'stock', 'equity', 'bond', 'mutual', 'sip', 'goal', 'preference',
                        'age', 'income', 'experience', 'allocation', 'strategy', 'advisor',
                        'knowledge', 'horizon', 'objective', 'asset', 'diversification'
                    ]
                    
                    relevant_columns = []
                    for col in df.columns:
                        col_lower = col.lower()
                        for keyword in investment_keywords:
                            if keyword in col_lower:
                                relevant_columns.append((col, keyword))
                                break
                    
                    if relevant_columns:
                        print(f"   💰 Investment-Related Columns:")
                        for col, keyword in relevant_columns:
                            print(f"      {col} (matches: {keyword})")
                        analysis_results[filename]['investment_columns'] = relevant_columns
                    
                    # Check for unique values in categorical columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        print(f"   🏷️  Categorical Data:")
                        for col in categorical_cols:
                            unique_count = df[col].nunique()
                            if unique_count <= 15:  # Show values if not too many
                                unique_vals = df[col].unique()
                                print(f"      {col}: {unique_count} unique values {list(unique_vals)}")
                            else:
                                print(f"      {col}: {unique_count} unique values")
                    
                    # Investment-specific analysis
                    investment_analysis = analyze_investment_patterns(df)
                    if investment_analysis:
                        print(f"   📊 Investment Patterns:")
                        for pattern, details in investment_analysis.items():
                            print(f"      {pattern}: {details}")
                        analysis_results[filename]['investment_patterns'] = investment_analysis
                    
                except Exception as e:
                    print(f"   ❌ Error loading file: {str(e)}")
                    analysis_results[filename] = {'error': str(e)}
        
        # Assess relevance for Smart Money AI Investment Engine
        print(f"\n🎯 Investment Engine Relevance Assessment:")
        print("=" * 50)
        
        assessment = assess_investment_relevance(analysis_results)
        print(assessment['detailed_report'])
        
        # Save complete analysis
        results = {
            'dataset_info': {
                'name': 'sudarsan27/investment-survey-dataset',
                'download_path': path,
                'analysis_date': datetime.now().isoformat(),
                'total_files': len(dataset_files),
                'file_list': dataset_files
            },
            'analysis_results': analysis_results,
            'investment_assessment': assessment,
            'integration_recommendations': generate_investment_recommendations(assessment)
        }
        
        # Save to file
        output_file = "data/investment_survey_analysis.json"
        os.makedirs("data", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Complete analysis saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {str(e)}")
        return None

def analyze_investment_patterns(df):
    """Analyze investment-specific patterns in the data"""
    
    patterns = {}
    
    # Look for risk-related patterns
    risk_cols = [col for col in df.columns if 'risk' in col.lower()]
    if risk_cols:
        for col in risk_cols:
            if df[col].dtype == 'object':
                risk_dist = df[col].value_counts()
                patterns[f'Risk Distribution ({col})'] = risk_dist.to_dict()
    
    # Look for age-based patterns
    age_cols = [col for col in df.columns if 'age' in col.lower()]
    if age_cols:
        for col in age_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                age_stats = df[col].describe()
                patterns[f'Age Range ({col})'] = f"{age_stats['min']:.0f}-{age_stats['max']:.0f} years (avg: {age_stats['mean']:.1f})"
    
    # Look for income patterns
    income_cols = [col for col in df.columns if any(word in col.lower() for word in ['income', 'salary', 'earning'])]
    if income_cols:
        for col in income_cols:
            if pd.api.types.is_numeric_dtype(df[col]):
                income_stats = df[col].describe()
                patterns[f'Income Range ({col})'] = f"₹{income_stats['min']:,.0f}-₹{income_stats['max']:,.0f} (avg: ₹{income_stats['mean']:,.0f})"
    
    # Look for investment preferences
    pref_cols = [col for col in df.columns if any(word in col.lower() for word in ['preference', 'choice', 'option'])]
    if pref_cols:
        for col in pref_cols:
            if df[col].dtype == 'object':
                pref_dist = df[col].value_counts()
                patterns[f'Investment Preferences ({col})'] = pref_dist.head().to_dict()
    
    return patterns

def assess_investment_relevance(analysis_results):
    """Assess how relevant this dataset is for Smart Money AI Investment Engine"""
    
    assessment = {
        'overall_score': 0,
        'file_scores': {},
        'strengths': [],
        'weaknesses': [],
        'use_cases': [],
        'integration_potential': [],
        'detailed_report': ''
    }
    
    total_score = 0
    file_count = 0
    
    for filename, data in analysis_results.items():
        if 'error' in data:
            continue
            
        file_score = 0
        file_strengths = []
        
        columns = [col.lower() for col in data.get('column_names', [])]
        
        # Score based on investment-relevant columns
        scoring_criteria = {
            'risk_profiling': (['risk', 'tolerance', 'appetite'], 30),
            'investment_preferences': (['investment', 'portfolio', 'allocation'], 25),
            'demographics': (['age', 'income', 'experience'], 20),
            'goals_objectives': (['goal', 'objective', 'horizon', 'target'], 15),
            'asset_classes': (['equity', 'bond', 'mutual', 'fund', 'stock'], 15),
            'advisory_data': (['advisor', 'recommendation', 'strategy'], 10)
        }
        
        for criteria_name, (keywords, points) in scoring_criteria.items():
            if any(keyword in ' '.join(columns) for keyword in keywords):
                file_score += points
                file_strengths.append(f"{criteria_name.replace('_', ' ').title()} data present")
        
        # Bonus points for data quality and size
        if data.get('rows', 0) > 500:
            file_score += 10
            file_strengths.append("Substantial dataset size")
        
        if len(data.get('column_names', [])) >= 10:
            file_score += 5
            file_strengths.append("Rich feature set")
        
        # Check for investment patterns
        if 'investment_patterns' in data:
            file_score += 10
            file_strengths.append("Investment patterns identified")
        
        assessment['file_scores'][filename] = {
            'score': file_score,
            'strengths': file_strengths
        }
        
        total_score += file_score
        file_count += 1
    
    # Calculate overall score
    if file_count > 0:
        assessment['overall_score'] = min(100, total_score / file_count)
    
    # Generate assessment categories
    if assessment['overall_score'] >= 70:
        relevance_level = "High"
        assessment['use_cases'] = [
            "Enhance risk profiling accuracy",
            "Improve investment recommendation algorithms", 
            "Build user preference prediction models",
            "Create demographic-based investment strategies",
            "Validate current recommendation logic"
        ]
        assessment['integration_potential'] = [
            "Direct integration into investment recommendation engine",
            "Enhancement of risk assessment algorithms",
            "Training data for preference prediction models",
            "Validation dataset for current investment logic"
        ]
    elif assessment['overall_score'] >= 40:
        relevance_level = "Medium"
        assessment['use_cases'] = [
            "Supplementary data for investment recommendations",
            "Risk profiling validation",
            "Investment preference insights",
            "Demographic correlation analysis"
        ]
        assessment['integration_potential'] = [
            "Selective integration of high-value features",
            "Cross-validation of existing investment models",
            "Enhancement of specific investment components"
        ]
    else:
        relevance_level = "Low"
        assessment['use_cases'] = [
            "Research and analysis only",
            "Insights for algorithm improvement"
        ]
        assessment['integration_potential'] = [
            "Limited utility for production investment system"
        ]
    
    # Generate detailed report
    assessment['detailed_report'] = f"""
📊 INVESTMENT DATASET RELEVANCE ANALYSIS
{"-" * 60}
Overall Score: {assessment['overall_score']:.1f}/100 ({relevance_level} Relevance)

📁 File Analysis:
{chr(10).join(f"   {filename}: {info['score']}/100 - {', '.join(info['strengths'])}" 
              for filename, info in assessment['file_scores'].items())}

💡 Potential Use Cases:
{chr(10).join(f"   • {use_case}" for use_case in assessment['use_cases'])}

🔧 Integration Opportunities:
{chr(10).join(f"   • {integration}" for integration in assessment['integration_potential'])}

🎯 Recommendation: {"HIGHLY RECOMMENDED" if assessment['overall_score'] >= 70 else "MODERATELY USEFUL" if assessment['overall_score'] >= 40 else "LIMITED VALUE"}
"""
    
    return assessment

def generate_investment_recommendations(assessment):
    """Generate specific recommendations for investment engine integration"""
    
    recommendations = []
    
    if assessment['overall_score'] >= 70:
        recommendations.extend([
            "Immediately integrate for investment recommendation enhancement",
            "Use for improving risk profiling accuracy",
            "Train preference prediction models with this data",
            "Create demographic-based investment strategies",
            "Validate and improve current investment algorithms"
        ])
    elif assessment['overall_score'] >= 40:
        recommendations.extend([
            "Selectively extract investment-relevant features",
            "Use for validating current investment logic",
            "Enhance risk assessment components",
            "Improve demographic correlation in recommendations"
        ])
    else:
        recommendations.extend([
            "Use for research and algorithm insights only",
            "Extract patterns to improve synthetic data generation"
        ])
    
    recommendations.extend([
        "Combine with existing personal finance dataset for comprehensive insights",
        "Consider for A/B testing investment recommendation improvements"
    ])
    
    return recommendations

if __name__ == "__main__":
    print("🚀 Starting Investment Survey Dataset Analysis...")
    print("This could significantly enhance our investment recommendation engine!\n")
    
    results = analyze_investment_survey_dataset()
    
    if results:
        print(f"\n✅ Analysis Complete!")
        print("Check the detailed report above and saved JSON file.")
        print("\n🎯 Next: If relevant, integrate with Smart Money AI investment engine!")
    else:
        print(f"\n❌ Analysis Failed!")