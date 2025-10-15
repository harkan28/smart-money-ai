#!/usr/bin/env python3
"""
Analyze Indian Personal Finance and Spending Habits Dataset
Test the dataset: shriyashjagtap/indian-personal-finance-and-spending-habits
"""

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
import json
import os
from datetime import datetime

def analyze_personal_finance_dataset():
    """Analyze the Indian personal finance dataset"""
    
    print("🔍 Analyzing Indian Personal Finance and Spending Habits Dataset")
    print("=" * 70)
    print("Dataset: shriyashjagtap/indian-personal-finance-and-spending-habits")
    print("Expected: Personal finance data relevant to Smart Money AI")
    print()
    
    try:
        # First, let's download and see what files are available
        print("📥 Downloading dataset to explore structure...")
        path = kagglehub.dataset_download("shriyashjagtap/indian-personal-finance-and-spending-habits")
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
        
        # Try to load each CSV file using the new API
        print(f"\n🔍 Analyzing Dataset Files:")
        print("=" * 50)
        
        analysis_results = {}
        
        for file_info in dataset_files:
            if file_info['extension'].lower() in ['.csv', '.xlsx']:
                filename = file_info['filename']
                print(f"\n📊 Loading: {filename}")
                print("-" * 30)
                
                try:
                    # Load using the new KaggleDatasetAdapter
                    df = kagglehub.load_dataset(
                        KaggleDatasetAdapter.PANDAS,
                        "shriyashjagtap/indian-personal-finance-and-spending-habits",
                        filename
                    )
                    
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
                        'file_size_mb': file_info['size_mb'],
                        'basic_stats': {}
                    }
                    
                    # Statistical analysis for numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns
                    if len(numeric_cols) > 0:
                        print(f"   📈 Numeric Column Statistics:")
                        for col in numeric_cols:
                            stats = df[col].describe()
                            print(f"      {col}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}, Range=[{stats['min']:.2f}, {stats['max']:.2f}]")
                            analysis_results[filename]['basic_stats'][col] = {
                                'mean': float(stats['mean']),
                                'std': float(stats['std']),
                                'min': float(stats['min']),
                                'max': float(stats['max']),
                                'count': int(stats['count'])
                            }
                    
                    # Check for relevant financial keywords
                    financial_keywords = [
                        'amount', 'price', 'cost', 'expense', 'income', 'salary', 'spending',
                        'transaction', 'payment', 'balance', 'debit', 'credit', 'budget',
                        'bank', 'account', 'category', 'merchant', 'date', 'time',
                        'sms', 'message', 'upi', 'card', 'investment', 'savings'
                    ]
                    
                    relevant_columns = []
                    for col in df.columns:
                        col_lower = col.lower()
                        for keyword in financial_keywords:
                            if keyword in col_lower:
                                relevant_columns.append((col, keyword))
                                break
                    
                    if relevant_columns:
                        print(f"   💰 Relevant Financial Columns:")
                        for col, keyword in relevant_columns:
                            print(f"      {col} (matches: {keyword})")
                        analysis_results[filename]['relevant_columns'] = relevant_columns
                    
                    # Check for unique values in categorical columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        print(f"   🏷️  Categorical Data:")
                        for col in categorical_cols[:3]:  # Show first 3 categorical columns
                            unique_count = df[col].nunique()
                            if unique_count <= 20:  # Show values if not too many
                                unique_vals = df[col].unique()[:10]  # First 10 unique values
                                print(f"      {col}: {unique_count} unique values {list(unique_vals)}")
                            else:
                                print(f"      {col}: {unique_count} unique values")
                    
                except Exception as e:
                    print(f"   ❌ Error loading file: {str(e)}")
                    analysis_results[filename] = {'error': str(e)}
        
        # Assess relevance for Smart Money AI
        print(f"\n🎯 Smart Money AI Relevance Assessment:")
        print("=" * 50)
        
        assessment = assess_smart_money_relevance(analysis_results)
        print(assessment['detailed_report'])
        
        # Save complete analysis
        results = {
            'dataset_info': {
                'name': 'shriyashjagtap/indian-personal-finance-and-spending-habits',
                'download_path': path,
                'analysis_date': datetime.now().isoformat(),
                'total_files': len(dataset_files),
                'file_list': dataset_files
            },
            'analysis_results': analysis_results,
            'relevance_assessment': assessment,
            'recommendations': generate_recommendations(assessment)
        }
        
        # Save to file
        output_file = "data/personal_finance_dataset_analysis.json"
        os.makedirs("data", exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n💾 Complete analysis saved to: {output_file}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {str(e)}")
        return None

def assess_smart_money_relevance(analysis_results):
    """Assess how relevant this dataset is for Smart Money AI"""
    
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
        
        # Score based on relevant columns
        scoring_criteria = {
            'spending/expense': (['spending', 'expense', 'cost', 'amount'], 25),
            'income': (['income', 'salary', 'earnings'], 20),
            'categories': (['category', 'type', 'classification'], 20),
            'transaction_data': (['transaction', 'payment', 'purchase'], 15),
            'banking': (['bank', 'account', 'card'], 15),
            'budgeting': (['budget', 'savings', 'plan'], 10),
            'temporal': (['date', 'time', 'month', 'year'], 5)
        }
        
        for criteria_name, (keywords, points) in scoring_criteria.items():
            if any(keyword in ' '.join(columns) for keyword in keywords):
                file_score += points
                file_strengths.append(f"{criteria_name} data present")
        
        # Bonus points for data quality
        if data.get('rows', 0) > 1000:
            file_score += 10
            file_strengths.append("Substantial dataset size")
        
        if len(data.get('column_names', [])) >= 5:
            file_score += 5
            file_strengths.append("Multiple data dimensions")
        
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
            "Primary training data for expense categorization",
            "Spending pattern analysis and insights",
            "Budget recommendation algorithm training",
            "User behavior modeling for personalization"
        ]
        assessment['integration_potential'] = [
            "Direct integration into ML categorization system",
            "Enhancement of smart budgeting algorithms", 
            "Training data for investment recommendations",
            "Validation dataset for current models"
        ]
    elif assessment['overall_score'] >= 40:
        relevance_level = "Medium"
        assessment['use_cases'] = [
            "Supplementary training data",
            "Model validation and testing",
            "Spending pattern validation",
            "Feature engineering insights"
        ]
        assessment['integration_potential'] = [
            "Selective integration of high-quality features",
            "Cross-validation of existing models",
            "Enhancement of specific components"
        ]
    else:
        relevance_level = "Low"
        assessment['use_cases'] = [
            "Research and analysis only",
            "Inspiration for synthetic data generation"
        ]
        assessment['integration_potential'] = [
            "Limited utility for production system"
        ]
    
    # Generate detailed report
    assessment['detailed_report'] = f"""
📊 DATASET RELEVANCE ANALYSIS
{"-" * 50}
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

def generate_recommendations(assessment):
    """Generate specific recommendations based on assessment"""
    
    recommendations = []
    
    if assessment['overall_score'] >= 70:
        recommendations.extend([
            "Immediately integrate this dataset into Smart Money AI",
            "Use for retraining ML categorization models",
            "Enhance budgeting algorithms with this spending data",
            "Create comparative analysis with current system performance"
        ])
    elif assessment['overall_score'] >= 40:
        recommendations.extend([
            "Selectively extract high-value features",
            "Use for model validation and testing",
            "Combine with existing synthetic data for training",
            "Focus on specific components that show highest relevance"
        ])
    else:
        recommendations.extend([
            "Consider for research purposes only",
            "Extract insights for improving current system",
            "Use patterns to enhance synthetic data generation"
        ])
    
    recommendations.append("Continue monitoring for newer, more relevant datasets")
    
    return recommendations

if __name__ == "__main__":
    print("🚀 Starting Personal Finance Dataset Analysis...")
    print("This could be exactly what Smart Money AI needs!\n")
    
    results = analyze_personal_finance_dataset()
    
    if results:
        print(f"\n✅ Analysis Complete!")
        print("Check the detailed report above and saved JSON file.")
    else:
        print(f"\n❌ Analysis Failed!")