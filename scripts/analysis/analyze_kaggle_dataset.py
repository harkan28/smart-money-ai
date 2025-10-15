#!/usr/bin/env python3
"""
Kaggle Finance India Dataset Analysis
Analyze the finance-india dataset to see if it's useful for Smart Money AI
"""

import kagglehub
import pandas as pd
import os
import json
from pathlib import Path

def download_and_analyze_dataset():
    """Download and analyze the Kaggle finance-india dataset"""
    
    print("🔍 Downloading Kaggle Finance India Dataset...")
    print("=" * 60)
    
    try:
        # Download latest version
        path = kagglehub.dataset_download("rajanand/finance-india")
        print(f"✅ Dataset downloaded to: {path}")
        
        # List all files in the dataset
        dataset_path = Path(path)
        print(f"\n📁 Dataset Structure:")
        print("-" * 30)
        
        all_files = []
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(dataset_path)
                file_size = file_path.stat().st_size / (1024 * 1024)  # MB
                all_files.append({
                    'file': str(relative_path),
                    'size_mb': round(file_size, 2),
                    'extension': file_path.suffix
                })
                print(f"   📄 {relative_path} ({file_size:.2f} MB)")
        
        # Analyze each CSV/Excel file
        print(f"\n🔍 Analyzing Dataset Contents:")
        print("=" * 60)
        
        analysis_results = {}
        
        for file_info in all_files:
            file_path = dataset_path / file_info['file']
            
            if file_info['extension'].lower() in ['.csv', '.xlsx', '.xls']:
                print(f"\n📊 Analyzing: {file_info['file']}")
                print("-" * 40)
                
                try:
                    # Read the file
                    if file_info['extension'].lower() == '.csv':
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_excel(file_path)
                    
                    # Basic info
                    rows, cols = df.shape
                    print(f"   📏 Dimensions: {rows} rows × {cols} columns")
                    print(f"   📋 Columns: {list(df.columns)}")
                    
                    # Sample data
                    print(f"   🔍 Sample Data (first 3 rows):")
                    print(df.head(3).to_string(index=False))
                    
                    # Data types
                    print(f"   📝 Data Types:")
                    for col, dtype in df.dtypes.items():
                        print(f"      {col}: {dtype}")
                    
                    # Store analysis
                    analysis_results[file_info['file']] = {
                        'rows': rows,
                        'columns': cols,
                        'column_names': list(df.columns),
                        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
                        'sample_data': df.head(3).to_dict('records'),
                        'null_counts': df.isnull().sum().to_dict(),
                        'file_size_mb': file_info['size_mb']
                    }
                    
                    # Check for relevant financial data
                    financial_keywords = [
                        'amount', 'price', 'cost', 'expense', 'income', 'salary',
                        'transaction', 'payment', 'balance', 'debit', 'credit',
                        'bank', 'account', 'category', 'merchant', 'date', 'time',
                        'sms', 'message', 'upi', 'card', 'atm', 'neft', 'rtgs'
                    ]
                    
                    relevant_columns = []
                    for col in df.columns:
                        col_lower = col.lower()
                        if any(keyword in col_lower for keyword in financial_keywords):
                            relevant_columns.append(col)
                    
                    if relevant_columns:
                        print(f"   💰 Relevant Financial Columns: {relevant_columns}")
                        analysis_results[file_info['file']]['relevant_columns'] = relevant_columns
                    
                except Exception as e:
                    print(f"   ❌ Error reading file: {str(e)}")
                    analysis_results[file_info['file']] = {'error': str(e)}
        
        # Generate assessment for Smart Money AI
        print(f"\n🎯 Smart Money AI Relevance Assessment:")
        print("=" * 60)
        
        assessment = assess_dataset_relevance(analysis_results)
        print(assessment['summary'])
        
        # Save analysis results
        results_file = Path("data/kaggle_dataset_analysis.json")
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump({
                'dataset_path': str(path),
                'analysis_results': analysis_results,
                'assessment': assessment,
                'download_timestamp': pd.Timestamp.now().isoformat()
            }, f, indent=2, default=str)
        
        print(f"\n💾 Analysis saved to: {results_file}")
        
        return path, analysis_results, assessment
        
    except Exception as e:
        print(f"❌ Error downloading dataset: {str(e)}")
        return None, None, None

def assess_dataset_relevance(analysis_results):
    """Assess how relevant this dataset is for Smart Money AI"""
    
    assessment = {
        'overall_score': 0,
        'relevance_areas': [],
        'potential_uses': [],
        'limitations': [],
        'recommendations': [],
        'summary': ''
    }
    
    total_files = len(analysis_results)
    relevant_files = 0
    
    for filename, data in analysis_results.items():
        if 'error' in data:
            continue
            
        file_score = 0
        file_uses = []
        
        # Check for SMS/transaction data
        columns = [col.lower() for col in data.get('column_names', [])]
        
        if any('sms' in col or 'message' in col for col in columns):
            file_score += 30
            file_uses.append("SMS parsing training data")
            assessment['relevance_areas'].append(f"{filename}: SMS data")
        
        if any('amount' in col or 'price' in col or 'cost' in col for col in columns):
            file_score += 25
            file_uses.append("Transaction amount analysis")
            assessment['relevance_areas'].append(f"{filename}: Transaction amounts")
        
        if any('category' in col or 'type' in col for col in columns):
            file_score += 20
            file_uses.append("Expense categorization training")
            assessment['relevance_areas'].append(f"{filename}: Category data")
        
        if any('bank' in col or 'account' in col for col in columns):
            file_score += 15
            file_uses.append("Banking pattern analysis")
            assessment['relevance_areas'].append(f"{filename}: Banking data")
        
        if any('date' in col or 'time' in col for col in columns):
            file_score += 10
            file_uses.append("Temporal analysis")
        
        if file_score > 20:
            relevant_files += 1
            assessment['potential_uses'].extend(file_uses)
        
        assessment['overall_score'] += file_score
    
    # Normalize score
    if total_files > 0:
        assessment['overall_score'] = min(100, assessment['overall_score'] / total_files)
    
    # Generate recommendations
    if assessment['overall_score'] >= 70:
        assessment['recommendations'].append("Highly recommended for Smart Money AI enhancement")
        assessment['recommendations'].append("Integrate as primary training dataset")
    elif assessment['overall_score'] >= 40:
        assessment['recommendations'].append("Moderately useful - extract relevant portions")
        assessment['recommendations'].append("Use as supplementary training data")
    else:
        assessment['recommendations'].append("Limited relevance - consider alternative datasets")
        assessment['recommendations'].append("May be useful for validation purposes only")
    
    # Generate summary
    relevance_level = "High" if assessment['overall_score'] >= 70 else "Medium" if assessment['overall_score'] >= 40 else "Low"
    
    assessment['summary'] = f"""
📊 Dataset Relevance Score: {assessment['overall_score']:.1f}/100 ({relevance_level})

🎯 Key Findings:
   • {total_files} files analyzed
   • {relevant_files} files with financial relevance
   • {len(assessment['relevance_areas'])} relevant data areas identified

💡 Potential Applications:
{chr(10).join(f'   • {use}' for use in set(assessment['potential_uses']))}

🔧 Recommendations:
{chr(10).join(f'   • {rec}' for rec in assessment['recommendations'])}
"""
    
    return assessment

if __name__ == "__main__":
    print("🚀 Starting Kaggle Finance India Dataset Analysis...")
    path, results, assessment = download_and_analyze_dataset()
    
    if path:
        print(f"\n✅ Analysis Complete!")
        print(f"Dataset Location: {path}")
    else:
        print(f"\n❌ Analysis Failed!")