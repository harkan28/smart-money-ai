#!/usr/bin/env python3
"""
Analyze Ramya's Personal Finance Dataset
Test if this dataset can enhance Smart Money AI further
"""

import kagglehub
import pandas as pd
import os
import json
from pathlib import Path

def download_and_analyze_ramya_dataset():
    """Download and analyze ramyapintchy/personal-finance-data dataset"""
    
    print("🔍 ANALYZING NEW DATASET: ramyapintchy/personal-finance-data")
    print("=" * 70)
    
    try:
        # Download the dataset
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("ramyapintchy/personal-finance-data")
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
        
        # Analyze each CSV file
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
                        'sample_data': df.head(3).to_dict('records'),
                        'data_types': df.dtypes.to_dict(),
                        'missing_values': df.isnull().sum().to_dict(),
                        'unique_values': {col: df[col].nunique() for col in df.columns}
                    }
                    
                    # Basic info
                    print(f"📏 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
                    print(f"📝 Columns: {', '.join(df.columns)}")
                    
                    # Show sample data
                    print(f"\n🔍 Sample Data:")
                    for i, row in df.head(3).iterrows():
                        print(f"   Row {i+1}: {dict(row)}")
                    
                    # Data quality
                    missing_count = df.isnull().sum().sum()
                    print(f"\n📈 Data Quality:")
                    print(f"   • Missing Values: {missing_count}")
                    print(f"   • Complete Records: {df.shape[0] - missing_count}")
                    
                    # Detect financial relevance
                    financial_keywords = [
                        'expense', 'income', 'budget', 'saving', 'investment',
                        'amount', 'cost', 'price', 'salary', 'spending',
                        'category', 'transaction', 'balance', 'account',
                        'financial', 'money', 'cash', 'credit', 'debit'
                    ]
                    
                    relevance_score = 0
                    relevant_columns = []
                    
                    for col in df.columns:
                        col_lower = col.lower()
                        for keyword in financial_keywords:
                            if keyword in col_lower:
                                relevance_score += 10
                                relevant_columns.append(col)
                                break
                    
                    # Check data content for financial terms
                    text_columns = df.select_dtypes(include=['object']).columns
                    for col in text_columns:
                        sample_text = ' '.join(df[col].astype(str).head(100).values).lower()
                        keyword_matches = sum(1 for keyword in financial_keywords if keyword in sample_text)
                        relevance_score += keyword_matches * 2
                    
                    file_analysis['relevance_score'] = min(relevance_score, 100)
                    file_analysis['relevant_columns'] = relevant_columns
                    
                    print(f"\n🎯 Smart Money AI Relevance: {file_analysis['relevance_score']}/100")
                    if relevant_columns:
                        print(f"💰 Financial Columns: {', '.join(relevant_columns)}")
                    
                    analysis_results[os.path.basename(file_path)] = file_analysis
                    
                except Exception as e:
                    print(f"❌ Error analyzing {file_path}: {e}")
                    analysis_results[os.path.basename(file_path)] = {
                        'error': str(e),
                        'relevance_score': 0
                    }
        
        # Overall assessment
        print(f"\n🏆 OVERALL DATASET ASSESSMENT")
        print("=" * 50)
        
        total_records = sum(result.get('shape', [0])[0] for result in analysis_results.values() if 'shape' in result)
        avg_relevance = sum(result.get('relevance_score', 0) for result in analysis_results.values()) / len(analysis_results) if analysis_results else 0
        
        print(f"📊 Total Records: {total_records:,}")
        print(f"📁 Files Analyzed: {len(analysis_results)}")
        print(f"🎯 Average Relevance Score: {avg_relevance:.1f}/100")
        
        # Smart Money AI integration potential
        integration_potential = assess_integration_potential(analysis_results, total_records, avg_relevance)
        print(f"\n🚀 INTEGRATION POTENTIAL: {integration_potential['level']}")
        print(f"💡 {integration_potential['recommendation']}")
        
        if integration_potential['level'] in ['HIGH', 'MEDIUM']:
            print(f"\n✅ NEXT STEPS:")
            for i, step in enumerate(integration_potential['next_steps'], 1):
                print(f"   {i}. {step}")
        
        # Save analysis results
        save_analysis_results(analysis_results, integration_potential)
        
        return analysis_results, integration_potential
        
    except Exception as e:
        print(f"❌ Error downloading or analyzing dataset: {e}")
        return None, None

def assess_integration_potential(analysis_results, total_records, avg_relevance):
    """Assess the potential for integrating this dataset with Smart Money AI"""
    
    integration_assessment = {
        'level': 'LOW',
        'recommendation': '',
        'next_steps': []
    }
    
    # Criteria for integration potential
    if avg_relevance >= 70 and total_records >= 1000:
        integration_assessment['level'] = 'HIGH'
        integration_assessment['recommendation'] = (
            f"Excellent integration potential! {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "This dataset could significantly enhance Smart Money AI's capabilities."
        )
        integration_assessment['next_steps'] = [
            "Perform detailed data analysis and mapping",
            "Design integration architecture",
            "Create enhanced features using this data",
            "Implement comprehensive testing",
            "Deploy integrated system"
        ]
    
    elif avg_relevance >= 50 and total_records >= 500:
        integration_assessment['level'] = 'MEDIUM'
        integration_assessment['recommendation'] = (
            f"Good integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "Could provide valuable enhancement to specific Smart Money AI features."
        )
        integration_assessment['next_steps'] = [
            "Analyze specific use cases for integration",
            "Identify most valuable data columns",
            "Design targeted feature enhancements",
            "Test integration with existing system"
        ]
    
    elif avg_relevance >= 30 or total_records >= 100:
        integration_assessment['level'] = 'LOW-MEDIUM'
        integration_assessment['recommendation'] = (
            f"Limited integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "May be useful for specific features but not a priority integration."
        )
        integration_assessment['next_steps'] = [
            "Focus on highest-relevance data subsets",
            "Consider data as supplementary enhancement only"
        ]
    
    else:
        integration_assessment['level'] = 'LOW'
        integration_assessment['recommendation'] = (
            f"Low integration potential. {total_records:,} records with {avg_relevance:.1f}/100 relevance. "
            "Not recommended for Smart Money AI integration."
        )
        integration_assessment['next_steps'] = [
            "Look for alternative datasets with higher financial relevance"
        ]
    
    return integration_assessment

def save_analysis_results(analysis_results, integration_potential):
    """Save analysis results to file"""
    
    results = {
        'dataset_name': 'ramyapintchy/personal-finance-data',
        'analysis_date': '2025-10-15',
        'file_analyses': analysis_results,
        'integration_assessment': integration_potential,
        'comparison_with_existing': {
            'current_dataset_1': {
                'name': 'shriyashjagtap/indian-personal-finance-and-spending-habits',
                'records': 20000,
                'relevance': 60,
                'status': 'integrated'
            },
            'current_dataset_2': {
                'name': 'sudarsan27/investment-survey-dataset', 
                'records': 100,
                'relevance': 75,
                'status': 'integrated'
            }
        }
    }
    
    # Save to data directory
    os.makedirs('data', exist_ok=True)
    
    with open('data/ramya_dataset_analysis.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Analysis saved to: data/ramya_dataset_analysis.json")

if __name__ == "__main__":
    print("🎯 Smart Money AI Dataset Analyzer")
    print("Testing: ramyapintchy/personal-finance-data")
    print()
    
    analysis, integration = download_and_analyze_ramya_dataset()
    
    if analysis:
        print(f"\n🎉 Analysis complete! Check data/ramya_dataset_analysis.json for detailed results")
    else:
        print(f"\n❌ Analysis failed. Please check the dataset name and try again.")