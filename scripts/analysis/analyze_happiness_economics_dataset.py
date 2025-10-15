#!/usr/bin/env python3
"""
Economics of Happiness Dataset Analysis for Smart Money AI
Dataset: nikbearbrown/the-economics-of-happiness-simple-data-20152019
Purpose: Assess relevance for personal finance and investment intelligence
"""

import kagglehub
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

def analyze_happiness_economics_dataset():
    """Comprehensive analysis of economics of happiness dataset"""
    
    print("🔍 ANALYZING: Economics of Happiness Dataset")
    print("=" * 60)
    
    try:
        # Download dataset
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("nikbearbrown/the-economics-of-happiness-simple-data-20152019")
        print(f"✅ Dataset downloaded to: {path}")
        
        # List available files
        print("\n📁 Available files:")
        for root, dirs, files in os.walk(path):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"  📄 {file} ({file_size:,} bytes)")
        
        # Load the main dataset
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            print("❌ No CSV files found!")
            return
            
        main_file = csv_files[0]  # Take first CSV file
        df = pd.read_csv(os.path.join(path, main_file))
        
        print(f"\n📊 DATASET ANALYSIS: {main_file}")
        print("=" * 50)
        
        # Basic info
        print(f"📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Column analysis
        print(f"\n📋 COLUMNS ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            print(f"  {i:2d}. {col:<30} | {dtype:<10} | Nulls: {null_count:>4} ({null_pct:5.1f}%)")
        
        # Data preview
        print(f"\n🔍 FIRST 5 RECORDS:")
        print(df.head())
        
        # Value ranges for numeric columns
        print(f"\n📈 NUMERIC COLUMNS SUMMARY:")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(df[numeric_cols].describe())
        else:
            print("  No numeric columns found")
        
        # Categorical analysis
        print(f"\n🏷️ CATEGORICAL COLUMNS:")
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols[:10]:  # Show first 10
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if unique_count <= 10:
                print(f"    Values: {list(df[col].unique())}")
        
        # Look for financial/economic indicators
        print(f"\n💰 FINANCIAL/ECONOMIC INDICATORS:")
        financial_keywords = ['gdp', 'income', 'wealth', 'money', 'economic', 'finance', 
                            'salary', 'wage', 'cost', 'price', 'inflation', 'unemployment',
                            'poverty', 'prosperity', 'spending', 'budget', 'investment']
        
        financial_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in financial_keywords):
                financial_columns.append(col)
        
        if financial_columns:
            print(f"  Found {len(financial_columns)} potentially relevant columns:")
            for col in financial_columns:
                print(f"    📊 {col}")
        else:
            print("  ⚠️ No obvious financial/economic columns found")
        
        # Assess Smart Money AI relevance
        relevance_score = assess_happiness_economics_relevance(df, financial_columns)
        
        # Save analysis
        analysis_result = {
            'dataset_name': 'nikbearbrown/the-economics-of-happiness-simple-data-20152019',
            'analysis_date': datetime.now().isoformat(),
            'records': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'financial_columns': financial_columns,
            'relevance_score': relevance_score,
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_data': {col: int(df[col].isnull().sum()) for col in df.columns}
        }
        
        save_happiness_economics_analysis(analysis_result)
        
        print(f"\n🎯 SMART MONEY AI RELEVANCE ASSESSMENT")
        print("=" * 50)
        print_relevance_assessment(relevance_score, df, financial_columns)
        
        return df, analysis_result
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {str(e)}")
        return None, None

def assess_happiness_economics_relevance(df, financial_columns):
    """Assess dataset relevance for Smart Money AI (0-100 scale)"""
    
    score = 0
    reasons = []
    
    # Base scoring criteria
    if len(financial_columns) > 0:
        score += 30
        reasons.append(f"Financial columns found: {len(financial_columns)}")
    
    # Check for specific valuable columns
    valuable_patterns = {
        'income': 25,
        'gdp': 15,
        'economic': 10,
        'wealth': 20,
        'unemployment': 15,
        'inflation': 20,
        'spending': 25,
        'cost': 15
    }
    
    for pattern, points in valuable_patterns.items():
        matching_cols = [col for col in df.columns if pattern in col.lower()]
        if matching_cols:
            score += points
            reasons.append(f"'{pattern}' indicators: {matching_cols}")
            break  # Don't double-count
    
    # Check data quality
    if df.shape[0] > 1000:
        score += 5
        reasons.append("Good sample size")
    
    if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) < 0.1:
        score += 5
        reasons.append("Low missing data")
    
    # Cap at 100
    score = min(score, 100)
    
    return score

def print_relevance_assessment(score, df, financial_columns):
    """Print detailed relevance assessment"""
    
    if score >= 70:
        status = "🟢 HIGH RELEVANCE - STRONGLY RECOMMEND INTEGRATION"
        recommendation = "This dataset should be integrated into Smart Money AI"
    elif score >= 50:
        status = "🟡 MODERATE RELEVANCE - SELECTIVE USE RECOMMENDED"
        recommendation = "Consider using specific components of this dataset"
    elif score >= 30:
        status = "🟠 LOW RELEVANCE - LIMITED UTILITY"
        recommendation = "Limited value for Smart Money AI core functionality"
    else:
        status = "🔴 NOT RELEVANT - NOT RECOMMENDED"
        recommendation = "This dataset is not suitable for Smart Money AI"
    
    print(f"📊 RELEVANCE SCORE: {score}/100")
    print(f"🎯 STATUS: {status}")
    print(f"💡 RECOMMENDATION: {recommendation}")
    
    print(f"\n🔍 ASSESSMENT DETAILS:")
    print(f"  📏 Data Size: {df.shape[0]:,} records × {df.shape[1]} columns")
    print(f"  💰 Financial Indicators: {len(financial_columns)} columns")
    print(f"  📊 Data Quality: {((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}% complete")
    
    # Integration potential
    if score >= 50:
        print(f"\n✅ INTEGRATION OPPORTUNITIES:")
        if any('income' in col.lower() for col in financial_columns):
            print("  💰 Income analysis and benchmarking")
        if any('gdp' in col.lower() for col in financial_columns):
            print("  📈 Economic context for investment decisions")
        if any('unemployment' in col.lower() for col in financial_columns):
            print("  🏢 Employment stability indicators")
    
    if score < 50:
        print(f"\n⚠️ LIMITATIONS:")
        if len(financial_columns) == 0:
            print("  💰 No clear financial indicators")
        if df.shape[0] < 1000:
            print("  📏 Limited sample size")
        print("  🎯 May not align with personal finance focus")

def save_happiness_economics_analysis(analysis_result):
    """Save analysis results"""
    
    output_dir = "/Users/harshitrawal/Downloads/SMART MONEY/smart_money_ai/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed analysis
    output_file = os.path.join(output_dir, "happiness_economics_dataset_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")

if __name__ == "__main__":
    print("🚀 SMART MONEY AI - HAPPINESS ECONOMICS DATASET ANALYSIS")
    print("=" * 60)
    
    df, analysis = analyze_happiness_economics_dataset()
    
    if df is not None:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Dataset contains {df.shape[0]:,} records with {df.shape[1]} columns")
    else:
        print(f"\n❌ Analysis failed!")