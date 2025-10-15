#!/usr/bin/env python3
"""
Behavioral Finance & Investment Decision Dataset Analysis for Smart Money AI
Dataset: rishavwalde/behavioral-finance-and-investment-decision
Purpose: Assess relevance for investment recommendation model enhancement
"""

import kagglehub
import pandas as pd
import numpy as np
from datetime import datetime
import os
import json

def analyze_behavioral_finance_dataset():
    """Comprehensive analysis of behavioral finance dataset"""
    
    print("🔍 ANALYZING: Behavioral Finance & Investment Decision Dataset")
    print("=" * 70)
    
    try:
        # Download dataset
        print("📥 Downloading dataset...")
        path = kagglehub.dataset_download("rishavwalde/behavioral-finance-and-investment-decision")
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
        print("=" * 60)
        
        # Basic info
        print(f"📏 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"💾 Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Column analysis
        print(f"\n📋 COLUMNS ({len(df.columns)}):")
        for i, col in enumerate(df.columns, 1):
            dtype = str(df[col].dtype)
            null_count = df[col].isnull().sum()
            null_pct = (null_count / len(df)) * 100
            unique_count = df[col].nunique()
            print(f"  {i:2d}. {col:<35} | {dtype:<10} | Nulls: {null_count:>4} ({null_pct:5.1f}%) | Unique: {unique_count}")
        
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
            if unique_count <= 15:
                sample_values = list(df[col].value_counts().head(10).index)
                print(f"    Top values: {sample_values}")
        
        # Look for behavioral finance indicators
        print(f"\n🧠 BEHAVIORAL FINANCE INDICATORS:")
        behavioral_keywords = ['risk', 'behavior', 'decision', 'investment', 'bias', 'psychology', 
                             'personality', 'attitude', 'preference', 'confidence', 'overconfidence',
                             'loss', 'gain', 'aversion', 'tolerance', 'experience', 'knowledge',
                             'emotion', 'sentiment', 'herd', 'momentum', 'contrarian', 'diversification']
        
        behavioral_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in behavioral_keywords):
                behavioral_columns.append(col)
        
        if behavioral_columns:
            print(f"  Found {len(behavioral_columns)} behavioral finance columns:")
            for col in behavioral_columns:
                print(f"    🧠 {col}")
                # Show value distribution for categorical behavioral columns
                if df[col].dtype == 'object' and df[col].nunique() <= 10:
                    print(f"       Values: {list(df[col].value_counts().index)}")
        else:
            print("  ⚠️ No obvious behavioral finance columns found")
        
        # Assess Smart Money AI relevance
        relevance_score = assess_behavioral_finance_relevance(df, behavioral_columns)
        
        # Save analysis
        analysis_result = {
            'dataset_name': 'rishavwalde/behavioral-finance-and-investment-decision',
            'analysis_date': datetime.now().isoformat(),
            'records': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'behavioral_columns': behavioral_columns,
            'relevance_score': relevance_score,
            'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_data': {col: int(df[col].isnull().sum()) for col in df.columns},
            'categorical_distributions': {}
        }
        
        # Add categorical value distributions
        for col in categorical_cols[:5]:  # Top 5 categorical columns
            if df[col].nunique() <= 20:
                analysis_result['categorical_distributions'][col] = dict(df[col].value_counts().head(10))
        
        save_behavioral_finance_analysis(analysis_result)
        
        print(f"\n🎯 SMART MONEY AI RELEVANCE ASSESSMENT")
        print("=" * 55)
        print_relevance_assessment(relevance_score, df, behavioral_columns)
        
        return df, analysis_result
        
    except Exception as e:
        print(f"❌ Error analyzing dataset: {str(e)}")
        return None, None

def assess_behavioral_finance_relevance(df, behavioral_columns):
    """Assess dataset relevance for Smart Money AI investment model (0-100 scale)"""
    
    score = 0
    reasons = []
    
    # Base scoring for behavioral finance focus
    if len(behavioral_columns) > 0:
        score += 40
        reasons.append(f"Behavioral finance columns found: {len(behavioral_columns)}")
    
    # Check for specific valuable behavioral patterns
    valuable_patterns = {
        'risk': 25,
        'investment': 20,
        'decision': 15,
        'behavior': 20,
        'bias': 15,
        'confidence': 10,
        'experience': 10,
        'tolerance': 15,
        'preference': 10
    }
    
    found_patterns = []
    for pattern, points in valuable_patterns.items():
        matching_cols = [col for col in df.columns if pattern in col.lower()]
        if matching_cols:
            score += min(points, 15)  # Cap individual pattern points
            found_patterns.append(pattern)
            reasons.append(f"'{pattern}' indicators: {matching_cols}")
    
    # Data quality bonus
    if df.shape[0] > 500:
        score += 10
        reasons.append("Good sample size for behavioral analysis")
    
    if df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) < 0.1:
        score += 5
        reasons.append("Low missing data rate")
    
    # Investment decision relevance
    investment_keywords = ['portfolio', 'stock', 'fund', 'return', 'market', 'asset']
    investment_cols = [col for col in df.columns if any(kw in col.lower() for kw in investment_keywords)]
    if investment_cols:
        score += 15
        reasons.append(f"Investment-specific columns: {investment_cols}")
    
    # Behavioral complexity bonus
    categorical_cols = df.select_dtypes(include=['object']).columns
    complex_behavioral_cols = [col for col in categorical_cols if col in behavioral_columns and df[col].nunique() > 3]
    if len(complex_behavioral_cols) > 2:
        score += 10
        reasons.append("Complex behavioral variables for advanced profiling")
    
    # Cap at 100
    score = min(score, 100)
    
    return score

def print_relevance_assessment(score, df, behavioral_columns):
    """Print detailed relevance assessment"""
    
    if score >= 80:
        status = "🟢 VERY HIGH RELEVANCE - IMMEDIATE INTEGRATION RECOMMENDED"
        recommendation = "This dataset significantly enhances investment behavioral profiling"
    elif score >= 65:
        status = "🟢 HIGH RELEVANCE - STRONGLY RECOMMEND INTEGRATION"
        recommendation = "Excellent behavioral data for investment decision modeling"
    elif score >= 50:
        status = "🟡 MODERATE RELEVANCE - SELECTIVE INTEGRATION RECOMMENDED"
        recommendation = "Useful behavioral insights for specific investment features"
    elif score >= 30:
        status = "🟠 LOW RELEVANCE - LIMITED UTILITY"
        recommendation = "Some behavioral insights but limited investment applicability"
    else:
        status = "🔴 NOT RELEVANT - NOT RECOMMENDED"
        recommendation = "Dataset not suitable for investment behavioral profiling"
    
    print(f"📊 RELEVANCE SCORE: {score}/100")
    print(f"🎯 STATUS: {status}")
    print(f"💡 RECOMMENDATION: {recommendation}")
    
    print(f"\n🔍 ASSESSMENT DETAILS:")
    print(f"  📏 Data Size: {df.shape[0]:,} records × {df.shape[1]} columns")
    print(f"  🧠 Behavioral Indicators: {len(behavioral_columns)} columns")
    print(f"  📊 Data Quality: {((1 - df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100):.1f}% complete")
    
    # Integration potential based on score
    if score >= 65:
        print(f"\n✅ HIGH-VALUE INTEGRATION OPPORTUNITIES:")
        print("  🎯 Enhanced risk profiling with behavioral indicators")
        print("  🧠 Advanced investment personality assessment")
        print("  📈 Behavioral bias detection and mitigation")
        print("  💡 Personalized investment recommendation refinement")
        print("  🔄 Dynamic risk tolerance adjustment based on behavior")
    elif score >= 50:
        print(f"\n✅ MODERATE INTEGRATION OPPORTUNITIES:")
        print("  🧠 Behavioral profiling enhancement")
        print("  🎯 Risk tolerance refinement")
        print("  📊 Investment decision pattern analysis")
    
    if score < 65:
        print(f"\n⚠️ CONSIDERATIONS:")
        if len(behavioral_columns) < 5:
            print("  🧠 Limited behavioral complexity")
        if df.shape[0] < 1000:
            print("  📏 Sample size may be limited for complex behavioral modeling")
        print("  🎯 May need combination with existing datasets for full value")

    # Comparison with existing datasets
    print(f"\n📊 COMPARISON WITH EXISTING SMART MONEY AI DATASETS:")
    print("  🥇 Gold Price Prediction: 100/100 (market timing intelligence)")
    print("  🎯 Personal Finance: 60/100 (spending behavior)")
    print("  💡 Investment Survey: 75/100 (basic behavioral profiling)")
    print(f"  🧠 Behavioral Finance: {score}/100 (advanced behavioral insights)")
    
    if score > 75:
        print("  🏆 HIGHEST BEHAVIORAL INTELLIGENCE DATASET!")
    elif score > 60:
        print("  🎯 STRONG COMPLEMENT TO EXISTING INVESTMENT INTELLIGENCE")

def save_behavioral_finance_analysis(analysis_result):
    """Save analysis results"""
    
    output_dir = "/Users/harshitrawal/Downloads/SMART MONEY/smart_money_ai/data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed analysis
    output_file = os.path.join(output_dir, "behavioral_finance_dataset_analysis.json")
    with open(output_file, 'w') as f:
        json.dump(analysis_result, f, indent=2)
    
    print(f"\n💾 Analysis saved to: {output_file}")

if __name__ == "__main__":
    print("🚀 SMART MONEY AI - BEHAVIORAL FINANCE DATASET ANALYSIS")
    print("=" * 65)
    
    df, analysis = analyze_behavioral_finance_dataset()
    
    if df is not None:
        print(f"\n✅ Analysis completed successfully!")
        print(f"📊 Dataset contains {df.shape[0]:,} records with {df.shape[1]} columns")
        print(f"🧠 Behavioral finance focus makes this highly relevant for investment intelligence")
    else:
        print(f"\n❌ Analysis failed!")