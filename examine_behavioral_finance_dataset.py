#!/usr/bin/env python3
"""
Quick examination of the behavioral finance dataset structure
"""

import kagglehub
import pandas as pd

def examine_behavioral_finance_dataset():
    """Quick examination of the dataset"""
    
    print("🔍 EXAMINING: Behavioral Finance Dataset")
    print("=" * 50)
    
    try:
        # Download dataset
        path = kagglehub.dataset_download("rishavwalde/behavioral-finance-and-investment-decision")
        
        # Load the dataset
        df = pd.read_csv(path + "/Behavioral Finance and Investment Decision.csv")
        
        print(f"📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"\n📋 Columns: {list(df.columns)}")
        
        print(f"\n🔍 Full dataset preview:")
        print(df.head(10))
        
        print(f"\n📈 Statistical summary:")
        print(df.describe(include='all'))
        
        print(f"\n🏷️ Unique values per column:")
        for col in df.columns:
            unique_count = df[col].nunique()
            print(f"  {col}: {unique_count} unique values")
            if df[col].dtype == 'object' and unique_count <= 10:
                print(f"    Values: {list(df[col].unique())}")
        
        print(f"\n📊 Data types:")
        print(df.dtypes)
        
        print(f"\n❓ This appears to be basic demographic data only.")
        print(f"   Looking for additional files or hidden columns...")
        
        # Check all columns for any behavioral indicators
        behavioral_keywords = ['risk', 'behavior', 'decision', 'investment', 'bias', 'personality']
        found_behavioral = False
        
        for col in df.columns:
            col_lower = col.lower()
            for keyword in behavioral_keywords:
                if keyword in col_lower:
                    print(f"   🧠 Found behavioral indicator: {col}")
                    found_behavioral = True
        
        if not found_behavioral:
            print(f"   ⚠️ No behavioral finance indicators found in column names")
            print(f"   📋 This dataset contains only: {', '.join(df.columns)}")
        
        return df
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    df = examine_behavioral_finance_dataset()