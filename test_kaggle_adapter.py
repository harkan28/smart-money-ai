#!/usr/bin/env python3
"""
Test the provided Kaggle dataset loading code
Testing kagglehub functionality with the economics of happiness dataset
"""

# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter

def test_kaggle_dataset_adapter():
    """Test the KaggleDatasetAdapter functionality"""
    
    print("🧪 TESTING KAGGLE DATASET ADAPTER")
    print("=" * 50)
    
    try:
        # Set the path to the file you'd like to load
        file_path = "TEH_World_Happiness_2015_2019_Imputed.csv"  # Specific file path
        
        print("📥 Loading dataset using KaggleDatasetAdapter...")
        
        # Load the latest version
        df = kagglehub.load_dataset(
            KaggleDatasetAdapter.PANDAS,
            "nikbearbrown/the-economics-of-happiness-simple-data-20152019",
            file_path,
            # Provide any additional arguments like 
            # sql_query or pandas_kwargs. See the 
            # documentation for more information:
            # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
        )
        
        print("✅ Dataset loaded successfully!")
        print(f"📊 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        
        print("\n🔍 First 5 records:")
        print(df.head())
        
        print(f"\n📋 Columns: {list(df.columns)}")
        
        print(f"\n📈 Basic statistics:")
        print(df.describe())
        
        return True, df
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        print(f"🔧 Error type: {type(e).__name__}")
        return False, None

def analyze_adapter_results(df):
    """Analyze the results from the adapter"""
    
    if df is not None:
        print(f"\n🎯 ADAPTER ANALYSIS RESULTS")
        print("=" * 40)
        
        print(f"✅ KaggleDatasetAdapter.PANDAS works correctly")
        print(f"📊 Successfully loaded {df.shape[0]:,} records")
        print(f"🏷️ Columns available: {df.shape[1]}")
        
        # Check data types
        print(f"\n📋 Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col}: {dtype}")
        
        # Check for missing data
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n⚠️ Missing Data:")
            for col, missing in missing_data.items():
                if missing > 0:
                    print(f"  {col}: {missing} missing values")
        else:
            print(f"\n✅ No missing data found")

if __name__ == "__main__":
    print("🚀 TESTING KAGGLEHUB DATASET ADAPTER")
    print("=" * 50)
    
    success, df = test_kaggle_dataset_adapter()
    
    if success:
        analyze_adapter_results(df)
        print(f"\n✅ KaggleDatasetAdapter test completed successfully!")
    else:
        print(f"\n❌ KaggleDatasetAdapter test failed!")
        print(f"💡 This might be due to API limitations or dataset access issues")