import kagglehub
import pandas as pd

# Download and examine the gold dataset
path = kagglehub.dataset_download("sid321axn/gold-price-prediction-dataset")
df = pd.read_csv(f"{path}/FINAL_USO.csv")

print("📊 GOLD PRICE DATASET DETAILED ANALYSIS:")
print(f"Shape: {df.shape}")
print(f"All Columns ({len(df.columns)}): {list(df.columns)}")

print(f"\n🔍 SAMPLE DATA (first 3 rows):")
print(df.head(3))

print(f"\n📈 KEY PRICE COLUMNS ANALYSIS:")
price_cols = [col for col in df.columns if "price" in col.lower() or col in ["Open", "High", "Low", "Close"]]
print(f"Price-related columns: {price_cols[:10]}")

for col in price_cols[:5]:  # Show first 5 price columns
    if col in df.columns:
        print(f"\n{col} statistics:")
        print(f"  Range: ${df[col].min():.2f} - ${df[col].max():.2f}")
        print(f"  Mean: ${df[col].mean():.2f}")
        print(f"  Recent values: {df[col].tail(3).tolist()}")

print(f"\n📅 DATE ANALYSIS:")
df["Date"] = pd.to_datetime(df["Date"])
print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Total days: {(df['Date'].max() - df['Date'].min()).days}")
print(f"Data frequency: ~{len(df) / ((df['Date'].max() - df['Date'].min()).days / 365.25):.0f} records per year")

print(f"\n💎 PRECIOUS METALS INDICATORS:")
metals_cols = [col for col in df.columns if any(metal in col.upper() for metal in ["GOLD", "SILVER", "PLT", "PLATINUM"])]
print(f"Metals-related columns: {metals_cols}")

print(f"\n🏭 OTHER ASSET CLASSES:")
other_cols = [col for col in df.columns if any(asset in col.upper() for asset in ["EU_", "OF_", "OS_", "SF_"])]
print(f"Other asset columns: {other_cols[:10]}")