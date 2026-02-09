import pandas as pd

# Load dataset
df = pd.read_csv("final_dynamic_pricing_dataset.csv")

# Show first 5 rows
print("Dataset Preview:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicates
df = df.drop_duplicates()

print("\nDataset Shape After Cleaning:", df.shape)
