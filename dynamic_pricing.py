import pandas as pd
import numpy as np

# Load parquet file from OneDrive Desktop
df = pd.read_parquet(
    r"C:\Users\karth\OneDrive\Desktop\nigerian_retail_and_ecommerce_dynamic_pricing_logs.parquet"
)

print("✅ Dataset Loaded Successfully!")
print("Total rows:", len(df))


# Rename columns
df = df.rename(columns={
    "current_price_ngn": "price",
    "stock_level": "inventory",
    "timestamp": "date"
})

# Generate sales quantity
df["sales_quantity"] = (
    ((df["price"].max() - df["price"]) / df["price"].max()) * 50
    + (df["inventory"] / df["inventory"].max()) * 30
    + np.random.randint(1, 10, len(df))
).astype(int)

# Final dataset
final_df = df[["product_id", "price", "sales_quantity", "inventory", "date"]]

# Save CSV in project folder
final_df.to_csv("final_dynamic_pricing_dataset.csv", index=False)

print("✅ Final CSV created successfully!")
final_df.head(5000).to_csv("sample_dynamic_pricing.csv", index=False)
print("✅ Sample file created!")

