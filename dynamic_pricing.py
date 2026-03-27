import pandas as pd
import numpy as np

# ==============================
# 1. Load Dataset
# ==============================

df = pd.read_parquet(
    r"C:\Users\karth\OneDrive\Desktop\nigerian_retail_and_ecommerce_dynamic_pricing_logs.parquet"
)

print("Dataset Loaded Successfully!")
print("Total rows:", len(df))

# ==============================
# 2. Rename Columns (Clean Names)
# ==============================

df = df.rename(columns={
    "current_price_ngn": "price",
    "competitor_price_ngn": "competitor_price",
    "stock_level": "inventory",
    "timestamp": "date"
})

print("\nColumns after renaming:\n", df.columns)

# ==============================
# 3. Convert Date Column
# ==============================

df["date"] = pd.to_datetime(df["date"])

# ==============================
# 4. Weekend Flag
# ==============================

df["weekend_flag"] = df["date"].dt.dayofweek.apply(
    lambda x: 1 if x >= 5 else 0
)

# ==============================
# 5. Extract Time Features
# ==============================

df["day"] = df["date"].dt.day
df["month"] = df["date"].dt.month
df["year"] = df["date"].dt.year
df["hour"] = df["date"].dt.hour
df["am_pm"] = df["date"].dt.strftime("%p")
df["day_name"] = df["date"].dt.day_name()

# ==============================
# 6. Seasonality Column
# ==============================

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

df["seasonality"] = df["month"].apply(get_season)

# ==============================
# 6A. Holiday Flag (NEW)
# ==============================

holiday_list = [
    "01-01",  # New Year
    "12-25",  # Christmas
    "10-01",  # Nigeria Independence Day
    "05-01"   # Workers Day
]

df["month_day"] = df["date"].dt.strftime("%m-%d")

df["holiday_flag"] = df["month_day"].apply(
    lambda x: 1 if x in holiday_list else 0
)

df.drop(columns=["month_day"], inplace=True)

# ==============================
# 7. Generate Sales Quantity (Synthetic)
# ==============================

df["sales_quantity"] = (
    ((df["price"].max() - df["price"]) / df["price"].max()) * 50
    + (df["inventory"] / df["inventory"].max()) * 30
    + np.random.randint(5, 15, len(df))
).astype(int)

# ==============================
# 8. Base Price + Discount Column
# ==============================

df["base_price"] = df["price"] + np.random.randint(50, 300, len(df))

df["discount"] = np.where(
    df["base_price"] > 0,
    ((df["base_price"] - df["price"]) / df["base_price"]) * 100,
    0
)

df["discount"] = df["discount"].clip(lower=0).round(2)

# ==============================
# 9. Weather Column
# ==============================

def season_weather(season):
    if season == "Winter":
        return np.random.choice(["Cloudy", "Rainy", "Snowy"])
    elif season == "Summer":
        return np.random.choice(["Sunny", "Cloudy"])
    else:
        return np.random.choice(["Sunny", "Rainy", "Cloudy"])

df["weather"] = df["seasonality"].apply(season_weather)

# ==============================
# 10. Region Column
# ==============================

regions = ["North", "South", "East", "West"]
df["region"] = np.random.choice(regions, size=len(df))

# ==============================
# 11. Feature Engineering
# ==============================

df["revenue"] = df["price"] * df["sales_quantity"]

df["stock_status"] = df["inventory"].apply(
    lambda x: "Low" if x < 100 else "Available"
)

# ==============================
# 12. Data Cleaning Checks
# ==============================

print("\nDuplicate rows:", df.duplicated().sum())
print("\nMissing values:\n", df.isnull().sum())

# ==============================
# 13. Final Dataset Export
# ==============================

final_df = df[[
    "product_id",
    "price",
    "base_price",
    "competitor_price",
    "discount",
    "sales_quantity",
    "revenue",

    "inventory",
    "stock_status",

    "region",
    "weather",
    "seasonality",

    "date",
    "year",
    "month",
    "day",
    "day_name",
    "hour",
    "am_pm",
    "weekend_flag",
    "holiday_flag"
]]

