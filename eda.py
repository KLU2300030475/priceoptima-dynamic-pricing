import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os


sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

# =========================================================
# 1. LOAD DATA
# =========================================================

file_path = "data/raw/final_dynamic_pricing_dataset_v2.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

# =========================================================
# 2. BASIC CLEANING
# =========================================================

print("\nMissing Values:")
print(df.isnull().sum())

df = df.dropna()
df['date'] = pd.to_datetime(df['date'])

# =========================================================
# 3. CREATE OUTPUT FOLDER
# =========================================================

os.makedirs("data/processed/eda_plots", exist_ok=True)

# =========================================================
# 4. SALES OVER TIME
# =========================================================

daily_sales = df.groupby('date')['sales_quantity'].sum()

plt.figure()
daily_sales.plot()
plt.title("Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Sales Quantity")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/sales_over_time.png", dpi=300)
plt.close()

# =========================================================
# 5. REVENUE OVER TIME
# =========================================================

daily_revenue = df.groupby('date')['revenue'].sum()

plt.figure()
daily_revenue.plot()
plt.title("Revenue Over Time")
plt.xlabel("Date")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/revenue_over_time.png", dpi=300)
plt.close()

# =========================================================
# 6. PRICE vs SALES
# =========================================================

sns.regplot(
    x='price',
    y='sales_quantity',
    data=df,
    scatter_kws={'alpha':0.2, 's':8},
    line_kws={'color':'red'},
    ci=None
)
plt.title("Price vs Sales Quantity")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/price_vs_sales.png", dpi=300)
plt.close()

# =========================================================
# 7. COMPETITOR PRICE vs SALES
# =========================================================

sns.regplot(
    x='competitor_price',
    y='sales_quantity',
    data=df,
    scatter_kws={'alpha':0.2, 's':8},
    line_kws={'color':'red'},
    ci=None
)
plt.title("Competitor Price vs Sales")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/competitor_price_vs_sales.png", dpi=300)
plt.close()

# =========================================================
# 8. INVENTORY vs SALES
# =========================================================

sns.regplot(
    x='inventory',
    y='sales_quantity',
    data=df,
    scatter_kws={'alpha':0.2, 's':8},
    line_kws={'color':'red'},
    ci=None
)
plt.title("Inventory vs Sales")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/inventory_vs_sales.png", dpi=300)
plt.close()

# =========================================================
# 9. PRODUCT-WISE SALES (Top 20)
# =========================================================

product_sales = df.groupby('product_id')['sales_quantity'].sum().sort_values(ascending=False)

plt.figure()
product_sales.head(20).plot(kind='bar')
plt.title("Top 20 Product Sales")
plt.xlabel("Product ID")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/product_wise_sales.png", dpi=300)
plt.close()

# =========================================================
# 10. PRICE DISTRIBUTION
# =========================================================

plt.figure()
sns.histplot(df['price'], bins=30, kde=True)
plt.title("Price Distribution")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/price_distribution.png", dpi=300)
plt.close()

# =========================================================
# 11. DISCOUNT vs SALES
# =========================================================
sns.regplot(
    x='discount',
    y='sales_quantity',
    data=df,
    scatter_kws={'alpha':0.2, 's':8},
    line_kws={'color':'red'},
    ci=None
)
plt.title("Discount vs Sales")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/discount_vs_sales.png", dpi=300)
plt.close()

# =========================================================
# 12. REGION-WISE SALES
# =========================================================

region_sales = df.groupby('region')['sales_quantity'].sum()

plt.figure()
region_sales.plot(kind='bar')
plt.title("Region Wise Sales")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/region_sales.png", dpi=300)
plt.close()

# =========================================================
# 13. WEATHER IMPACT
# =========================================================

weather_sales = df.groupby('weather')['sales_quantity'].sum()

plt.figure()
weather_sales.plot(kind='bar')
plt.title("Weather Impact on Sales")
plt.xlabel("Weather")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/weather_sales.png", dpi=300)
plt.close()

# =========================================================
# 14. WEEKEND IMPACT
# =========================================================

weekend_sales = df.groupby('weekend_flag')['sales_quantity'].mean()

plt.figure()
weekend_sales.plot(kind='bar')
plt.title("Weekend vs Weekday Average Sales")
plt.xlabel("Weekend Flag (0=No, 1=Yes)")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/weekend_sales.png", dpi=300)
plt.close()

# =========================================================
# 15. HOLIDAY IMPACT
# =========================================================

holiday_sales = df.groupby('holiday_flag')['sales_quantity'].mean()

plt.figure()
holiday_sales.plot(kind='bar')
plt.title("Holiday Impact on Sales")
plt.xlabel("Holiday Flag (0=No, 1=Yes)")
plt.ylabel("Average Sales")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/holiday_sales.png", dpi=300)
plt.close()

# =========================================================
# 16. CORRELATION HEATMAP
# =========================================================

numeric_df = df.select_dtypes(include=['int64', 'float64'])

plt.figure(figsize=(12,8))
sns.heatmap(
    numeric_df.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    linewidths=0.5
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("data/processed/eda_plots/correlation_heatmap.png", dpi=300)
plt.close()

# =========================================================
# 17. DEMAND ELASTICITY (Log-Log Model)
# =========================================================

elastic_df = df[(df['price'] > 0) & (df['sales_quantity'] > 0)].copy()

elastic_df['log_price'] = np.log(elastic_df['price'])
elastic_df['log_quantity'] = np.log(elastic_df['sales_quantity'])

X = sm.add_constant(elastic_df['log_price'])
y = elastic_df['log_quantity']

model = sm.OLS(y, X).fit()

print("\nDemand Elasticity Result:")
print(model.summary())

elasticity_value = model.params['log_price']
print("\nPrice Elasticity Coefficient:", elasticity_value)

# =========================================================
# 18. DEMAND SEGMENTATION
# =========================================================

mean_sales = df['sales_quantity'].mean()
df['demand_category'] = np.where(df['sales_quantity'] > mean_sales, "High Demand", "Low Demand")

# =========================================================
# 19. REVENUE SEGMENTATION
# =========================================================

mean_revenue = df['revenue'].mean()
df['revenue_category'] = np.where(df['revenue'] > mean_revenue, "High Revenue", "Low Revenue")

# =========================================================
# 20. SAVE FINAL DATASET
# =========================================================

df.to_csv("data/processed/eda_ready_dataset.csv", index=False)

print("\nEDA Completed Successfully")
print("All plots saved inside: data/processed/eda_plots/")