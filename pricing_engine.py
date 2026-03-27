import pandas as pd
import matplotlib.pyplot as plt

# =========================
# 1. LOAD DATA
# =========================

file_path = "data/processed/eda_ready_dataset.csv"
df = pd.read_csv(file_path)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)


# =========================
# 2. STATIC PRICING
# =========================

df["original_revenue"] = df["price"] * df["sales_quantity"]


# =========================
# 3. CREATE ADJUSTED PRICE
# =========================

df["adjusted_price"] = df["price"]


# =========================
# 4. TIME-BASED PRICING
# =========================

# Weekend demand → increase price slightly
df.loc[df["weekend_flag"] == 1, "adjusted_price"] *= 1.10

# Peak hours pricing (6PM–10PM)
df.loc[(df["hour"] >= 18) & (df["hour"] <= 22), "adjusted_price"] *= 1.07


# =========================
# 5. INVENTORY-BASED PRICING
# =========================

# Low inventory → higher price (scarcity)
df.loc[df["inventory"] < 50, "adjusted_price"] *= 1.12

# High inventory → small discount only
df.loc[df["inventory"] > 500, "adjusted_price"] *= 0.97


# =========================
# 6. DYNAMIC REVENUE
# =========================

df["dynamic_revenue"] = df["adjusted_price"] * df["sales_quantity"]


# =========================
# 7. PRICE COMPARISON
# =========================

print("\n===== Original Price → Adjusted Price =====")
print(df[["product_id","price","adjusted_price"]].head(10))


# =========================
# 8. REVENUE COMPARISON
# =========================

print("\n===== Original Revenue → New Revenue =====")
print(df[["product_id","original_revenue","dynamic_revenue"]].head(10))


# =========================
# 9. REVENUE IMPROVEMENT
# =========================

original_revenue = df["original_revenue"].sum()
new_revenue = df["dynamic_revenue"].sum()

revenue_improvement = ((new_revenue - original_revenue) / original_revenue) * 100

print("\n===== Revenue Improvement =====")
print("Original Revenue :", original_revenue)
print("New Revenue :", new_revenue)
print("Revenue Improvement (%) :", round(revenue_improvement, 2))


# =========================
# 10. SAVE RESULTS
# =========================

output_path = "data/processed/pricing_engine_results.csv"
df.to_csv(output_path, index=False)

print("\nDynamic pricing results saved successfully!")
print("File location:", output_path)


# =========================
# 11. VISUALIZATION
# =========================

df = pd.read_csv(output_path)

original_total = df["original_revenue"].sum()
dynamic_total = df["dynamic_revenue"].sum()

plt.figure()
plt.bar(["Static Pricing", "Dynamic Pricing"],
        [original_total, dynamic_total])
plt.title("Revenue Comparison")
plt.ylabel("Total Revenue")
plt.show()

plt.figure()
plt.scatter(df["price"], df["adjusted_price"], alpha=0.3)
plt.xlabel("Original Price")
plt.ylabel("Adjusted Price")
plt.title("Original Price vs Adjusted Price")
plt.show()

# =========================
# FUNCTION FOR STREAMLIT APP
# =========================

def get_optimal_price(price, demand, inventory=100, hour=12, weekend_flag=0):
    """
    Lightweight version of pricing logic for real-time prediction
    """

    adjusted_price = price

    # Weekend pricing
    if weekend_flag == 1:
        adjusted_price *= 1.10

    # Peak hours
    if 18 <= hour <= 22:
        adjusted_price *= 1.07

    # Inventory logic
    if inventory < 50:
        adjusted_price *= 1.12
    elif inventory > 500:
        adjusted_price *= 0.97

    # Demand-based tweak (optional)
    if demand > 50:
        adjusted_price *= 1.05
    elif demand < 20:
        adjusted_price *= 0.95

    return adjusted_price