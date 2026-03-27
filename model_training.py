import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import joblib
import os

print("STARTING PROGRAM")

# ===============================
# Load Data
# ===============================
df = pd.read_csv("data/processed/eda_ready_dataset.csv")

# ===============================
# Basic Cleaning
# ===============================
df = df.dropna()

# ===============================
# Feature Engineering
# ===============================
df['price_diff'] = df['price'] - df['competitor_price']

# ===============================
# Handle Categorical Columns
# ===============================
categorical_cols = [
    'stock_status',
    'region',
    'weather',
    'seasonality',
    'day_name',
    'am_pm'
]

df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# ===============================
# Features & Target
# ===============================
features = [
    'price',
    'base_price',
    'competitor_price',
    'discount',
    'inventory',
    'year',
    'month',
    'day',
    'hour',
    'weekend_flag',
    'holiday_flag',
    'price_diff'
]

target = 'sales_quantity'

X = df[features]
y = df[target]

# ===============================
# Train Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# Models
# ===============================
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
lgb_model = LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)

xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)

# ===============================
# Evaluation
# ===============================
def evaluate(model, name):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"\n{name} Performance")
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

evaluate(xgb_model, "XGBoost")
evaluate(lgb_model, "LightGBM")

# ===============================
# Save Models
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
models_path = os.path.join(BASE_DIR, "saved_models")

os.makedirs(models_path, exist_ok=True)

joblib.dump(xgb_model, os.path.join(models_path, "xgboost_model.pkl"))
joblib.dump(lgb_model, os.path.join(models_path, "lightgbm_model.pkl"))

from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('model', xgb_model)
])

joblib.dump(pipeline, "pipeline.pkl")


# ===============================
# Backtesting Revenue
# ===============================
df['static_revenue'] = df['price'] * df['sales_quantity']

df['xgb_pred'] = xgb_model.predict(X)
df['xgb_revenue'] = df['price'] * df['xgb_pred']

df['lgb_pred'] = lgb_model.predict(X)
df['lgb_revenue'] = df['price'] * df['lgb_pred']


# ===============================
# Revenue Comparison
# ===============================
static_rev = df['static_revenue'].sum()
xgb_rev = df['xgb_revenue'].sum()
lgb_rev = df['lgb_revenue'].sum()

print("\nRevenue Comparison")
print("Static Revenue :", static_rev)
print("XGBoost Revenue:", xgb_rev)
print("LightGBM Revenue:", lgb_rev)

# ===============================
# Revenue Lift
# ===============================
def revenue_lift(new, original):
    return ((new - original) / original) * 100

print("\nRevenue Lift")
print("XGBoost Lift :", revenue_lift(xgb_rev, static_rev), "%")
print("LightGBM Lift:", revenue_lift(lgb_rev, static_rev), "%")

# ===============================
# PRICE OPTIMIZATION (NEW 🔥)
# ===============================
def find_optimal_price(row, model):
    base_price = row['price']

    # controlled price range
    price_options = np.linspace(base_price * 0.85, base_price * 1.15, 8)

    best_price = base_price
    best_revenue = 0

    for p in price_options:
        temp = row[features].copy()

        temp['price'] = p
        temp['price_diff'] = p - row['competitor_price']

        temp_df = pd.DataFrame([temp.values], columns=features)

        demand = model.predict(temp_df)[0]

        # controlled demand
        demand = max(0, min(demand, 1.2 * row['sales_quantity']))

        # penalty for large change
        penalty = abs(p - base_price) * 0.05

        revenue = (p * demand) - penalty

        if revenue > best_revenue:
            best_revenue = revenue
            best_price = p

    return best_price, best_revenue
# ===============================
# Apply Optimization (Sample)
# ===============================
print("\nRunning Price Optimization...")

sample_df = df.sample(1000, random_state=42).copy()

opt_prices = []
opt_revenues = []

for i, (_, row) in enumerate(sample_df.iterrows()):

    p, r = find_optimal_price(row, xgb_model)
    opt_prices.append(p)
    opt_revenues.append(r)

sample_df['optimized_price'] = opt_prices
sample_df['optimized_revenue'] = opt_revenues

# ===============================
# Optimized Revenue Comparison
# ===============================
static_sample_rev = (sample_df['price'] * sample_df['sales_quantity']).sum()
optimized_sample_rev = sample_df['optimized_revenue'].sum()

print("\nOptimized Revenue Comparison (Sample)")
print("Static Revenue :", static_sample_rev)
print("Optimized Revenue :", optimized_sample_rev)

opt_lift = ((optimized_sample_rev - static_sample_rev) / static_sample_rev) * 100

print("Optimized Revenue Lift:", opt_lift, "%")

import matplotlib.pyplot as plt

# Use your actual values
labels = ["Static", "Optimized"]
values = [static_sample_rev, optimized_sample_rev]

plt.figure()
plt.bar(labels, values)
plt.title("Revenue Comparison")
plt.xlabel("Pricing Strategy")
plt.ylabel("Revenue")

plt.show()