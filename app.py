import streamlit as st
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="PriceOptima", layout="wide")

# -----------------------------
# TITLE
# -----------------------------
st.title("💰 PriceOptima - Dynamic Pricing System")
st.markdown("### Intelligent Price Recommendation Dashboard")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("⚙️ Advanced Settings")

competitor_price = st.sidebar.number_input("Competitor Price", value=95.0)
discount = st.sidebar.number_input("Discount (%)", value=5.0)
weekend_flag = st.sidebar.selectbox("Weekend?", [0, 1])
holiday_flag = st.sidebar.selectbox("Holiday?", [0, 1])

# -----------------------------
# MAIN INPUTS
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    price = st.number_input("Current Price", min_value=0.0, value=100.0)

with col2:
    inventory = st.number_input("Inventory Level", min_value=0, value=50)

# -----------------------------
# AUTO MODE
# -----------------------------
auto_mode = st.checkbox("⚡ Auto Mode (Recommended)")

if auto_mode:
    competitor_price = price * 0.95
    discount = 5.0
    weekend_flag = 0
    holiday_flag = 0

price_diff = price - competitor_price

# -----------------------------
# DEFAULT VALUES
# -----------------------------
base_price = price
year = 2024
month = 1
day = 1
hour = 12

# -----------------------------
# LOAD MODEL
# -----------------------------
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    pipeline = joblib.load(os.path.join(BASE_DIR, "pipeline.pkl"))
    st.success("✅ Model Ready")
    pipeline_loaded = True
except:
    st.error("❌ Model failed to load")
    pipeline_loaded = False

# -----------------------------
# BUTTON
# -----------------------------
if st.button("🚀 Get Recommendation"):

    if pipeline_loaded:
        # -----------------------------
        # ORIGINAL DEMAND
        # -----------------------------
        input_df = pd.DataFrame([{
            "price": price,
            "base_price": base_price,
            "competitor_price": competitor_price,
            "discount": discount,
            "inventory": inventory,
            "year": year,
            "month": month,
            "day": day,
            "hour": hour,
            "weekend_flag": weekend_flag,
            "holiday_flag": holiday_flag,
            "price_diff": price_diff
        }])

        demand = pipeline.predict(input_df)[0]

        # -----------------------------
        # SMART PRICE ADJUSTMENT
        # -----------------------------
        # Dynamic logic based on demand
        if demand > 60:
            recommended_price = price * 1.10
        elif demand > 40:
            recommended_price = price * 1.05
        else:
            recommended_price = price * 0.95

        # -----------------------------
        # RE-CALCULATE DEMAND (IMPORTANT FIX)
        # -----------------------------
        new_input_df = input_df.copy()
        new_input_df["price"] = recommended_price
        new_input_df["price_diff"] = recommended_price - competitor_price

        new_demand = pipeline.predict(new_input_df)[0]

        # 🔥 ADJUST DEMAND BASED ON PRICE CHANGE
        price_change_ratio = (recommended_price - price) / price

        new_demand = new_demand * (1 - 0.5 * price_change_ratio)

        # -----------------------------
        # REVENUE CALCULATIONS
        # -----------------------------
        static_price = price
        rule_price = competitor_price * (1 - discount / 100)

        static_revenue = static_price * demand
        rule_revenue = rule_price * demand
        ml_revenue = recommended_price * new_demand

        improvement = ((ml_revenue - static_revenue) / static_revenue) * 100

        # -----------------------------
        # KPI CARDS
        # -----------------------------
        st.subheader("📊 Key Metrics")

        k1, k2, k3 = st.columns(3)

        k1.metric("Demand", f"{new_demand:.2f}")
        k2.metric("Recommended Price", f"{recommended_price:.2f}")
        k3.metric("Revenue Improvement", f"{improvement:.2f}%")

        # -----------------------------
        # PRICE COMPARISON
        # -----------------------------
        st.subheader("📊 Pricing Comparison")

        c1, c2, c3 = st.columns(3)

        c1.metric("Original Price", f"{price:.2f}")
        c2.metric("Rule-Based Price", f"{rule_price:.2f}")
        c3.metric("ML Price", f"{recommended_price:.2f}")

        # -----------------------------
        # REVENUE GRAPH
        # -----------------------------
        st.subheader("📈 Revenue Comparison")

        fig, ax = plt.subplots()

        labels = ["Static", "Rule-Based", "ML-Based"]
        values = [static_revenue, rule_revenue, ml_revenue]

        ax.bar(labels, values)
        ax.set_ylabel("Revenue")

        st.pyplot(fig)

        st.write(f"ML Revenue: {ml_revenue:.2f}")
        ax.set_title("Revenue Comparison")
        # -----------------------------
        # BUSINESS INSIGHT
        # -----------------------------
        st.subheader("📌 Business Insight")

        if recommended_price > price:
            st.success("Increase price → High demand expected 📈")
        else:
            st.warning("Decrease price → Boost sales 📉")

        # -----------------------------
        # EXPLANATION
        # -----------------------------
        st.info("""
💡 This system:
- Predicts demand using Machine Learning
- Adjusts price dynamically based on demand
- Recalculates demand after price change
- Shows real revenue improvement
""")