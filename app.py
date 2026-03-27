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
# SIDEBAR (ADVANCED INPUTS)
# -----------------------------
st.sidebar.header("⚙️ Advanced Settings")

base_price = st.sidebar.number_input("Base Price", value=100.0)
competitor_price = st.sidebar.number_input("Competitor Price", value=95.0)
discount = st.sidebar.number_input("Discount (%)", value=5.0)

year = st.sidebar.number_input("Year", value=2024)
month = st.sidebar.number_input("Month", 1, 12, value=1)
day = st.sidebar.number_input("Day", 1, 31, value=1)
hour = st.sidebar.number_input("Hour", 0, 23, value=12)

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

price_diff = price - competitor_price

auto_mode = st.checkbox("⚡ Auto Mode (Recommended)")

if auto_mode:
    base_price = price
    competitor_price = price * 0.95
    discount = 5
    weekend_flag = 0
    holiday_flag = 0
# -----------------------------
# LOAD PIPELINE
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

        from pricing_engine import get_optimal_price
        recommended_price = get_optimal_price(price, demand)

        old_revenue = price * demand
        new_revenue = recommended_price * demand
        improvement = ((new_revenue - old_revenue) / old_revenue) * 100

        # -----------------------------
        # KPI CARDS
        # -----------------------------
        st.subheader("📊 Key Metrics")

        k1, k2, k3 = st.columns(3)

        k1.metric("Demand", f"{demand:.2f}")
        k2.metric("Recommended Price", f"{recommended_price:.2f}")
        k3.metric("Revenue Improvement", f"{improvement:.2f}%")

        # -----------------------------
        # GRAPH
        # -----------------------------
        st.subheader("📈 Revenue Comparison")

        fig, ax = plt.subplots()
        ax.bar(["Old", "New"], [old_revenue, new_revenue])

        st.pyplot(fig)