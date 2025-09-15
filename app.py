# 🥑 streamlit_app_funky_prophet.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
import joblib
from prophet import Prophet

MODEL_PATH = "model_svr.joblib"

# --- Page Config ---
st.set_page_config(
    page_title="Avocado Price Dashboard — Funk & Forecast",
    page_icon="🥑",
    layout="wide"
)

# --- Custom CSS for Styling ---
st.markdown(
    """
    <style>
        .main {
            background-color: #f4fff6;
        }
        h1, h2, h3, h4, h5 {
            color: #1b4332;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 1.1rem;
            padding: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Header ---
st.title("🥑 Avocado Price Dashboard — Funk & Forecast")

# --- Load Model ---
@st.cache_resource
def load_artifact(path: str):
    return joblib.load(path)

artifact = load_artifact(MODEL_PATH)
pipeline = artifact["pipeline"]
selected_features = artifact["selected_features"]
region_dummy_cols = artifact["region_dummy_cols"]

known_regions = sorted([c.replace("region_", "") for c in region_dummy_cols])
baseline_label = "Other/Unknown (baseline)"
region_options = [baseline_label] + known_regions

# --- Sidebar Fun ---
st.sidebar.header("🥑 Fun Facts")
st.sidebar.success("Avocados are technically berries!")
st.sidebar.info("The word 'avocado' comes from the Aztec word 'ahuacatl'.")
st.sidebar.warning("Prices can spike around Super Bowl Sunday! 🏈")

# --- Tabs Layout ---
tabs = st.tabs(["⚙️ Inputs", "📊 Predictions & Charts", "🔮 Forecast"])

# ================= Inputs Tab =================
with tabs[0]:
    st.subheader("⚙️ Input Parameters")
    with st.form("input_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            dt = st.date_input("📅 Date", value=date(2017, 1, 1))
            typ = st.radio("🍃 Type", options=["conventional", "organic"], horizontal=True)
            region_choice = st.selectbox("🌎 Region", options=region_options)
            prev_avg = st.number_input("⏪ Previous Avg Price (momentum)", value=0.0, step=0.01, format="%.2f")
        with c2:
            total_volume = st.number_input("📦 Total Volume", value=50000.0, step=100.0, format="%.2f")
            vol_4046 = st.number_input("🥑 PLU 4046", value=1000.0, step=10.0, format="%.2f")
            vol_4225 = st.number_input("🥑 PLU 4225", value=40000.0, step=100.0, format="%.2f")
            vol_4770 = st.number_input("🥑 PLU 4770", value=50.0, step=1.0, format="%.2f")
        with c3:
            total_bags = st.number_input("🛍 Total Bags", value=8000.0, step=10.0, format="%.2f")
            small_bags = st.number_input("👜 Small Bags", value=7900.0, step=10.0, format="%.2f")
            large_bags = st.number_input("🎒 Large Bags", value=100.0, step=10.0, format="%.2f")
            xlarge_bags = st.number_input("🧳 XLarge Bags", value=0.0, step=1.0, format="%.2f")

        submitted = st.form_submit_button("🚀 Predict")

# ================= Prediction Tab =================
with tabs[1]:
    st.subheader("📊 Predictions & Visuals")
    if 'submitted' in locals() and submitted:
        try:
            year_val = dt.year
            region_val = region_choice if region_choice != baseline_label else "BASELINE"

            raw = pd.DataFrame([{
                "Date": pd.to_datetime(dt),
                "type": typ,
                "region": region_val,
                "AveragePrice": np.nan,
                "Total Volume": float(total_volume),
                "4046": float(vol_4046),
                "4225": float(vol_4225),
                "4770": float(vol_4770),
                "Total Bags": float(total_bags),
                "Small Bags": float(small_bags),
                "Large Bags": float(large_bags),
                "XLarge Bags": float(xlarge_bags),
                "year": int(year_val)
            }])

            df = raw.copy()
            df["type"] = df["type"].map({"conventional": 0, "organic": 1})
            df["month"] = pd.to_datetime(df["Date"]).dt.month
            df["Price_Momentum"] = float(prev_avg) if prev_avg is not None else 0.0
            df["Volume_per_Bag"] = np.where(df["Total Bags"] > 0, df["Total Volume"] / df["Total Bags"], 0.0)
            df["Small_Bags_Ratio"] = np.where(df["Total Bags"] > 0, df["Small Bags"] / df["Total Bags"], 0.0)
            df["Large_Bags_Ratio"] = np.where(df["Total Bags"] > 0, df["Large Bags"] / df["Total Bags"], 0.0)
            df["Total_PLU_Volume"] = df["4046"] + df["4225"] + df["4770"]
            df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
            df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

            region_dummies = pd.get_dummies(df["region"], prefix="region", drop_first=True)
            for col in region_dummy_cols:
                if col not in region_dummies.columns:
                    region_dummies[col] = 0
            region_dummies = region_dummies[region_dummy_cols]
            df = pd.concat([df, region_dummies], axis=1)
            df.drop(columns=["region"], inplace=True)

            df = df.ffill()
            for col in selected_features:
                if col not in df.columns:
                    df[col] = 0
            X = df[selected_features].copy()

            pred = pipeline.predict(X)
            pred_val = float(np.asarray(pred).ravel())

            st.success(f"🎉 Predicted Avocado Price: **{pred_val:.3f} USD**")
            st.balloons()

            # --- Matplotlib Visual ---
            fig, ax = plt.subplots()
            ax.bar(["Predicted Price"], [pred_val], color="green")
            ax.set_ylabel("Price (USD)")
            ax.set_title("Avocado Price Prediction")
            st.pyplot(fig)

            # --- Prophet Forecast (quick one for context) ---
            st.markdown("### 🔮 Short-term Forecast with Prophet")
            demo_hist = pd.DataFrame({
                "ds": pd.date_range(start="2016-01-01", periods=52, freq="W"),
                "y": np.random.uniform(0.8, 1.6, size=52)
            })
            m = Prophet()
            m.fit(demo_hist)
            future = m.make_future_dataframe(periods=12, freq='W')
            forecast = m.predict(future)

            fig3, ax3 = plt.subplots(figsize=(8, 4))
            ax3.plot(demo_hist["ds"], demo_hist["y"], label="Historical", color="black")
            ax3.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="green")
            ax3.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="lightgreen", alpha=0.5)
            ax3.axhline(pred_val, color="red", linestyle="--", label="Predicted Point")
            ax3.legend()
            ax3.set_title("Prophet Forecast with Current Prediction")
            st.pyplot(fig3)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ================= Forecast Tab =================
with tabs[2]:
    st.subheader("🔮 Prophet Forecast")
    uploaded = st.file_uploader("Upload historical avocado CSV (with Date, AveragePrice)", type=["csv"])

    if uploaded:
        hist = pd.read_csv(uploaded)
        hist["Date"] = pd.to_datetime(hist["Date"])
        hist = hist.rename(columns={"Date": "ds", "AveragePrice": "y"})

        horizon = st.slider("Forecast horizon (weeks)", 4, 52, 12)
        m = Prophet()
        m.fit(hist)
        future = m.make_future_dataframe(periods=horizon, freq='W')
        forecast = m.predict(future)

        # Plot forecast
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(hist["ds"], hist["y"], label="Historical", color="black")
        ax2.plot(forecast["ds"], forecast["yhat"], label="Forecast", color="green")
        ax2.fill_between(forecast["ds"], forecast["yhat_lower"], forecast["yhat_upper"], color="lightgreen", alpha=0.5)
        ax2.set_title("Prophet Forecast of Avocado Prices")
        ax2.set_ylabel("Price (USD)")
        ax2.legend()
        st.pyplot(fig2)

        st.download_button("📥 Download Forecast CSV", forecast.to_csv(index=False).encode(), "forecast.csv", "text/csv")
    else:
        st.info("Upload a CSV with 'Date' and 'AveragePrice' to see Prophet forecast.")

# --- API Query Params (hidden) ---
query_params = st.query_params
if query_params:
    st.write("🔌 API mode params:", query_params)