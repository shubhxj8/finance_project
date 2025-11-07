import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import os, json, subprocess
from datetime import datetime

# Import pipeline functions
from strategy_pipeline import main as run_strategy_pipeline
from data_extraction_backup import run_data_pipeline

# -------------------------------
# Streamlit Config
# -------------------------------
st.set_page_config(page_title="NIFTY Regime Detection Dashboard", layout="wide")
os.makedirs("outputs", exist_ok=True)

st.markdown("<h1 style='text-align:center;'>📊 NIFTY Regime Detection & Strategy Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Built with ❤️ using HMM, XGBoost, and RL</p>", unsafe_allow_html=True)

# -------------------------------
# Sidebar Controls
# -------------------------------
st.sidebar.header("⚙️ Controls")

out_folder = "outputs"
input_csv = "nifty_regime_hmm.csv"

run_data = st.sidebar.button("📥 Run Data Extraction")
run_pipeline = st.sidebar.button("🚀 Run Full Strategy Pipeline")

st.sidebar.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# -------------------------------
# Handle Run Buttons
# -------------------------------
if run_data:
    st.info("📡 Extracting and preparing fresh market data... Please wait ⏳")
    try:
        run_data_pipeline(output_file=input_csv)
        st.success(f"✅ Data extraction complete! Saved as `{input_csv}`")
        # input_csv = 'outputs/' + input_csv
        if os.path.exists(input_csv):
            df_preview = pd.read_csv(input_csv)
            st.subheader("📂 New Data Preview")
            st.dataframe(df_preview.head(10))
            st.caption(f"Shape: {df_preview.shape}")
        else:
            st.warning("⚠️ Output file not found after extraction.")
    except Exception as e:
        st.error(f"❌ Data extraction failed: {e}")

elif run_pipeline:
    if not os.path.exists(input_csv):
        st.error(f"⚠️ `{input_csv}` not found — run data extraction first.")
    else:
        st.info("🚀 Running full strategy pipeline... This may take a few minutes ⏳")
        try:
            run_strategy_pipeline(out_folder=out_folder, retrain=True)
            st.success("✅ Strategy pipeline retraining complete! New results saved in 'outputs/'")
        except Exception as e:
            st.error(f"❌ Strategy pipeline failed: {e}")

# -------------------------------
# Utility: Load CSV Safely
# -------------------------------
def load_csv(file):
    path = os.path.join(out_folder, file)
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

# -------------------------------
# Load Saved Results (Static Home View)
# -------------------------------
index_df = load_csv("bars_with_emas.csv")
# trades_df = load_csv("trades.csv")
# daily_df = load_csv("daily_metrics.csv")
context_df = load_csv("trade_contexts_with_anomalies.csv")
option_chain_df = load_csv("option_chain.csv")
total_oi_df = load_csv("total_oi_timeseries.csv")

train_trades_df = load_csv("train_trades.csv")
train_daily_df = load_csv("train_daily_metrics.csv")
test_trades_df = load_csv("test_trades.csv")
test_daily_df = load_csv("test_daily_metrics.csv")


# If nothing exists — show welcome
if all(df.empty for df in [index_df, train_trades_df, train_daily_df, test_trades_df, test_daily_df, context_df]):
    st.info("👋 Welcome! No results yet — click **Run Data Extraction** and then **Run Full Strategy Pipeline** to generate insights.")
    st.stop()

# -------------------------------
# Dashboard Tabs
# -------------------------------
tabs = st.tabs([
    "📈 Market & Regimes",
    "🔥 Option & OI Analytics",
    "💰 Backtest Performance",
    "🧠 Feature Importance",
    "🤖 Model Metrics",
    "🚨 Anomalies"
])

# === 1️⃣ Market & Regimes ===
with tabs[0]:
    st.subheader("📈 Market OHLC + Regime Overlay")

    if not index_df.empty:
        fig = go.Figure()

        # Candlestick
        if all(c in index_df.columns for c in ['open','high','low','close']):
            fig.add_trace(go.Candlestick(
                x=index_df['timestamp'],
                open=index_df['open'],
                high=index_df['high'],
                low=index_df['low'],
                close=index_df['close'],
                name='Price'
            ))
        else:
            fig.add_trace(go.Scatter(x=index_df['timestamp'], y=index_df['close'], name='Close Price'))

        # EMA overlays
        for ema_col, color in [('ema_5', 'orange'), ('ema_15', 'blue')]:
            if ema_col in index_df.columns:
                fig.add_trace(go.Scatter(
                    x=index_df['timestamp'],
                    y=index_df[ema_col],
                    mode='lines',
                    name=ema_col,
                    line=dict(color=color)
                ))

        # Regime overlay
        if 'regime' in index_df.columns:
            regime_colors = {0:"gray",1:"green",2:"red"}
            fig.add_trace(go.Scatter(
                x=index_df['timestamp'],
                y=index_df['close'],
                mode='markers',
                marker=dict(color=[regime_colors.get(r,'blue') for r in index_df['regime']], size=4, opacity=0.6),
                name='Regime'
            ))

        fig.update_layout(title="NIFTY Regimes with 5EMA/15EMA", xaxis_title="Date", yaxis_title="Price")
        st.plotly_chart(fig, use_container_width=True)

        # Regime distribution
        if 'regime' in index_df.columns:
            st.markdown("#### 📊 Regime Distribution")
            regime_count = index_df['regime'].value_counts(normalize=True) * 100
            fig_regime = px.bar(
                x=regime_count.index,
                y=regime_count.values,
                color=regime_count.index.map({0:"gray",1:"green",2:"red"}),
                labels={'x':'Regime', 'y':'% of Time'},
                title="Regime Frequency (%)"
            )
            st.plotly_chart(fig_regime, use_container_width=True)
    else:
        st.warning("No price data available.")

# === 2️⃣ Option & OI Analytics ===
with tabs[1]:
    st.subheader("🔥 Option Chain & OI Analytics")

    if not option_chain_df.empty:
        st.markdown("#### 🧭 Option Chain OI Heatmap")
        heatmap_df = option_chain_df.pivot_table(
            index="strike_price", columns="expiry_date", values="open_interest", aggfunc="mean"
        )

        fig_heat = go.Figure(data=go.Heatmap(
            z=heatmap_df.values,
            x=heatmap_df.columns,
            y=heatmap_df.index,
            colorscale="Viridis"
        ))
        fig_heat.update_layout(title="OI by Strike & Expiry", xaxis_title="Expiry", yaxis_title="Strike Price")
        st.plotly_chart(fig_heat, use_container_width=True)

    if not total_oi_df.empty:
        st.markdown("#### 📊 Total OI & Change-in-OI Over Time")
        fig_oi = go.Figure()
        fig_oi.add_trace(go.Scatter(
            x=total_oi_df["timestamp"], y=total_oi_df["total_oi"],
            mode="lines", name="Total OI"
        ))
        if "change_in_oi" in total_oi_df.columns:
            fig_oi.add_trace(go.Bar(
                x=total_oi_df["timestamp"], y=total_oi_df["change_in_oi"],
                name="ΔOI", opacity=0.5
            ))
        fig_oi.update_layout(title="OI Trends", xaxis_title="Date", yaxis_title="Open Interest")
        st.plotly_chart(fig_oi, use_container_width=True)
    else:
        st.info("No OI data found.")

# === 3️⃣ Backtest Performance ===
with tabs[2]:
    st.subheader("💹 Backtest Performance (Train vs Test)")

    if not (train_trades_df.empty and test_trades_df.empty):
        for phase_name, trades, daily in [
            ("Training (First 50%)", train_trades_df, train_daily_df),
            ("Backtest (Last 50%)", test_trades_df, test_daily_df),
        ]:
            if trades.empty or daily.empty:
                continue

            st.markdown(f"### 📊 {phase_name}")
            total_pnl = daily['daily_pnl'].sum() if 'daily_pnl' in daily.columns else 0
            sharpe = (daily['daily_pnl'].mean() / daily['daily_pnl'].std() * (252**0.5)) if 'daily_pnl' in daily.columns and daily['daily_pnl'].std() != 0 else 0
            max_dd = daily['daily_pnl'].cumsum().sub(daily['daily_pnl'].cumsum().cummax()).min() if 'daily_pnl' in daily.columns else 0
            winrate = trades['win'].mean() * 100 if 'win' in trades.columns else 0

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Total PNL", f"{total_pnl:,.0f}")
            col2.metric("⚡ Sharpe", f"{sharpe:.2f}")
            col3.metric("📉 Max Drawdown", f"{max_dd:,.0f}")
            col4.metric("🏆 Win Rate", f"{winrate:.1f}%")

            if 'daily_pnl' in daily.columns:
                daily["cum_equity"] = daily["daily_pnl"].cumsum()
                eq_fig = go.Figure()
                eq_fig.add_trace(go.Scatter(x=daily["entry_date"], y=daily["cum_equity"], name="Equity Curve"))
                eq_fig.update_layout(title=f"{phase_name} Equity Curve")
                st.plotly_chart(eq_fig, use_container_width=True)

            if 'pnl' in trades.columns:
                fig_hist = px.histogram(trades, x='pnl', nbins=30, title=f'{phase_name} Trade P&L Distribution')
                st.plotly_chart(fig_hist, use_container_width=True)

            st.markdown("---")
    else:
        st.info("No training or backtest trade results found — run the strategy pipeline first.")

# === 4️⃣ Feature Importance ===
with tabs[3]:
    st.subheader("🧠 Feature Importance & Correlations")
    for imp_name, title in [
        ("xgb_feature_importance.png", "XGBoost Importance"),
        ("permutation_importance.png", "Permutation Importance")
    ]:
        imp_path = os.path.join(out_folder, imp_name)
        if os.path.exists(imp_path):
            st.image(imp_path, caption=title, use_container_width=True)

    if not context_df.empty:
        st.markdown("#### 🔍 Feature Correlation Heatmap")
        corr = context_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, cmap='coolwarm', center=0)
        st.pyplot(fig)

# === 5️⃣ Model Metrics ===
with tabs[4]:
    st.subheader("🤖 Model Metrics Radar")
    models = []
    for model_name in [r"rl_metrics.json", r"xgb_metrics.json"]:
        path = os.path.join(out_folder, model_name)
        if os.path.exists(path):
            with open(path) as f:
                metrics = json.load(f)
            models.append({"Model": model_name.split("_")[0].upper(), **metrics})

    if models:
        df = pd.DataFrame(models)
        df_melt = df.melt(id_vars="Model", var_name="Metric", value_name="Score")
        fig = px.line_polar(df_melt, r="Score", theta="Metric", color="Model", line_close=True)
        fig.update_traces(fill='toself')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No model metric files found yet.")

# === 6️⃣ Anomalies ===
with tabs[5]:
    st.subheader("🚨 Anomaly Detection")

    if not context_df.empty and "anomaly" in context_df.columns:
        total_anoms = int(context_df["anomaly"].sum())
        st.success(f"✅ Loaded anomaly data — {total_anoms} anomalies detected")

        context_df["bubble_size"] = context_df["anomaly_score"].abs() * 50 if "anomaly_score" in context_df.columns else 20
        fig = px.scatter(
            context_df,
            x="return_last" if "return_last" in context_df.columns else "pnl",
            y="volatility_last" if "volatility_last" in context_df.columns else "volatility_mean",
            color="anomaly",
            size="bubble_size",
            hover_data=["pnl", "regime", "anomaly_score"] if "anomaly_score" in context_df.columns else ["pnl", "regime"],
            color_discrete_map={0: "blue", 1: "red"},
            title="Anomalous Trade Contexts"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 Top Anomalies")
        st.dataframe(context_df[context_df["anomaly"] == 1].sort_values(by="anomaly_score", ascending=False).head(20))
    else:
        st.info("No anomaly data yet — retrain the pipeline to detect anomalies.")

st.markdown("---")
st.caption("💡 Built with Streamlit, Plotly, HMM, XGBoost, and RL.")
