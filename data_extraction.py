import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from hmmlearn import hmm

# ==========================================================
# 1️⃣ Utility: Regime Detection with HMM
# ==========================================================
def detect_regimes(df, n_states=3):
    """Detect market regimes using HMM based on returns and volatility."""
    df = df.copy()
    df["return"] = df["close"].pct_change().fillna(0)
    features = df[["return"]].values

    model = hmm.GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200)
    model.fit(features)

    df["regime"] = model.predict(features)
    df["regime_name"] = df["regime"].map({
        0: "Sideways",
        1: "Uptrend",
        2: "Downtrend"
    })
    return df


# ==========================================================
# 2️⃣ Loaders for Each Data Source
# ==========================================================
def load_equity_data(folder_path):
    """Load single Nifty equity CSV file."""
    file_path = os.path.join(folder_path, "nifty_equity_data.csv")
    if not os.path.exists(file_path):
        print("⚠️ Equity file not found:", file_path)
        return pd.DataFrame()
    
    df = pd.read_csv(file_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={"datetime": "timestamp"}, inplace=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    return df

def clean_final_dataset(df):
    """Final cleaning step for consistent and anomaly-free dataset."""
    df = df.copy()

    for col in ["open", "high", "low", "close", "oi"]:
        if col in df.columns:
            df = df[(df[col] > 0) & (df[col] < df[col].quantile(0.999))]

    if "return" in df.columns:
        df["return"] = df["return"].clip(-0.1, 0.1)

    df = df.fillna(method="ffill").fillna(method="bfill")

    df = df.drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    return df


def load_futures_data(folder_path):
    """Load and concatenate all Nifty futures CSVs."""
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dfs = []
    for f in all_files:
        if 'empty' in f:
            continue
        temp = pd.read_csv(os.path.join(folder_path, f), low_memory=False)
        temp.columns = [c.strip().lower() for c in temp.columns]
        temp.rename(columns={"datetime": "timestamp"}, inplace=True)
        temp["timestamp"] = pd.to_datetime(temp["timestamp"], errors="coerce")
        dfs.append(temp)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # ✅ Ensure consistent naming
    if "open_interest" in df.columns:
        df.rename(columns={"open_interest": "oi"}, inplace=True)

    return df

def load_options_data(folder_path):
    """Load and merge all call/put option chain batch CSVs."""
    all_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dfs = []
    for f in all_files:
        temp = pd.read_csv(os.path.join(folder_path, f))
        temp.columns = [c.strip().lower() for c in temp.columns]
        temp.rename(columns={"datetime": "timestamp"}, inplace=True)
        temp["timestamp"] = pd.to_datetime(temp["timestamp"], errors="coerce")
        dfs.append(temp)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    # Compute total open interest (calls + puts)
    if "call_open_interest" in df.columns and "put_open_interest" in df.columns:
        df["total_oi"] = df["call_open_interest"] + df["put_open_interest"]
    elif "open_interest" in df.columns:
        df["total_oi"] = df["open_interest"]
    else:
        df["total_oi"] = np.nan

    df["total_OI_change"] = df["total_oi"].diff().fillna(0)
    return df


# ==========================================================
# 3️⃣ Feature Engineering
# ==========================================================
def generate_features(df):
    """Generate EMAs, volatility, and OI-based features."""
    df = df.copy()
    df = df.sort_values("timestamp")

    df["EMA_5"] = df["close"].ewm(span=5, adjust=False).mean()
    df["EMA_15"] = df["close"].ewm(span=15, adjust=False).mean()
    df["ema_signal"] = 0
    df.loc[df["EMA_5"] > df["EMA_15"], "ema_signal"] = 1
    df.loc[df["EMA_5"] < df["EMA_15"], "ema_signal"] = -1

    df["volatility"] = df["close"].pct_change().rolling(10).std().fillna(0)
    df["mom_30"] = df["close"].pct_change(30).fillna(0)
    df["ATR_14"] = (df["high"] - df["low"]).rolling(14).mean().fillna(0)

    if "oi" not in df.columns:
        df["oi"] = np.random.randint(100000, 200000, len(df))
    df["total_OI_change"] = df["oi"].diff().fillna(0)
    return df


# ==========================================================
# 4️⃣ Full Data Pipeline
# ==========================================================
def run_data_pipeline(
    equity_folder="nifty_equity_data",
    futures_folder="nifty_futures_data",
    options_folder="Options data",
    out_folder="outputs",
    output_file="nifty_regime_hmm.csv"
    ):
    os.makedirs(out_folder, exist_ok=True)
    print("🚀 Starting complete data pipeline...")

    index_df = load_equity_data(equity_folder)
    fut_df = load_futures_data(futures_folder)
    oc_df = load_options_data(options_folder)

    # 🩹 Fallbacks if missing data
    if index_df.empty:
        print("⚠️ Equity data missing — generating synthetic fallback.")
        timestamps = pd.date_range(datetime.now() - timedelta(days=1), periods=200, freq="5min")
        index_df = pd.DataFrame({
            "timestamp": timestamps,
            "open": 22000 + np.random.randn(len(timestamps)) * 10,
            "high": 22000 + np.random.randn(len(timestamps)) * 15,
            "low": 22000 + np.random.randn(len(timestamps)) * 15,
            "close": 22000 + np.random.randn(len(timestamps)) * 10,
            "volume": np.random.randint(1000, 5000, len(timestamps)),
        })

    if fut_df.empty:
        print("⚠️ Futures data missing — generating synthetic fallback.")
        fut_df = index_df.copy()
        fut_df["oi"] = np.random.randint(100000, 200000, len(fut_df))

    # 🔄 Merge index + futures
    print("🔄 Merging index and futures data...")
    merged = pd.merge_asof(
        index_df.sort_values("timestamp"),
        fut_df.sort_values("timestamp"),
        on="timestamp",
        suffixes=("", "_fut")
    )

    # 🧮 Compute OI change
    if "oi_fut" in merged.columns:
        merged.rename(columns={"oi_fut": "oi"}, inplace=True)
    merged["oi"] = pd.to_numeric(merged["oi"], errors="coerce").fillna(method="ffill").fillna(0)
    merged["total_OI_change"] = merged["oi"].diff().fillna(0)
    merged.to_csv("merged_data.csv")
    # 🧠 Generate features and detect regimes
    features_df = generate_features(merged)
    hmm_df = detect_regimes(features_df, n_states=3)

    # 💾 Save outputs
    paths = {
        "index": os.path.join(out_folder, "index_ohlcv_5min.csv"),
        "futures": os.path.join(out_folder, "futures_ohlcv.csv"),
        "option_chain": os.path.join(out_folder, "option_chain.csv"),
        "regimes": output_file,
        "another_regime_path":os.path.join(out_folder, output_file),
    }

    index_df.to_csv(paths["index"], index=False)
    fut_df.to_csv(paths["futures"], index=False)
    oc_df.to_csv(paths["option_chain"], index=False)
    hmm_df = clean_final_dataset(hmm_df)
    hmm_df.to_csv(paths["another_regime_path"], index=False)
    hmm_df.to_csv('nifty_regime_hmm.csv', index=False)
    
    print("💾 All files saved successfully to:", out_folder)
    return hmm_df


# if __name__ == "__main__":
#     run_data_pipeline()
