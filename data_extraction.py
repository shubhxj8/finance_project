import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from nsepython import nse_optionchain_scrapper, nse_fno
from tqdm import tqdm
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import warnings, os
warnings.filterwarnings("ignore")

# -----------------------------------------
# 1. Fetch NIFTY Index OHLC (5-min)
# -----------------------------------------
def get_index_ohlc(symbol="^NSEI", period="30d", interval="5m"):
    print("📊 Fetching NIFTY 5-min index data...")
    df = yf.download(symbol, period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]
    df.reset_index(inplace=True)
    df.rename(columns={
        'Datetime': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)
    df['symbol'] = 'NIFTY'
    print(f"✅ Index data fetched: {df.shape}")
    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']]


# -----------------------------------------
# 2. Fetch Futures OHLC
# -----------------------------------------
def get_futures_ohlc(symbol="NIFTY"):
    try:
        data = nse_fno(symbol)
        fut_data = []
        expiries = []
        for record in data["stocks"]:
            if "FUTIDX" in record["metadata"]["instrumentType"]:
                expiries.append(record["metadata"]["expiryDate"])

        expiries = sorted(set(expiries))
        if not expiries:
            raise ValueError("No futures data found in NSE response.")

        latest_expiry = expiries[0]
        print(f"📆 Using nearest expiry: {latest_expiry}")

        for record in data["stocks"]:
            md = record["metadata"]
            if md["instrumentType"] == "FUTIDX" and md["expiryDate"] == latest_expiry:
                trade = record["marketDeptOrderBook"]["tradeInfo"]
                fut_data.append({
                    "timestamp": pd.to_datetime(md["lastUpdateTime"]),
                    "symbol": md["symbol"],
                    "expiry": md["expiryDate"],
                    "open": trade["open"],
                    "high": trade["high"],
                    "low": trade["low"],
                    "close": trade["close"],
                    "volume": trade["quantityTraded"],
                    "oi": trade["openInterest"],
                })

        df = pd.DataFrame(fut_data)
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"✅ Futures data fetched: {df.shape}")
        return df

    except Exception as e:
        print(f"⚠️ NSE futures fetch failed: {e}")
        print("🧪 Simulating synthetic futures data for demo...")
        import random
        timestamps = pd.date_range(datetime.now() - timedelta(hours=4), periods=48, freq="5min")
        base_price = 22000
        simulated = {
            "timestamp": timestamps,
            "symbol": [symbol + "_FUT"] * len(timestamps),
            "open": [base_price + random.uniform(-10, 10) for _ in timestamps],
            "high": [base_price + random.uniform(10, 20) for _ in timestamps],
            "low": [base_price + random.uniform(-20, -10) for _ in timestamps],
            "close": [base_price + random.uniform(-5, 5) for _ in timestamps],
            "volume": [random.randint(1000, 5000) for _ in timestamps],
            "oi": [random.randint(100000, 200000) for _ in timestamps],
        }
        return pd.DataFrame(simulated)


# -----------------------------------------
# 3. Fetch Option Chain
# -----------------------------------------
def get_option_chain(symbol="NIFTY"):
    try:
        print("🧩 Fetching option chain...")
        data = nse_optionchain_scrapper(symbol)
        all_records = data['records']['data']
        oc_list = []
        for rec in all_records:
            for side in ['CE', 'PE']:
                if side in rec:
                    item = rec[side]
                    oc_list.append({
                        'timestamp': datetime.now(),
                        'strike_price': rec.get('strikePrice'),
                        'type': side,
                        'expiry_date': item.get('expiryDate'),
                        'last_price': item.get('lastPrice'),
                        'open_interest': item.get('openInterest'),
                        'change_in_oi': item.get('changeinOpenInterest'),
                        'implied_volatility': item.get('impliedVolatility'),
                        'underlying_value': data['records']['underlyingValue']
                    })
        print(f"✅ Option chain fetched: {len(oc_list)} records")
        return pd.DataFrame(oc_list)
    except Exception as e:
        print(f"⚠️ Option chain fetch failed: {e}")
        return pd.DataFrame()


# -----------------------------------------
# 4. Feature Engineering + HMM
# -----------------------------------------
def generate_features(df):
    print("⚙️ Generating features...")
    df = df.copy()
    df['return'] = df['close'].pct_change()
    n = len(df)
    win_short = max(5, min(30, n // 10))
    win_long = max(10, min(60, n // 5))
    df['volatility'] = df['return'].rolling(win_short).std()
    df['mom_30'] = df['close'] / df['close'].shift(win_short) - 1
    df['mom_60'] = df['close'] / df['close'].shift(win_long) - 1
    df['ATR_14'] = (df['high'] - df['low']).rolling(max(5, min(14, n // 15))).mean()
    df['return_skewness'] = df['return'].rolling(win_long).skew()
    df['total_OI_change'] = df['oi'].diff()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    print(f"✅ Features ready. Shape: {df.shape}")
    return df


def detect_regimes(df, n_states=3):
    print("🧠 Running HMM regime detection...")
    feature_cols = ['volatility', 'mom_30', 'ATR_14', 'total_OI_change']
    X = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
    X = StandardScaler().fit_transform(X)
    model = GaussianHMM(n_components=n_states, covariance_type='full', n_iter=200, random_state=42)
    model.fit(X)
    df['regime'] = model.predict(X)
    print(f"✅ Regime detection complete ({n_states} states).")
    return df


# -----------------------------------------
# 5. Run Entire Data Pipeline
# -----------------------------------------
def run_data_pipeline(out_folder="outputs", output_file="nifty_regime_hmm.csv"):
    os.makedirs(out_folder, exist_ok=True)

    index_df = get_index_ohlc()
    fut_df = get_futures_ohlc()
    oc_df = get_option_chain()

    # Merge
    print("🔄 Merging index and futures data...")

    # Ensure timestamps exist and are datetime
    if 'timestamp' not in index_df.columns or index_df.empty:
        raise ValueError("Index data missing or empty. Check Yahoo Finance fetch.")
    if 'timestamp' not in fut_df.columns or fut_df.empty:
        raise ValueError("Futures data missing or empty. Check NSE fetch.")

    index_df['timestamp'] = pd.to_datetime(index_df['timestamp'], errors='coerce').dt.tz_localize(None)
    fut_df['timestamp'] = pd.to_datetime(fut_df['timestamp'], errors='coerce').dt.tz_localize(None)

    # Drop any invalid timestamps
    index_df = index_df.dropna(subset=['timestamp']).sort_values('timestamp')
    fut_df = fut_df.dropna(subset=['timestamp']).sort_values('timestamp')

    # Merge safely
    merged = pd.merge_asof(index_df, fut_df, on='timestamp', direction='nearest', suffixes=('', '_fut'))

    print(f"✅ Merged dataset shape: {merged.shape}")

    merged = pd.merge_asof(index_df.sort_values('timestamp'),
                           fut_df.sort_values('timestamp'),
                           on='timestamp', suffixes=('', '_fut'))

    if 'oi_fut' in merged.columns:
        merged.rename(columns={'oi_fut': 'oi'}, inplace=True)
    merged['oi'] = pd.to_numeric(merged.get('oi', np.nan), errors='coerce').fillna(method='ffill')

    features_df = generate_features(merged)
    hmm_df = detect_regimes(features_df, n_states=3)

    # Save outputs in `outputs/`
    paths = {
        "index": os.path.join(out_folder, "index_ohlcv_5min.csv"),
        "futures": os.path.join(out_folder, "futures_ohlcv.csv"),
        "option_chain": os.path.join(out_folder, "option_chain.csv"),
        "regimes": os.path.join(out_folder, output_file)
    }
    index_df.to_csv(paths["index"], index=False)
    fut_df.to_csv(paths["futures"], index=False)
    oc_df.to_csv(paths["option_chain"], index=False)
    hmm_df.to_csv(paths["regimes"], index=False)

    print("💾 All files saved to:", out_folder)
    return hmm_df


if __name__ == "__main__":
    run_data_pipeline()
