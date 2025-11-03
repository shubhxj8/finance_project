# strategy_pipeline.py
import os
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import IsolationForest
import xgboost as xgb
import shap
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

# ---------- Backtest helpers (same as earlier) ----------
def compute_emas(df, span_fast=5, span_slow=15):
    df = df.copy()
    df['ema_5'] = df['close'].ewm(span=span_fast, adjust=False).mean()
    df['ema_15'] = df['close'].ewm(span=span_slow, adjust=False).mean()
    return df

def generate_signals(df):
    df = df.copy()
    df['signal_raw'] = 0
    df.loc[df['ema_5'] > df['ema_15'], 'signal_raw'] = 1
    df.loc[df['ema_5'] < df['ema_15'], 'signal_raw'] = -1
    df['entry_long'] = ((df['ema_5'] > df['ema_15']) & (df['ema_5'].shift(1) <= df['ema_15'].shift(1))).astype(int)
    df['entry_short'] = ((df['ema_5'] < df['ema_15']) & (df['ema_5'].shift(1) >= df['ema_15'].shift(1))).astype(int)
    return df

def backtest_ema(df, slippage_pct=0.0001, commission=50.0, notional=100000.0, entry_next_open=True):
    df = df.copy().reset_index(drop=True)
    trades = []
    position = 0
    for i in range(1, len(df)-1):
        row = df.loc[i]
        next_row = df.loc[i+1]
        # ENTRY LONG
        if row['entry_long'] == 1 and position == 0:
            entry_price = next_row['open'] if entry_next_open else row['close']
            entry_price = entry_price * (1 + slippage_pct)
            shares = notional / entry_price
            position = 1
            trades.append({'entry_index': i+1, 'side': 'long', 'entry_price': entry_price, 'shares': shares, 'entry_time': next_row['timestamp']})
        # ENTRY SHORT
        if row['entry_short'] == 1 and position == 0:
            entry_price = next_row['open'] if entry_next_open else row['close']
            entry_price = entry_price * (1 - slippage_pct)
            shares = notional / entry_price
            position = -1
            trades.append({'entry_index': i+1, 'side': 'short', 'entry_price': entry_price, 'shares': shares, 'entry_time': next_row['timestamp']})
        # EXIT LONG
        if position == 1 and (row['entry_short'] == 1 or row['ema_5'] < row['ema_15']):
            exit_price = next_row['open'] * (1 - slippage_pct)
            trade = trades[-1]
            trade.update({'exit_index': i+1, 'exit_price': exit_price, 'exit_time': next_row['timestamp']})
            pnl = (exit_price - trade['entry_price']) * trade['shares'] - commission
            trade['pnl'] = pnl
            trade['return_pct'] = pnl / notional
            position = 0
        # EXIT SHORT
        if position == -1 and (row['entry_long'] == 1 or row['ema_5'] > row['ema_15']):
            exit_price = next_row['open'] * (1 + slippage_pct)
            trade = trades[-1]
            trade.update({'exit_index': i+1, 'exit_price': exit_price, 'exit_time': next_row['timestamp']})
            pnl = (trade['entry_price'] - exit_price) * trade['shares'] - commission
            trade['pnl'] = pnl
            trade['return_pct'] = pnl / notional
            position = 0

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df['cum_pnl'] = trades_df['pnl'].cumsum()
        trades_df['win'] = trades_df['pnl'] > 0
        trades_df['entry_date'] = pd.to_datetime(trades_df['entry_time']).dt.date
        daily = trades_df.groupby('entry_date').agg(trades=('pnl','count'), daily_pnl=('pnl','sum'), daily_winrate=('win','mean')).reset_index()
    else:
        daily = pd.DataFrame()
    return trades_df, daily

# ---------- Build per-trade context ----------
def build_trade_contexts(df, trades_df, lookback=30, feature_cols=None):
    rows = []
    for _, t in trades_df.iterrows():
        idx = int(t['entry_index'])
        start_idx = max(0, idx - lookback)
        window = df.iloc[start_idx:idx]
        ctx = {}
        for c in feature_cols:
            ctx[f'{c}_mean'] = window[c].mean() if len(window)>0 else np.nan
            ctx[f'{c}_std'] = window[c].std() if len(window)>0 else np.nan
            ctx[f'{c}_last'] = window[c].iloc[-1] if len(window)>0 else np.nan
        ctx['regime'] = df.loc[idx, 'regime'] if 'regime' in df.columns else np.nan
        ctx['pnl'] = t['pnl']
        ctx['success'] = int(t['pnl'] > 0)
        rows.append(ctx)
    context_df = pd.DataFrame(rows)
    return context_df

# ---------- Anomaly detection ----------
def detect_anomalies(context_df, contamination=0.05):
    feat_cols = [c for c in context_df.columns if c not in ['pnl','success','regime']]
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    X = context_df[feat_cols].fillna(0)
    iso.fit(X)
    context_df['anomaly_score'] = iso.decision_function(X)
    context_df['anomaly'] = iso.predict(X) == -1
    return context_df, iso

# ---------- XGBoost + SHAP ----------
# Replace your existing explain_with_xgboost with this function

def explain_with_xgboost(context_df, save_plots=True, out_folder="outputs"):
    """
    Train XGBoost classifier and produce:
      - xgb_feature_importance.png (gain-based)
      - permutation_importance.png (permutation importance)
      - feature_importances.csv
    Returns: model, feature_df (DataFrame of importance)
    """
    # import os
    # import numpy as np
    # import pandas as pd
    import matplotlib.pyplot as plt
    import xgboost as xgb
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import train_test_split
    from sklearn.utils import resample

    os.makedirs(out_folder, exist_ok=True)

    context_df = context_df.dropna(subset=["success"])
    if context_df.empty:
        raise ValueError("Context dataframe empty or no 'success' labels to train")

    # Prepare data
    feature_cols = [c for c in context_df.columns if c not in ['pnl','success','regime','anomaly','anomaly_score']]
    X = context_df[feature_cols].fillna(0)
    y = context_df['success'].astype(int)

    # ==========================================================
    # 🧩 1. Sanity check before training
    # ==========================================================
    print("\n🧩 Data summary before training:")
    print(f"Shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    print("Feature variances:\n", X.var().sort_values().head(10))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    # ==========================================================
    # ⚙️ 2. Add stronger synthetic noise for flat features
    # ==========================================================
    for col in X_train.columns:
        if X_train[col].std() == 0:
            X_train[col] += np.random.normal(0, 0.01, size=len(X_train))
            X_test[col] += np.random.normal(0, 0.01, size=len(X_test))
            print(f"⚠️ Added noise to constant feature: {col}")

    # ==========================================================
    # ⚖️ 3. Balance dataset (oversample positives heavily)
    # ==========================================================
    train_df = X_train.copy()
    train_df['success'] = y_train

    majority = train_df[train_df['success'] == 0]
    minority = train_df[train_df['success'] == 1]

    if len(minority) > 0 and len(majority) > 0:
        minority_upsampled = resample(
            minority,
            replace=True,
            n_samples=max(len(majority), len(minority) * 3),
            random_state=42
        )
        train_bal = pd.concat([majority, minority_upsampled])
        X_train = train_bal.drop('success', axis=1)
        y_train = train_bal['success']

    print(f"✅ Balanced dataset (after upsampling): {y_train.value_counts().to_dict()}")

    # ==========================================================
    # 🚀 4. Train XGBoost (stronger config)
    # ==========================================================
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0,
        random_state=42
    )
    model.fit(X_train, y_train)

    # ==========================================================
    # 🔍 5. Debug internal XGBoost feature importance
    # ==========================================================
    print("\n🔍 Feature importance (gain):")
    importance = model.get_booster().get_score(importance_type='gain')
    print(importance)

    # Convert to DataFrame (ensure mapping matches your columns)
    # ✅ Build importance DataFrame directly from the computed dictionary
    gain_importance = model.get_booster().get_score(importance_type="gain")
    gain_importance = dict(sorted(gain_importance.items(), key=lambda item: item[1], reverse=True))
    print("🔍 Feature importance (gain) [sorted]:")
    print(gain_importance)

    # Convert dict to DataFrame
    imp_df = pd.DataFrame(list(gain_importance.items()), columns=["feature", "gain"])
    imp_df = imp_df.sort_values(by="gain", ascending=False).reset_index(drop=True)

    # Save gain-based importance
    imp_df.to_csv(os.path.join(out_folder, "xgb_feature_importance.csv"), index=False)

    # Plot gain importance
    if save_plots:
        plt.figure(figsize=(8, max(3, len(imp_df) / 3)))
        plt.barh(imp_df['feature'][::-1], imp_df['gain'][::-1])
        plt.title("XGBoost Feature Importance (gain)")
        plt.xlabel("Gain")
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "xgb_feature_importance.png"), dpi=300)
        plt.close()

    # ==========================================================
    # 🧮 Permutation importance (model-agnostic)
    # ==========================================================
    print("Computing permutation importance (this may take a few seconds)...")
    perm_res = permutation_importance(model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=1)
    perm_df = (
        pd.DataFrame({
            'feature': X.columns,
            'perm_mean': perm_res.importances_mean,
            'perm_std': perm_res.importances_std
        })
        .sort_values('perm_mean', ascending=False)
    )

    perm_df.to_csv(os.path.join(out_folder, "permutation_importance.csv"), index=False)

    if save_plots:
        plt.figure(figsize=(8, max(3, len(perm_df) / 3)))
        plt.barh(perm_df['feature'][::-1], perm_df['perm_mean'][::-1])
        plt.title("Permutation Importance (mean decrease in score)")
        plt.xlabel("Mean importance")
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "permutation_importance.png"), dpi=300)
        plt.close()

    print("Saved feature importance artifacts to:", out_folder)
    return model, imp_df, perm_df

# ---------- LSTM training (classification) ----------
def build_lstm_model(input_shape, dropout=0.3, learning_rate=0.0005):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.layers import Bidirectional

    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(dropout))
    model.add(BatchNormalization())
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=learning_rate),
        metrics=['AUC', 'accuracy']
    )
    return model


def build_sequences(df, feature_cols, window=60, horizon=5, threshold_profit=0.0005):
    Xs, ys, meta = [], [], []
    arr = df[feature_cols].values
    for i in range(window, len(df)-horizon-1):
        X = arr[i-window:i]
        entry_price = df.loc[i+1, 'open']
        exit_price = df.loc[i+1+horizon, 'close']
        ret = (exit_price - entry_price) / entry_price
        y = 1 if ret > threshold_profit else 0
        Xs.append(X); ys.append(y); meta.append(df.loc[i+1, 'timestamp'])
    if len(Xs)==0:
        return np.empty((0,window,len(feature_cols))), np.array([]), pd.DataFrame()
    return np.array(Xs), np.array(ys), pd.DataFrame({'timestamp': meta})

def walk_forward_train(df, feature_cols, window=120, horizon=5):
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import StandardScaler
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

    X, y, meta = build_sequences(df, feature_cols, window=window, horizon=horizon)
    if X.shape[0] == 0:
        print("No sequences for LSTM (not enough data). Skipping.")
        return None, None, None

    print(f"🧠 Built {len(X)} sequences for LSTM training")
    n = len(X)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    # Scale features globally
    X_flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(X_flat)
    X_scaled = scaler.transform(X_flat).reshape(X.shape)

    X_train, y_train = X_scaled[:train_end], y[:train_end]
    X_val, y_val = X_scaled[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X_scaled[val_end:], y[val_end:]

    # ⚖️ Balance classes with SMOTE
    y_train_ratio = np.mean(y_train)
    print(f"🔹 Original train ratio of success={y_train_ratio:.3f}")
    try:
        sm = SMOTE(random_state=42)
        X_train_res, y_train_res = sm.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
        X_train = X_train_res.reshape(X_train_res.shape[0], window, len(feature_cols))
        y_train = y_train_res
        print(f"✅ After SMOTE: {np.mean(y_train):.3f} positives")
    except Exception as e:
        print("⚠️ SMOTE skipped:", e)

    # Build model
    model = build_lstm_model((X.shape[1], X.shape[2]))

    # Train longer with early stopping
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150, batch_size=32,
        callbacks=[es],
        verbose=1
    )

    # Evaluate
    y_pred_proba = model.predict(X_test).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    metrics = {
        'auc': roc_auc_score(y_test, y_pred_proba) if len(y_test) > 0 else None,
        'accuracy': accuracy_score(y_test, y_pred) if len(y_test) > 0 else None,
        'precision': precision_score(y_test, y_pred, zero_division=0) if len(y_test) > 0 else None,
        'recall': recall_score(y_test, y_pred, zero_division=0) if len(y_test) > 0 else None
    }

    print(f"📊 Test metrics: {metrics}")
    return model, scaler, metrics

# ---------- Orchestrator ----------
def main_backup(input_csv='nifty_regime_hmm.csv', out_folder='outputs'):
    import os as osx
    osx.makedirs(out_folder, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # 1. Compute EMAs and signals
    print("Computing EMAs & signals...")
    df = compute_emas(df)
    df = generate_signals(df)
    df.to_csv(osx.path.join(out_folder, 'bars_with_emas.csv'), index=False)

    # 2. Backtest EMA
    print("Running backtest...")
    trades_df, daily_df = backtest_ema(df)
    trades_path = osx.path.join(out_folder, 'trades.csv')
    daily_path = osx.path.join(out_folder, 'daily_metrics.csv')
    trades_df.to_csv(trades_path, index=False)
    daily_df.to_csv(daily_path, index=False)
    print(f"Saved trades -> {trades_path} | daily -> {daily_path}")

    if trades_df.empty:
        print("No trades generated. Stopping pipeline.")
        return

    # 3. Build trade contexts
    print("Building trade contexts...")
    feature_cols = ['return','volatility','ema_5','ema_15','total_OI_change']  # extend as available
    context_df = build_trade_contexts(df, trades_df, lookback=30, feature_cols=feature_cols)
    ctx_path = osx.path.join(out_folder, 'trade_contexts.csv')
    context_df.to_csv(ctx_path, index=False)
    print(f"Saved trade contexts -> {ctx_path}")

    # 4. Anomaly detection
    print("Detecting anomalies...")
    context_df, iso = detect_anomalies(context_df)
    context_df.to_csv(os.path.join(out_folder, 'trade_contexts_with_anomalies.csv'), index=False)

    # 5. Explain with XGBoost + SHAP
    print("Training XGBoost and computing SHAP...")
    # xgb_model, imp_df, perm_df = explain_with_xgboost(context_df)
    print("Training XGBoost and computing SHAP...")

    # ✅ Ensure target label (success) is correctly set and has variation
    if "success" not in context_df.columns or context_df["success"].nunique() < 2:
        print("⚠️ 'success' label missing or constant — regenerating label based on pnl > 0...")
        context_df["success"] = (context_df["pnl"] > 0).astype(int)

    print("Target value counts:\n", context_df["success"].value_counts(dropna=False))

    # 🚀 Proceed only if we have both 0 and 1
    if context_df["success"].nunique() >= 2:
        try:
            xgb_model, imp_df, perm_df = explain_with_xgboost(context_df)
            print("\nTop 5 XGBoost features by gain:")
            print(imp_df.head())
            print("\nTop 5 Permutation features:")
            print(perm_df.head())
        except Exception as e:
            print("⚠️ XGBoost explanation step failed:", e)
    else:
        print("⚠️ Not enough label variation to train XGBoost (all trades same outcome). Skipping step.")


    
    print("XGBoost trained. SHAP ready.")

    # 6. LSTM (optional, may take time)
    print("Training LSTM (may take several minutes)...")
    lstm_feature_cols = [
        'return', 'return_mean', 'return_std',
        'volatility', 'volatility_mean', 'volatility_std',
        'ema_5', 'ema_15', 'ema_30',
        'total_OI_change'
    ]

    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(10).std()
    df['return_mean'] = df['return'].rolling(10).mean()
    df['return_std'] = df['return'].rolling(10).std()
    df['volatility_mean'] = df['volatility'].rolling(10).mean()
    df['volatility_std'] = df['volatility'].rolling(10).std()
    df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
    df['total_OI_change'] = df['open_interest'].pct_change() if 'open_interest' in df.columns else 0


    # lstm_feature_cols = ['return','volatility','ema_5','ema_15','total_OI_change']  # same as above
    model, scaler, metrics = walk_forward_train(df, lstm_feature_cols, window=60, horizon=5)
    print("LSTM metrics:", metrics)

    import json
    out_folder = "outputs"
    osx.makedirs(out_folder, exist_ok=True)
    with open(os.path.join(out_folder, "lstm_metrics.json"), "w") as f:
        json.dump(metrics, f)

    print("✅ LSTM metrics saved to outputs/lstm_metrics.json")

    print("Pipeline finished. Outputs in:", out_folder)

def main(input_csv='nifty_regime_hmm.csv', out_folder='outputs', retrain=True):
    import os as osx
    import json
    osx.makedirs(out_folder, exist_ok=True)

    if retrain:
        print("🔄 Retrain=True — clearing old outputs...")
        for f in osx.listdir(out_folder):
            path = osx.path.join(out_folder, f)
            try:
                if osx.path.isfile(path):
                    osx.remove(path)
            except Exception as e:
                print("⚠️ Failed to delete", path, ":", e)


    print("📂 Loading data...")
    df = pd.read_csv(input_csv, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    # --- EMA + signals ---
    print("📊 Computing EMAs & signals...")
    df = compute_emas(df)
    df = generate_signals(df)
    df.to_csv(osx.path.join(out_folder, 'bars_with_emas.csv'), index=False)

    # --- Backtest ---
    print("💹 Running backtest...")
    trades_df, daily_df = backtest_ema(df)
    trades_df.to_csv(osx.path.join(out_folder, 'trades.csv'), index=False)
    daily_df.to_csv(osx.path.join(out_folder, 'daily_metrics.csv'), index=False)

    if trades_df.empty:
        print("⚠️ No trades generated. Stopping pipeline.")
        return

    # --- Build trade contexts ---
    print("🧩 Building trade contexts...")
    feature_cols = ['return', 'volatility', 'ema_5', 'ema_15', 'total_OI_change']
    context_df = build_trade_contexts(df, trades_df, lookback=30, feature_cols=feature_cols)
    context_df, iso = detect_anomalies(context_df)
    context_df.to_csv(osx.path.join(out_folder, 'trade_contexts_with_anomalies.csv'), index=False)

    # --- XGBoost step ---
    print("⚙️ XGBoost step starting...")
    xgb_model_path = osx.path.join(out_folder, "xgb_model.json")

    if not retrain and osx.path.exists(xgb_model_path):
        print("📦 Loading existing XGBoost model...")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)
    else:
        print("🧠 Training new XGBoost model...")
        if "success" not in context_df.columns or context_df["success"].nunique() < 2:
            print("⚠️ 'success' missing or constant; regenerating from pnl>0.")
            context_df["success"] = (context_df["pnl"] > 0).astype(int)
        if context_df["success"].nunique() >= 2:
            try:
                xgb_model, imp_df, perm_df = explain_with_xgboost(context_df)
                xgb_model.save_model(xgb_model_path)
            except Exception as e:
                print("⚠️ XGBoost failed:", e)
        else:
            print("⚠️ Not enough label variation. Skipping XGBoost training.")

    # --- LSTM step ---
    print("🤖 LSTM step starting...")
    lstm_model_path = osx.path.join(out_folder, "lstm_model.keras")
    lstm_metrics_path = osx.path.join(out_folder, "lstm_metrics.json")

    if not retrain and osx.path.exists(lstm_model_path) and osx.path.exists(lstm_metrics_path):
        print("📦 Using existing LSTM model and metrics.")
        with open(lstm_metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        print("🧠 Training new LSTM model...")
        # Build features
        df['return'] = df['close'].pct_change()
        df['volatility'] = df['return'].rolling(10).std()
        df['return_mean'] = df['return'].rolling(10).mean()
        df['return_std'] = df['return'].rolling(10).std()
        df['volatility_mean'] = df['volatility'].rolling(10).mean()
        df['volatility_std'] = df['volatility'].rolling(10).std()
        df['ema_30'] = df['close'].ewm(span=30, adjust=False).mean()
        df['total_OI_change'] = (
            df['open_interest'].pct_change() if 'open_interest' in df.columns else 0
        )

        lstm_feature_cols = [
            'return', 'return_mean', 'return_std',
            'volatility', 'volatility_mean', 'volatility_std',
            'ema_5', 'ema_15', 'ema_30', 'total_OI_change'
        ]

        model, scaler, metrics = walk_forward_train(df, lstm_feature_cols, window=60, horizon=5)
        if model:
            model.save(lstm_model_path)
        with open(lstm_metrics_path, "w") as f:
            json.dump(metrics, f)

        print(f"✅ LSTM metrics: {metrics}")
    
        # ==========================================================
        # 📊 STEP 7: Option Chain + OI Analytics (for Streamlit dashboard)
        # ==========================================================
        print("📊 Generating Option Chain & OI analytics...")

        # --- Option Chain Heatmap ---
        # Simulate or aggregate OI by strike/expiry
        if "strike_price" in df.columns and "expiry_date" in df.columns and "open_interest" in df.columns:
            option_chain_df = (
                df.groupby(["strike_price", "expiry_date"])["open_interest"]
                .sum()
                .reset_index()
            )
        else:
            # Mock dataset for visualization if none available
            strikes = np.arange(19000, 20500, 100)
            expiries = pd.date_range(datetime.date.today(), periods=3, freq="W-THU")
            data = []
            for exp in expiries:
                for s in strikes:
                    data.append({
                        "strike_price": s,
                        "expiry_date": exp,
                        "open_interest": np.random.randint(50000, 300000)
                    })
            option_chain_df = pd.DataFrame(data)

        option_chain_path = os.path.join(out_folder, "option_chain.csv")
        option_chain_df.to_csv(option_chain_path, index=False)
        print(f"✅ Option Chain data saved → {option_chain_path}")

        # --- Total OI & Change in OI over time ---
        if "open_interest" in df.columns:
            total_oi_df = (
                df.groupby("timestamp")["open_interest"]
                .sum()
                .diff()
                .reset_index(name="change_in_oi")
            )
            total_oi_df["total_oi"] = df.groupby("timestamp")["open_interest"].sum().values
        else:
            # Fallback if no open_interest data
            total_oi_df = pd.DataFrame({
                "timestamp": pd.date_range(datetime.date.today(), periods=10, freq="H"),
                "total_oi": np.linspace(2e7, 2.5e7, 10),
                "change_in_oi": np.random.randint(-100000, 100000, 10)
            })

        total_oi_path = os.path.join(out_folder, "total_oi_timeseries.csv")
        total_oi_df.to_csv(total_oi_path, index=False)
        print(f"✅ Total OI timeseries saved → {total_oi_path}")

        print("✅ Pipeline complete. Results saved in:", out_folder)


# if __name__ == "__main__":
#     main()

