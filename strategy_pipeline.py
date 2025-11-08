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
    from sklearn.ensemble import IsolationForest
    import numpy as np

    print("🚀 Detecting anomalies (dropping non-numeric columns)...")
    df = context_df.copy()

    # --- Drop all non-numeric columns automatically ---
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        print(f"⚠️ Dropping non-numeric columns before fitting IsolationForest: {non_numeric_cols}")
        df = df.drop(columns=non_numeric_cols)

    # --- Drop target-like columns too ---
    drop_cols = [c for c in ['anomaly', 'anomaly_reason', 'profit_anomaly', 'loss_anomaly'] if c in df.columns]
    if drop_cols:
        print(f"⚠️ Dropping derived anomaly columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    df = df.fillna(0)

    # --- Numeric features only ---
    feature_cols = [c for c in df.columns if c not in ['pnl', 'success', 'regime']]
    X = df[feature_cols].select_dtypes(include=[np.number])

    # --- Fit model ---
    iso = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    iso.fit(X)

    context_df['anomaly_score'] = iso.decision_function(X)
    context_df['anomaly'] = (iso.predict(X) == -1).astype(int)

    print("✅ Anomalies computed successfully.")
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
    # inside explain_with_xgboost()
    context_df = context_df.select_dtypes(include=[np.number])

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

from stable_baselines3 import PPO
from rl_environment import TradingEnv

from stable_baselines3.common.vec_env import DummyVecEnv

def train_rl_agent(df, model_path="outputs/ppo_trading_agent.zip"):
    print("🤖 Training RL trading agent...")
    env = DummyVecEnv([lambda: TradingEnv(df)])  # Multi-instance wrapper
    model = PPO("MlpPolicy", env, verbose=0, n_steps=512, batch_size=128, learning_rate=3e-4)
    model.learn(total_timesteps=50000)
    model.save(model_path)
    print(f"✅ RL model trained and saved: {model_path}")
    return model



def generate_rl_trades(df, model_path="outputs/ppo_trading_agent.zip"):
    from stable_baselines3 import PPO
    from rl_environment import TradingEnv
    import numpy as np

    print("🎯 Generating trades using RL agent...")

    try:
        model = PPO.load(model_path)
        env = TradingEnv(df)
        obs, _ = env.reset()
        actions = []
        balances = []

        for _ in range(len(df) - env.window - 1):
            action, _ = model.predict(obs, deterministic=True)

            # ✅ Convert NumPy array to int safely
            if isinstance(action, np.ndarray):
                action = int(action.item())

            obs, reward, done, _, _ = env.step(action)
            actions.append(action)
            balances.append(env.balance)
            if done:
                break

        # --- Safe padding ---
        df = df.copy()
        padded_actions = [np.nan] * env.window + actions
        padded_balances = [np.nan] * env.window + balances

        # ✅ Align lengths safely
        if len(padded_actions) < len(df):
            pad_len = len(df) - len(padded_actions)
            padded_actions += [np.nan] * pad_len
            padded_balances += [np.nan] * pad_len

        df["action"] = padded_actions[:len(df)]
        df["balance"] = padded_balances[:len(df)]

        # ✅ Map actions safely
        df["signal"] = df["action"].apply(
            lambda x: {0: "HOLD", 1: "BUY", 2: "SELL"}.get(int(x), np.nan)
            if pd.notnull(x) else np.nan
        )

        print("✅ RL trade signals generated successfully.")
        return df

    except Exception as e:
        print(f"⚠️ RL agent step failed: {e}")
        return df

# ---------- Orchestrator ----------
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

    split_idx = len(df) // 2
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)

    print("📊 Computing EMAs & signals...")
    train_df = compute_emas(train_df)
    train_df = generate_signals(train_df)
    test_df = compute_emas(test_df)
    test_df = generate_signals(test_df)

    # Save both sets
    train_df.to_csv(os.path.join(out_folder, "train_bars.csv"), index=False)
    test_df.to_csv(os.path.join(out_folder, "test_bars.csv"), index=False)

    df.to_csv(osx.path.join(out_folder, 'bars_with_emas.csv'), index=False)

    # --- Backtest ---
    print("💹 Running backtest...")
    # trades_df, daily_df = backtest_ema(df)
    # trades_df.to_csv(osx.path.join(out_folder, 'trades.csv'), index=False)
    # daily_df.to_csv(osx.path.join(out_folder, 'daily_metrics.csv'), index=False)

    print("💹 Running backtest on training data...")
    train_trades_df, train_daily_df = backtest_ema(train_df)
    train_trades_df.to_csv(os.path.join(out_folder, "train_trades.csv"), index=False)
    train_daily_df.to_csv(os.path.join(out_folder, "train_daily_metrics.csv"), index=False)

    if train_trades_df.empty:
        print("⚠️ No trades generated. Stopping pipeline.")
        return

    # --- Build trade contexts ---
    # print("🧩 Building trade contexts...")
    # feature_cols = ['return', 'volatility', 'ema_5', 'ema_15', 'total_OI_change']
    # context_df = build_trade_contexts(df, train_trades_df, lookback=30, feature_cols=feature_cols)

    print("🧩 Building trade contexts (train)...")
    feature_cols = ['return', 'volatility', 'ema_5', 'ema_15', 'total_OI_change']
    train_context_df  = build_trade_contexts(train_df, train_trades_df, lookback=30, feature_cols=feature_cols)
    # context_df, iso = detect_anomalies(context_df)

    train_context_df.to_csv(osx.path.join(out_folder, 'trade_contexts_with_anomalies.csv'), index=False)
        # --- Profit-based anomaly detection ---
    print("🚀 Detecting profit-based outperforming anomalies...")

    if 'pnl' in train_context_df .columns and not train_context_df ['pnl'].empty:
        mean_pnl = train_context_df ['pnl'].mean()
        std_pnl = train_context_df ['pnl'].std()

        train_context_df ['profit_anomaly'] = train_context_df ['pnl'] > mean_pnl + 2 * std_pnl
        train_context_df ['loss_anomaly'] = train_context_df ['pnl'] < mean_pnl - 2 * std_pnl

        train_context_df ['anomaly_reason'] = np.where(
            train_context_df ['profit_anomaly'], 'Outperforming trade',
            np.where(train_context_df ['loss_anomaly'], 'Underperforming trade', 'Normal')
        )

        if 'regime' in train_context_df.columns:
            train_context_df['regime_pnl_mean'] = train_context_df.groupby('regime')['pnl'].transform('mean')
            train_context_df['regime_pnl_std'] = train_context_df.groupby('regime')['pnl'].transform('std')
            train_context_df['regime_profit_anomaly'] = (
                train_context_df['pnl'] > train_context_df['regime_pnl_mean'] + 2 * train_context_df['regime_pnl_std']
            )

        print("✅ Profit-based anomalies detected and labeled.")
    else:
        print("⚠️ Skipping profit anomaly detection (no pnl data).")

    # Save updated context with anomalies
    train_context_df.to_csv(osx.path.join(out_folder, 'trade_contexts_with_anomalies.csv'), index=False)

    # --- XGBoost step ---
    print("⚙️ XGBoost step starting...")
    xgb_model_path = osx.path.join(out_folder, "xgb_model.json")

    if not retrain and osx.path.exists(xgb_model_path):
        print("📦 Loading existing XGBoost model...")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)
    else:
        
    # --- LSTM step ---
        context_df, iso = detect_anomalies(train_context_df)

        context_df.to_csv(osx.path.join(out_folder, 'trade_contexts_with_anomalies.csv'), index=False)
            # --- Profit-based anomaly detection ---
        print("🚀 Detecting profit-based outperforming anomalies...")

        if 'pnl' in context_df.columns and not context_df['pnl'].empty:
            mean_pnl = context_df['pnl'].mean()
            std_pnl = context_df['pnl'].std()

            context_df['profit_anomaly'] = context_df['pnl'] > mean_pnl + 2 * std_pnl
            context_df['loss_anomaly'] = context_df['pnl'] < mean_pnl - 2 * std_pnl

            context_df['anomaly_reason'] = np.where(
                context_df['profit_anomaly'], 'Outperforming trade',
                np.where(context_df['loss_anomaly'], 'Underperforming trade', 'Normal')
            )

            if 'regime' in context_df.columns:
                context_df['regime_pnl_mean'] = context_df.groupby('regime')['pnl'].transform('mean')
                context_df['regime_pnl_std'] = context_df.groupby('regime')['pnl'].transform('std')
                context_df['regime_profit_anomaly'] = (
                    context_df['pnl'] > context_df['regime_pnl_mean'] + 2 * context_df['regime_pnl_std']
                )

            print("✅ Profit-based anomalies detected and labeled.")
        else:
            print("⚠️ Skipping profit anomaly detection (no pnl data).")

        # Save updated context with anomalies
        context_df.to_csv(osx.path.join(out_folder, 'trade_contexts_with_anomalies.csv'), index=False)

        # --- XGBoost step ---
        print("⚙️ XGBoost step starting...")
        xgb_model_path = osx.path.join(out_folder, "xgb_model.json")
        print("🧠 Training new XGBoost model...")

        if "success" not in context_df.columns or context_df["success"].nunique() < 2:
            print("⚠️ 'success' missing or constant; regenerating from pnl>0.")
            context_df["success"] = (context_df["pnl"] > 0).astype(int)

        if context_df["success"].nunique() >= 2:
            try:
                # --- Clean non-numeric columns before XGBoost ---
                non_numeric_cols = context_df.select_dtypes(exclude=[np.number]).columns.tolist()
                if non_numeric_cols:
                    print(f"⚠️ Dropping non-numeric columns before XGBoost: {non_numeric_cols}")
                    context_df = context_df.drop(columns=non_numeric_cols)

                xgb_model, imp_df, perm_df = explain_with_xgboost(context_df)
                xgb_model.save_model(xgb_model_path)
                # ==========================================================
                # 📊 Save XGBoost evaluation metrics
                # ==========================================================
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                import json

                print("📈 Evaluating XGBoost performance...")

                # Get test split and predictions (reuse same data split as explain_with_xgboost)
                feature_cols = [c for c in context_df.columns if c not in ['pnl','success','regime','anomaly','anomaly_score']]
                X = context_df[feature_cols].fillna(0)
                y = context_df['success'].astype(int)

                # Use last 25% as test (same split logic)
                split = int(len(X) * 0.75)
                X_train, X_test = X.iloc[:split], X.iloc[split:]
                y_train, y_test = y.iloc[:split], y.iloc[split:]

                y_pred = xgb_model.predict(X_test)

                xgb_metrics = {
                    "accuracy": float(accuracy_score(y_test, y_pred)),
                    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                    "f1_score": float(f1_score(y_test, y_pred, zero_division=0))
                }

                metrics_path = os.path.join(out_folder, "xgb_metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(xgb_metrics, f, indent=4)

                print(f"✅ XGBoost metrics saved → {metrics_path}")
            except Exception as e:
                print("⚠️ XGBoost failed:", e)
        else:
            print("⚠️ Not enough label variation. Skipping XGBoost training.")

        # --- Reinforcement Learning step ---
        print("🤖 Training Reinforcement Learning (RL) agent...")
        try:
            rl_model_path = osx.path.join(out_folder, "ppo_trading_agent.zip")
            model = train_rl_agent(train_df, model_path=rl_model_path)
            df_rl = generate_rl_trades(test_df, model_path=rl_model_path)
            df_rl.to_csv(osx.path.join(out_folder, "rl_trades.csv"), index=False)
            print("✅ RL trades generated and saved → rl_trades.csv")
            # ==========================================================
            # 📊 Evaluate RL Agent Performance
            # ==========================================================
            print("📊 Evaluating RL agent performance...")

            try:
                total_rewards = []
                num_eval_episodes = 5

                env_eval = TradingEnv(test_df)
                for i in range(num_eval_episodes):
                    obs, _ = env_eval.reset()
                    episode_reward = 0.0
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, _, _ = env_eval.step(int(action))
                        episode_reward += reward
                    total_rewards.append(episode_reward)


                    for _ in range(len(test_df) - env_eval.window - 1):
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, done, _, _ = env_eval.step(int(action))
                        episode_reward += reward
                        if done:
                            break
                    total_rewards.append(episode_reward)

                rl_metrics = {
                    "average_reward": float(np.mean(total_rewards)),
                    "max_reward": float(np.max(total_rewards)),
                    "min_reward": float(np.min(total_rewards)),
                    "episodes": num_eval_episodes
                }

                metrics_path = os.path.join(out_folder, "rl_metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(rl_metrics, f, indent=4)

                print(f"✅ RL metrics saved → {metrics_path}")

            except Exception as e:
                print(f"⚠️ RL evaluation failed: {e}")


            if "pnl" in train_context_df.columns:
                train_context_df["rl_signal"] = df_rl["signal"]
                train_context_df.to_csv(osx.path.join(out_folder, "trade_contexts_with_anomalies.csv"), index=False)
        except Exception as e:
            print("⚠️ RL agent step failed:", e)

        print("💹 Running backtest on test data...")
        test_trades_df, test_daily_df = backtest_ema(test_df)
        test_trades_df.to_csv(os.path.join(out_folder, "test_trades.csv"), index=False)
        test_daily_df.to_csv(os.path.join(out_folder, "test_daily_metrics.csv"), index=False)


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

