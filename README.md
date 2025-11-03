# 📊 NIFTY Regime Detection & Strategy Dashboard

A complete machine-learning-driven trading research project that combines **financial market data**, **statistical modeling**, and **deep learning** to understand and trade NIFTY market regimes.  
The entire process — from **data extraction** to **LSTM-based prediction** — is automated and visualized through a **Streamlit dashboard**.

---

## 🧠 What This Project Does

The goal of this project is to identify how the NIFTY index behaves under different market conditions and build an intelligent system that can:
1. Detect market **regimes** (bullish, bearish, sideways) using a **Hidden Markov Model (HMM)**  
2. Train an **XGBoost** classifier to predict short-term moves based on engineered features  
3. Train an **LSTM** model to capture time-series dependencies and improve prediction accuracy  
4. Simulate a **backtested trading strategy** and visualize results interactively  

Everything — including feature extraction, model training, evaluation, and visualization — runs through a single streamlined pipeline.

---

## ⚙️ Key Components

| Component | File | Description |
|------------|------|-------------|
| **Data Extraction** | `data_extraction.py` | Downloads and cleans NIFTY OHLC data, adds EMAs and technical indicators |
| **Modeling & Strategy** | `strategy_pipeline.py` | Runs HMM, XGBoost, and LSTM models; performs backtesting; saves metrics and plots |
| **Dashboard** | `streamlit_run.py` | Streamlit app for exploring market regimes, model outputs, and trading results |

---

## 🧩 Tools and Libraries

- **Python 3.10+**
- **pandas, numpy** – data handling  
- **hmmlearn** – regime detection  
- **xgboost** – feature-based classification  
- **tensorflow / keras** – LSTM model  
- **plotly, seaborn, matplotlib** – interactive charts  
- **streamlit** – dashboard interface  

---

## 🪜 How to Set Up and Run

### 1️⃣ Clone the repository
```bash
Clone the reqpository
cd nifty-regime-detection
install required libraries from requirement.txt

First Click on "Run Data Extraction" Button to load data then Click on "Run Full Strategy Pipeline" to run on latest extracted data