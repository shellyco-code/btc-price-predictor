import streamlit as st
import pandas as pd
import numpy as np
import requests
import scipy.stats as stats
from arch import arch_model
import matplotlib.pyplot as plt
import json
import warnings
warnings.filterwarnings('ignore')

# --- Backend Helper Functions ---
@st.cache_data(ttl=60) # Cache for 60 seconds
def get_binance_data(symbol="BTCUSDT", interval="1h", limit=500):
    url = "https://data-api.binance.vision/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()
    
    df = pd.DataFrame(data, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['date'] = pd.to_datetime(df['close_time'], unit='ms')
    df['close'] = df['close'].astype(float)
    df.set_index('date', inplace=True)
    return df['close'].sort_index()

def rolling_entropy(x, window=60, bins=20):
    def ent(v):
        p, _ = np.histogram(v, bins=bins, density=True)
        p = p[p > 0]
        return -np.sum(p * np.log(p))
    return x.rolling(window).apply(ent, raw=True)

def update_params(p, sigma2, bar_sigma2, t):
    err = sigma2 - bar_sigma2
    lr  = p['eta'] / (1 + t**0.55)
    p['gamma'] = np.clip(p['gamma'] + lr * err, 0.01, 0.5)
    return p

def simulate_cyber_gbm(S0, mu, sigma_fig, H, M, redundancy, info_filter, nu,
                       params, bar_sigma2, n_steps, dt=1, eps=1e-6):
    S = np.zeros(n_steps + 1)
    V = np.zeros(n_steps + 1)
    S[0] = S0
    sigma2 = sigma_fig.iloc[-1] ** 2
    H_max = H.max() if H.max() > 0 else 1.0
    M_max = M.max() if M.max() > 0 else 1.0
    for t in range(1, n_steps + 1):
        current = -1
        H_val = min(H.iloc[current] / H_max, 1.0) if not H.empty else 0.5
        M_val = min(M.iloc[current] / M_max, 1.0) if not M.empty else 0.5
        crisis  = (H_val > 0.8) or (M_val > 0.8)
        delta_t = params['delta'] if crisis else 0.0
        sigma2 = (
            sigma_fig.iloc[current]**2 * (1 + params['alpha'] * H_val + delta_t * M_val)
            + params['gamma'] * (bar_sigma2 - sigma2)
        )
        r_val = max(1e-12, redundancy.iloc[current]) if not redundancy.empty else 1.0
        sigma2 *= r_val
        i_val = info_filter.iloc[current] if not info_filter.empty else 0.0
        sigma2 *= 1 + 0.5 * i_val
        sigma2 = max(eps, min(sigma2, 0.5))
        Z   = np.random.standard_t(nu) * np.sqrt((nu - 2) / nu)
        S[t]= S[t-1] * np.exp((mu - 0.5 * sigma2) * dt + np.sqrt(sigma2 * dt) * Z)
        V[t]= sigma2
        params = update_params(params, sigma2, bar_sigma2, t)
    return S, V

def simulate_mc(S0, mu, sigma_fig, H, M, redundancy, info_filter, nu, bar_sigma2, base_params,
                n_sims=10_000, n_days=1):
    out = np.zeros((n_sims, n_days + 1))
    for i in range(n_sims):
        paths, _ = simulate_cyber_gbm(
            S0, mu, sigma_fig, H, M, redundancy, info_filter, nu,
            base_params.copy(),
            bar_sigma2, n_days, dt=1
        )
        out[i] = paths
    return out

@st.cache_data(ttl=60)
def predict_next_hour(prices):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    train_ret = log_ret
    
    am  = arch_model(train_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
    res = am.fit(disp='off')
    
    sigma_fig = res.conditional_volatility / 100
    resid = (train_ret * 100 - res.params['mu']) / res.conditional_volatility
    nu_bt = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
    
    H_bt = rolling_entropy(resid)
    M_bt = train_ret.abs().rolling(60).mean()
    redundancy_bt = 1 + 0.1 * np.log1p(prices.rolling(5).var() / prices.rolling(20).var())
    info_filter_bt = (H_bt > H_bt.mean()).astype(float)
    
    H_bt_clean = H_bt.dropna()
    M_bt_clean = M_bt.dropna()
    redundancy_bt_clean = redundancy_bt.dropna()
    info_filter_bt_clean = info_filter_bt.dropna()
    
    bar_sigma2_bt = (sigma_fig**2).mean()
    S0_bt = prices.iloc[-1]
    
    H_max = H_bt_clean.max() if not H_bt_clean.empty else 1.0
    M_max = M_bt_clean.max() if not M_bt_clean.empty else 1.0
    α0, δ0 = 0.5, 0.3
    if α0 * H_max + δ0 * M_max >= 1:
        fac = 0.95 / (α0 * H_max + δ0 * M_max)
        α0 *= fac
        δ0 *= fac
    base_params = {'alpha': α0, 'delta': δ0, 'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3}
    
    paths_bt = simulate_mc(S0_bt, train_ret.mean(),
                           sigma_fig, H_bt_clean, M_bt_clean, 
                           redundancy_bt_clean, info_filter_bt_clean, nu_bt,
                           bar_sigma2_bt, base_params,
                           n_sims=10000, n_days=1)
    
    S_t1 = paths_bt[:, 1]
    low95, high95 = np.percentile(S_t1, [2.5, 97.5])
    return S0_bt, low95, high95

def get_backtest_metrics():
    # Attempt to load backtest results from Part A
    try:
        data = []
        with open("backtest_results.jsonl", "r") as f:
            for line in f:
                data.append(json.loads(line))
        df = pd.DataFrame(data)
        return df['coverage_95'].mean(), df['width_95'].mean(), df['winkler'].mean()
    except FileNotFoundError:
        return None, None, None

# --- UI Setup ---
st.set_page_config(page_title="BTC Price Predictor", layout="wide")

st.title("Bitcoin 1-Hour Price Range Predictor")
st.markdown("AlphaI × Polaris Challenge Submission")

# 1. Fetch Backtest Metrics
cov, width, winkler = get_backtest_metrics()
st.subheader("Part-A Backtest Metrics (30 Days)")
col1, col2, col3 = st.columns(3)
if cov is not None:
    col1.metric("Coverage (Target: ~0.95)", f"{cov:.4f}")
    col2.metric("Average Width", f"${width:,.2f}")
    col3.metric("Winkler Score", f"{winkler:,.2f}")
else:
    st.warning("backtest_results.jsonl not found. Run part_a.py first to generate metrics.")

# 2. Fetch Live Data & Predict
st.markdown("---")
st.subheader("Live Prediction")
with st.spinner("Fetching live data from Binance..."):
    prices = get_binance_data(limit=500)

with st.spinner("Running Geometric Brownian Motion simulation (10k paths)..."):
    current_price, low95, high95 = predict_next_hour(prices)

c1, c2, c3 = st.columns(3)
c1.metric("Current BTC Price", f"${current_price:,.2f}")
c2.metric("Predicted Next Hour Low", f"${low95:,.2f}")
c3.metric("Predicted Next Hour High", f"${high95:,.2f}")

# 3. Chart
st.markdown("### Last 50 Hours & Next Hour Prediction")
last_50 = prices.iloc[-50:]
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(last_50.index, last_50.values, label="BTC Price", color="blue")
ax.scatter(last_50.index[-1], last_50.values[-1], color="red", zorder=5)

# Calculate next hour time
next_hour_time = last_50.index[-1] + pd.Timedelta(hours=1)
ax.plot([last_50.index[-1], next_hour_time], [current_price, (low95+high95)/2], linestyle="--", color="gray", label="Trajectory")
ax.fill_between([last_50.index[-1], next_hour_time], 
                [current_price, low95], 
                [current_price, high95], 
                color="orange", alpha=0.3, label="95% Confidence Range")

ax.set_ylabel("Price (USDT)")
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend()
st.pyplot(fig)

# --- Part C: Persistence (Optional) ---
# For simplicity, we just save the latest prediction to a file and display history
st.markdown("---")
st.subheader("Prediction History (Part C)")

history_file = "prediction_history.jsonl"
new_pred = {
    "timestamp": str(next_hour_time),
    "current_price": current_price,
    "pred_low": low95,
    "pred_high": high95
}

# Load existing
history = []
try:
    with open(history_file, "r") as f:
        for line in f:
            history.append(json.loads(line))
except FileNotFoundError:
    pass

# Check if we already added a prediction for this exact next_hour_time
if not history or history[-1]["timestamp"] != str(next_hour_time):
    history.append(new_pred)
    with open(history_file, "a") as f:
        f.write(json.dumps(new_pred) + "\n")

if history:
    df_hist = pd.DataFrame(history)
    st.dataframe(df_hist)
