import numpy as np
import pandas as pd
import requests
import scipy.stats as stats
from arch import arch_model
from tqdm import tqdm
import json
import warnings
warnings.filterwarnings('ignore')

def get_binance_data(symbol="BTCUSDT", interval="1h", limit=1000):
    url = "https://data-api.binance.vision/api/v3/klines"
    
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params)
    r.raise_for_status()
    data2 = r.json()
    
    params["endTime"] = data2[0][0] - 1
    r = requests.get(url, params=params)
    r.raise_for_status()
    data1 = r.json()
    
    data = data1 + data2
    
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

def backtest_confidence_intervals(prices, test_size=720, window_size=500):
    log_ret = np.log(prices / prices.shift(1)).dropna()
    res_li  = []
    
    start_idx = len(prices) - test_size - 1
    
    # We will simulate 720 bars. To speed it up for testing, we can limit n_sims to 5000 
    # but the assignment asks for 10,000 in original. Let's stick to 10000.
    for i in tqdm(range(start_idx, len(prices) - 1)):
        # STRICT NO PEEKING: Use data up to index i
        prices_train = prices.iloc[i - window_size + 1 : i + 1]
        train_ret = log_ret.iloc[i - window_size + 1 : i + 1]
        
        am  = arch_model(train_ret * 100, vol='FIGARCH', p=1, o=0, q=1, dist='studentst')
        res = am.fit(disp='off')
        
        sigma_fig = res.conditional_volatility / 100
        resid = (train_ret * 100 - res.params['mu']) / res.conditional_volatility
        
        nu_bt = max(4, stats.t.fit(resid, floc=0, fscale=1)[0])
        
        H_bt = rolling_entropy(resid)
        M_bt = train_ret.abs().rolling(60).mean()
        
        redundancy_bt = 1 + 0.1 * np.log1p(prices_train.rolling(5).var() / prices_train.rolling(20).var())
        info_filter_bt = (H_bt > H_bt.mean()).astype(float)
        
        H_bt_clean = H_bt.dropna()
        M_bt_clean = M_bt.dropna()
        redundancy_bt_clean = redundancy_bt.dropna()
        info_filter_bt_clean = info_filter_bt.dropna()
        
        sigma_bt = sigma_fig
        bar_sigma2_bt = (sigma_bt**2).mean()
        S0_bt = prices_train.iloc[-1]
        
        H_max = H_bt_clean.max() if not H_bt_clean.empty else 1.0
        M_max = M_bt_clean.max() if not M_bt_clean.empty else 1.0
        α0, δ0 = 0.5, 0.3
        if α0 * H_max + δ0 * M_max >= 1:
            fac = 0.95 / (α0 * H_max + δ0 * M_max)
            α0 *= fac
            δ0 *= fac
        base_params = {'alpha': α0, 'delta': δ0, 'gamma': 0.2, 'kappa': 0.1, 'eta': 1e-3}
        
        paths_bt = simulate_mc(S0_bt, train_ret.mean(),
                               sigma_bt, H_bt_clean, M_bt_clean, 
                               redundancy_bt_clean, info_filter_bt_clean, nu_bt,
                               bar_sigma2_bt, base_params,
                               n_sims=10000, n_days=1)
        
        S_t1 = paths_bt[:, 1]
        low95, high95 = np.percentile(S_t1, [2.5, 97.5])
        
        actual = prices.iloc[i + 1]
        width95  = high95 - low95
        alpha = 0.05
        winkler = (width95 + (2/alpha)*(low95-actual)) if actual < low95 else \
                  (width95 + (2/alpha)*(actual-high95)) if actual > high95 else \
                  width95
                  
        res_li.append({
            'timestamp': str(prices.index[i + 1]),
            'actual': float(actual),
            'low_95': float(low95), 
            'high_95': float(high95),
            'coverage_95': int(low95 <= actual <= high95),
            'width_95': float(width95),
            'winkler': float(winkler)
        })
        
    return res_li

if __name__ == "__main__":
    print("Fetching Binance data...")
    prices = get_binance_data()
    print(f"Got {len(prices)} bars.")
    
    print("Starting backtest for 720 bars...")
    results = backtest_confidence_intervals(prices, test_size=720, window_size=500)
    
    print("Writing results to backtest_results.jsonl...")
    with open("backtest_results.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
            
    df_res = pd.DataFrame(results)
    print("------------------------------------------")
    print(f"Couverture 95% (Coverage) : {df_res['coverage_95'].mean():.4f}")
    print(f"Largeur moyenne (Avg Width): {df_res['width_95'].mean():.2f}")
    print(f"Score de Winkler (Winkler) : {df_res['winkler'].mean():.2f}")
    print("------------------------------------------")
    print("Done!")
