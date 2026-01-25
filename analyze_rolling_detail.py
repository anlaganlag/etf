
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.data_fetcher import DataFetcher
from config import config

# Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def analyze_rolling_t14():
    print("=== Analyzing Rolling Position Curve (T=14) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Setup Data & Universe
    # Load Strong List
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_curated = pd.read_excel(excel_path)
    df_curated.columns = df_curated.columns.str.strip()
    df_curated = df_curated.rename(columns={'symbol': 'etf_code', 'sec_name': 'etf_name', '主题': 'theme'})
    curated_codes = set(df_curated['etf_code'])
    
    # Load Prices
    print("[Data] Loading Prices...")
    all_codes = list(fetcher.get_all_etfs()['etf_code'])
    price_data = {}
    start_load = "2023-01-01"
    
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except: pass
            
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"  Matrix: {prices_df.shape}")
    
    # 2. Daily Returns Matrix (for fast simulation)
    # Return at index t is: (Price_t+1 - Price_t) / Price_t
    # aligned with date_t
    daily_rets = prices_df.pct_change(1).shift(-1) 
    
    # 3. Score Calculation
    print("[Score] Calculating Scores...")
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = config.SECTOR_TOP_N_THRESHOLD
    
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        ranks = prices_df.pct_change(period).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    
    # 4. Simulation T=14
    # We want to compare:
    # A. Periodic (Standard T=14, starting day 0)
    # B. Rolling (Average of 14 staggered periodic strategies)
    
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    # Pre-compute Picks for every day (Greedy optimization)
    # picks_cache[d] = list of codes to buy if we rebalanced on day d
    picks_cache = {}
    
    print("[Sim] Pre-computing daily picks...")
    for d in valid_dates:
        if d not in total_scores.index:
            picks_cache[d] = []
            continue
            
        s = total_scores.loc[d]
        r = r20_matrix.loc[d]
        metric = s * 10000 + r
        metric = metric.dropna()
        metric = metric[metric > -99999]
        sorted_codes = metric.sort_values(ascending=False).index
        # Filter curated
        top10 = [c for c in sorted_codes if c in curated_codes][:10]
        picks_cache[d] = top10

    print("[Sim] Simulating 14 Tranches...")
    
    T = 14
    tranche_curves = [] # List of numpy arrays
    
    for k in range(T):
        # Tranche k rebalances on day i if (i - k) % T == 0
        # Wait, if k=0, rebal at 0, 14...
        # If k=1, rebal at 1, 15...
        # On days before first rebalance (e.g. day 0 for k=5), 
        # we assume it bought on day 0? 
        # Consistent logic: EVERY tranche buys on Day 0 (Start of Sim).
        # Then Tranche k performs its NEXT rebalance on day k.
        # Then k+T.
        
        curve = [1.0]
        curr_val = 1.0
        
        # Initial Holdings (Day 0) - Common for all, to minimize start-up noise
        curr_holdings = picks_cache.get(valid_dates[0], [])
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i]
            
            # 1. Calculate Day Return (from t to t+1) based on current holdings
            if not curr_holdings:
                r = 0.0
            else:
                if date_t in daily_rets.index:
                    rr = daily_rets.loc[date_t, curr_holdings]
                    r = rr.mean()
                    if pd.isna(r): r = 0.0
                else:
                    r = 0.0
            
            curr_val *= (1.0 + r)
            curve.append(curr_val)
            
            # 2. Check for Rebalance (Close of Day i)
            # We need to decide holdings for Day i+1 onwards
            # Rebalance if we are at a boundary
            # logic: rebalance at step i if i matches pattern
            # Pattern: k, k+T, k+2T
            
            # If we are at day i=k, or i=k+T...
            # Note: k is 0..13. 
            # If k=0: rebal at i=0, 14...
            # If k=13: rebal at i=13, 27...
            
            if (i - k) % T == 0 and i >= k:
                 # perform rebalance using data from date_t
                 # holdings apply for t+1 returns
                 curr_holdings = picks_cache.get(date_t, []) # Actually date_t is valid_dates[i]
            
            # Special case for start-up:
            # We initialized holdings at Day 0 (before loop).
            # If k=0, we rebalance at i=0 (End of Day 0).
            # This means Day 0 return was based on 'Initial Holdings'.
            # At End of Day 0, we re-pick based on Day 0 scores.
            # Usually Day 0 Initial Holdings should BE the Day 0 picks.
            # So rebalancing at i=0 is redundant but harmless (same picks).
            
            # What about k=5?
            # i=0: No rebal. Holds Day 0 picks.
            # ...
            # i=5: Rebal. switch to Day 5 picks.
            # Correct.
            
        tranche_curves.append(np.array(curve))
        
    # Aggregate Rolling Curve
    # Sum of tranches / T
    rolling_curve = np.sum(tranche_curves, axis=0) / T
    
    # Periodic Baseline (Tranche 0 - usually the standard "Start at 0" view)
    periodic_curve = tranche_curves[0]
    
    # Verification: Print Day 0 Holdings
    day0_picks = picks_cache.get(valid_dates[0], [])
    print(f"\n[Alignment Check] Day 0 Holdings (N={len(day0_picks)}): {day0_picks}")
    print(f"  Confirming both Periodic and Rolling use this set for the first period (Tranche 0).")
    
    
    # Save Data
    dates = valid_dates[:len(rolling_curve)] # Should be equal len
    
    # Stats
    roll_ret = (rolling_curve[-1] - 1) * 100
    per_ret = (periodic_curve[-1] - 1) * 100
    
    print(f"  Rolling Return: {roll_ret:.2f}%")
    print(f"  Periodic Return: {per_ret:.2f}%")
    
    # Export CSV
    res_df = pd.DataFrame({
        'Date': dates,
        'Rolling_NAV': rolling_curve,
        'Periodic_NAV': periodic_curve
    })
    # Add all tranches
    for k in range(T):
        res_df[f'Tranche_{k}'] = tranche_curves[k]
        
    csv_path = os.path.join(config.DATA_OUTPUT_DIR, "rolling_t14_analysis.csv")
    res_df.to_csv(csv_path, index=False)
    print(f"  Data saved to {csv_path}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot Tranches (Thin, Grey)
    for k in range(T):
        ax.plot(dates, tranche_curves[k], color='gray', alpha=0.15, linewidth=1)
        
    # Plot Periodic (Blue, Dashed)
    ax.plot(dates, periodic_curve, color='blue', linestyle='--', alpha=0.6, linewidth=1.5, label=f'定期调仓 (T=14, Batch 0) +{per_ret:.1f}%')
    
    # Plot Rolling (Red, Thick)
    ax.plot(dates, rolling_curve, color='#e74c3c', linewidth=2.5, label=f'滚动持仓 (T=14, 14 tranches) +{roll_ret:.1f}%')
    
    ax.set_title("T=14 滚动持仓 vs 定期调仓 (含所有批次分布)", fontsize=16)
    ax.set_ylabel("净值 (NAV)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    img_path = os.path.join(config.CHART_OUTPUT_DIR, "rolling_t14_curve.png")
    plt.savefig(img_path, dpi=300)
    print(f"  Chart saved to {img_path}")

if __name__ == "__main__":
    analyze_rolling_t14()
