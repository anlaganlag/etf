
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config
from datetime import datetime, timedelta

def run_6month_comparison():
    print("=== Last 6 Months Comparison (2025-07-25 to Present) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Setup
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Data loading
    print("[Data] Loading Prices...")
    price_data = {}
    start_load = "2024-06-01" # Earlier for R250
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except: pass
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    # 3. Global Ranking Scores (8-period)
    print("[Score] Calculating Global Ranking...")
    periods = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for p, pts in periods.items():
        ranks = prices_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= 15) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)

    # 4. Simulation window: Last 6 months
    # Current date is 2026-01-25. 6 months ago is roughly 2025-07-25.
    start_sim = "2025-07-25"
    sim_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    print(f"Simulating from {sim_dates[0].date()} to {sim_dates[-1].date()} ({len(sim_dates)} trading days)")
    
    t_values = [10, 14, 20]
    min_score = 150
    
    for T in t_values:
        val = 1.0
        holdings = []
        for i in range(len(sim_dates) - 1):
            if i % T == 0:
                dt = sim_dates[i]
                s = total_scores.loc[dt]
                valid = s[s >= min_score].index
                metric = (s * 10000 + r20_matrix.loc[dt])[valid].dropna()
                holdings = [c for c in metric.sort_values(ascending=False).index if c in strong_codes][:10]
            
            if holdings:
                day_ret = daily_rets.loc[sim_dates[i], holdings].mean()
                val *= (1 + (day_ret if not pd.isna(day_ret) else 0))
        
        print(f"  T={T:<2}: Return = {(val-1)*100:>6.1f}%")

if __name__ == "__main__":
    run_6month_comparison()
