
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_definitive_comparison():
    print("=== Definitive T-Period Comparison: Yield vs Adaptivity ===")
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

    # 2. Data
    print("[Data] Loading Prices...")
    price_data = {}
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
    
    # 3. Scoring (8-periods)
    print("[Score] Calculating Global Ranking...")
    periods = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for p, pts in periods.items():
        ranks = prices_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= 15) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)

    # 4. Windows
    windows = {
        'Full (Nov 2024 - Present)': '2024-11-01',
        'Recent (Nov 2025 - Present)': '2025-11-01'
    }
    
    t_values = [10, 14, 20]
    
    for win_name, start_date in windows.items():
        print(f"\nWindow: {win_name}")
        sim_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
        
        for T in t_values:
            val = 1.0
            holdings = []
            for i in range(len(sim_dates) - 1):
                if i % T == 0:
                    dt = sim_dates[i]
                    s = total_scores.loc[dt]
                    valid = s[s >= 150].index
                    metric = (s * 10000 + r20_matrix.loc[dt])[valid].dropna()
                    holdings = [c for c in metric.sort_values(ascending=False).index if c in strong_codes][:10]
                if holdings:
                    val *= (1 + daily_rets.loc[sim_dates[i], holdings].mean())
            
            print(f"  T={T:<2}: Return = {(val-1)*100:>6.1f}%")

if __name__ == "__main__":
    run_definitive_comparison()
