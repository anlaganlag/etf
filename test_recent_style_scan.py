
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_recent_style_scan():
    print("=== Recent Style Scan (T=6-14) | Nov 2025 - Jan 2026 ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # Setup Universe (Strong List)
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # Build Price Matrix (Fast Loading)
    print("[Data] Loading Prices...")
    price_data = {}
    start_load = "2024-01-01" # Enough for R250
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
    
    # Scoring (All 8 periods)
    print("[Score] Calculating Global Scores...")
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for p, pts in periods_rule.items():
        ranks = prices_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    # Scan Config
    start_sim = "2025-11-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    t_range = range(6, 15) # T=6 to T=14
    min_score = 150
    
    print(f"\nSim Period: {start_sim} to Present ({len(valid_dates)} days)")
    
    results = []
    
    for T in t_range:
        val = 1.0
        curr_holdings = []
        for i in range(len(valid_dates) - 1):
            if i % T == 0:
                dt = valid_dates[i]
                s = total_scores.loc[dt]
                metric = (s * 10000 + r20_matrix.loc[dt]).dropna()
                valid = s[s >= min_score].index
                metric = metric[metric.index.isin(valid)]
                sorted_codes = metric.sort_values(ascending=False).index.tolist()
                curr_holdings = [c for c in sorted_codes if c in strong_codes][:10]
            
            if curr_holdings:
                d_ret = daily_rets.loc[valid_dates[i], curr_holdings].mean()
                val *= (1 + (d_ret if not pd.isna(d_ret) else 0))
        
        ret_pct = (val - 1.0) * 100
        results.append({'T': T, 'Return': ret_pct})
        print(f"  T={T:<2}: {ret_pct:>6.1f}%")

    print("\nSummary (Sorted by Return):")
    df_results = pd.DataFrame(results).sort_values('Return', ascending=False)
    print(df_results.to_markdown(index=False))

if __name__ == "__main__":
    run_recent_style_scan()
