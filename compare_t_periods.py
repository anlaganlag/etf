
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_t_comparison():
    print("=== T=10 vs T=15 vs T=20 Comparison ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # Load Strong List
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # Build Price Matrix
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
    
    # Scoring (All 8 periods)
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for p, pts in periods_rule.items():
        ranks = prices_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    def pick_top10(date_t, whitelist, min_score, top_n):
        if date_t not in total_scores.index: return []
        s = total_scores.loc[date_t]
        r = r20_matrix.loc[date_t]
        metric = s * 10000 + r
        valid = s[s >= min_score].index
        metric = metric[metric.index.isin(valid)]
        sorted_codes = metric.sort_values(ascending=False).index.tolist()
        return [c for c in sorted_codes if c in whitelist][:top_n]
    
    # Testing Config
    test_ranges = {
        'Full (24-09起)': '2024-09-01',
        'Post-924 (24-11起)': '2024-11-01',
        'Year-2025 (25-01起)': '2025-01-01'
    }
    t_values = [10, 15, 20]
    min_score = 150
    
    results = []
    
    for range_name, start_date in test_ranges.items():
        print(f"\nTesting Range: {range_name}")
        valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
        
        for T in t_values:
            val = 1.0
            curr_holdings = []
            for i in range(len(valid_dates) - 1):
                if i % T == 0:
                    curr_holdings = pick_top10(valid_dates[i], strong_codes, min_score, 10)
                if curr_holdings:
                    day_ret = daily_rets.loc[valid_dates[i], curr_holdings].mean()
                    val *= (1 + (day_ret if not pd.isna(day_ret) else 0))
            
            total_ret = (val - 1.0) * 100
            print(f"  T={T}: {total_ret:.1f}%")
            results.append({'Range': range_name, 'T': T, 'Return': total_ret})

if __name__ == "__main__":
    run_t_comparison()
