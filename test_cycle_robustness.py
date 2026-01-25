
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_cycle_robustness():
    print("=== Cycle Robustness Test: Moving Start Date (Offset Scan) ===")
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
    start_load = "2024-06-01"
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
    
    # 3. Scoring
    print("[Score] Pre-calculating Global Ranks...")
    periods = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for p, pts in periods.items():
        ranks = prices_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= 15) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)

    # 4. Multi-Start test
    start_sim = "2025-07-25" # Using the 6-month window as requested previously
    sim_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    t_values = [10, 14, 20]
    min_score = 150
    
    final_comparison = []
    
    for T in t_values:
        print(f"\nTesting T={T} with all {T} possible start-date offsets...")
        all_offsets = []
        
        for offset in range(T):
            val = 1.0
            holdings = []
            # We skip the first 'offset' days to stagger the rebalance cycle
            sub_dates = sim_dates[offset:]
            
            for i in range(len(sub_dates) - 1):
                if i % T == 0:
                    dt = sub_dates[i]
                    s = total_scores.loc[dt]
                    valid = s[s >= min_score].index
                    metric = (s * 10000 + r20_matrix.loc[dt])[valid].dropna()
                    holdings = [c for c in metric.sort_values(ascending=False).index if c in strong_codes][:10]
                
                if holdings:
                    val *= (1 + daily_rets.loc[sub_dates[i], holdings].mean())
            
            ret = (val - 1.0) * 100
            all_offsets.append(ret)
            
        avg_ret = np.mean(all_offsets)
        std_ret = np.std(all_offsets)
        min_ret = np.min(all_offsets)
        max_ret = np.max(all_offsets)
        
        print(f"  Result: Avg={avg_ret:.1f}%, Min={min_ret:.1f}%, Max={max_ret:.1f}%, Std={std_ret:.1f}")
        final_comparison.append({'T': T, 'Avg_Return': avg_ret, 'Stability': 1/(std_ret+1), 'Best': max_ret, 'Worst': min_ret})

    print("\n" + "="*50)
    print(f"{'T':<5} | {'Avg Return':<12} | {'Worst Case':<12} | {'Best Case':<12}")
    print("-"*50)
    for res in final_comparison:
        print(f"{res['T']:<5} | {res['Avg_Return']:>10.1f}% | {res['Worst']:>10.1f}% | {res['Best']:>10.1f}%")
    print("="*50)

if __name__ == "__main__":
    run_cycle_robustness()
