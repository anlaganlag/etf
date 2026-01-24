
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_fetcher import DataFetcher
from config import config

def run_merged_test():
    print("=== Testing Merged Lists (Strong + Weak) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Load Both Lists
    rename_map = {
        'symbol': 'etf_code', 'sec_name': 'etf_name', 
        'name_cleaned': 'theme', '主题': 'theme'
    }
    
    # Strong
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    # Weak
    weak_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果-弱.xlsx")
    df_weak = pd.read_excel(weak_path)
    df_weak.columns = df_weak.columns.str.strip()
    df_weak = df_weak.rename(columns=rename_map)
    weak_codes = set(df_weak['etf_code'])
    
    # Merged
    merged_codes = strong_codes.union(weak_codes)
    print(f"Strong: {len(strong_codes)}, Weak: {len(weak_codes)}, Merged: {len(merged_codes)}")
    
    # All Market for Global Ranking
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Build Price Matrix
    print("\n[Data] Building Price Matrix...")
    price_data = {}
    start_load = "2022-09-01"
    
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except:
                pass
            
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"  Matrix: {prices_df.shape}")
    
    # 3. Vectorized Scoring
    print("\n[Score] Calculating Scores...")
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = config.SECTOR_TOP_N_THRESHOLD
    
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        period_ret = prices_df.pct_change(period)
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    print("  Scoring Complete.")
    
    # 4. Simulation (Fixed T=20)
    print("\n[Sim] Running Simulations (T=20 Fixed)...")
    
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    T = 20
    
    # Helper
    def pick_top10(date_t, whitelist):
        if date_t not in total_scores.index: return []
        s = total_scores.loc[date_t]
        r = r20_matrix.loc[date_t]
        metric = s * 10000 + r
        metric = metric.dropna()
        metric = metric[metric > -99999]
        sorted_codes = metric.sort_values(ascending=False).index.tolist()
        candidates = [c for c in sorted_codes if c in whitelist]
        return candidates[:10]
    
    # Run for Strong, Weak, Merged
    universes = {
        'Strong': strong_codes,
        'Weak': weak_codes,
        'Merged': merged_codes
    }
    
    results = []
    curves = {}
    
    for name, whitelist in universes.items():
        val = 1.0
        vals = [1.0]
        curr_holdings = []
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i]
            
            # Rebalance every T days
            if i % T == 0:
                curr_holdings = pick_top10(date_t, whitelist)
            
            # Calculate return
            if not curr_holdings:
                day_ret = 0.0
            else:
                if date_t in daily_rets.index:
                    day_ret = daily_rets.loc[date_t, curr_holdings].mean()
                    if pd.isna(day_ret): day_ret = 0.0
                else:
                    day_ret = 0.0
            
            val *= (1 + day_ret)
            vals.append(val)
            
        curves[name] = vals
        
        # Stats
        arr = np.array(vals)
        total_ret = (arr[-1] - 1.0) * 100
        peak = np.maximum.accumulate(arr)
        dd = np.min((arr - peak)/peak) * 100
        
        print(f"  {name:<10}: Return={total_ret:>6.1f}%, MaxDD={dd:>6.1f}%")
        results.append({'Universe': name, 'Return': total_ret, 'MaxDD': dd})

    # Add Benchmarks for reference
    bm_codes = {'CSI300': 'SHSE.510300', 'ChiNext': 'SZSE.159915'}
    for bm_name, bm_code in bm_codes.items():
        if bm_code in prices_df.columns:
            bm_prices = prices_df.loc[valid_dates, bm_code]
            bm_ret = (bm_prices.iloc[-1] / bm_prices.iloc[0] - 1) * 100
            print(f"  {bm_name:<10}: Return={bm_ret:>6.1f}%")
            results.append({'Universe': bm_name, 'Return': bm_ret, 'MaxDD': 0})

    # Save
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(config.DATA_OUTPUT_DIR, "merged_list_test.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(12, 6))
    for name, vals in curves.items():
        min_len = min(len(vals)-1, len(valid_dates))
        plt.plot(valid_dates[:min_len], vals[1:min_len+1], label=f"{name} ({(vals[-1]-1)*100:.1f}%)")
    plt.title("Merged List Performance (T=20 Fixed)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.CHART_OUTPUT_DIR, "merged_list_test.svg"))
    print(f"\nChart saved.")

if __name__ == "__main__":
    run_merged_test()
