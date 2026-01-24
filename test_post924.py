
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_fetcher import DataFetcher
from config import config

def run_post924_analysis():
    """
    Test strategies AFTER the 924 rally (start from 2024-11-01).
    Also test a "Score Threshold" filter (only buy if score > X, else cash).
    """
    print("=== Post-924 Rally Strategy Analysis ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # Load Weak List (best performer)
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    weak_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果-弱.xlsx")
    df_weak = pd.read_excel(weak_path)
    df_weak.columns = df_weak.columns.str.strip()
    df_weak = df_weak.rename(columns=rename_map)
    weak_codes = set(df_weak['etf_code'])
    
    # All Market
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # Build Price Matrix
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
    
    # Scoring
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
    
    # Helper
    def pick_top10(date_t, whitelist, min_score=0):
        if date_t not in total_scores.index: return []
        s = total_scores.loc[date_t]
        r = r20_matrix.loc[date_t]
        metric = s * 10000 + r
        metric = metric.dropna()
        metric = metric[metric > -99999]
        
        # Apply score threshold
        if min_score > 0:
            # Filter by raw score (not metric)
            valid = s[s >= min_score].index
            metric = metric[metric.index.isin(valid)]
        
        sorted_codes = metric.sort_values(ascending=False).index.tolist()
        candidates = [c for c in sorted_codes if c in whitelist]
        return candidates[:10]
    
    # Test Periods
    periods = {
        'Full (2024-09-01)': '2024-09-01',
        'Post-924 (2024-11-01)': '2024-11-01',
        'Post-Year (2025-01-01)': '2025-01-01'
    }
    
    # Strategies
    strategies = {
        'Weak_T20': {'whitelist': weak_codes, 'T': 20, 'min_score': 0},
        'Weak_T10': {'whitelist': weak_codes, 'T': 10, 'min_score': 0},
        'Weak_T20_Score150': {'whitelist': weak_codes, 'T': 20, 'min_score': 150}, # Only buy if score >= 150
        'Weak_T20_Score200': {'whitelist': weak_codes, 'T': 20, 'min_score': 200}, # Higher threshold
    }
    
    results = []
    
    for period_name, start_date in periods.items():
        print(f"\n=== Period: {period_name} ===")
        valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
        
        # Benchmark (ChiNext)
        chinext_code = 'SZSE.159915'
        if chinext_code in prices_df.columns:
            chinext_prices = prices_df.loc[valid_dates, chinext_code]
            chinext_ret = (chinext_prices.iloc[-1] / chinext_prices.iloc[0] - 1) * 100
        else:
            chinext_ret = 0
        
        print(f"  ChiNext Benchmark: {chinext_ret:.1f}%")
        
        for strat_name, strat_conf in strategies.items():
            whitelist = strat_conf['whitelist']
            T = strat_conf['T']
            min_score = strat_conf['min_score']
            
            val = 1.0
            curr_holdings = []
            
            for i in range(len(valid_dates) - 1):
                date_t = valid_dates[i]
                
                if i % T == 0:
                    curr_holdings = pick_top10(date_t, whitelist, min_score)
                
                if not curr_holdings:
                    day_ret = 0.0  # Cash - 0 return
                else:
                    if date_t in daily_rets.index:
                        day_ret = daily_rets.loc[date_t, curr_holdings].mean()
                        if pd.isna(day_ret): day_ret = 0.0
                    else:
                        day_ret = 0.0
                
                val *= (1 + day_ret)
            
            # Stats
            total_ret = (val - 1.0) * 100
            excess = total_ret - chinext_ret
            
            print(f"    {strat_name:<20}: Return={total_ret:>6.1f}%, vs ChiNext: {excess:>+6.1f}%")
            results.append({
                'Period': period_name,
                'Strategy': strat_name,
                'Return': total_ret,
                'ChiNext': chinext_ret,
                'Excess': excess
            })

    # Save
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(config.DATA_OUTPUT_DIR, "post924_analysis.csv"), index=False)
    
    # Summary
    print("\n=== Summary: Best Strategy per Period ===")
    for period_name in periods.keys():
        sub = df_res[df_res['Period'] == period_name]
        best = sub.loc[sub['Excess'].idxmax()]
        print(f"  {period_name}: {best['Strategy']} (Excess: {best['Excess']:+.1f}%)")

if __name__ == "__main__":
    run_post924_analysis()
