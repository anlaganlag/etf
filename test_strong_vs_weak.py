
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_strong_vs_weak():
    print("=== Strong vs Weak with Score Threshold ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    
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
    
    print(f"Strong: {len(strong_codes)}, Weak: {len(weak_codes)}")
    
    # All Market
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # Build Price Matrix
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
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = config.SECTOR_TOP_N_THRESHOLD
    
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        period_ret = prices_df.pct_change(period)
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    def pick_top10(date_t, whitelist, min_score=0):
        if date_t not in total_scores.index: return []
        s = total_scores.loc[date_t]
        r = r20_matrix.loc[date_t]
        metric = s * 10000 + r
        metric = metric.dropna()
        metric = metric[metric > -99999]
        
        if min_score > 0:
            valid = s[s >= min_score].index
            metric = metric[metric.index.isin(valid)]
        
        sorted_codes = metric.sort_values(ascending=False).index.tolist()
        candidates = [c for c in sorted_codes if c in whitelist]
        return candidates[:10]
    
    # Test Periods
    periods = {
        'Full': '2024-09-01',
        'Post-924': '2024-11-01',
    }
    
    # Strategies
    strategies = [
        ('Strong_T20', strong_codes, 20, 0),
        ('Strong_T20_Score150', strong_codes, 20, 150),
        ('Weak_T20', weak_codes, 20, 0),
        ('Weak_T20_Score150', weak_codes, 20, 150),
    ]
    
    print("\n" + "="*70)
    print(f"{'Strategy':<25} | {'Full (09-01)':<15} | {'Post-924 (11-01)':<15}")
    print("="*70)
    
    for strat_name, whitelist, T, min_score in strategies:
        row = f"{strat_name:<25}"
        
        for period_name, start_date in periods.items():
            valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
            
            val = 1.0
            curr_holdings = []
            
            for i in range(len(valid_dates) - 1):
                date_t = valid_dates[i]
                
                if i % T == 0:
                    curr_holdings = pick_top10(date_t, whitelist, min_score)
                
                if not curr_holdings:
                    day_ret = 0.0
                else:
                    if date_t in daily_rets.index:
                        day_ret = daily_rets.loc[date_t, curr_holdings].mean()
                        if pd.isna(day_ret): day_ret = 0.0
                    else:
                        day_ret = 0.0
                
                val *= (1 + day_ret)
            
            total_ret = (val - 1.0) * 100
            row += f" | {total_ret:>13.1f}%"
        
        print(row)
    
    # Benchmark
    chinext_code = 'SZSE.159915'
    row = f"{'ChiNext (基准)':<25}"
    for period_name, start_date in periods.items():
        valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
        if chinext_code in prices_df.columns:
            chinext_prices = prices_df.loc[valid_dates, chinext_code]
            chinext_ret = (chinext_prices.iloc[-1] / chinext_prices.iloc[0] - 1) * 100
        else:
            chinext_ret = 0
        row += f" | {chinext_ret:>13.1f}%"
    print(row)
    print("="*70)

if __name__ == "__main__":
    run_strong_vs_weak()
