
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_scoring_comparison():
    print("=== Scoring Strategy Comparison ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
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
    
    # Define different scoring strategies
    scoring_strategies = {
        'Original (R1-R250)': {1:100, 3:70, 5:50, 10:30, 20:20, 60:15, 120:10, 250:5},
        'Client (R1-R20)': {1:100, 3:70, 5:50, 10:30, 20:20},
        'Short Only (R1-R5)': {1:100, 3:70, 5:50},
        'Medium (R5-R60)': {5:100, 10:70, 20:50, 60:30},
        'Balanced (Equal Weight)': {1:50, 3:50, 5:50, 10:50, 20:50},
        'Aggressive (R1 Heavy)': {1:200, 3:50, 5:30, 10:20, 20:10},
    }
    
    threshold = 15  # Top N threshold
    
    # Pre-compute returns for all periods
    all_periods = [1, 3, 5, 10, 20, 60, 120, 250]
    period_returns = {}
    period_ranks = {}
    
    for p in all_periods:
        period_returns[p] = prices_df.pct_change(p)
        period_ranks[p] = period_returns[p].rank(axis=1, ascending=False, method='min')
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    def compute_scores(scores_rule):
        total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
        for period, pts in scores_rule.items():
            total_scores = total_scores.add((period_ranks[period] <= threshold) * pts, fill_value=0)
        return total_scores
    
    def pick_top10(date_t, total_scores, whitelist, min_score):
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
    
    # Test Config
    start_date = '2024-11-01'  # Post-924
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
    T = 20
    min_score = 150
    
    print(f"\nPeriod: {start_date} ~ Present")
    print(f"Settings: T={T}, Score≥{min_score}, Top 10\n")
    
    results = []
    
    for strat_name, scores_rule in scoring_strategies.items():
        # Compute total scores for this strategy
        total_scores = compute_scores(scores_rule)
        
        val = 1.0
        curr_holdings = []
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i]
            
            if i % T == 0:
                curr_holdings = pick_top10(date_t, total_scores, strong_codes, min_score)
            
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
        print(f"  {strat_name:<25}: Return = {total_ret:>6.1f}%")
        results.append({'Strategy': strat_name, 'Return': total_ret})

    # Benchmark
    chinext_code = 'SZSE.159915'
    if chinext_code in prices_df.columns:
        chinext_prices = prices_df.loc[valid_dates, chinext_code]
        chinext_ret = (chinext_prices.iloc[-1] / chinext_prices.iloc[0] - 1) * 100
        print(f"\n  ChiNext (基准): {chinext_ret:.1f}%")
    
    # Find best
    df_res = pd.DataFrame(results)
    best = df_res.loc[df_res['Return'].idxmax()]
    print(f"\n🏆 Best: {best['Strategy']} with {best['Return']:.1f}%")

if __name__ == "__main__":
    run_scoring_comparison()
