
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_with_costs():
    print("=== Strategy Test WITH Transaction Costs ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # Cost per side (buy or sell)
    COST_PER_SIDE = 0.0005  # 0.05% = 5 bps
    
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    
    # Strong
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
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
    
    # Test Config
    start_date = '2024-11-01'  # Post-924
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
    
    strategies = [
        ('Strong_T20_Score150 (No Cost)', 20, 150, False),
        ('Strong_T20_Score150 (With Cost)', 20, 150, True),
    ]
    
    print(f"\nPeriod: {start_date} ~ Present")
    print(f"Transaction Cost: {COST_PER_SIDE*100:.2f}% per side\n")
    
    for strat_name, T, min_score, apply_cost in strategies:
        val = 1.0
        curr_holdings = set()
        prev_holdings = set()
        total_cost = 0.0
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i]
            
            if i % T == 0:
                new_holdings = set(pick_top10(date_t, strong_codes, min_score))
                
                # Calculate turnover cost
                if apply_cost and (new_holdings or prev_holdings):
                    # Stocks sold: prev - new
                    sold = prev_holdings - new_holdings
                    # Stocks bought: new - prev
                    bought = new_holdings - prev_holdings
                    
                    # Cost = (sold_count + bought_count) / 10 * cost_per_side
                    # Assuming equal weight, each position is 10%
                    turnover_frac = (len(sold) + len(bought)) / 10
                    cost = turnover_frac * COST_PER_SIDE
                    val *= (1 - cost)
                    total_cost += cost * 100
                
                prev_holdings = new_holdings
                curr_holdings = new_holdings
            
            # Calculate return
            if not curr_holdings:
                day_ret = 0.0
            else:
                if date_t in daily_rets.index:
                    day_ret = daily_rets.loc[date_t, list(curr_holdings)].mean()
                    if pd.isna(day_ret): day_ret = 0.0
                else:
                    day_ret = 0.0
            
            val *= (1 + day_ret)
        
        total_ret = (val - 1.0) * 100
        print(f"  {strat_name:<35}: Return={total_ret:>6.1f}%, Cost Paid={total_cost:.2f}%")

    # Benchmark
    chinext_code = 'SZSE.159915'
    if chinext_code in prices_df.columns:
        chinext_prices = prices_df.loc[valid_dates, chinext_code]
        chinext_ret = (chinext_prices.iloc[-1] / chinext_prices.iloc[0] - 1) * 100
        print(f"  {'ChiNext (基准)':<35}: Return={chinext_ret:>6.1f}%")

if __name__ == "__main__":
    run_with_costs()
