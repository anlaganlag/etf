
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_score_scan():
    print("=== Score Threshold Scan (T=14, 2024-09-01 Start) ===")
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
            except:
                pass
            
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    # Scoring
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = 15 # Top 15
    
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        period_ret = prices_df.pct_change(period)
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    def pick_top10(date_t, whitelist, min_score, max_dd_trace=[]):
        if date_t not in total_scores.index: return []
        s = total_scores.loc[date_t]
        
        # Logic: If no threshold, we just pick top 10 from whitelist
        if min_score == 0:
            # Simple sorting by score then r20
            # Note: total_scores are already calculated based on top 15 rank
            metric = s * 10000 + r20_matrix.loc[date_t]
            # Filter whitelist
            valid_metric = metric[metric.index.isin(whitelist)].sort_values(ascending=False)
            return valid_metric.head(10).index.tolist()
        
        # Logic: With threshold
        r = r20_matrix.loc[date_t]
        metric = s * 10000 + r
        valid = s[s >= min_score].index
        metric = metric[metric.index.isin(valid)]
        sorted_codes = metric.sort_values(ascending=False).index.tolist()
        candidates = [c for c in sorted_codes if c in whitelist]
        return candidates[:10]
    
    # Test Config
    start_date = '2024-09-01'
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_date)]
    T = 14
    
    print(f"\nPeriod: {start_date} ~ Present")
    print(f"Strategy: T={T}, Top 10\n")
    
    # Scan thresholds: 0 is no threshold
    thresholds = [0, 50, 100, 150, 200, 250, 300]
    
    results = []
    
    # Calculate MaxDD function
    def calc_max_dd(vals):
        arr = np.array(vals)
        peak = np.maximum.accumulate(arr)
        drawdowns = (arr - peak) / peak
        return np.min(drawdowns) * 100

    for min_score in thresholds:
        val = 1.0
        val_curve = [1.0]
        curr_holdings = []
        cash_days = 0
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i]
            
            if i % T == 0:
                curr_holdings = pick_top10(date_t, strong_codes, min_score)
            
            if not curr_holdings:
                day_ret = 0.0
                cash_days += 1
            else:
                if date_t in daily_rets.index:
                    day_ret = daily_rets.loc[date_t, curr_holdings].mean()
                    if pd.isna(day_ret): day_ret = 0.0
                else:
                    day_ret = 0.0
            
            val *= (1 + day_ret)
            val_curve.append(val)
        
        total_ret = (val - 1.0) * 100
        cash_pct = cash_days / len(valid_dates) * 100
        max_dd = calc_max_dd(val_curve)
        
        print(f"  Score≥{min_score:<3}: Return = {total_ret:>6.1f}%, MaxDD = {max_dd:>6.1f}%, CashFreq = {cash_pct:>4.1f}%")
        results.append({'Score': min_score, 'Return': total_ret, 'MaxDD': max_dd, 'CashPct': cash_pct})

    # Find best
    df_res = pd.DataFrame(results)
    best = df_res.loc[df_res['Return'].idxmax()]
    print(f"\n🏆 Best: Score≥{int(best['Score'])} with {best['Return']:.1f}%")
    
    # Benchmark
    chinext_code = 'SZSE.159915'
    if chinext_code in prices_df.columns:
        chinext_prices = prices_df.loc[valid_dates, chinext_code]
        chinext_ret = (chinext_prices.iloc[-1] / chinext_prices.iloc[0] - 1) * 100
        print(f"   ChiNext: {chinext_ret:.1f}%")

if __name__ == "__main__":
    run_score_scan()
