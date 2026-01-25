
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_fetcher import DataFetcher
from config import config

def run_holdings_count_scan():
    print("=== Scanning Holdings Count (3, 5, 10) vs Period (1-20) NO THRESHOLD ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Load Universe
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_curated = pd.read_excel(excel_path)
    df_curated.columns = df_curated.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    df_curated = df_curated.rename(columns=rename_map)
    curated_codes = set(df_curated['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Build Price Matrix
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
    
    # 3. Score Calc
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = 15
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        period_ret = prices_df.pct_change(period)
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    # 4. Scanning
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    holdings_counts = [3, 5, 10]
    scan_range = range(1, 21) # 1 to 20
    
    all_results = []
    
    for top_n in holdings_counts:
        print(f"\n--- Testing Top {top_n} ---")
        
        for T in scan_range:
            portfolio_value = 1.0
            portfolio_values = [1.0]
            current_holdings = []
            
            for i in range(len(valid_dates) - 1):
                date_t = valid_dates[i]
                
                # Rebalance
                if i % T == 0:
                    if date_t in total_scores.index:
                        s = total_scores.loc[date_t]
                        r = r20_matrix.loc[date_t]
                        metric = s * 10000 + r
                        metric = metric.dropna()
                        
                        sorted_codes = metric.sort_values(ascending=False).index.tolist()
                        candidates = [c for c in sorted_codes if c in curated_codes]
                        current_holdings = candidates[:top_n] # Dynamic N
                
                if not current_holdings:
                    day_ret = 0.0
                else:
                    if date_t in daily_rets.index:
                        day_ret = daily_rets.loc[date_t, current_holdings].mean()
                        if pd.isna(day_ret): day_ret = 0.0
                    else:
                        day_ret = 0.0
                
                portfolio_value *= (1 + day_ret)
                portfolio_values.append(portfolio_value)
            
            arr = np.array(portfolio_values)
            total_ret = (arr[-1] - 1.0) * 100
            peak = np.maximum.accumulate(arr)
            dd = np.min((arr - peak)/peak) * 100
            
            print(f"  N={top_n:<2}, T={T:<2}: Return={total_ret:>6.1f}%, MaxDD={dd:>6.1f}%")
            all_results.append({
                'TopN': top_n,
                'Period': T,
                'Return': total_ret,
                'MaxDD': dd
            })

    # Save
    df_res = pd.DataFrame(all_results)
    df_res.to_csv(os.path.join(config.DATA_OUTPUT_DIR, "tuning_holdings_count.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    colors = {3: 'red', 5: 'green', 10: 'blue'}
    
    for n in holdings_counts:
        sub = df_res[df_res['TopN'] == n]
        plt.plot(sub['Period'], sub['Return'], marker='o', label=f'Top {n} Holdings', color=colors[n])
        
    plt.xlabel('Holding Period (Days)')
    plt.ylabel('Total Return (%)')
    plt.title(f'Impact of Holdings Count & Period (No Threshold, 2024-09~)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(config.CHART_OUTPUT_DIR, "tuning_holdings_count.png"))
    print(f"\nChart saved to {os.path.join(config.CHART_OUTPUT_DIR, 'tuning_holdings_count.png')}")

if __name__ == "__main__":
    run_holdings_count_scan()
