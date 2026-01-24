
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_fetcher import DataFetcher
from src.etf_ranker import EtfRanker
from config import config

def run_holding_period_scan():
    print("=== Reformulating Holding Period (1-20 Days) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    ranker = EtfRanker(fetcher)
    
    # 1. Load Universe: Curated (Strong or Weak? Let's use Strong as baseline)
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_curated = pd.read_excel(excel_path)
    df_curated.columns = df_curated.columns.str.strip()
    rename_map = {
        'symbol': 'etf_code', 'sec_name': 'etf_name', 
        'name_cleaned': 'theme', '主题': 'theme'
    }
    df_curated = df_curated.rename(columns=rename_map)
    curated_codes = set(df_curated['etf_code'])
    
    # All Market codes for Global Ranking
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])
    print(f"Universe: {len(curated_codes)} Curated / {len(all_codes)} Global")

    # 2. Build Price Matrix (Fast I/O)
    print("\n[2/4] Building Price Matrix...")
    price_data = {}
    start_load = "2022-09-01"
    
    count = 0
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
        count += 1
        if count % 500 == 0: print(f"  Processed {count}...")
            
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"  Matrix: {prices_df.shape}")
    
    # 3. Vectorized Scoring
    print("\n[3/4] Calculating Scores...")
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = config.SECTOR_TOP_N_THRESHOLD
    
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        period_ret = prices_df.pct_change(period)
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    print("  Scoring Complete.")
    
    # 4. Scanning Loop
    print("\n[4/4] Scanning Holding Periods (1..20)...")
    
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    results = [] # {Period, Return, MaxDD}
    
    # We test periods T in [1, 2, ..., 60]
    scan_range = range(1, 61)
    
    # To be comparable, all strategies start on Buy Day 0 (valid_dates[0])
    
    # Pre-calc daily returns for all assets to speed up
    daily_rets = prices_df.pct_change(1).shift(-1) # Return for T -> T+1 stored at T
    
    for T in scan_range:
        # Simulation
        portfolio_value = 1.0
        portfolio_values = [1.0]
        
        current_holdings = []
        
        # We walk day by day
        # But we only rebalance when (i % T == 0)
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i]
            # date_next = valid_dates[i+1] # Not needed for signal, used for return
            
            # Rebalance Signal?
            if i % T == 0:
                # 1. Get Scores
                if date_t in total_scores.index:
                    s = total_scores.loc[date_t]
                    r = r20_matrix.loc[date_t]
                    metric = s * 10000 + r
                    metric = metric.dropna()
                    metric = metric[metric > -99999]
                    sorted_codes = metric.sort_values(ascending=False).index.tolist()
                    
                    # 2. Filter (Curated Strong)
                    candidates = [c for c in sorted_codes if c in curated_codes]
                    current_holdings = candidates[:10]
            
            # Calculate Return for T -> T+1
            # Using pre-calced daily_rets
            if not current_holdings:
                day_ret = 0.0
            else:
                # Look up returns for these assets at date_t
                # Efficient lookup?
                # daily_rets.loc[date_t, current_holdings] -> Series
                # mean()
                try:
                    # Check if date_t exists in daily_rets
                    if date_t in daily_rets.index:
                        rets = daily_rets.loc[date_t, current_holdings]
                        day_ret = rets.mean()
                        if pd.isna(day_ret): day_ret = 0.0
                    else:
                        day_ret = 0.0
                except:
                    day_ret = 0.0
            
            portfolio_value *= (1 + day_ret)
            portfolio_values.append(portfolio_value)
            
        # Stats
        arr = np.array(portfolio_values)
        total_ret = (arr[-1] - 1.0) * 100
        peak = np.maximum.accumulate(arr)
        dd = np.min((arr - peak)/peak) * 100
        
        print(f"  T={T:<2}: Return={total_ret:>6.1f}%, MaxDD={dd:>6.1f}%")
        results.append({'Period': T, 'Return': total_ret, 'MaxDD': dd})

    # Save
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(config.DATA_OUTPUT_DIR, "tuning_holding_period.csv"), index=False)
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(df_res['Period'], df_res['Return'], 'b-o', label='Total Return')
    ax1.set_xlabel('Holding Period (Days)')
    ax1.set_ylabel('Total Return (%)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(df_res['Period'], df_res['MaxDD'], 'r--s', label='Max Drawdown')
    ax2.set_ylabel('Max Drawdown (%)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    plt.title(f"Impact of Holding Period (Rebalance Frequency)")
    plt.savefig(os.path.join(config.CHART_OUTPUT_DIR, "tuning_holding_period.svg"))
    print(f"\nChart saved to {config.CHART_OUTPUT_DIR}")

if __name__ == "__main__":
    run_holding_period_scan()
