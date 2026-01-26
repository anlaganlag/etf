
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from src.etf_ranker import EtfRanker
from config import config

def analyze_period():
    print("=== Analyzing Best Holding Period for ETF Strategy ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    ranker = EtfRanker(fetcher)
    
    # 1. Load Candidates and Filter Cross-border
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    candidates = df_excel.rename(columns={'symbol': 'etf_code', 'sec_name': 'etf_name', '主题': 'theme'})
    exclude_kws = ['纳斯达克', '标普', '恒生', '日经', '德国', '港股', '海外', '国外', '纳指', '亚洲', '美国', '道琼斯']
    df_candidates = candidates[~candidates['etf_name'].str.contains('|'.join(exclude_kws), na=False)].copy()
    candidate_codes = set(df_candidates['etf_code'])
    print(f"Candidates: {len(df_candidates)} (Cross-border excluded)")

    # 2. Pre-load History Data (Fast Matrix)
    print("\n[Data] Loading price matrix...")
    all_etf_list = fetcher.get_all_etfs()
    all_codes = list(all_etf_list['etf_code'])
    
    price_data = {}
    start_load = "2023-09-01" # Enough lookback for r250
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
    print(f"Matrix shape: {prices_df.shape}")
    
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    # 3. Pre-calculate Daily Ranks (to speed up simulation)
    print("[Score] Vectorizing daily rankings (within candidates)...")
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = config.SECTOR_TOP_N_THRESHOLD
    
    # Slice price matrix to candidates only for ranking - USE INTERSECTION TO AVOID KEYERROR
    valid_candidates = list(candidate_codes.intersection(set(prices_df.columns)))
    print(f"Valid candidates in matrix: {len(valid_candidates)}")
    prices_candidates = prices_df[valid_candidates].copy()
    
    total_scores = pd.DataFrame(0.0, index=prices_candidates.index, columns=prices_candidates.columns)
    for period, pts in scores_rule.items():
        period_ret = prices_candidates.pct_change(period)
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    print(f"Max candidate score found: {total_scores.max().max()}")
    
    # 20-day momentum for tie-breaking
    r20_matrix = prices_candidates.pct_change(20).fillna(-999)
    
    # Helper for picking ranked codes on a specific date
    def get_daily_ranks(date_t):
        if date_t not in total_scores.index: return []
        s = total_scores.loc[date_t]
        r = r20_matrix.loc[date_t]
        # Multi-score metric: (Score * large_number) + Momentum
        metric = s * 10000 + r
        # Sort by metric
        return metric.sort_values(ascending=False).index.tolist()

    # 4. Simulation Parameters
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    print(f"Simulation dates: {len(valid_dates)}")
    
    test_periods = [5, 10, 14, 20, 25, 30, 40]
    top_n = 10
    buffer_n = 10
    
    results = []
    
    print(f"\n[Sim] Scanning Periods {test_periods}...")
    
    for T in test_periods:
        portfolio_value = 1.0
        portfolio_values = [1.0]
        current_holdings = []
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i]
            
            # Rebalance Logic (Buffer-based)
            if i % T == 0:
                all_ranks = get_daily_ranks(date_t)
                if not all_ranks:
                    if i < 10: print(f"DEBUG: No ranks on {date_t}")
                    pass 
                else:
                    top_n_new = all_ranks[:top_n]
                    buffer_list = all_ranks[:buffer_n]
                    
                    final_targets = []
                    for code in current_holdings:
                        if code in buffer_list:
                            final_targets.append(code)
                    
                    for code in top_n_new:
                        if len(final_targets) >= top_n: break
                        if code not in final_targets:
                            final_targets.append(code)
                    
                    current_holdings = final_targets
                    if i < 10 * T: print(f"DEBUG: T={T} Day {i} {date_t} Holdings: {len(current_holdings)}")
            
            # Calculate daily return
            if not current_holdings:
                day_ret = 0.0
            else:
                day_ret = daily_rets.loc[date_t, current_holdings].mean()
                if pd.isna(day_ret): 
                    # if i < 10: print(f"DEBUG: NaN return for {current_holdings}")
                    day_ret = 0.0
                
            portfolio_value *= (1 + day_ret)
            portfolio_values.append(portfolio_value)
            
        # Stats
        arr = np.array(portfolio_values)
        total_ret = (arr[-1] - 1.0) * 100
        peak = np.maximum.accumulate(arr)
        max_dd = np.min((arr - peak)/peak) * 100
        
        # Win Rate
        diffs = np.diff(arr)
        win_rate = np.sum(diffs > 0) / len(diffs) * 100
        
        print(f"  T={T:<2}: Return={total_ret:>6.2f}%, MaxDD={max_dd:>6.2f}%, WinRate={win_rate:.1f}%")
        results.append({'Period': T, 'Return': total_ret, 'MaxDD': max_dd, 'WinRate': win_rate})

    # Summary Table
    df_res = pd.DataFrame(results).sort_values('Return', ascending=False)
    print("\n=== RANKING RESULTS ===")
    print(df_res.to_string(index=False))
    
    # Save to report
    report_path = os.path.join(config.DATA_OUTPUT_DIR, "best_holding_period_analysis.csv")
    df_res.to_csv(report_path, index=False)
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.plot(df_res['Period'], df_res['Return'], 'b-s', label='Total Return %')
    plt.axhline(0, color='black', alpha=0.3)
    plt.xlabel('Rebalance Period (Trading Days)')
    plt.ylabel('Return (%)')
    plt.title('Return vs. Rebalance Period (Top 10 Buffer Logic)')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.CHART_OUTPUT_DIR, "period_analysis.svg"))
    print(f"\nAnalysis saved to {report_path}")

if __name__ == "__main__":
    analyze_period()
