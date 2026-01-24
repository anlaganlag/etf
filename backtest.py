
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from src.etf_ranker import EtfRanker
from src.data_fetcher import DataFetcher
from config import config

def run_backtest():
    print("=== Starting Backtest (2024-09-01 to Present) ===")
    
    # 1. Setup
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    ranker = EtfRanker(fetcher)
    
    candidate_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    if not os.path.exists(candidate_path):
        print("Candidate file not found")
        return
        
    df_candidates = pd.read_excel(candidate_path)
    df_candidates = df_candidates.rename(columns={
        'symbol': 'etf_code',
        'sec_name': 'etf_name',
        '主题': 'theme'
    })
    
    print(f"Candidate Universe: {len(df_candidates)} ETFs")

    # 2. Pre-load History for Performance
    print("\nPre-loading history for all candidates...")
    history_cache = {}
    
    # Range: Need enough lookback for r250 from 2024-09-01, so fetch from 2022-09-01
    start_history_range = "2022-09-01"
    end_history_range = datetime.now().strftime("%Y-%m-%d")
    
    for idx, row in df_candidates.iterrows():
        code = row['etf_code']
        if (idx + 1) % 50 == 0:
            print(f"Loaded {idx+1}/{len(df_candidates)}...")
        
        df = fetcher.get_etf_daily_history(code, start_history_range, end_history_range)
        if df is not None and not df.empty:
             history_cache[code] = df
    
    print(f"Cache ready: {len(history_cache)} ETFs loaded.")

    # 3. Define Timeline & Benchmarks
    start_date_str = "2024-09-01"
    
    # Benchmarks
    print("\nLoading Benchmarks...")
    benchmarks = {
        'CSI300': 'SHSE.510300',
        'ChiNext': 'SZSE.159915'
    }
    benchmark_data = {}
    
    for name, code in benchmarks.items():
        df = fetcher.get_etf_daily_history(code, start_date_str, end_history_range)
        if df is not None and not df.empty:
            # Ensure Date is datetime
            df['日期'] = pd.to_datetime(df['日期'])
            benchmark_data[name] = df.sort_values('日期').set_index('日期')
            print(f"Loaded {name}: {len(df)} days")
        else:
            print(f"Failed to load {name}")

    # Use CSI300 as calendar reference
    if 'CSI300' not in benchmark_data:
        print("Critical Error: CSI300 data missing for calendar reference")
        return
        
    ref_df = benchmark_data['CSI300']
    
    # Filter dates >= 2024-09-01
    # Note: ref_df index is datetime
    trading_days = ref_df.index[ref_df.index >= pd.to_datetime(start_date_str)].strftime('%Y-%m-%d').tolist()
    
    print(f"Backtest Period: {len(trading_days)} trading days")
    
    # 4. Simulation Loop
    portfolio_value = 1.0 
    portfolio_values = []
    
    # Track Benchmark Values (Normalized to 1.0)
    bm_values = {name: [1.0] for name in benchmarks}
    
    print("\nStarting Daily Loop...")
    
    daily_logs = []

    for i, current_date in enumerate(trading_days[:-1]): 
        next_date = trading_days[i+1]
        
        # --- Strategy Update ---
        top_df = ranker.select_top_etfs(
            df_candidates, 
            top_n=10, 
            reference_date=current_date, 
            history_cache=history_cache
        )
        
        avg_return = 0.0
        selected_codes = []
        
        if not top_df.empty:
            selected_codes = top_df['etf_code'].tolist()
            day_returns = []
            for code in selected_codes:
                hist = history_cache.get(code)
                if hist is None: continue
                
                # Optimized lookup
                # slice by date range (fast enough for small df)
                sub = hist[(hist['日期'] >= pd.to_datetime(current_date)) & (hist['日期'] <= pd.to_datetime(next_date))]
                
                if len(sub) < 2:
                    day_returns.append(0.0)
                    continue
                
                # Assume sorted
                p_t = float(sub.iloc[0]['收盘'])
                p_t1 = float(sub.iloc[-1]['收盘']) # Should be next_date
                
                if p_t == 0: ret = 0
                else: ret = (p_t1 - p_t) / p_t
                day_returns.append(ret)
            
            if day_returns:
                avg_return = sum(day_returns) / len(day_returns)
            
        # Update Portfolio
        portfolio_value *= (1 + avg_return)
        portfolio_values.append(portfolio_value)
        
        # --- Benchmark Update ---
        for name, code in benchmarks.items():
            bm_df = benchmark_data.get(name)
            bm_ret = 0.0
            if bm_df is not None:
                try:
                    p_t = float(bm_df.loc[current_date]['收盘'])
                    p_t1 = float(bm_df.loc[next_date]['收盘'])
                    if p_t != 0:
                        bm_ret = (p_t1 - p_t) / p_t
                except KeyError:
                    pass # Missing data for date
            
            # Update BM Value
            last_val = bm_values[name][-1]
            bm_values[name].append(last_val * (1 + bm_ret))

        # Log
        daily_logs.append({
            'Date': next_date, 
            'Return': avg_return, 
            'Value': portfolio_value,
            'Holdings': ",".join(selected_codes)
        })
        
        if (i+1) % 20 == 0:
            print(f"Date: {current_date}, Strategy: {portfolio_value:.4f}, CSI300: {bm_values['CSI300'][-1]:.4f}")

    # 5. Report
    print("\n=== Backtest Results ===")
    final_value = portfolio_values[-1] if portfolio_values else 1.0
    total_return = (final_value - 1.0) * 100
    
    def calc_max_dd(vals):
        arr = np.array(vals)
        peak = np.maximum.accumulate(arr)
        drawdowns = (arr - peak) / peak
        return np.min(drawdowns) * 100

    max_dd = calc_max_dd([1.0] + [x['Value'] for x in daily_logs])
    
    print(f"{'Strategy':<10} | Return: {total_return:>6.2f}% | MaxDD: {max_dd:>6.2f}%")
    
    for name in benchmarks:
        vals = bm_values[name]
        ret = (vals[-1] - 1.0) * 100
        dd = calc_max_dd(vals)
        print(f"{name:<10} | Return: {ret:>6.2f}% | MaxDD: {dd:>6.2f}%")
    
    # Save Report
    df_log = pd.DataFrame(daily_logs)
    # Add benchmark columns to log
    for name in benchmarks:
        # sliced to match daily_logs length (which is len(trading_days)-1)
        # bm_values has init 1.0 + len(trading_days)-1 entries
        # so bm_values[1:] matches daily_logs
        df_log[name] = bm_values[name][1:]
        
    output_path = os.path.join(config.DATA_OUTPUT_DIR, "backtest_result.csv")
    df_log.to_csv(output_path, index=False)
    print(f"Detailed logs saved to {output_path}")
    
    # Save Plot
    plt.figure(figsize=(12, 6))
    dates = pd.to_datetime(df_log['Date'])
    plt.plot(dates, df_log['Value'], label=f'Strategy (+{total_return:.1f}%)', linewidth=2)
    
    for name in benchmarks:
        vals = bm_values[name][1:]
        ret = (vals[-1] - 1.0) * 100
        plt.plot(dates, vals, label=f'{name} (+{ret:.1f}%)', alpha=0.7)

    plt.title(f"ETF Strategy vs Benchmarks ({start_date_str} ~ Present)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = os.path.join(config.CHART_OUTPUT_DIR, "backtest_curve.svg")
    plt.savefig(plot_path)
    print(f"Equity curve saved to {plot_path}")

if __name__ == "__main__":
    run_backtest()
