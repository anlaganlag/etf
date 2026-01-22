import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Strategy Config
# Strategy Config
CACHE_DIR = "data_cache"
BENCHMARK_CODE = "sh510300" # HS300 ETF as benchmark
# Backtest past 1 year
START_DATE = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Scoring Weights (Same as config)
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10

def load_data():
    price_dict = {}
    print("Loading cached data...")
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv") and "etf_list" not in f]
    
    for f in files:
        code = f.replace(".csv", "")
        # Skip weird files
        if not (code.startswith('sh') or code.startswith('sz')): continue
        
        path = os.path.join(CACHE_DIR, f)
        try:
            df = pd.read_csv(path)
            if '日期' not in df.columns or '收盘' not in df.columns:
                continue
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            price_dict[code] = df['收盘']
        except:
            pass
            
    print(f"Loaded {len(price_dict)} ETFs. Aligning dates...")
    full_df = pd.DataFrame(price_dict)
    full_df = full_df.sort_index()
    return full_df

def run_backtest():
    # 1. Prepare Data
    prices = load_data()
    
    # Filter dates
    prices = prices[START_DATE:END_DATE]
    if prices.empty:
        print("No data in date range.")
        return

    # Benchmark
    if BENCHMARK_CODE in prices.columns:
        bench_prices = prices[BENCHMARK_CODE]
    else:
        print(f"Benchmark {BENCHMARK_CODE} not found. Using average.")
        bench_prices = prices.mean(axis=1)

    # 2. Benchmark Returns
    bench_ret = bench_prices.pct_change().fillna(0)
    bench_equity = (1 + bench_ret).cumprod()

    # 3. Strategy Loop (Weekly Rotation)
    print("Calculating rolling returns...")
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = prices.pct_change(periods=d).fillna(-999)

    strategy_rets = []
    dates = prices.index
    
    # Position holding
    current_holdings = [] # List of codes
    
    print("Running simulation (Weekly Rotation)...")
    
    # Rebalance on Mondays (or first trading day of week)
    # We check if today's week != previous day's week
    
    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i+1] # Trade close-to-close or open-to-open
        
        # Check if we should rebalance today (for tomorrow's holdings)
        # Rebalance every 5 days? Or check weekday.
        # Simple logic: Rebalance if 'tomorrow' is Monday (isoweekday=1) 
        # But markets might be closed on Monday.
        # Better: Rebalance every N days. Let's say every 5 trading days.
        rebalance = (i % 5 == 0)
        
        if rebalance:
            # Rank and Select
            daily_scores = pd.Series(0, index=prices.columns)
            valid_data_mask = prices.loc[today].notna()
            
            for d, weight in SCORES.items():
                r_d = roll_rets[d].loc[today]
                # Filter valid and not extreme crash (simple filter)
                valid_r = r_d[valid_data_mask & (r_d > -100)]
                if valid_r.empty: continue
                
                # Dynamic Threshold: Top 10%
                threshold_n = max(10, int(len(valid_r) * 0.1))
                top_codes = valid_r.nlargest(threshold_n).index
                daily_scores.loc[top_codes] += weight

            # Select Top N
            current_holdings = daily_scores.nlargest(TOP_N).index.tolist()
        
        # Use simple variable name to avoid confusion
        current_daily_ret = 0.0
        
        if current_holdings:
            # Calculate tomorrows return for current holdings
            # We assume we hold them from Today Close to Tomorrow Close
            # Handle warnings by using fill_method=None in future, but for now just direct calculation
            # prices.pct_change(1) gives (P_t / P_{t-1}) - 1
            # We want return at 'tomorrow' based on 'today's close.
            # pct_change(1) at tomorrow is exactly this.
            
            # Extract returns for held ETFs at tomorrow
            # Vectorized calc
            # Note: pct_change is pre-calced or calced on fly?
            # Re-calc small slice correctly
            p_tomorrow = prices.loc[tomorrow, current_holdings]
            p_today = prices.loc[today, current_holdings]
            
            # Avoid division by zero
            rets = (p_tomorrow - p_today) / p_today
            current_daily_ret = rets.mean()
            
            if pd.isna(current_daily_ret):
                current_daily_ret = 0.0
        
        strategy_rets.append(current_daily_ret)

    # 4. Results
    strategy_rets = pd.Series(strategy_rets, index=dates[1:])
    strategy_equity = (1 + strategy_rets).cumprod()
    
    # Align Bench
    bench_equity = bench_equity.reindex(strategy_equity.index)
    
    # Metrics
    total_ret = (strategy_equity.iloc[-1] - 1) * 100
    bench_total_ret = (bench_equity.iloc[-1] - 1) * 100
    
    # Max Drawdown
    roll_max = strategy_equity.cummax()
    drawdown = strategy_equity / roll_max - 1
    max_dd = drawdown.min() * 100

    print("\n=== Backtest Results (Strategy vs HS300) ===")
    print(f"Period: {dates[0].date()} to {dates[-1].date()}")
    print(f"Strategy Return: {total_ret:.2f}%")
    print(f"Benchmark Return: {bench_total_ret:.2f}%")
    print(f"Excess Return:   {total_ret - bench_total_ret:.2f}%")
    print(f"Max Drawdown:    {max_dd:.2f}%")
    
    # Save curve
    res_df = pd.DataFrame({
        "Strategy": strategy_equity,
        "Benchmark": bench_equity
    })
    res_df.to_csv("backtest_results.csv")
    print("Equity curve saved to backtest_results.csv")

if __name__ == "__main__":
    run_backtest()
