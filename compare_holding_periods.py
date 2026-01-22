import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# Config
CACHE_DIR = "data_cache"
BENCHMARK_CODE = "sh510300"
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10

def load_data():
    price_dict = {}
    print("Loading cached data for comparison...")
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv") and "etf_list" not in f]
    
    for f in files:
        code = f.replace(".csv", "")
        if not (code.startswith('sh') or code.startswith('sz')): continue
        
        path = os.path.join(CACHE_DIR, f)
        try:
            df = pd.read_csv(path)
            if '日期' not in df.columns or '收盘' not in df.columns: continue
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            price_dict[code] = df['收盘']
        except: pass
            
    full_df = pd.DataFrame(price_dict).sort_index()
    return full_df[START_DATE:END_DATE]

def backtest_with_period(prices, roll_rets, holding_period):
    """
    Run backtest with a specific holding period (rebalance every N days).
    """
    dates = prices.index
    strategy_rets = []
    current_holdings = []
    
    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i+1]
        
        # Check Rebalance
        # Rebalance on day 0, day N, day 2N...
        if i % holding_period == 0:
            daily_scores = pd.Series(0, index=prices.columns)
            valid_mask = prices.loc[today].notna()
            
            for d, weight in SCORES.items():
                r_d = roll_rets[d].loc[today]
                valid_r = r_d[valid_mask & (r_d > -100)]
                if valid_r.empty: continue
                
                # Top 10%
                threshold_n = max(10, int(len(valid_r) * 0.1))
                top_codes = valid_r.nlargest(threshold_n).index
                daily_scores.loc[top_codes] += weight

            current_holdings = daily_scores.nlargest(TOP_N).index.tolist()
            
        # Calc Return
        current_daily_ret = 0.0
        if current_holdings:
            p_tomorrow = prices.loc[tomorrow, current_holdings]
            p_today = prices.loc[today, current_holdings]
            try:
                # Simple return: (P_t+1 - P_t) / P_t
                rets = (p_tomorrow - p_today) / p_today
                current_daily_ret = rets.mean()
            except:
                current_daily_ret = 0.0
        
        if pd.isna(current_daily_ret): current_daily_ret = 0.0
        strategy_rets.append(current_daily_ret)
        
    # Metrics
    equity = (1 + pd.Series(strategy_rets)).cumprod()
    total_ret = (equity.iloc[-1] - 1) * 100
    
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    max_dd = drawdown.min() * 100
    
    return total_ret, max_dd

def main():
    prices = load_data()
    if prices.empty:
        print("No data loaded.")
        return
    
    print("Pre-calculating rolling returns...")
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = prices.pct_change(periods=d).fillna(-999)
        
    print(f"\n=== Comparing Holding Periods (T1 to T20) ===")
    print(f"Date Range: {START_DATE} to {END_DATE}")
    print(f"{'Period':<10} | {'Total Return':<15} | {'Max Drawdown':<15} | {'Score'}")
    print("-" * 60)
    
    results = []
    
    # Test T1 to T20
    for p in range(1, 21):
        ret, dd = backtest_with_period(prices, roll_rets, p)
        
        # Simple score: Return / |DD| (Calmar-like)
        score = ret / abs(dd) if dd != 0 else 0
        
        print(f"T{p:<9} | {ret:>11.2f}%    | {dd:>11.2f}%    | {score:.2f}")
        results.append({"Period": f"T{p}", "Return": ret, "MaxDD": dd, "Score": score})

    # Best Period
    best = max(results, key=lambda x: x['Return'])
    print("-" * 60)
    print(f"Best Period by Return: {best['Period']} ({best['Return']:.2f}%)")

if __name__ == "__main__":
    main()
