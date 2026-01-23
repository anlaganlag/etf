import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

# --- Config ---
INITIAL_CAPITAL = 1_000_000.0
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_DIR = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10

def load_data():
    price_dict = {}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv") and "etf_list" not in f]
    for f in files:
        code = f.replace(".csv", "")
        if not (code.startswith('sh') or code.startswith('sz')): continue
        try:
            df = pd.read_csv(os.path.join(CACHE_DIR, f))
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            price_dict[code] = df['收盘']
        except: pass
    full_df = pd.DataFrame(price_dict).sort_index()
    return full_df[START_DATE:END_DATE]

def backtest_stats(prices, roll_rets, holding_period):
    dates = prices.index
    strategy_rets = []
    current_holdings = []
    win_count = 0
    trade_count = 0
    
    for i in range(len(dates) - 1):
        today = dates[i]
        tomorrow = dates[i+1]
        
        # Rebalance check
        if i % holding_period == 0:
            daily_scores = pd.Series(0, index=prices.columns)
            valid_mask = prices.loc[today].notna()
            for d, weight in SCORES.items():
                r_d = roll_rets[d].loc[today]
                valid_r = r_d[valid_mask & (r_d > -100)]
                if not valid_r.empty:
                    thresh = max(10, int(len(valid_r) * 0.1))
                    top_codes = valid_r.nlargest(thresh).index
                    daily_scores.loc[top_codes] += weight
            current_holdings = daily_scores.nlargest(TOP_N).index.tolist()
            trade_count += 1

        # Calc Return
        if current_holdings:
            p_tomorrow = prices.loc[tomorrow, current_holdings]
            p_today = prices.loc[today, current_holdings]
            rets = (p_tomorrow - p_today) / p_today
            day_ret = rets.mean()
            if day_ret > 0: win_count += 1
            strategy_rets.append(day_ret)
        else:
            strategy_rets.append(0)

    equity = (1 + pd.Series(strategy_rets)).cumprod()
    final_equity = equity.iloc[-1] * INITIAL_CAPITAL
    total_ret = (equity.iloc[-1] - 1) * 100
    max_dd = (equity / equity.cummax() - 1).min() * 100
    win_rate = (win_count / len(strategy_rets)) * 100 if strategy_rets else 0
    
    return {
        "Period": f"T{holding_period}",
        "Total_Return_%": round(total_ret, 2),
        "Max_Drawdown_%": round(max_dd, 2),
        "Final_Asset": round(final_equity, 2),
        "Daily_Win_Rate_%": round(win_rate, 2),
        "Score": round(total_ret / abs(max_dd), 2) if max_dd != 0 else 0
    }

def main():
    prices = load_data()
    roll_rets = {d: prices.pct_change(periods=d).fillna(-999) for d in SCORES.keys()}
    
    all_results = []
    for p in range(1, 21):
        stats = backtest_stats(prices, roll_rets, p)
        all_results.append(stats)
    
    res_df = pd.DataFrame(all_results)
    # Output Table
    print("\n=== Holding Period Performance: T1 to T20 (Detailed) ===")
    print(res_df.to_string(index=False))
    
    # Save to CSV
    res_df.to_csv("holding_period_comparison.csv", index=False)
    print("\nDetailed verification data saved to: holding_period_comparison.csv")

if __name__ == "__main__":
    main()
