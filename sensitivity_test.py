import pandas as pd
import numpy as np
import os
from compare_rolling_vs_periodic import load_data, get_signals, calculate_metrics

CAPITAL = 10_000_000.0
COST = 0.0001
TOP_N = 10
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}

def run_periodic_with_offset(T, offset, closes, opens, roll_rets, name_map):
    cash = CAPITAL
    holdings = {}
    history = []
    dates = closes.index
    
    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]
        
        val = cash + sum(q * closes.loc[today].get(c, 0) for c, q in holdings.items())
        history.append(val)
        
        # 调仓逻辑：(i + offset) % T == 0
        if (i + offset) % T == 0:
            target_codes = get_signals(today, closes, roll_rets, name_map)
            exec_prices = opens.loc[next_day]
            
            # Sells
            for code in list(holdings.keys()):
                if code not in target_codes:
                    p = exec_prices.get(code, 0)
                    if not pd.isna(p) and p > 0:
                        cash += holdings[code] * p * (1 - COST)
                        del holdings[code]
            
            # Buys
            if target_codes:
                current_val = cash + sum(q * exec_prices.get(c, 0) for c, q in holdings.items())
                target_per_etf = current_val / TOP_N
                for code in target_codes:
                    price = exec_prices.get(code, 0)
                    if pd.isna(price) or price <= 0: continue
                    curr_qty = holdings.get(code, 0)
                    curr_val = curr_qty * price
                    if curr_val < target_per_etf * 0.95:
                        to_buy_val = target_per_etf - curr_val
                        shares = int(to_buy_val / (price * (1 + COST))) // 100 * 100
                        if shares > 0:
                            cash -= shares * price * (1 + COST)
                            holdings[code] = holdings.get(code, 0) + shares
    
    val = cash + sum(q * closes.iloc[-1].get(c, 0) for c, q in holdings.items())
    history.append(val)
    return history

def main():
    closes, opens, name_map = load_data()
    roll_rets = {d: closes.pct_change(periods=d).fillna(-1) for d in SCORES.keys()}
    
    T = 12
    print(f"--- Periodic T={T} 灵敏度测试 (偏移量 0 ~ {T-1}) ---")
    results = []
    for offset in range(T):
        hist = run_periodic_with_offset(T, offset, closes, opens, roll_rets, name_map)
        ret, dd, sharpe = calculate_metrics(hist)
        results.append({'offset': offset, 'return': ret, 'max_dd': dd})
        print(f"偏移 {offset:2d} 天 | 收益: {ret:7.2f}% | 回撤: {dd:7.2f}%")

    df = pd.DataFrame(results)
    print("\n灵敏度分析:")
    print(f"最大收益: {df['return'].max():.2f}%")
    print(f"最小收益: {df['return'].min():.2f}%")
    print(f"收益标准差: {df['return'].std():.2f}%")
    print(f"收益中位数: {df['return'].median():.2f}%")

if __name__ == "__main__":
    main()
