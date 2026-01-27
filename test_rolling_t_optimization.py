# coding=utf-8
"""
Rolling策略T值优化测试
对比 T=6,7,8,10,12,14 的完整表现
"""
from __future__ import print_function, absolute_import
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

from config import config

load_dotenv()

# --- Rolling Backtest Engine ---
class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {}
        self.pos_records = {}
        self.total_value = initial_cash
        self.rest_days = 0

    def update_value(self, price_map):
        val = self.cash
        for sym, shares in list(self.holdings.items()):
            price = price_map.get(sym, 0)
            if price > 0:
                val += shares * price
                if sym in self.pos_records:
                    self.pos_records[sym]['high_price'] = max(
                        self.pos_records[sym]['high_price'], price
                    )
        self.total_value = val

    def check_guard(self, price_map, stop_loss=0.20, trailing_trigger=0.10, trailing_drop=0.05):
        to_sell = []
        is_tp = False
        for sym, rec in self.pos_records.items():
            if sym not in self.holdings:
                continue
            curr_price = price_map.get(sym, 0)
            if curr_price <= 0:
                continue

            entry = rec['entry_price']
            high = rec['high_price']

            # Stop Loss
            if curr_price < entry * (1 - stop_loss):
                to_sell.append(sym)
                continue

            # Trailing Profit
            if high > entry * (1 + trailing_trigger):
                if curr_price < high * (1 - trailing_drop):
                    to_sell.append(sym)
                    is_tp = True
        return to_sell, is_tp

    def sell(self, symbol, price):
        if symbol in self.holdings:
            shares = self.holdings[symbol]
            self.cash += shares * price
            del self.holdings[symbol]
            if symbol in self.pos_records:
                del self.pos_records[symbol]

    def buy(self, symbol, cash_allocated, price):
        if price <= 0:
            return
        shares = int(cash_allocated / price / 100) * 100
        cost = shares * price
        if shares > 0 and self.cash >= cost:
            self.cash -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
            self.pos_records[symbol] = {'entry_price': price, 'high_price': price}


def get_ranking(prices_df, whitelist, theme_map, current_dt, min_score=20):
    """标准5P评分 + Theme Boost"""
    history_prices = prices_df[prices_df.index <= current_dt]
    if len(history_prices) < 251:
        return None, None

    # Base Scoring
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    base_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        ranks = rets.rank(ascending=False, method='min')
        base_scores += (ranks <= threshold) * pts

    valid_base = base_scores[base_scores.index.isin(whitelist)]

    # Theme Boost
    strong_etfs = valid_base[valid_base >= 150]
    theme_counts = {}
    for code in strong_etfs.index:
        t = theme_map.get(code, 'Unknown')
        theme_counts[t] = theme_counts.get(t, 0) + 1

    strong_themes = {t for t, count in theme_counts.items() if count >= 3}

    final_scores = valid_base.copy()
    for code in final_scores.index:
        if theme_map.get(code, 'Unknown') in strong_themes:
            final_scores[code] += 50

    valid_final = final_scores[final_scores >= min_score]
    if valid_final.empty:
        return None, base_scores

    r20 = (history_prices.iloc[-1] / history_prices.iloc[-21] - 1) if len(history_prices) > 20 else pd.Series(0.0, index=history_prices.columns)

    df = pd.DataFrame({
        'score': valid_final,
        'r20': r20[valid_final.index],
        'theme': [theme_map.get(c, 'Unknown') for c in valid_final.index]
    })
    return df.sort_values(by=['score', 'r20'], ascending=False), base_scores


def get_market_exposure(prices_df, current_dt, total_scores):
    """市场环境评估"""
    mkt_idx = 'SHSE.000001'
    if mkt_idx in prices_df.columns:
        mkt_prices = prices_df[prices_df.index <= current_dt][mkt_idx]
        if len(mkt_prices) >= 20:
            ma20 = mkt_prices.rolling(20).mean().iloc[-1]
            curr_mkt = mkt_prices.iloc[-1]
            if curr_mkt < ma20:
                return 0.0

    strong = (total_scores >= 150).sum() if total_scores is not None else 0
    return 1.0 if strong >= 5 else 0.3


def backtest_rolling_strategy(T, top_n=5, start_date='2024-09-01', end_date='2026-01-27'):
    """单次Rolling回测"""
    print(f"\n{'='*60}")
    print(f"Testing Rolling Strategy: T={T}, Top_N={top_n}")
    print(f"{'='*60}")

    # Load Data
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns:
        df_excel['theme'] = df_excel['etf_name']
    whitelist = set(df_excel['etf_code'])
    theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    # Build Price Matrix
    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR)
             if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            if code.startswith('sh'):
                code = 'SHSE.' + code[2:]
            elif code.startswith('sz'):
                code = 'SZSE.' + code[2:]
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except:
            pass

    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"Price matrix loaded: {prices_df.shape}")

    # Filter Date Range
    prices_df = prices_df[(prices_df.index >= start_date) & (prices_df.index <= end_date)]

    # Initialize Tranches
    initial_cash = 1000000
    tranches = [Tranche(i, initial_cash / T) for i in range(T)]

    # Backtest Loop
    equity_curve = []
    days_count = 0

    for current_dt in prices_df.index:
        days_count += 1
        price_map = prices_df.loc[current_dt].to_dict()

        # Update all tranches
        for tranche in tranches:
            tranche.update_value(price_map)

            if tranche.rest_days > 0:
                tranche.rest_days -= 1

            to_sell_list, is_tp = tranche.check_guard(price_map)
            if to_sell_list:
                if is_tp:
                    tranche.rest_days = 1
                for sym in to_sell_list:
                    tranche.sell(sym, price_map.get(sym, 0))

        # Rebalance active tranche
        rebalance_idx = (days_count - 1) % T
        active_tranche = tranches[rebalance_idx]

        # Sell all holdings in active tranche
        for sym in list(active_tranche.holdings.keys()):
            price = price_map.get(sym, 0)
            if price > 0:
                active_tranche.sell(sym, price)

        # Buy new targets
        ranking_df, total_scores = get_ranking(prices_df, whitelist, theme_map, current_dt)
        exposure = get_market_exposure(prices_df, current_dt, total_scores)

        if ranking_df is not None and active_tranche.rest_days == 0:
            if exposure > 0:
                targets = []
                seen = set()
                for code, row in ranking_df.iterrows():
                    if row['theme'] not in seen:
                        targets.append(code)
                        seen.add(row['theme'])
                    if len(targets) >= top_n:
                        break

                if targets:
                    invest_amt = active_tranche.cash * exposure
                    per_amt = invest_amt / len(targets)
                    for sym in targets:
                        price = price_map.get(sym, 0)
                        if price > 0:
                            active_tranche.buy(sym, per_amt, price)

        active_tranche.update_value(price_map)

        # Calculate total portfolio value
        total_value = sum(t.total_value for t in tranches)
        equity_curve.append({
            'date': current_dt,
            'total_value': total_value,
            'return_pct': (total_value / initial_cash - 1) * 100
        })

    # Calculate Metrics
    equity_df = pd.DataFrame(equity_curve)
    final_return = equity_df['return_pct'].iloc[-1]
    equity_df['cummax'] = equity_df['total_value'].cummax()
    equity_df['drawdown'] = (equity_df['total_value'] / equity_df['cummax'] - 1) * 100
    max_drawdown = equity_df['drawdown'].min()

    # Sharpe Ratio (简化计算)
    daily_returns = equity_df['total_value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    return {
        'T': T,
        'Final_Return_Pct': final_return,
        'Max_Drawdown_Pct': max_drawdown,
        'Sharpe_Ratio': sharpe,
        'Final_Value': equity_df['total_value'].iloc[-1],
        'equity_curve': equity_df
    }


def main():
    """运行完整对比测试"""
    print("\n" + "="*80)
    print("Rolling Strategy T-Value Optimization Test")
    print("测试区间: 2024-09-01 至 2026-01-27")
    print("="*80)

    # Test different T values
    T_values = [6, 7, 8, 10, 12, 14]
    results = []

    for T in T_values:
        result = backtest_rolling_strategy(T, top_n=5)
        results.append(result)

        print(f"\n[T={T}] Results:")
        print(f"  累计收益: {result['Final_Return_Pct']:.2f}%")
        print(f"  最大回撤: {result['Max_Drawdown_Pct']:.2f}%")
        print(f"  夏普比率: {result['Sharpe_Ratio']:.2f}")

    # Summary Table
    print("\n" + "="*80)
    print("Summary Comparison Table")
    print("="*80)
    summary_df = pd.DataFrame([{
        'T值': r['T'],
        '累计收益(%)': f"{r['Final_Return_Pct']:.2f}",
        '最大回撤(%)': f"{r['Max_Drawdown_Pct']:.2f}",
        '夏普比率': f"{r['Sharpe_Ratio']:.2f}",
        '终值(万元)': f"{r['Final_Value']/10000:.2f}"
    } for r in results])
    print(summary_df.to_string(index=False))

    # Save Results
    output_path = os.path.join(config.BASE_DIR, 'output', 'data', 'rolling_t_optimization_results.csv')
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")

    # Find Best T
    best_result = max(results, key=lambda x: x['Final_Return_Pct'])
    print(f"\n🏆 最优T值: T={best_result['T']}")
    print(f"   收益: {best_result['Final_Return_Pct']:.2f}%")
    print(f"   回撤: {best_result['Max_Drawdown_Pct']:.2f}%")
    print(f"   夏普: {best_result['Sharpe_Ratio']:.2f}")

    # Save detailed equity curves
    for result in results:
        curve_path = os.path.join(
            config.BASE_DIR, 'output', 'data',
            f'rolling_T{result["T"]}_equity_curve.csv'
        )
        result['equity_curve'].to_csv(curve_path, index=False)

    print(f"\n详细净值曲线已保存至 output/data/ 目录")


if __name__ == '__main__':
    main()
