# coding=utf-8
"""
Rolling策略TOP_N优化测试
对比 T=6 vs T=10 在不同持仓数量下的表现
"""
from __future__ import print_function, absolute_import
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv
from config import config

load_dotenv()

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
            if curr_price < entry * (1 - stop_loss):
                to_sell.append(sym)
                continue
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
    history_prices = prices_df[prices_df.index <= current_dt]
    if len(history_prices) < 251:
        return None, None

    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    base_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        ranks = rets.rank(ascending=False, method='min')
        base_scores += (ranks <= threshold) * pts

    valid_base = base_scores[base_scores.index.isin(whitelist)]

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


def backtest_rolling(T, top_n, start_date='2021-12-01', end_date='2026-01-27'):
    """执行单次回测"""
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
    prices_df = prices_df[(prices_df.index >= start_date) & (prices_df.index <= end_date)]

    initial_cash = 1000000
    tranches = [Tranche(i, initial_cash / T) for i in range(T)]

    equity_curve = []
    days_count = 0

    for current_dt in prices_df.index:
        days_count += 1
        price_map = prices_df.loc[current_dt].to_dict()

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

        rebalance_idx = (days_count - 1) % T
        active_tranche = tranches[rebalance_idx]

        for sym in list(active_tranche.holdings.keys()):
            price = price_map.get(sym, 0)
            if price > 0:
                active_tranche.sell(sym, price)

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

        total_value = sum(t.total_value for t in tranches)
        equity_curve.append({
            'date': current_dt,
            'total_value': total_value,
            'return_pct': (total_value / initial_cash - 1) * 100
        })

    equity_df = pd.DataFrame(equity_curve)
    final_return = equity_df['return_pct'].iloc[-1]
    equity_df['cummax'] = equity_df['total_value'].cummax()
    equity_df['drawdown'] = (equity_df['total_value'] / equity_df['cummax'] - 1) * 100
    max_drawdown = equity_df['drawdown'].min()

    daily_returns = equity_df['total_value'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0

    years = len(equity_df) / 252
    annualized_return = ((equity_df['total_value'].iloc[-1] / initial_cash) ** (1/years) - 1) * 100 if years > 0 else 0

    risk_adj_ratio = final_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'T': T,
        'TOP_N': top_n,
        'Final_Return': final_return,
        'Annualized_Return': annualized_return,
        'Max_Drawdown': max_drawdown,
        'Sharpe': sharpe,
        'Risk_Adj_Ratio': risk_adj_ratio
    }


def main():
    print("\n" + "="*80)
    print("Rolling策略 T值 × TOP_N 组合优化测试")
    print("测试区间: 2021-12-01 至 2026-01-27")
    print("="*80)

    # Test combinations
    T_values = [6, 10]
    TOP_N_values = [3, 5, 8, 10]

    results = []

    for T in T_values:
        for top_n in TOP_N_values:
            print(f"\n测试 T={T}, TOP_N={top_n}...")
            result = backtest_rolling(T, top_n)
            results.append(result)
            print(f"  收益: {result['Final_Return']:.2f}%")
            print(f"  年化: {result['Annualized_Return']:.2f}%")
            print(f"  回撤: {result['Max_Drawdown']:.2f}%")
            print(f"  夏普: {result['Sharpe']:.2f}")
            print(f"  风险调整比: {result['Risk_Adj_Ratio']:.2f}")

    # Summary
    print("\n" + "="*80)
    print("完整对比表")
    print("="*80)
    summary_df = pd.DataFrame(results)
    summary_df = summary_df.sort_values(by='Risk_Adj_Ratio', ascending=False)
    print(summary_df.to_string(index=False, float_format='%.2f'))

    # Find Best
    best_return = max(results, key=lambda x: x['Final_Return'])
    best_sharpe = max(results, key=lambda x: x['Sharpe'])
    best_risk_adj = max(results, key=lambda x: x['Risk_Adj_Ratio'])

    print("\n" + "="*80)
    print("🏆 优化结论")
    print("="*80)
    print(f"最高收益: T={best_return['T']}, TOP_N={best_return['TOP_N']} → {best_return['Final_Return']:.2f}%")
    print(f"最佳夏普: T={best_sharpe['T']}, TOP_N={best_sharpe['TOP_N']} → {best_sharpe['Sharpe']:.2f}")
    print(f"最优风险调整: T={best_risk_adj['T']}, TOP_N={best_risk_adj['TOP_N']} → {best_risk_adj['Risk_Adj_Ratio']:.2f}")

    # Save
    output_path = os.path.join(config.BASE_DIR, 'output', 'data', 'rolling_T_TopN_optimization.csv')
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
