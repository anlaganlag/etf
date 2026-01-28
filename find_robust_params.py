# coding=utf-8
"""
寻找稳健参数配置
测试多种参数组合，找到在牛市和全周期都表现优秀的配置
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
                    self.pos_records[sym]['high_price'] = max(self.pos_records[sym]['high_price'], price)
        self.total_value = val

    def check_guard(self, price_map, stop_loss, trailing_trigger, trailing_drop):
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


def get_ranking(prices_df, whitelist, theme_map, current_dt, min_score):
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


def backtest_rolling(T, top_n, stop_loss, trailing_trigger, trailing_drop, min_score,
                     start_date, end_date, prices_df, whitelist, theme_map):
    filtered_df = prices_df[(prices_df.index >= start_date) & (prices_df.index <= end_date)]

    initial_cash = 1000000
    tranches = [Tranche(i, initial_cash / T) for i in range(T)]

    equity_curve = []
    days_count = 0

    for current_dt in filtered_df.index:
        days_count += 1
        price_map = filtered_df.loc[current_dt].to_dict()

        for tranche in tranches:
            tranche.update_value(price_map)
            if tranche.rest_days > 0:
                tranche.rest_days -= 1

            to_sell_list, is_tp = tranche.check_guard(price_map, stop_loss, trailing_trigger, trailing_drop)
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

        ranking_df, total_scores = get_ranking(filtered_df, whitelist, theme_map, current_dt, min_score)
        exposure = get_market_exposure(filtered_df, current_dt, total_scores)

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
        equity_curve.append(total_value)

    equity_series = pd.Series(equity_curve)
    final_return = (equity_series.iloc[-1] / initial_cash - 1) * 100
    cummax = equity_series.cummax()
    drawdown = (equity_series / cummax - 1) * 100
    max_drawdown = drawdown.min()

    daily_returns = equity_series.pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() > 0 else 0
    risk_adj_ratio = final_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'Return': final_return,
        'Drawdown': max_drawdown,
        'Sharpe': sharpe,
        'Risk_Adj': risk_adj_ratio
    }


def main():
    print("\n" + "="*80)
    print("🔍 寻找稳健参数配置")
    print("="*80)

    # Load Data
    print("\n加载数据...")
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
    print(f"数据加载完成: {prices_df.shape}")

    # Test parameter combinations
    print("\n测试参数组合...")

    param_combos = [
        ("原始配置", 0.20, 0.10, 0.05, 20),
        ("激进止盈", 0.20, 0.08, 0.03, 20),
        ("高门槛", 0.20, 0.10, 0.05, 50),
        ("平衡配置", 0.18, 0.09, 0.04, 30),
        ("全优化", 0.15, 0.08, 0.03, 50),
    ]

    T = 6
    top_n = 5

    bull_start = '2024-09-01'
    bull_end = '2026-01-27'
    long_start = '2021-12-01'
    long_end = '2026-01-27'

    results = []

    for name, sl, tt, td, ms in param_combos:
        print(f"\n测试 {name} (SL={sl}, TT={tt}, TD={td}, MS={ms})...")

        bull_result = backtest_rolling(T, top_n, sl, tt, td, ms,
                                       bull_start, bull_end, prices_df, whitelist, theme_map)
        long_result = backtest_rolling(T, top_n, sl, tt, td, ms,
                                       long_start, long_end, prices_df, whitelist, theme_map)

        # 计算稳健性得分 (两个环境都好才是真的好)
        stability_score = min(bull_result['Risk_Adj'], long_result['Risk_Adj'])

        results.append({
            '配置': name,
            'STOP_LOSS': sl,
            'TRAILING_TRIGGER': tt,
            'TRAILING_DROP': td,
            'MIN_SCORE': ms,
            '牛市收益': bull_result['Return'],
            '牛市回撤': bull_result['Drawdown'],
            '牛市风险调整比': bull_result['Risk_Adj'],
            '全周期收益': long_result['Return'],
            '全周期回撤': long_result['Drawdown'],
            '全周期风险调整比': long_result['Risk_Adj'],
            '稳健性得分': stability_score
        })

        print(f"  牛市: 收益{bull_result['Return']:.2f}% 回撤{bull_result['Drawdown']:.2f}% 风险调整比{bull_result['Risk_Adj']:.2f}")
        print(f"  全周期: 收益{long_result['Return']:.2f}% 回撤{long_result['Drawdown']:.2f}% 风险调整比{long_result['Risk_Adj']:.2f}")
        print(f"  稳健性得分: {stability_score:.2f}")

    # 排序并展示
    print("\n" + "="*80)
    print("📊 参数配置对比 (按稳健性排序)")
    print("="*80)

    df_results = pd.DataFrame(results).sort_values(by='稳健性得分', ascending=False)
    print("\n" + df_results[['配置', '牛市风险调整比', '全周期风险调整比', '稳健性得分']].to_string(index=False, float_format='%.2f'))

    # 最佳配置
    best = df_results.iloc[0]

    print("\n" + "="*80)
    print("🏆 推荐配置")
    print("="*80)
    print(f"\n配置名称: {best['配置']}")
    print(f"\n参数设置:")
    print(f"  STOP_LOSS = {best['STOP_LOSS']}")
    print(f"  TRAILING_TRIGGER = {best['TRAILING_TRIGGER']}")
    print(f"  TRAILING_DROP = {best['TRAILING_DROP']}")
    print(f"  MIN_SCORE = {best['MIN_SCORE']}")

    print(f"\n性能指标:")
    print(f"  牛市: 收益{best['牛市收益']:.2f}% 回撤{best['牛市回撤']:.2f}% 风险调整比{best['牛市风险调整比']:.2f}")
    print(f"  全周期: 收益{best['全周期收益']:.2f}% 回撤{best['全周期回撤']:.2f}% 风险调整比{best['全周期风险调整比']:.2f}")
    print(f"  稳健性得分: {best['稳健性得分']:.2f}")

    # Save
    output_path = os.path.join(config.BASE_DIR, 'output', 'data', 'robust_params_comparison.csv')
    df_results.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
