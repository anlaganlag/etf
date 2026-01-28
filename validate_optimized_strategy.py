# coding=utf-8
"""
优化后策略全面回测验证
对比原始配置 vs 优化配置在不同市场环境下的表现
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


def backtest_rolling(config_name, T, top_n, stop_loss, trailing_trigger, trailing_drop, min_score,
                     start_date, end_date, prices_df, whitelist, theme_map):
    """执行回测"""
    filtered_df = prices_df[(prices_df.index >= start_date) & (prices_df.index <= end_date)]

    initial_cash = 1000000
    tranches = [Tranche(i, initial_cash / T) for i in range(T)]

    equity_curve = []
    days_count = 0
    trade_count = 0
    stop_loss_count = 0
    take_profit_count = 0
    win_count = 0
    loss_count = 0
    total_pnl = 0

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
                    take_profit_count += len(to_sell_list)
                else:
                    stop_loss_count += len(to_sell_list)

                for sym in to_sell_list:
                    if sym in tranche.pos_records:
                        entry = tranche.pos_records[sym]['entry_price']
                        curr = price_map.get(sym, 0)
                        pnl = (curr / entry - 1) if entry > 0 else 0
                        if pnl > 0:
                            win_count += 1
                        else:
                            loss_count += 1
                        total_pnl += pnl
                    tranche.sell(sym, price_map.get(sym, 0))

        rebalance_idx = (days_count - 1) % T
        active_tranche = tranches[rebalance_idx]

        # Sell all
        for sym in list(active_tranche.holdings.keys()):
            price = price_map.get(sym, 0)
            if price > 0:
                active_tranche.sell(sym, price)

        # Buy new
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
                            trade_count += 1

        active_tranche.update_value(price_map)

        total_value = sum(t.total_value for t in tranches)
        equity_curve.append({
            'date': current_dt,
            'config': config_name,
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
    risk_adj_ratio = final_return / abs(max_drawdown) if max_drawdown != 0 else 0

    win_rate = (win_count / (win_count + loss_count) * 100) if (win_count + loss_count) > 0 else 0

    return {
        'Config': config_name,
        'Period': f"{start_date} to {end_date}",
        'Final_Return': final_return,
        'Max_Drawdown': max_drawdown,
        'Sharpe': sharpe,
        'Risk_Adj_Ratio': risk_adj_ratio,
        'Trade_Count': trade_count,
        'Stop_Loss_Count': stop_loss_count,
        'Take_Profit_Count': take_profit_count,
        'Win_Rate': win_rate,
        'Days': len(equity_df),
        'equity_curve': equity_df
    }


def main():
    print("\n" + "="*80)
    print("🔬 优化后策略全面回测验证")
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

    # Define test periods
    test_periods = [
        ("🐂 强势牛市", '2024-09-01', '2026-01-27'),
        ("📊 全周期", '2021-12-01', '2026-01-27'),
        ("📉 2022熊市", '2022-01-01', '2022-12-31'),
        ("📈 2023震荡", '2023-01-01', '2023-12-31'),
        ("🚀 最近2个月", '2025-11-27', '2026-01-27'),
    ]

    # Configurations
    configs = [
        ("原始配置", 6, 5, 0.20, 0.10, 0.05, 20),
        ("优化配置", 6, 5, 0.15, 0.08, 0.03, 50),
    ]

    all_results = []
    all_curves = []

    for period_name, start, end in test_periods:
        print(f"\n{'='*80}")
        print(f"{period_name}: {start} 至 {end}")
        print(f"{'='*80}")

        for cfg_name, T, top_n, sl, tt, td, ms in configs:
            print(f"\n测试 {cfg_name}...")
            result = backtest_rolling(cfg_name, T, top_n, sl, tt, td, ms,
                                     start, end, prices_df, whitelist, theme_map)

            result['Period_Name'] = period_name
            all_results.append(result)
            all_curves.append(result['equity_curve'])

            print(f"  收益: {result['Final_Return']:.2f}%")
            print(f"  回撤: {result['Max_Drawdown']:.2f}%")
            print(f"  夏普: {result['Sharpe']:.2f}")
            print(f"  风险调整比: {result['Risk_Adj_Ratio']:.2f}")
            print(f"  胜率: {result['Win_Rate']:.1f}%")
            print(f"  止盈/止损: {result['Take_Profit_Count']}/{result['Stop_Loss_Count']}")

    # Summary comparison
    print("\n" + "="*80)
    print("📊 全面对比汇总")
    print("="*80)

    summary_df = pd.DataFrame([{
        '测试区间': r['Period_Name'],
        '配置': r['Config'],
        '收益(%)': f"{r['Final_Return']:.2f}",
        '回撤(%)': f"{r['Max_Drawdown']:.2f}",
        '夏普': f"{r['Sharpe']:.2f}",
        '风险调整比': f"{r['Risk_Adj_Ratio']:.2f}",
        '胜率(%)': f"{r['Win_Rate']:.1f}",
        '交易次数': r['Trade_Count']
    } for r in all_results])

    print("\n" + summary_df.to_string(index=False))

    # Calculate improvements
    print("\n" + "="*80)
    print("📈 优化效果统计")
    print("="*80)

    for period_name, _, _ in test_periods:
        period_results = [r for r in all_results if r['Period_Name'] == period_name]
        if len(period_results) == 2:
            baseline = period_results[0]
            optimized = period_results[1]

            return_imp = optimized['Final_Return'] - baseline['Final_Return']
            dd_imp = optimized['Max_Drawdown'] - baseline['Max_Drawdown']
            risk_adj_imp = ((optimized['Risk_Adj_Ratio'] - baseline['Risk_Adj_Ratio']) /
                           baseline['Risk_Adj_Ratio'] * 100) if baseline['Risk_Adj_Ratio'] != 0 else 0

            print(f"\n{period_name}:")
            print(f"  收益提升: {return_imp:+.2f}% ({baseline['Final_Return']:.2f}% → {optimized['Final_Return']:.2f}%)")
            print(f"  回撤改善: {dd_imp:+.2f}% ({baseline['Max_Drawdown']:.2f}% → {optimized['Max_Drawdown']:.2f}%)")
            print(f"  风险调整比提升: {risk_adj_imp:+.1f}% ({baseline['Risk_Adj_Ratio']:.2f} → {optimized['Risk_Adj_Ratio']:.2f})")

    # Save results
    output_dir = os.path.join(config.BASE_DIR, 'output', 'data')

    summary_df.to_csv(os.path.join(output_dir, 'validation_summary.csv'), index=False, encoding='utf-8-sig')

    # Save equity curves
    equity_combined = pd.concat(all_curves, ignore_index=True)
    equity_combined.to_csv(os.path.join(output_dir, 'validation_equity_curves.csv'), index=False)

    print(f"\n✅ 结果已保存至 {output_dir}")

    # Final recommendation
    print("\n" + "="*80)
    print("🏆 最终结论")
    print("="*80)

    # Count wins
    wins = 0
    total = 0
    for period_name, _, _ in test_periods:
        period_results = [r for r in all_results if r['Period_Name'] == period_name]
        if len(period_results) == 2:
            total += 1
            if period_results[1]['Risk_Adj_Ratio'] > period_results[0]['Risk_Adj_Ratio']:
                wins += 1

    print(f"\n优化配置 vs 原始配置:")
    print(f"  胜出次数: {wins}/{total} ({wins/total*100:.0f}%)")

    if wins >= total * 0.8:
        print(f"\n✅ 强烈推荐: 优化配置在{wins}个测试区间中表现更优")
        print(f"  建议立即使用优化参数")
    elif wins >= total * 0.6:
        print(f"\n✅ 推荐: 优化配置在多数区间表现更好")
        print(f"  可以使用优化参数")
    else:
        print(f"\n⚠️ 谨慎: 优化效果不稳定，建议进一步测试")


if __name__ == '__main__':
    main()
