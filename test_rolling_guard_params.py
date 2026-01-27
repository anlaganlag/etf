# coding=utf-8
"""
Rolling策略防御参数优化 - Low Hanging Fruit
测试止损、止盈、评分阈值的最优组合
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
                     start_date='2024-09-01', end_date='2026-01-27'):
    """执行回测"""
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
    trade_count = 0
    stop_loss_count = 0
    take_profit_count = 0

    for current_dt in prices_df.index:
        days_count += 1
        price_map = prices_df.loc[current_dt].to_dict()

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
                    tranche.sell(sym, price_map.get(sym, 0))

        rebalance_idx = (days_count - 1) % T
        active_tranche = tranches[rebalance_idx]

        # Sell all
        for sym in list(active_tranche.holdings.keys()):
            price = price_map.get(sym, 0)
            if price > 0:
                active_tranche.sell(sym, price)

        # Buy new
        ranking_df, total_scores = get_ranking(prices_df, whitelist, theme_map, current_dt, min_score)
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
                            trade_count += 1

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
    risk_adj_ratio = final_return / abs(max_drawdown) if max_drawdown != 0 else 0

    return {
        'Final_Return': final_return,
        'Max_Drawdown': max_drawdown,
        'Sharpe': sharpe,
        'Risk_Adj_Ratio': risk_adj_ratio,
        'Trade_Count': trade_count,
        'Stop_Loss_Count': stop_loss_count,
        'Take_Profit_Count': take_profit_count
    }


def main():
    print("\n" + "="*80)
    print("🔧 Rolling策略防御参数优化 (Low Hanging Fruit)")
    print("基准配置: T=6, TOP_N=5, 测试区间: 2024-09至今")
    print("="*80)

    T = 6
    top_n = 5

    # 当前基准
    baseline = backtest_rolling(T, top_n, 0.20, 0.10, 0.05, 20)
    print(f"\n📊 当前基准 (STOP_LOSS=0.20, TRAILING_TRIGGER=0.10, TRAILING_DROP=0.05, MIN_SCORE=20):")
    print(f"  收益: {baseline['Final_Return']:.2f}%")
    print(f"  回撤: {baseline['Max_Drawdown']:.2f}%")
    print(f"  风险调整比: {baseline['Risk_Adj_Ratio']:.2f}")
    print(f"  止损次数: {baseline['Stop_Loss_Count']}")
    print(f"  止盈次数: {baseline['Take_Profit_Count']}")

    print("\n" + "="*80)
    print("测试1: 止损参数优化 (STOP_LOSS)")
    print("="*80)

    stop_loss_results = []
    for sl in [0.15, 0.18, 0.20, 0.22, 0.25]:
        result = backtest_rolling(T, top_n, sl, 0.10, 0.05, 20)
        stop_loss_results.append({
            'STOP_LOSS': sl,
            'Return': result['Final_Return'],
            'Drawdown': result['Max_Drawdown'],
            'Risk_Adj': result['Risk_Adj_Ratio'],
            'SL_Count': result['Stop_Loss_Count']
        })
        print(f"SL={sl:.2f}: 收益{result['Final_Return']:.2f}% 回撤{result['Max_Drawdown']:.2f}% 风险调整比{result['Risk_Adj_Ratio']:.2f} 止损{result['Stop_Loss_Count']}次")

    sl_df = pd.DataFrame(stop_loss_results).sort_values(by='Risk_Adj', ascending=False)
    best_sl = sl_df.iloc[0]
    print(f"\n✅ 最优止损: {best_sl['STOP_LOSS']:.2f} (风险调整比: {best_sl['Risk_Adj']:.2f})")

    print("\n" + "="*80)
    print("测试2: 追踪止盈触发点优化 (TRAILING_TRIGGER)")
    print("="*80)

    trigger_results = []
    for tt in [0.08, 0.10, 0.12, 0.15]:
        result = backtest_rolling(T, top_n, best_sl['STOP_LOSS'], tt, 0.05, 20)
        trigger_results.append({
            'TRAILING_TRIGGER': tt,
            'Return': result['Final_Return'],
            'Drawdown': result['Max_Drawdown'],
            'Risk_Adj': result['Risk_Adj_Ratio'],
            'TP_Count': result['Take_Profit_Count']
        })
        print(f"TT={tt:.2f}: 收益{result['Final_Return']:.2f}% 回撤{result['Max_Drawdown']:.2f}% 风险调整比{result['Risk_Adj_Ratio']:.2f} 止盈{result['Take_Profit_Count']}次")

    tt_df = pd.DataFrame(trigger_results).sort_values(by='Risk_Adj', ascending=False)
    best_tt = tt_df.iloc[0]
    print(f"\n✅ 最优触发点: {best_tt['TRAILING_TRIGGER']:.2f} (风险调整比: {best_tt['Risk_Adj']:.2f})")

    print("\n" + "="*80)
    print("测试3: 追踪止盈回撤幅度优化 (TRAILING_DROP)")
    print("="*80)

    drop_results = []
    for td in [0.03, 0.05, 0.07, 0.10]:
        result = backtest_rolling(T, top_n, best_sl['STOP_LOSS'], best_tt['TRAILING_TRIGGER'], td, 20)
        drop_results.append({
            'TRAILING_DROP': td,
            'Return': result['Final_Return'],
            'Drawdown': result['Max_Drawdown'],
            'Risk_Adj': result['Risk_Adj_Ratio']
        })
        print(f"TD={td:.2f}: 收益{result['Final_Return']:.2f}% 回撤{result['Max_Drawdown']:.2f}% 风险调整比{result['Risk_Adj_Ratio']:.2f}")

    td_df = pd.DataFrame(drop_results).sort_values(by='Risk_Adj', ascending=False)
    best_td = td_df.iloc[0]
    print(f"\n✅ 最优回撤幅度: {best_td['TRAILING_DROP']:.2f} (风险调整比: {best_td['Risk_Adj']:.2f})")

    print("\n" + "="*80)
    print("测试4: 最低评分阈值优化 (MIN_SCORE)")
    print("="*80)

    score_results = []
    for ms in [10, 20, 50, 100]:
        result = backtest_rolling(T, top_n, best_sl['STOP_LOSS'], best_tt['TRAILING_TRIGGER'], best_td['TRAILING_DROP'], ms)
        score_results.append({
            'MIN_SCORE': ms,
            'Return': result['Final_Return'],
            'Drawdown': result['Max_Drawdown'],
            'Risk_Adj': result['Risk_Adj_Ratio'],
            'Trade_Count': result['Trade_Count']
        })
        print(f"MS={ms:3d}: 收益{result['Final_Return']:.2f}% 回撤{result['Max_Drawdown']:.2f}% 风险调整比{result['Risk_Adj_Ratio']:.2f} 交易{result['Trade_Count']}次")

    ms_df = pd.DataFrame(score_results).sort_values(by='Risk_Adj', ascending=False)
    best_ms = ms_df.iloc[0]
    print(f"\n✅ 最优评分阈值: {best_ms['MIN_SCORE']:.0f} (风险调整比: {best_ms['Risk_Adj']:.2f})")

    # Final Test
    print("\n" + "="*80)
    print("🏆 最终对比")
    print("="*80)

    optimal = backtest_rolling(T, top_n, best_sl['STOP_LOSS'], best_tt['TRAILING_TRIGGER'],
                               best_td['TRAILING_DROP'], best_ms['MIN_SCORE'])

    print(f"\n原始配置:")
    print(f"  参数: SL=0.20, TT=0.10, TD=0.05, MS=20")
    print(f"  收益: {baseline['Final_Return']:.2f}%")
    print(f"  回撤: {baseline['Max_Drawdown']:.2f}%")
    print(f"  风险调整比: {baseline['Risk_Adj_Ratio']:.2f}")

    print(f"\n优化后配置:")
    print(f"  参数: SL={best_sl['STOP_LOSS']:.2f}, TT={best_tt['TRAILING_TRIGGER']:.2f}, TD={best_td['TRAILING_DROP']:.2f}, MS={best_ms['MIN_SCORE']:.0f}")
    print(f"  收益: {optimal['Final_Return']:.2f}%")
    print(f"  回撤: {optimal['Max_Drawdown']:.2f}%")
    print(f"  风险调整比: {optimal['Risk_Adj_Ratio']:.2f}")

    improvement = ((optimal['Risk_Adj_Ratio'] - baseline['Risk_Adj_Ratio']) / baseline['Risk_Adj_Ratio']) * 100
    print(f"\n📈 风险调整比提升: {improvement:+.1f}%")

    # Save
    summary = pd.DataFrame([{
        'Config': 'Baseline',
        'STOP_LOSS': 0.20,
        'TRAILING_TRIGGER': 0.10,
        'TRAILING_DROP': 0.05,
        'MIN_SCORE': 20,
        'Return': baseline['Final_Return'],
        'Drawdown': baseline['Max_Drawdown'],
        'Risk_Adj': baseline['Risk_Adj_Ratio']
    }, {
        'Config': 'Optimized',
        'STOP_LOSS': best_sl['STOP_LOSS'],
        'TRAILING_TRIGGER': best_tt['TRAILING_TRIGGER'],
        'TRAILING_DROP': best_td['TRAILING_DROP'],
        'MIN_SCORE': best_ms['MIN_SCORE'],
        'Return': optimal['Final_Return'],
        'Drawdown': optimal['Max_Drawdown'],
        'Risk_Adj': optimal['Risk_Adj_Ratio']
    }])

    output_path = os.path.join(config.BASE_DIR, 'output', 'data', 'rolling_guard_params_optimization.csv')
    summary.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✅ 结果已保存至: {output_path}")


if __name__ == '__main__':
    main()
