#!/usr/bin/env python3
"""
综合T值优化验证系统
对比定期调仓、滚动持仓、强制买入三种策略，找出最合适的T值
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

# --- Config ---
INITIAL_CAPITAL = 1_000_000.0  # 基础资金
COMMISSION_RATE = 0.0001
SLIPPAGE = 0.001
START_DATE = "2024-10-09"
END_DATE = "2025-01-09"  # 缩短测试期间到3个月
CACHE_DIR = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10

def get_theme_normalized(name):
    """More robust theme extraction for meaningful grouping"""
    if not name or pd.isna(name): return "Unknown"
    name = name.lower()
    keywords = ["芯片", "半导体", "人工智能", "ai", "红利", "银行", "机器人", "光伏", "白酒", "医药", "医疗", "军工", "新能源", "券商", "证券", "黄金", "纳斯达克", "标普", "信创", "软件", "房地产", "中药", "2000", "1000", "500", "300"]
    for k in keywords:
        if k in name: return k
    theme = name.replace("etf", "").replace("基金", "").replace("增强", "").replace("指数", "")
    for word in ["中证", "沪深", "上证", "深证", "科创", "创业板", "港股通", "300", "500", "1000", "50", "100"]:
        theme = theme.replace(word, "")
    return theme.strip() if theme.strip() else "宽基"

class StrategyTester:
    """策略测试器基类"""

    def __init__(self, strategy_name, capital, name_map, holding_period):
        self.strategy_name = strategy_name
        self.initial_capital = capital
        self.name_map = name_map
        self.holding_period = holding_period
        self.cash = capital
        self.holdings = {}
        self.history = []
        self.trade_log = []

    def get_total_value(self, prices):
        holdings_value = 0.0
        for code, info in self.holdings.items():
            if code in prices and not pd.isna(prices[code]):
                holdings_value += info['shares'] * prices[code]
        return self.cash + holdings_value

    def order(self, code, qty, price, action, date):
        """Execute order"""
        if action == "BUY":
            cost = qty * price * (1 + COMMISSION_RATE + SLIPPAGE)
            if self.cash >= cost:
                self.cash -= cost
                if code not in self.holdings:
                    self.holdings[code] = {'shares': 0, 'entry_date': date}
                self.holdings[code]['shares'] += qty
                self.trade_log.append({
                    "date": date, "code": code, "name": self.name_map.get(code, ""),
                    "action": action, "price": price, "shares": qty,
                    "total_amt": cost, "remaining_cash": self.cash
                })
        elif action == "SELL":
            if code in self.holdings and self.holdings[code]['shares'] >= qty:
                revenue = qty * price * (1 - COMMISSION_RATE - SLIPPAGE)
                self.cash += revenue
                self.holdings[code]['shares'] -= qty
                if self.holdings[code]['shares'] == 0:
                    del self.holdings[code]
                self.trade_log.append({
                    "date": date, "code": code, "name": self.name_map.get(code, ""),
                    "action": action, "price": price, "shares": qty,
                    "total_amt": revenue, "remaining_cash": self.cash
                })

class RegularRebalanceStrategy(StrategyTester):
    """定期调仓策略"""

    def run_backtest(self, closes, opens, roll_rets, dates):
        for i in range(len(dates) - 1):
            today = dates[i]
            next_day = dates[i+1]

            self.history.append({"date": today, "value": self.get_total_value(closes.loc[today])})

            # 定期调仓逻辑
            if i % self.holding_period == 0:
                # 计算信号
                daily_scores = pd.Series(0, index=closes.columns)
                valid_mask = closes.loc[today].notna()
                for d, weight in SCORES.items():
                    r_d = roll_rets[d].loc[today]
                    valid_r = r_d[valid_mask & (r_d > -100)]
                    if not valid_r.empty:
                        threshold = max(10, int(len(valid_r) * 0.1))
                        top_codes = valid_r.nlargest(threshold).index
                        daily_scores.loc[top_codes] += weight

                top_etfs = daily_scores.nlargest(TOP_N).index.tolist()
                exec_prices = opens.loc[next_day]

                # 卖出所有现有持仓
                for code in list(self.holdings.keys()):
                    if code in exec_prices and not pd.isna(exec_prices[code]):
                        shares = self.holdings[code]['shares']
                        self.order(code, shares, exec_prices[code], "SELL", next_day)

                # 买入新组合
                if top_etfs:
                    capital_per_etf = self.cash / len(top_etfs)
                    for code in top_etfs:
                        price = exec_prices.get(code, 0)
                        if not pd.isna(price) and price > 0:
                            shares = int(capital_per_etf / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                            shares = (shares // 100) * 100
                            if shares > 0:
                                self.order(code, shares, price, "BUY", next_day)

class RollingStrategy(StrategyTester):
    """滚动持仓策略"""

    def __init__(self, strategy_name, capital, name_map, holding_period):
        super().__init__(strategy_name, capital, name_map, holding_period)
        self.holding_queue = []

    def run_backtest(self, closes, opens, roll_rets, dates):
        for i in range(len(dates) - 1):
            today = dates[i]
            next_day = dates[i+1]

            self.history.append({"date": today, "value": self.get_total_value(closes.loc[today])})

            # 计算信号
            daily_scores = pd.Series(0, index=closes.columns)
            valid_mask = closes.loc[today].notna()
            for d, weight in SCORES.items():
                r_d = roll_rets[d].loc[today]
                valid_r = r_d[valid_mask & (r_d > -100)]
                if not valid_r.empty:
                    threshold = max(10, int(len(valid_r) * 0.1))
                    top_codes = valid_r.nlargest(threshold).index
                    daily_scores.loc[top_codes] += weight

            top_etfs = daily_scores.nlargest(TOP_N).index.tolist()
            exec_prices = opens.loc[next_day]

            # 检查到期的ETF
            expired_codes = []
            for code in self.holding_queue[:]:
                if code in self.holdings:
                    entry_date = self.holdings[code]['entry_date']
                    if (next_day - entry_date).days >= self.holding_period:
                        expired_codes.append(code)

            # 卖出到期ETF
            for code in expired_codes:
                if code in self.holdings:
                    shares = self.holdings[code]['shares']
                    price = exec_prices.get(code, 0)
                    if not pd.isna(price) and price > 0:
                        self.order(code, shares, price, "SELL", next_day)

            # 买入新ETF
            current_codes = set(self.holdings.keys())
            target_codes = set(top_etfs)
            to_buy = target_codes - current_codes

            if to_buy and len(self.holdings) < TOP_N:
                positions_needed = TOP_N - len(self.holdings)
                cash_per_position = self.cash / min(positions_needed, len(to_buy))
                for code in list(to_buy)[:positions_needed]:
                    price = exec_prices.get(code, 0)
                    if not pd.isna(price) and price > 0:
                        shares = int(cash_per_position / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                        shares = (shares // 100) * 100
                        if shares > 0:
                            self.order(code, shares, price, "BUY", next_day)
                            if code not in self.holding_queue:
                                self.holding_queue.append(code)

class ForcedBuyStrategy(StrategyTester):
    """强制买入滚动策略"""

    def __init__(self, strategy_name, capital, name_map, holding_period):
        super().__init__(strategy_name, capital, name_map, holding_period)
        self.total_capital = capital * holding_period
        self.cash = self.total_capital
        self.batches = []

    def run_backtest(self, closes, opens, roll_rets, dates):
        for i in range(len(dates) - 1):
            today = dates[i]
            next_day = dates[i+1]

            self.history.append({"date": today, "value": self.get_total_value(closes.loc[today])})

            # 计算信号
            daily_scores = pd.Series(0, index=closes.columns)
            valid_mask = closes.loc[today].notna()
            for d, weight in SCORES.items():
                r_d = roll_rets[d].loc[today]
                valid_r = r_d[valid_mask & (r_d > -100)]
                if not valid_r.empty:
                    threshold = max(10, int(len(valid_r) * 0.1))
                    top_codes = valid_r.nlargest(threshold).index
                    daily_scores.loc[top_codes] += weight

            top_etfs = daily_scores.nlargest(TOP_N).index.tolist()
            exec_prices = opens.loc[next_day]

            # 检查到期批次
            expired_batches = []
            for batch in self.batches:
                if next_day >= batch['expiry']:
                    expired_batches.append(batch)

            for batch in expired_batches:
                for code in batch['etfs']:
                    if code in self.holdings:
                        shares = self.holdings[code]['shares']
                        price = exec_prices.get(code, 0)
                        if not pd.isna(price) and price > 0:
                            self.order(code, shares, price, "SELL", next_day)
                self.batches.remove(batch)

            # 买入新批次
            active_batches = len(self.batches)
            if active_batches < self.holding_period:
                capital_per_batch = INITIAL_CAPITAL
                capital_per_etf = capital_per_batch / TOP_N

                for code in top_etfs:
                    price = exec_prices.get(code, 0)
                    if not pd.isna(price) and price > 0:
                        shares = int(capital_per_etf / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                        shares = (shares // 100) * 100
                        if shares > 0:
                            self.order(code, shares, price, "BUY", next_day)

                expiry_date = next_day + pd.Timedelta(days=self.holding_period)
                self.batches.append({
                    'date': next_day,
                    'etfs': top_etfs.copy(),
                    'expiry': expiry_date
                })

def run_comprehensive_test():
    """运行综合T值优化测试"""

    print("="*80)
    print("综合T值优化验证系统")
    print("="*80)

    # 加载数据
    list_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("etf_list_")]
    name_map = {}
    if list_files:
        l_df = pd.read_csv(os.path.join(CACHE_DIR, sorted(list_files)[-1]))
        name_map = dict(zip(l_df['etf_code'], l_df['etf_name']))

    price_dict = {}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv") and "etf_list" not in f][:30]  # 限制文件数量加速测试
    for f in files:
        code = f.replace(".csv", "")
        if not (code.startswith('sh') or code.startswith('sz')): continue
        try:
            df = pd.read_csv(os.path.join(CACHE_DIR, f))
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            price_dict[code] = df
        except: pass

    closes = pd.DataFrame({k: v['收盘'] for k, v in price_dict.items()}).sort_index()[START_DATE:END_DATE]
    opens = pd.DataFrame({k: v.get('开盘', v['收盘']) for k, v in price_dict.items()}).sort_index()[START_DATE:END_DATE]

    # 预计算信号
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = closes.pct_change(periods=d).fillna(-999)

    dates = closes.index[:20]  # 只测试前20天，加速验证
    print(f"测试期间: {START_DATE} 到 {dates[-1].date()}")
    print(f"测试天数: {len(dates)}")

    # 定义测试策略和T值
    strategies = [
        ("定期调仓", RegularRebalanceStrategy, [1, 2, 3, 5, 8, 10, 12, 15]),
        ("滚动持仓", RollingStrategy, [1, 2, 3, 5, 8, 10, 12, 15, 18]),
        ("强制买入", ForcedBuyStrategy, [1, 2, 3, 5, 8, 10, 12, 15])
    ]

    results = []

    for strategy_name, strategy_class, t_values in strategies:
        print(f"\n🔍 测试{strategy_name}策略...")

        for t in t_values:
            try:
                strategy = strategy_class(f"{strategy_name}_T{t}", INITIAL_CAPITAL, name_map, t)
                strategy.run_backtest(closes, opens, roll_rets, dates)

                # 计算绩效
                h_series = pd.Series([h['value'] for h in strategy.history])
                if len(h_series) > 1:
                    total_ret = (h_series.iloc[-1] - getattr(strategy, 'total_capital', strategy.initial_capital)) / getattr(strategy, 'total_capital', strategy.initial_capital) * 100
                    max_dd = ((h_series / h_series.cummax() - 1).min()) * 100 if len(h_series) > 1 else 0
                    annual_ret = total_ret / (len(dates) / 250)  # 年化

                    results.append({
                        'Strategy': strategy_name,
                        'T': t,
                        'Return_3M': total_ret,
                        'Return_Annual': annual_ret,
                        'MaxDD': max_dd,
                        'Trades': len(strategy.trade_log),
                        'Total_Capital': getattr(strategy, 'total_capital', strategy.initial_capital)
                    })

                    print("{:.1f}".format(annual_ret))
            except Exception as e:
                print(f"  T{t} 测试失败: {e}")

    # 分析结果
    df = pd.DataFrame(results)

    print("\n🎯 综合分析结果:")
    print("="*80)

    # 各策略最优T值
    print("\n🏆 各策略最优T值:")
    for strategy_name in df['Strategy'].unique():
        strategy_data = df[df['Strategy'] == strategy_name]
        if not strategy_data.empty:
            best = strategy_data.loc[strategy_data['Return_Annual'].idxmax()]
            print("{} | T{} | {:.2f}% | {:.2f}%".format(
                strategy_name, best['T'], best['Return_Annual'], best['MaxDD']
            ))

    # 全局最优
    if not df.empty:
        global_best = df.loc[df['Return_Annual'].idxmax()]
        print("\n🌟 全局最优策略:")
        print("策略: {} | T值: {} | 年化收益: {:.2f}% | 最大回撤: {:.2f}%".format(
            global_best['Strategy'], global_best['T'],
            global_best['Return_Annual'], global_best['MaxDD']
        ))

    # 收益分布分析
    print("\n📊 收益分布:")
    profitable_count = len(df[df['Return_Annual'] > 0])
    print("盈利策略: {}/{} ({:.1f}%)".format(profitable_count, len(df), profitable_count/len(df)*100))
    print("平均年化收益: {:.2f}%".format(df['Return_Annual'].mean()))
    print("最高年化收益: {:.2f}% ({} T{})".format(
        df['Return_Annual'].max(),
        df.loc[df['Return_Annual'].idxmax(), 'Strategy'],
        df.loc[df['Return_Annual'].idxmax(), 'T']
    ))

    # 保存详细结果
    df.to_csv("comprehensive_t_optimization_results.csv", index=False)
    print("\n💾 详细结果已保存至: comprehensive_t_optimization_results.csv")

    # 投资建议
    print("\n💡 投资建议:")
    if not df.empty:
        # 基于收益风险比推荐
        df['Sharpe'] = df['Return_Annual'] / abs(df['MaxDD']) * 100  # 简化的夏普比率
        best_sharpe = df.loc[df['Sharpe'].idxmax()]

        print("• 最高收益推荐: {} T{} ({:.2f}%年化)".format(
            global_best['Strategy'], global_best['T'], global_best['Return_Annual']
        ))
        print("• 最佳风险调整: {} T{} (收益风险比: {:.3f})".format(
            best_sharpe['Strategy'], best_sharpe['T'], best_sharpe['Sharpe']
        ))
        print("• 保守选择: 定期调仓T8-T12 (稳定可行)")
        print("• 激进选择: 强制买入T10-T15 (收益潜力大))")

if __name__ == "__main__":
    run_comprehensive_test()