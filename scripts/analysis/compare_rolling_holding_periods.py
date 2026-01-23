import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Config ---
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0001  # 万1佣金
SLIPPAGE = 0.001         # 千1滑点
TOP_N = 10               # 始终保持10只ETF
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_DIR = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}

def get_theme_normalized(name):
    """More robust theme extraction for meaningful grouping"""
    if not name or pd.isna(name): return "Unknown"
    name = name.lower()
    # Priority keywords for grouping
    keywords = ["芯片", "半导体", "人工智能", "ai", "红利", "银行", "机器人", "光伏", "白酒", "医药", "医疗", "军工", "新能源", "券商", "证券", "黄金", "纳斯达克", "标普", "信创", "软件", "房地产", "中药", "2000", "1000", "500", "300"]
    for k in keywords:
        if k in name: return k
    # Fallback
    theme = name.replace("etf", "").replace("基金", "").replace("增强", "").replace("指数", "")
    for word in ["中证", "沪深", "上证", "深证", "科创", "创业板", "港股通", "300", "500", "1000", "50", "100"]:
        theme = theme.replace(word, "")
    return theme.strip() if theme.strip() else "宽基"

class RollingPortfolioTester:
    """
    滚动持仓策略测试器：测试不同持有期下的滚动策略表现
    """
    def __init__(self, capital, name_map, holding_period):
        self.initial_capital = capital
        self.name_map = name_map
        self.holding_period = holding_period  # T值：每个ETF的持有天数
        self.reset()

    def reset(self):
        self.cash = self.initial_capital
        self.holdings = {}  # code -> {'shares': qty, 'entry_date': date, 'hold_days': days}
        self.holding_queue = []  # 持仓队列，按进入时间排序 [oldest, ..., newest]
        self.history = []
        self.trade_log = []

    def get_total_value(self, prices):
        holdings_value = 0.0
        for code, info in self.holdings.items():
            if code in prices and not pd.isna(prices[code]):
                holdings_value += info['shares'] * prices[code]
        return self.cash + holdings_value

    def update_holding_days(self):
        """更新所有持仓的持有天数"""
        for code in self.holdings:
            self.holdings[code]['hold_days'] += 1

    def get_expired_holdings(self):
        """获取持有超过holding_period的ETF"""
        expired = []
        for code in self.holding_queue:
            if code in self.holdings and self.holdings[code]['hold_days'] >= self.holding_period:
                expired.append(code)
        return expired

    def is_fully_invested(self):
        """检查是否已满仓"""
        return len(self.holdings) >= TOP_N

    def order(self, code, qty, price, action, date):
        """Execute order"""
        if action == "BUY":
            cost = qty * price * (1 + COMMISSION_RATE + SLIPPAGE)
            if self.cash >= cost:
                self.cash -= cost
                if code not in self.holdings:
                    self.holdings[code] = {'shares': 0, 'entry_date': date, 'hold_days': 0}
                    self.holding_queue.append(code)
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
                    if code in self.holding_queue:
                        self.holding_queue.remove(code)
                self.trade_log.append({
                    "date": date, "code": code, "name": self.name_map.get(code, ""),
                    "action": action, "price": price, "shares": qty,
                    "total_amt": revenue, "remaining_cash": self.cash
                })

    def run_backtest(self, closes, opens, roll_rets, dates):
        """运行滚动持仓回测"""
        for i in range(len(dates) - 1):
            today = dates[i]
            next_day = dates[i+1]

            # 更新持有天数
            self.update_holding_days()

            # 记录每日价值
            self.history.append({"date": today, "value": self.get_total_value(closes.loc[today])})

            # 计算每日信号
            daily_scores = pd.Series(0, index=closes.columns)
            valid_mask = closes.loc[today].notna()
            for d, weight in SCORES.items():
                r_d = roll_rets[d].loc[today]
                valid_r = r_d[valid_mask & (r_d > -100)]
                if not valid_r.empty:
                    threshold = max(10, int(len(valid_r) * 0.1))
                    top_codes = valid_r.nlargest(threshold).index
                    daily_scores.loc[top_codes] += weight

            # 获取当前top holdings (with sector limit)
            sorted_candidates = daily_scores.sort_values(ascending=False).index
            target_holdings = []
            theme_counts = {}

            for code in sorted_candidates:
                if len(target_holdings) >= TOP_N: break
                theme = get_theme_normalized(self.name_map.get(code, ""))
                count = theme_counts.get(theme, 0)
                if count < 1:  # 每行业限1只
                    target_holdings.append(code)
                    theme_counts[theme] = count + 1

            # 执行滚动交易
            exec_prices = opens.loc[next_day]

            if not self.is_fully_invested():
                # 建仓阶段
                current_codes = set(self.holding_queue)
                target_codes = set(target_holdings)
                to_buy = target_codes - current_codes

                if to_buy:
                    positions_needed = TOP_N - len(self.holding_queue)
                    if positions_needed > 0:
                        cash_per_position = self.cash / positions_needed
                        for code in list(to_buy)[:positions_needed]:
                            price = exec_prices.get(code, 0)
                            if not pd.isna(price) and price > 0:
                                shares = int(cash_per_position / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                                shares = (shares // 100) * 100
                                if shares > 0:
                                    self.order(code, shares, price, "BUY", next_day)
            else:
                # 滚动阶段：替换过期的ETF
                expired_codes = self.get_expired_holdings()
                current_codes = set(self.holding_queue)
                target_codes = set(target_holdings)

                to_sell = set(expired_codes) - target_codes  # 需要卖出的
                to_buy = target_codes - current_codes      # 需要买入的

                # 卖出过期ETF
                total_sell_value = 0
                for code in to_sell:
                    if code in self.holdings:
                        price = exec_prices.get(code, 0)
                        if not pd.isna(price) and price > 0:
                            shares = self.holdings[code]['shares']
                            self.order(code, shares, price, "SELL", next_day)
                            total_sell_value += shares * price * (1 - COMMISSION_RATE - SLIPPAGE)

                # 买入新ETF
                if to_buy and total_sell_value > 0:
                    cash_per_new = total_sell_value / len(to_buy)
                    for code in to_buy:
                        price = exec_prices.get(code, 0)
                        if not pd.isna(price) and price > 0:
                            shares = int(cash_per_new / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                            shares = (shares // 100) * 100
                            if shares > 0:
                                self.order(code, shares, price, "BUY", next_day)

def load_data():
    """加载数据"""
    list_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("etf_list_")]
    name_map = {}
    if list_files:
        l_df = pd.read_csv(os.path.join(CACHE_DIR, sorted(list_files)[-1]))
        name_map = dict(zip(l_df['etf_code'], l_df['etf_name']))

    price_dict = {}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv") and "etf_list" not in f]
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

    return closes, opens, name_map

def run_rolling_comparison():
    """运行T1-T20滚动持仓策略对比"""
    print("开始T1-T20滚动持仓策略对比测试...")

    # 加载数据
    closes, opens, name_map = load_data()

    # 预计算信号
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = closes.pct_change(periods=d).fillna(-999)

    dates = closes.index
    results = []

    # 测试T1到T20
    for t in range(1, 21):
        print(f"\n测试 Rolling T{t} 策略...")

        # 创建策略实例
        strategy = RollingPortfolioTester(INITIAL_CAPITAL, name_map, t)
        strategy.run_backtest(closes, opens, roll_rets, dates)

        # 计算绩效指标
        h_series = pd.Series([h['value'] for h in strategy.history])
        total_ret = (h_series.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

        roll_max = h_series.cummax()
        drawdown = h_series / roll_max - 1
        max_dd = drawdown.min() * 100

        # 简单风险调整得分
        score = total_ret / abs(max_dd) if max_dd != 0 else 0

        results.append({
            "Period": f"T{t}",
            "T_Value": t,
            "Return": total_ret,
            "MaxDD": max_dd,
            "Score": score,
            "Trades": len(strategy.trade_log)
        })

        print(".2f"
              ".2f"
              ".3f"
              f"交易次数: {len(strategy.trade_log)}")

    return results

def analyze_results(results):
    """分析结果"""
    df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("T1-T20 滚动持仓策略对比分析报告")
    print("="*80)

    # 最佳策略
    best_overall = df.loc[df['Score'].idxmax()]
    best_return = df.loc[df['Return'].idxmax()]
    best_risk = df.loc[df['MaxDD'].idxmax()]

    print("\n=== 最佳策略 ===")
    print("最佳综合: T{} (得分: {:.3f})".format(best_overall['T_Value'], best_overall['Score']))
    print("最高收益: T{} ({:.2f}%)".format(best_return['T_Value'], best_return['Return']))
    print("最低风险: T{} (回撤: {:.2f}%)".format(best_risk['T_Value'], best_risk['MaxDD']))

    # 统计分析
    profitable = len(df[df['Return'] > 0])
    print("\n=== 统计分析 ===")
    print("盈利策略: {}/20 ({:.1f}%)".format(profitable, profitable/20*100))
    print("平均收益率: {:.2f}%".format(df['Return'].mean()))
    print("平均最大回撤: {:.2f}%".format(df['MaxDD'].mean()))
    print("平均交易次数: {:.0f}".format(df['Trades'].mean()))

    # 趋势分析
    corr_return = df['T_Value'].corr(df['Return'])
    corr_risk = df['T_Value'].corr(df['MaxDD'])
    print("\n=== 趋势分析 ===")
    print("持有期与收益相关性: {:.3f}".format(corr_return))
    print("持有期与风险相关性: {:.3f}".format(corr_risk))

    # 保存结果
    df.to_csv("rolling_holding_period_comparison.csv", index=False)
    print("\n结果已保存至: rolling_holding_period_comparison.csv")
    print("\n=== 详细数据 ===")
    print("T | 收益率% | 回撤% | 得分 | 交易次数")
    print("-" * 40)
    for _, row in df.iterrows():
        print("{} | {:>8.2f} | {:>6.2f} | {:.3f} | {:>8.0f}".format(
            row['T_Value'], row['Return'], row['MaxDD'], row['Score'], row['Trades']
        ))

if __name__ == "__main__":
    results = run_rolling_comparison()
    analyze_results(results)
    print("\nT1-T20滚动持仓策略对比测试完成!")