import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Config ---
INITIAL_CAPITAL = 1_000_000.0  # 基础资金
COMMISSION_RATE = 0.0001
SLIPPAGE = 0.001
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")
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

class ForcedBuyRollingPortfolio:
    """强制买入滚动策略：每天都买入10支ETF，T份资金滚动"""

    def __init__(self, capital, name_map, holding_period):
        self.initial_capital = capital
        self.name_map = name_map
        self.holding_period = holding_period  # T值：同时持有的资金份数
        self.total_capital = capital * holding_period  # 总共需要T倍资金

        # ETF批次管理：记录每批买入的ETF及其到期时间
        self.batches = []  # [{'date': date, 'etfs': [code1, code2, ...], 'expiry': expiry_date}]

        self.cash = self.total_capital
        self.holdings = {}  # code -> {'shares': qty, 'batch_date': date}
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
                    self.holdings[code] = {'shares': 0, 'batch_date': date}
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

    def get_expired_batches(self, current_date):
        """获取到期的批次"""
        expired = []
        for batch in self.batches:
            if current_date >= batch['expiry']:
                expired.append(batch)
        return expired

    def remove_batch(self, batch):
        """从批次列表中移除"""
        if batch in self.batches:
            self.batches.remove(batch)

def run_forced_buy_strategy_t(holding_period):
    """运行单个T值的强制买入滚动策略"""
    print(f"测试强制买入滚动策略 T{holding_period}...")

    # 加载数据
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

    # 初始化策略
    pf = ForcedBuyRollingPortfolio(INITIAL_CAPITAL, name_map, holding_period)
    dates = closes.index

    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]

        # 记录每日价值
        pf.history.append({"date": today, "value": pf.get_total_value(closes.loc[today])})

        # 计算每日top10信号
        daily_scores = pd.Series(0, index=closes.columns)
        valid_mask = closes.loc[today].notna()
        for d, weight in SCORES.items():
            r_d = closes.pct_change(periods=d).loc[today]
            valid_r = r_d[valid_mask & (r_d > -100)]
            if not valid_r.empty:
                threshold = max(10, int(len(valid_r) * 0.1))
                top_codes = valid_r.nlargest(threshold).index
                daily_scores.loc[top_codes] += weight

        # 获取top10 ETF（允许重复）
        top_etfs = daily_scores.nlargest(TOP_N).index.tolist()
        exec_prices = opens.loc[next_day]

        # 检查是否有到期的批次需要卖出
        expired_batches = pf.get_expired_batches(next_day)
        for batch in expired_batches:
            # 卖出到期批次的ETF
            for code in batch['etfs']:
                if code in pf.holdings:
                    shares = pf.holdings[code]['shares']
                    price = exec_prices.get(code, 0)
                    if not pd.isna(price) and price > 0 and shares > 0:
                        pf.order(code, shares, price, "SELL", next_day)
            pf.remove_batch(batch)

        # 计算当前活跃批次数
        active_batches = len(pf.batches)

        # 如果活跃批次少于T，买入新一批ETF
        if active_batches < holding_period:
            # 计算每只ETF的资金
            capital_per_batch = INITIAL_CAPITAL  # 每批次使用基础资金量
            capital_per_etf = capital_per_batch / TOP_N

            # 买入top10 ETF（强制买入，即使重复）
            for code in top_etfs:
                price = exec_prices.get(code, 0)
                if not pd.isna(price) and price > 0:
                    shares = int(capital_per_etf / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                    shares = (shares // 100) * 100
                    if shares > 0:
                        pf.order(code, shares, price, "BUY", next_day)

            # 记录这一批次的买入
            expiry_date = next_day + pd.Timedelta(days=holding_period)
            pf.batches.append({
                'date': next_day,
                'etfs': top_etfs.copy(),
                'expiry': expiry_date
            })

    # 计算绩效指标
    h_series = pd.Series([h['value'] for h in pf.history])
    total_ret = (h_series.iloc[-1] - pf.total_capital) / pf.total_capital * 100
    max_dd = ((h_series / h_series.cummax() - 1).min()) * 100

    return total_ret, max_dd, len(pf.trade_log), pf.total_capital

def compare_all_t1_t20():
    """比较T1-T20的所有强制买入滚动策略"""
    print("="*80)
    print("强制买入滚动策略 T1-T20 完整对比分析")
    print("="*80)

    results = []

    # 测试T1-T20
    for t in range(1, 21):  # T1 to T20
        print(f"\n测试 T{t} 策略...")
        try:
            ret, dd, trades, total_capital = run_forced_buy_strategy_t(t)
            results.append({
                'T': t,
                'Return': ret,
                'MaxDD': dd,
                'Trades': trades,
                'Total_Capital': total_capital,
                'Capital_Multiple': t
            })
            print("收益率: {:.2f}%, 最大回撤: {:.2f}%, 交易次数: {}, 总资金: {:.0f}万".format(
                ret, dd, trades, total_capital/10000))
        except Exception as e:
            print(f"T{t} 测试失败: {e}")
            continue

    print("\n" + "="*80)
    print("T1-T20 强制买入滚动策略综合排名")
    print("="*80)

    # 按收益率排序
    sorted_by_return = sorted(results, key=lambda x: x['Return'], reverse=True)
    print("\n=== 按收益率排名 (最高到最低) ===")
    print("排名 | T值 | 收益率% | 回撤% | 得分 | 交易次数 | 资金倍数")
    print("-" * 70)
    for i, r in enumerate(sorted_by_return[:10], 1):  # 只显示前10名
        score = r['Return'] / abs(r['MaxDD']) if r['MaxDD'] != 0 else 0
        print("{} | {} | {:>8.2f} | {:>6.2f} | {:.3f} | {:>8d} | {}x".format(
            i, r['T'], r['Return'], r['MaxDD'], score, r['Trades'], r['Capital_Multiple']
        ))

    # 按风险调整得分排序
    sorted_by_score = sorted(results, key=lambda x: x['Return'] / abs(x['MaxDD']) if x['MaxDD'] != 0 else 0, reverse=True)
    print("\n=== 按风险调整得分排名 (最高到最低) ===")
    print("排名 | T值 | 收益率% | 回撤% | 得分 | 交易次数 | 资金倍数")
    print("-" * 70)
    for i, r in enumerate(sorted_by_score[:10], 1):  # 只显示前10名
        score = r['Return'] / abs(r['MaxDD']) if r['MaxDD'] != 0 else 0
        print("{} | {} | {:>8.2f} | {:>6.2f} | {:.3f} | {:>8d} | {}x".format(
            i, r['T'], r['Return'], r['MaxDD'], score, r['Trades'], r['Capital_Multiple']
        ))

    # 统计分析
    returns = [r['Return'] for r in results]
    maxdds = [r['MaxDD'] for r in results]
    trades = [r['Trades'] for r in results]

    print("\n=== 统计汇总 ===")
    print("测试策略数量: {}".format(len(results)))
    print("平均收益率: {:.2f}%".format(np.mean(returns)))
    print("最高收益率: {:.2f}% (T{})".format(max(returns), results[returns.index(max(returns))]['T']))
    print("最低收益率: {:.2f}% (T{})".format(min(returns), results[returns.index(min(returns))]['T']))
    print("平均最大回撤: {:.2f}%".format(np.mean(maxdds)))
    print("平均交易次数: {:.0f}".format(np.mean(trades)))

    # 趋势分析
    t_values = [r['T'] for r in results]
    print("\n=== 趋势分析 ===")
    print("T值与收益率相关性: {:.3f}".format(np.corrcoef(t_values, returns)[0, 1]))
    print("T值与回撤相关性: {:.3f}".format(np.corrcoef(t_values, maxdds)[0, 1]))
    print("T值与交易次数相关性: {:.3f}".format(np.corrcoef(t_values, trades)[0, 1]))

    print("\n=== 关键发现 ===")
    print("1. 最佳策略: T{} ({:.2f}%)".format(sorted_by_score[0]['T'], sorted_by_score[0]['Return']))
    print("2. 最差策略: T{} ({:.2f}%)".format(sorted_by_return[-1]['T'], sorted_by_return[-1]['Return']))
    print("3. 资金效率: T值越大，资金需求越高，但收益不一定更好")

    # 保存完整结果
    df = pd.DataFrame(results)
    df.to_csv("forced_buy_t1_t20_comparison.csv", index=False)
    print("\n完整结果已保存至: forced_buy_t1_t20_comparison.csv")

if __name__ == "__main__":
    compare_all_t1_t20()