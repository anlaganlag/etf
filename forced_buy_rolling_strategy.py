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
    """
    强制买入滚动策略：每天都买入10支ETF，T份资金滚动
    策略逻辑：
    - T2: 2份资金，每天买入10支ETF
    - Day 1: 买入10支ETF (1/2资金)
    - Day 2: 买入10支ETF (1/2资金)
    - Day 3: 卖出Day 1的10支，买入新的10支
    - Day 4: 卖出Day 2的10支，买入新的10支
    """
    def __init__(self, capital, name_map, holding_period):
        self.initial_capital = capital
        self.name_map = name_map
        self.holding_period = holding_period  # T值：持有天数，也是资金份数
        self.total_capital = capital * holding_period  # 总共需要T倍资金

        # ETF批次管理：记录每批买入的ETF及其到期时间
        self.batches = []  # [{'date': date, 'etfs': [code1, code2, ...], 'expiry': expiry_date}]

        self.cash = self.total_capital
        self.holdings = {}  # code -> {'shares': qty, 'batch_date': date}
        self.history = []
        self.trade_log = []
        self.daily_holdings_log = []

    def get_total_value(self, prices):
        holdings_value = 0.0
        for code, info in self.holdings.items():
            if code in prices and not pd.isna(prices[code]):
                holdings_value += info['shares'] * prices[code]
        return self.cash + holdings_value

    def record_daily_snapshot(self, date, prices):
        """Record daily holdings snapshot"""
        total_val = self.get_total_value(prices)
        if not self.holdings:
            self.daily_holdings_log.append({
                "date": date, "code": "CASH", "name": "现金", "theme": "现金",
                "qty": 0, "value": f"{self.cash:.2f}", "weight": "100%"
            })
        else:
            for code, info in self.holdings.items():
                name = self.name_map.get(code, "Unknown")
                theme = get_theme_normalized(name)
                value = info['shares'] * prices.get(code, 0)
                weight = f"{value/total_val*100:.1f}%" if total_val > 0 else "0%"
                self.daily_holdings_log.append({
                    "date": date, "code": code, "name": name, "theme": theme,
                    "qty": info['shares'], "value": f"{value:.2f}", "weight": weight
                })

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

def run_forced_buy_rolling_strategy(holding_period):
    """
    运行强制买入滚动策略
    holding_period (T值): 同时持有的资金份数/ETF批次数
    """
    print(f"运行强制买入滚动策略 T{holding_period}...")

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

    print(f"策略配置: T{holding_period}, 总资金: ${pf.total_capital:,.0f}")

    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]

        # 记录每日价值
        pf.record_daily_snapshot(today, closes.loc[today])
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

        # 打印进度
        if i % 30 == 0:
            current_value = pf.get_total_value(closes.loc[today])
            print("{:.1f}".format(current_value))
    # 保存结果
    pd.DataFrame(pf.daily_holdings_log).to_csv(f"daily_holdings_forced_buy_T{holding_period}.csv", index=False)
    pd.DataFrame(pf.trade_log).to_csv(f"trade_log_forced_buy_T{holding_period}.csv", index=False)

    # 计算绩效
    h_series = pd.Series([h['value'] for h in pf.history])
    total_ret = (h_series.iloc[-1] - pf.total_capital) / pf.total_capital * 100
    max_dd = ((h_series / h_series.cummax() - 1).min()) * 100

    print("\n=== 强制买入滚动策略 T{} 结果 ===".format(holding_period))
    print("总收益率: {:.2f}%".format(total_ret))
    print("最大回撤: {:.2f}%".format(max_dd))
    print("总交易次数: {}".format(len(pf.trade_log)))
    print("资金使用: ${:,.0f} (基础资金 ${:,.0f} × {}份)".format(
        pf.total_capital, INITIAL_CAPITAL, holding_period))

    return total_ret, max_dd, len(pf.trade_log)

def compare_forced_buy_strategies():
    """比较不同T值的强制买入滚动策略"""
    print("="*60)
    print("强制买入滚动策略对比分析")
    print("="*60)

    results = []
    for t in range(2, 6):  # 测试T2, T3, T4, T5
        print(f"\n测试强制买入滚动策略 T{t}...")
        ret, dd, trades = run_forced_buy_rolling_strategy(t)
        results.append({
            'T': t,
            'Return': ret,
            'MaxDD': dd,
            'Trades': trades,
            'Total_Capital': INITIAL_CAPITAL * t
        })

    print("\n=== 策略对比结果 ===")
    print("T | 收益率% | 回撤% | 交易次数 | 总资金(万)")
    print("-" * 45)
    for r in results:
        print("{} | {:>8.2f} | {:>6.2f} | {:>8d} | {:>10.0f}".format(
            r['T'], r['Return'], r['MaxDD'], r['Trades'], r['Total_Capital']/10000
        ))

if __name__ == "__main__":
    compare_forced_buy_strategies()