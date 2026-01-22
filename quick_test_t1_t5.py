import pandas as pd
import numpy as np
import os
from datetime import datetime

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

def run_quick_test(t):
    """快速测试单个T值"""
    print(f"快速测试 T{t} 策略...")

    # 加载数据
    list_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("etf_list_")]
    name_map = {}
    if list_files:
        l_df = pd.read_csv(os.path.join(CACHE_DIR, sorted(list_files)[-1]))
        name_map = dict(zip(l_df['etf_code'], l_df['etf_name']))

    price_dict = {}
    files = [f for f in os.listdir(CACHE_DIR) if f.endswith(".csv") and "etf_list" not in f][:50]  # 只加载前50个文件加速测试
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
    pf = ForcedBuyRollingPortfolio(INITIAL_CAPITAL, name_map, t)
    dates = closes.index[:30]  # 只测试前30天

    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]

        # 记录每日价值
        pf.history.append({"date": today, "value": pf.get_total_value(closes.loc[today])})

        # 简化的信号计算（为了加速）
        top_etfs = list(closes.columns[:TOP_N])  # 直接取前10个ETF作为top
        exec_prices = opens.loc[next_day]

        # 检查到期批次
        expired_batches = pf.get_expired_batches(next_day)
        for batch in expired_batches:
            for code in batch['etfs']:
                if code in pf.holdings:
                    shares = pf.holdings[code]['shares']
                    price = exec_prices.get(code, 0)
                    if not pd.isna(price) and price > 0 and shares > 0:
                        # 简化的卖出逻辑
                        pf.cash += shares * price * (1 - COMMISSION_RATE - SLIPPAGE)
                        pf.holdings[code]['shares'] -= shares
                        if pf.holdings[code]['shares'] == 0:
                            del pf.holdings[code]
            pf.remove_batch(batch)

        # 买入新批次
        active_batches = len(pf.batches)
        if active_batches < t:
            capital_per_batch = INITIAL_CAPITAL
            capital_per_etf = capital_per_batch / TOP_N

            for code in top_etfs:
                price = exec_prices.get(code, 0)
                if not pd.isna(price) and price > 0:
                    shares = int(capital_per_etf / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                    shares = (shares // 100) * 100
                    if shares > 0:
                        pf.order(code, shares, price, "BUY", next_day)

            expiry_date = next_day + pd.Timedelta(days=t)
            pf.batches.append({
                'date': next_day,
                'etfs': top_etfs.copy(),
                'expiry': expiry_date
            })

    # 计算结果
    h_series = pd.Series([h['value'] for h in pf.history])
    total_ret = (h_series.iloc[-1] - pf.total_capital) / pf.total_capital * 100 if len(h_series) > 0 else 0
    max_dd = ((h_series / h_series.cummax() - 1).min()) * 100 if len(h_series) > 1 else 0

    return {
        'T': t,
        'Return': total_ret,
        'MaxDD': max_dd,
        'Trades': len(pf.trade_log),
        'Total_Capital': pf.total_capital
    }

def quick_compare_t1_t5():
    """快速比较T1-T5"""
    print("快速测试 T1-T5 强制买入滚动策略")
    print("="*50)

    results = []
    for t in range(1, 6):
        result = run_quick_test(t)
        results.append(result)
        print("T{}: 收益率{:.2f}%, 回撤{:.2f}%, 交易{}次, 资金{}万".format(
            result['T'], result['Return'], result['MaxDD'],
            result['Trades'], result['Total_Capital']/10000
        ))

    # 简单排序
    sorted_results = sorted(results, key=lambda x: x['Return'], reverse=True)
    print("\n收益率排名:")
    for i, r in enumerate(sorted_results, 1):
        print("{}. T{}: {:.2f}%".format(i, r['T'], r['Return']))

if __name__ == "__main__":
    quick_compare_t1_t5()