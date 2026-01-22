import pandas as pd
import numpy as np
import os
from datetime import datetime

# --- Config ---
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0001  # 万1佣金
SLIPPAGE = 0.001         # 千1滑点
ROLLING_WINDOW = 10      # T10: 10只ETF滚动
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_DIR = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10

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

class RollingPortfolio:
    """
    滚动持仓策略：始终保持TOP_N只ETF，滚动替换最老的持仓
    """
    def __init__(self, capital, name_map):
        self.cash = capital
        self.holdings = {}  # code -> {'shares': qty, 'entry_date': date, 'entry_price': price}
        self.name_map = name_map
        self.history = []
        self.trade_log = []
        self.daily_holdings_log = []
        self.holding_queue = []  # 持仓队列，按进入时间排序 [oldest, ..., newest]

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
                    self.holdings[code] = {'shares': 0, 'entry_date': date, 'entry_price': price}
                    self.holding_queue.append(code)  # 新持仓加入队列末尾
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
                        self.holding_queue.remove(code)  # 从队列中移除
                self.trade_log.append({
                    "date": date, "code": code, "name": self.name_map.get(code, ""),
                    "action": action, "price": price, "shares": qty,
                    "total_amt": revenue, "remaining_cash": self.cash
                })

    def get_oldest_holding(self):
        """获取最老的持仓（队列头部）"""
        if self.holding_queue:
            return self.holding_queue[0]
        return None

    def get_current_holding_codes(self):
        """获取当前所有持仓代码"""
        return list(self.holdings.keys())

    def is_fully_invested(self):
        """检查是否已满仓（持有TOP_N只ETF）"""
        return len(self.holdings) >= TOP_N

def run_rolling_portfolio_strategy():
    """
    执行滚动持仓策略
    策略逻辑：
    1. 建仓期：逐步买入ETF直到持有10只
    2. 滚动期：每天卖出最老的ETF，买入最新的ETF
    """
    # Load data
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

    # Initialize portfolio
    pf = RollingPortfolio(INITIAL_CAPITAL, name_map)

    # Pre-calculate signals
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = closes.pct_change(periods=d).fillna(-999)

    dates = closes.index
    print(f"Running Rolling Portfolio Strategy from {START_DATE}...")

    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]

        # Record daily snapshot
        pf.record_daily_snapshot(today, closes.loc[today])
        pf.history.append({"date": today, "value": pf.get_total_value(closes.loc[today])})

        # Calculate daily scores and ranking
        daily_scores = pd.Series(0, index=closes.columns)
        valid_mask = closes.loc[today].notna()
        for d, weight in SCORES.items():
            r_d = roll_rets[d].loc[today]
            valid_r = r_d[valid_mask & (r_d > -100)]
            if not valid_r.empty:
                threshold = max(10, int(len(valid_r) * 0.1))
                top_codes = valid_r.nlargest(threshold).index
                daily_scores.loc[top_codes] += weight

        # Get current top holdings (with sector limit)
        sorted_candidates = daily_scores.sort_values(ascending=False).index
        target_holdings = []
        theme_counts = {}

        for code in sorted_candidates:
            if len(target_holdings) >= TOP_N: break
            theme = get_theme_normalized(name_map.get(code, ""))
            count = theme_counts.get(theme, 0)
            if count < 1:  # 每行业限1只
                target_holdings.append(code)
                theme_counts[theme] = count + 1

        # Execute rolling trades
        exec_prices = opens.loc[next_day]

        if not pf.is_fully_invested():
            # 建仓阶段：买入缺失的ETF
            current_codes = pf.get_current_holding_codes()
            positions_to_fill = TOP_N - len(pf.holdings)
            if positions_to_fill > 0:
                cash_per_position = pf.cash / positions_to_fill
                for code in target_holdings:
                    if code not in current_codes:
                        price = exec_prices.get(code, 0)
                        if not pd.isna(price) and price > 0:
                            # 计算可买入数量
                            shares = int(cash_per_position / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                            shares = (shares // 100) * 100
                            if shares > 0:
                                pf.order(code, shares, price, "BUY", next_day)
        else:
            # 滚动阶段：卖出最老的，买入最新的
            current_codes = set(pf.get_current_holding_codes())
            target_codes = set(target_holdings)

            # 需要卖出的（不在新目标中）
            to_sell = current_codes - target_codes
            # 需要买入的（不在当前持仓中）
            to_buy = target_codes - current_codes

            # 优先卖出最老的持仓，为买入新ETF腾出资金
            total_sell_value = 0
            for code in list(to_sell):
                if code in pf.holdings:
                    price = exec_prices.get(code, 0)
                    if not pd.isna(price) and price > 0:
                        shares = pf.holdings[code]['shares']
                        sell_value = shares * price * (1 - COMMISSION_RATE - SLIPPAGE)
                        pf.order(code, shares, price, "SELL", next_day)
                        total_sell_value += sell_value

            # 用卖出资金买入新ETF
            if to_buy and total_sell_value > 0:
                cash_per_new = total_sell_value / len(to_buy) if to_buy else 0
                for code in to_buy:
                    price = exec_prices.get(code, 0)
                    if not pd.isna(price) and price > 0:
                        shares = int(cash_per_new / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                        shares = (shares // 100) * 100
                        if shares > 0:
                            pf.order(code, shares, price, "BUY", next_day)

        # 打印进度
        if i % 50 == 0:
            current_value = pf.get_total_value(closes.loc[today])
            print(".1f")
    # Save results
    pd.DataFrame(pf.daily_holdings_log).to_csv("daily_holdings_rolling.csv", index=False)
    pd.DataFrame(pf.trade_log).to_csv("trade_log_rolling.csv", index=False)
    print("\nRolling portfolio results saved!")
    print(f"Final portfolio value: ${pf.get_total_value(closes.iloc[-1]):,.2f}")

    # Calculate performance metrics
    h_series = pd.Series([h['value'] for h in pf.history])
    total_ret = (h_series.iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    max_dd = ((h_series / h_series.cummax() - 1).min()) * 100

    print(".2f")
    return total_ret, max_dd

if __name__ == "__main__":
    run_rolling_portfolio_strategy()