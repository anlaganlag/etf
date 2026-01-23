import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add root to path so we can import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

# --- Config ---
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0001
SLIPPAGE = 0.001
REBALANCE_DAYS = 10  # T10
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")
CACHE_DIR = config.DATA_CACHE_DIR
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10
SECTOR_LIMIT = 1

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

class Portfolio:
    def __init__(self, capital, name_map):
        self.cash = capital
        self.holdings = {} # code -> quantity
        self.name_map = name_map
        self.history = []
        self.trade_log = []
        self.daily_holdings_log = [] # New: Detailed daily snapshot

    def get_total_value(self, prices):
        holdings_value = 0.0
        for code, qty in self.holdings.items():
            if code in prices and not pd.isna(prices[code]):
                holdings_value += qty * prices[code]
        return self.cash + holdings_value

    def record_daily_snapshot(self, date, prices):
        """Record what is held today with names and themes"""
        total_val = self.get_total_value(prices)
        if not self.holdings:
            self.daily_holdings_log.append({
                "date": date, "code": "CASH", "name": "现金", "theme": "现金",
                "qty": 0, "value": f"{self.cash:.2f}", "weight": "100%"
            })
        else:
            total_equity = sum(q * prices.get(c, 0) for c, q in self.holdings.items())
            for code, qty in self.holdings.items():
                if qty > 0:
                    name = self.name_map.get(code, "Unknown")
                    theme = get_theme_normalized(name)
                    value = qty * prices.get(code, 0)
                    weight = f"{value/total_val*100:.1f}%" if total_val > 0 else "0%"
                    self.daily_holdings_log.append({
                        "date": date, "code": code, "name": name, "theme": theme,
                        "qty": qty, "value": f"{value:.2f}", "weight": weight
                    })

    def order(self, code, qty, price, action, date):
        """Execute order"""
        if action == "BUY":
            cost = qty * price * (1 + COMMISSION_RATE + SLIPPAGE)
            if self.cash >= cost:
                self.cash -= cost
                self.holdings[code] = self.holdings.get(code, 0) + qty
                self.trade_log.append({
                    "date": date, "code": code, "name": self.name_map.get(code, ""),
                    "action": action, "price": price, "shares": qty,
                    "total_amt": cost, "remaining_cash": self.cash
                })
        elif action == "SELL":
            if code in self.holdings and self.holdings[code] >= qty:
                revenue = qty * price * (1 - COMMISSION_RATE - SLIPPAGE)
                self.cash += revenue
                self.holdings[code] -= qty
                if self.holdings[code] == 0:
                    del self.holdings[code]
                self.trade_log.append({
                    "date": date, "code": code, "name": self.name_map.get(code, ""),
                    "action": action, "price": price, "shares": qty,
                    "total_amt": revenue, "remaining_cash": self.cash
                })

def load_data_with_names():
    """Load ETF data with names"""
    # Load List for Names
    list_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("etf_list_")]
    name_map = {}
    if list_files:
        l_df = pd.read_csv(os.path.join(CACHE_DIR, sorted(list_files)[-1]))
        name_map = dict(zip(l_df['etf_code'], l_df['etf_name']))

    # Load Prices
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

def run_corrected_simulation():
    closes, opens, name_map = load_data_with_names()
    dates = closes.index
    pf = Portfolio(INITIAL_CAPITAL, name_map)

    print("Pre-calculating signals...")
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = closes.pct_change(periods=d).fillna(-999)

    print(f"Simulating Trade with Sector Limit from {START_DATE}...")

    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]

        # Record Snapshot for Today
        pf.record_daily_snapshot(today, closes.loc[today])

        # Update History Val
        pf.history.append({"date": today, "value": pf.get_total_value(closes.loc[today])})

        # Rebalance Logic with Sector Limit
        if i % REBALANCE_DAYS == 0:
            daily_scores = pd.Series(0, index=closes.columns)
            valid_mask = closes.loc[today].notna()
            for d, weight in SCORES.items():
                r_d = roll_rets[d].loc[today]
                valid_r = r_d[valid_mask & (r_d > -100)]
                if not valid_r.empty:
                    threshold = max(10, int(len(valid_r) * 0.1))
                    top_codes = valid_r.nlargest(threshold).index
                    daily_scores.loc[top_codes] += weight

            # Selection with Sector Limit
            sorted_candidates = daily_scores.sort_values(ascending=False).index
            target_holdings = []
            theme_counts = {}

            for code in sorted_candidates:
                if len(target_holdings) >= TOP_N: break

                theme = get_theme_normalized(name_map.get(code, ""))
                count = theme_counts.get(theme, 0)

                if count < SECTOR_LIMIT:
                    target_holdings.append(code)
                    theme_counts[theme] = count + 1

            exec_prices = opens.loc[next_day]

            # Sells
            for code in list(pf.holdings.keys()):
                if code not in target_holdings:
                    qty = pf.holdings[code]
                    if code in exec_prices and not pd.isna(exec_prices[code]):
                        pf.order(code, qty, exec_prices[code], "SELL", next_day)

            # Buys
            current_equity = pf.get_total_value(exec_prices)
            target_per_pos = current_equity / TOP_N
            for code in target_holdings:
                price = exec_prices.get(code, 0)
                curr_qty = pf.holdings.get(code, 0)
                curr_val = curr_qty * price
                if curr_val < target_per_pos * 0.9:
                    shortfall = target_per_pos - curr_val
                    shares = int(shortfall / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                    shares = (shares // 100) * 100
                    if shares > 0:
                        pf.order(code, shares, price, "BUY", next_day)

            themes = [get_theme_normalized(name_map.get(c, "")) for c in target_holdings]
            print(f"Rebalance {next_day.date()}: Themes = {', '.join(set(themes))}")

    # Save results
    detail_path = os.path.join(config.DATA_OUTPUT_DIR, "daily_holdings_detail_corrected.csv")
    trade_path = os.path.join(config.DATA_OUTPUT_DIR, "trade_log_corrected.csv")
    
    pd.DataFrame(pf.daily_holdings_log).to_csv(detail_path, index=False)
    pd.DataFrame(pf.trade_log).to_csv(trade_path, index=False)
    print(f"\nCorrected daily snapshots saved to {detail_path}")
    print(f"Corrected trade log saved to {trade_path}")

if __name__ == "__main__":
    run_corrected_simulation()