import pandas as pd
import numpy as np
import os
from datetime import datetime
import math

# --- Simulation Config ---
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.0001 # 万1
SLIPPAGE = 0.001         # 千1滑点
REBALANCE_DAYS = 10      # T10调仓
START_DATE = "2024-10-09"
END_DATE = datetime.now().strftime("%Y-%m-%d")

CACHE_DIR = "data_cache"
SCORES = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
TOP_N = 10

def get_theme(name):
    """Simple heuristic to extract sector/theme from ETF name"""
    if not name or pd.isna(name): return "Unknown"
    theme = name.replace("ETF", "").replace("基金", "").replace("增强", "").replace("指数", "")
    # Further clean common prefixes/suffixes
    for word in ["中证", "沪深", "上证", "深证", "科创", "创业板", "港股通"]:
        theme = theme.replace(word, "")
    return theme if theme else "宽基"

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
            return

        for code, qty in self.holdings.items():
            price = prices.get(code, 0)
            val = qty * price
            weight = (val / total_val * 100) if total_val > 0 else 0
            name = self.name_map.get(code, "Unknown")
            self.daily_holdings_log.append({
                "date": date,
                "code": code,
                "name": name,
                "theme": get_theme(name),
                "qty": qty,
                "value": f"{val:.2f}",
                "weight": f"{weight:.1f}%"
            })

    def order(self, code, shares, price, action, date):
        amount = shares * price
        if action == "BUY":
            cost = amount * (1 + COMMISSION_RATE + SLIPPAGE)
            if self.cash >= cost:
                self.cash -= cost
                self.holdings[code] = self.holdings.get(code, 0) + shares
                self._log(date, code, "BUY", price, shares, cost)
        elif action == "SELL":
            revenue = amount * (1 - COMMISSION_RATE - SLIPPAGE)
            if self.holdings.get(code, 0) >= shares:
                self.holdings[code] -= shares
                if self.holdings[code] == 0: del self.holdings[code]
                self.cash += revenue
                self._log(date, code, "SELL", price, shares, revenue)

    def _log(self, date, code, action, price, shares, total_amt):
        self.trade_log.append({
            "date": date, "code": code, "name": self.name_map.get(code, ""), 
            "action": action, "price": f"{price:.3f}", "shares": shares,
            "total_amt": f"{total_amt:.2f}", "remaining_cash": f"{self.cash:.2f}"
        })

def load_data_with_names():
    print("Loading ETF names and history...")
    # 1. Load names from the list cache
    # Find the latest list cache
    list_files = [f for f in os.listdir(CACHE_DIR) if f.startswith("etf_list_")]
    name_map = {}
    if list_files:
        list_df = pd.read_csv(os.path.join(CACHE_DIR, sorted(list_files)[-1]))
        # Ensure code format matches (e.g. sh510300)
        name_map = dict(zip(list_df['etf_code'], list_df['etf_name']))

    # 2. Load History
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

def run_simulation():
    closes, opens, name_map = load_data_with_names()
    dates = closes.index
    pf = Portfolio(INITIAL_CAPITAL, name_map)
    
    print("Pre-calculating signals...")
    roll_rets = {}
    for d in SCORES.keys():
        roll_rets[d] = closes.pct_change(periods=d).fillna(-999)

    print(f"Simulating Trade from {START_DATE}...")
    
    for i in range(len(dates) - 1):
        today = dates[i]
        next_day = dates[i+1]
        
        # Record Snapshot for Today
        pf.record_daily_snapshot(today, closes.loc[today])
        
        # Update History Val
        pf.history.append({"date": today, "value": pf.get_total_value(closes.loc[today])})
        
        # Rebalance Logic
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
            
            target_holdings = daily_scores.nlargest(TOP_N).index.tolist()
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
                if pd.isna(price) or price <= 0: continue
                curr_val = pf.holdings.get(code, 0) * price
                if curr_val < target_per_pos * 0.9:
                    shortfall = target_per_pos - curr_val
                    shares = int(shortfall / (price * (1 + COMMISSION_RATE + SLIPPAGE)))
                    shares = (shares // 100) * 100
                    if shares > 0:
                        pf.order(code, shares, price, "BUY", next_day)
            
            # Print Rebalance Info for Analysis
            themes = [get_theme(name_map.get(c, "")) for c in target_holdings]
            print(f"Rebalance {next_day.date()}: Themes = {', '.join(set(themes))}")

    # Final record
    pf.record_daily_snapshot(dates[-1], closes.iloc[-1])
    
    # Save Results
    pd.DataFrame(pf.trade_log).to_csv("trade_log.csv", index=False)
    pd.DataFrame(pf.daily_holdings_log).to_csv("daily_holdings_detail.csv", index=False)
    print("\nDetailed daily snapshots saved to daily_holdings_detail.csv")
    print(f"Final Account Value: {pf.history[-1]['value']:,.2f}")

if __name__ == "__main__":
    run_simulation()
