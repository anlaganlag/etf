import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from config import config

# --- Simulation Settings ---
TOP_N_RANGE = [3, 4, 5, 6, 7, 8, 9, 10]
T_RANGE = [10, 11, 12, 13, 14, 15, 16]

# --- Shared Constants from gm_strategy_rolling0.py ---
STOP_LOSS = 0.05
TRAILING_TRIGGER = 0.06
TRAILING_DROP = 0.02
MAX_PER_THEME = 1
MIN_SCORE = 20
START_DATE = '2021-12-03'
END_DATE = '2026-01-23'

class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {} # {symbol: shares}
        self.pos_records = {} # {symbol: {'entry_price': x, 'high_price': y}}
        self.total_value = initial_cash
        self.guard_triggered_today = False 

    def update_value(self, price_map):
        val = self.cash
        for sym, shares in self.holdings.items():
            if sym in price_map:
                price = price_map[sym]
                val += shares * price
                if sym in self.pos_records:
                    self.pos_records[sym]['high_price'] = max(self.pos_records[sym]['high_price'], price)
        self.total_value = val

    def check_guard(self, price_map):
        to_sell = []
        for sym, rec in self.pos_records.items():
            if sym not in self.holdings: continue
            curr_price = price_map.get(sym, 0)
            if curr_price <= 0: continue
            entry, high = rec['entry_price'], rec['high_price']
            if (curr_price < entry * (1 - STOP_LOSS)) or \
               (high > entry * (1 + TRAILING_TRIGGER) and curr_price < high * (1 - TRAILING_DROP)):
                to_sell.append(sym)
        return to_sell

    def sell(self, symbol, price):
        if symbol in self.holdings:
            shares = self.holdings[symbol]
            self.cash += shares * price
            del self.holdings[symbol]
            if symbol in self.pos_records: del self.pos_records[symbol]

    def buy(self, symbol, cash_allocated, price):
        if price <= 0: return
        shares = int(cash_allocated / price / 100) * 100
        cost = shares * price
        if shares > 0 and self.cash >= cost:
            self.cash -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
            self.pos_records[symbol] = {'entry_price': price, 'high_price': price}

def get_ranking(prices_df, whitelist, theme_map, current_idx):
    # Use index instead of date for speed
    history = prices_df.iloc[:current_idx + 1]
    if len(history) < 251: return None
    
    base_scores = pd.Series(0.0, index=history.columns)
    periods_rule = {1: 20, 3: 30, 5: 50, 10: 70, 20: 100}
    
    last_row = history.iloc[-1]
    rets_dict = {}
    for p, pts in periods_rule.items():
        # Matching gm_strategy_rolling0.py: rets = (current / t-p) - 1
        rets = (last_row / history.iloc[-(p+1)]) - 1
        rets_dict[f'r{p}'] = rets
        ranks = rets.rank(ascending=False, method='min')
        base_scores += (ranks <= 15) * pts
    
    valid_scores = base_scores[base_scores.index.isin(whitelist)]
    valid_scores = valid_scores[valid_scores >= MIN_SCORE]
    if valid_scores.empty: return None

    data_to_df = {
        'score': valid_scores, 
        'theme': [theme_map.get(c, 'Unknown') for c in valid_scores.index],
        'etf_code': valid_scores.index
    }
    for p in periods_rule.keys():
        data_to_df[f'r{p}'] = rets_dict[f'r{p}'][valid_scores.index]

    df = pd.DataFrame(data_to_df)
    sort_cols = ['score', 'r1', 'r3', 'r5', 'r10', 'r20', 'etf_code']
    asc_order = [False, False, False, False, False, False, True]
    return df.sort_values(by=sort_cols, ascending=asc_order)

def simulate(top_n, t_period, prices_df, whitelist, theme_map):
    # Start and End indices
    all_dates = prices_df.index
    start_idx = all_dates.get_indexer([pd.Timestamp(START_DATE)], method='bfill')[0]
    end_idx = all_dates.get_indexer([pd.Timestamp(END_DATE)], method='ffill')[0]
    
    initial_cash = 1000000.0
    tranches = [Tranche(i, initial_cash / t_period) for i in range(t_period)]
    nav_history = []
    
    days_count = 0
    for i in range(start_idx, end_idx + 1):
        days_count += 1
        
        # CURRENT PRICE for update and guard
        price_map_today = prices_df.iloc[i].to_dict()
        
        # NEXT PRICE for buy/sell (Simulating the gm_strategy_rolling0.py look-ahead bug)
        # In gm_strategy_rolling0.py, today_prices = prices_df.iloc[searchsorted(now)]
        # Since 'now' is 15:00 and index is 00:00, searchsorted returns the next index (i+1)
        next_i = min(i + 1, len(prices_df) - 1)
        price_map_next = prices_df.iloc[next_i].to_dict()
        
        # 1. Update & Guard
        for t in tranches:
            t.update_value(price_map_today)
            to_sell = t.check_guard(price_map_today)
            t.guard_triggered_today = len(to_sell) > 0
            for sym in to_sell:
                # Sell at NEXT price (buggy behavior)
                t.sell(sym, price_map_next.get(sym, price_map_today.get(sym, 0)))
                
        # 2. Rolling Rebalance
        active_tranche = tranches[(days_count - 1) % t_period]
        # Sell all in active tranche at NEXT price
        for sym in list(active_tranche.holdings.keys()):
            active_tranche.sell(sym, price_map_next.get(sym, price_map_today.get(sym, 0)))
            
        # Buy new
        if not active_tranche.guard_triggered_today:
            ranking_df = get_ranking(prices_df, whitelist, theme_map, i)
            if ranking_df is not None:
                targets = []
                theme_count = {}
                for code, row in ranking_df.iterrows():
                    theme = row['theme']
                    if theme_count.get(theme, 0) < MAX_PER_THEME:
                        targets.append(code)
                        theme_count[theme] = theme_count.get(theme, 0) + 1
                    if len(targets) >= top_n: break
                
                if targets:
                    allocate_cash = active_tranche.cash
                    per_amt = allocate_cash / len(targets)
                    for sym in targets:
                        # Buy at NEXT price (buggy behavior)
                        active_tranche.buy(sym, per_amt, price_map_next.get(sym, price_map_today.get(sym, 0)))
        
        # 3. Calculate NAV (using today's price)
        total_val = 0
        for t in tranches:
            t.update_value(price_map_today)
            total_val += t.total_value
        nav_history.append(total_val)
        
    nav_series = pd.Series(nav_history)
    total_return = (nav_series.iloc[-1] / nav_series.iloc[0] - 1) * 100
    
    cummax = nav_series.cummax()
    max_dd = ((nav_series - cummax) / cummax).min() * 100
    
    daily_rets = nav_series.pct_change().dropna()
    sharpe = np.sqrt(252) * daily_rets.mean() / daily_rets.std() if daily_rets.std() > 0 else 0
    
    return total_return, max_dd, sharpe

def main():
    print("Loading data...")
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
    whitelist = set(df_excel['etf_code'])
    theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            code = ('SHSE.' if code.startswith('sh') else 'SZSE.') + code[2:]
        if code not in whitelist: continue
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except: pass
    prices_df = pd.DataFrame(price_data).sort_index().ffill().dropna(how='all')
    print(f"Data Loaded: {prices_df.shape}")

    results = []
    print(f"{'N':<4} {'T':<4} {'Ret':<8} {'MDD':<8} {'Shp':<8}")
    for top_n in TOP_N_RANGE:
        row_results = []
        for t_p in T_RANGE:
            res = simulate(top_n, t_p, prices_df, whitelist, theme_map)
            row_results.append(res)
            print(f"{top_n:<4} {t_p:<4} {res[0]:<8.2f} {res[1]:<8.2f} {res[2]:<8.2f}")
        results.append(row_results)

    # Output Matrices
    print("\n=== RETURN MATRIX (%) ===")
    header = "      " + "".join([f"T={t:<8}" for t in T_RANGE])
    print(header)
    for i, top_n in enumerate(TOP_N_RANGE):
        row = f"N={top_n:<4} " + "".join([f"{results[i][j][0]:<10.2f}" for j in range(len(T_RANGE))])
        print(row)

    print("\n=== MAX DRAWDOWN MATRIX (%) ===")
    print(header)
    for i, top_n in enumerate(TOP_N_RANGE):
        row = f"N={top_n:<4} " + "".join([f"{results[i][j][1]:<10.2f}" for j in range(len(T_RANGE))])
        print(row)

    print("\n=== SHARPE MATRIX ===")
    print(header)
    for i, top_n in enumerate(TOP_N_RANGE):
        row = f"N={top_n:<4} " + "".join([f"{results[i][j][2]:<10.2f}" for j in range(len(T_RANGE))])
        print(row)

if __name__ == "__main__":
    main()
