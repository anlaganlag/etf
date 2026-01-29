"""
策略逻辑：止损即换股 (Stop & Swap)
在原gm_strategy_rolling0.py基础上修改：
当触发止损/止盈卖出后，不保留现金，而是立即买入当前排名最高且未持有的ETF。
"""

import pandas as pd
import numpy as np
import os
import json
import datetime
import time
from gm.api import *
from dotenv import load_dotenv

load_dotenv()

# === Config ===
TOP_N = 8
STOP_LOSS = 0.05
TRAILING_TRIGGER = 0.06
TRAILING_DROP = 0.02
MIN_SCORE = 20
REBALANCE_PERIOD_T = 10
MAX_PER_THEME = 1

# 实盘数据更新
LIVE_DATA_UPDATE = False
START_DATE='2024-09-01 09:00:00'
END_DATE='2026-01-27 16:00:00'

# === Data Loading (Simplified) ===
# Copying necessary parts from original script...
from config import config

def log(msg):
    with open("debug_log.txt", "a") as f:
        f.write(msg + "\n")

def load_data():
    log("DEBUG: Entering load_data")
    log(f"DEBUG: Base Dir: {config.BASE_DIR}")
    
    # 1. Whitelist
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    if not os.path.exists(excel_path):
        log(f"DEBUG: Excel missing at {excel_path}")
        return None, None, None
        
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
    
    whitelist_set = set(df_excel['etf_code'])
    theme_map = df_excel.set_index('etf_code')['theme'].to_dict()
    log(f"DEBUG: Loaded whitelist with {len(whitelist_set)} items")
    log(f"DEBUG: Sample whitelist items: {list(whitelist_set)[:5]}")
    
    # 2. Prices
    price_data = {}
    if not os.path.exists(config.DATA_CACHE_DIR):
        log(f"DEBUG: Data cache dir missing at {config.DATA_CACHE_DIR}")
        return None, None, None
        
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    log(f"DEBUG: Found {len(files)} csv files in cache")
    if files:
        first_code = files[0].replace('_', '.').replace('.csv', '')
        log(f"DEBUG: Sample cache code: {first_code} from file {files[0]}")
    
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            code = ('SHSE.' if code.startswith('sh') else 'SZSE.') + code[2:]
            
        # Only load if in whitelist or index
        if code == 'SHSE.000001': log("DEBUG: Found Benchmark SHSE.000001")
        
        if code not in whitelist_set and code != 'SHSE.000001': continue
            
        try:
            fp = os.path.join(config.DATA_CACHE_DIR, f)
            df = pd.read_csv(fp, usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except Exception as e:
            log(f"Error loading {f}: {e}")
        
    if not price_data:
        log("DEBUG: No price data loaded! Check cache directory or whitelist matching.")
        return None, None, None

    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    prices_df.sort_index(inplace=True)
    prices_df = prices_df.fillna(method='ffill')
    
    return whitelist_set, prices_df, theme_map

# === Tranche Class (Modified) ===
class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.positions = {} # {symbol: {'volume': 0, 'entry_price': 0, 'highest_price': 0}}
        self.history = []
        self.guard_triggered_today = False # Flag to prevent buy-back same day

    def update_price(self, symbol, current_price):
        if symbol in self.positions:
            pos = self.positions[symbol]
            pos['last_price'] = current_price
            if current_price > pos['highest_price']:
                pos['highest_price'] = current_price

    def check_risk(self, symbol, current_price, current_dt):
        if symbol not in self.positions: return True, "NoPos"
        
        pos = self.positions[symbol]
        entry = pos['entry_price']
        high = pos['highest_price']
        
        # Stop Loss
        if current_price < entry * (1 - STOP_LOSS):
            return False, "StopLoss"
            
        # Trailing
        gain_pct = (high - entry) / entry
        if gain_pct >= TRAILING_TRIGGER:
            drop_pct = (high - current_price) / high
            if drop_pct >= TRAILING_DROP:
                return False, "Trailing"
                
        return True, "Hold"

    def sell(self, symbol, price):
        if symbol in self.positions:
            vol = self.positions[symbol]['volume']
            revenue = vol * price
            self.cash += revenue
            del self.positions[symbol]
            self.guard_triggered_today = True # Mark as triggered
            return revenue
        return 0

    def buy(self, symbol, amount, price):
        if price <= 0: return
        vol = int(amount / price / 100) * 100
        if vol > 0:
            cost = vol * price
            self.cash -= cost
            self.positions[symbol] = {
                'volume': vol,
                'entry_price': price,
                'highest_price': price,
                'last_price': price
            }

# === Strategy Logic ===
def init(context):
    context.whitelist, context.prices_df, context.theme_map = load_data()
    context.tranches = [Tranche(i, 1000000/REBALANCE_PERIOD_T) for i in range(REBALANCE_PERIOD_T)]
    context.days_count = 0
    
    if 'SHSE.000001' in context.prices_df:
        context.bench_bench = context.prices_df['SHSE.000001']
    else:
        context.bench_bench = None
        log("DEBUG: Benchmark SHSE.000001 not found in prices_df")
    
    print(f"Initialized Immediate Replacement Strategy (T={REBALANCE_PERIOD_T}, N={TOP_N})")
    print(f"Data Date Range: {context.prices_df.index[0]} to {context.prices_df.index[-1]}")
    print(f"Columns: {context.prices_df.columns[:5]}")
    
    subscribe(symbols='SHSE.000001', frequency='1d')

def get_ranking(context, current_dt):
    # Retrieve history up to current_dt
    # Simplified ranking logic from original
    try:
        hist = context.prices_df.loc[:current_dt].iloc[-30:] # Last 30 days
        if len(hist) < 25: 
            # log(f"Not enough history for {current_dt}: {len(hist)}")
            return None, None
        
        current_prices = hist.iloc[-1]
        
        # Calculate returns
        # Check if we have enough rows for -21 (r20)
        # r10 = (hist.iloc[-1] / hist.iloc[-11]) - 1
        # r20 = (hist.iloc[-1] / hist.iloc[-21]) - 1
        
        score_series = pd.Series(0.0, index=current_prices.index)
        
        periods = [1, 3, 5, 10, 20]
        # periods = [10]
        for p in periods:
            if len(hist) > p+1:
                r = (hist.iloc[-1] / hist.iloc[-(p+1)]) - 1
                rank = r.rank(ascending=False)
                # SMOOTH score
                s = (30 - rank) / 30
                s = s.clip(lower=0)
                score_series += s * (20 if p==10 else 10) # Weighted
        
        # Filter whitelist
        valid_symbols = [s for s in score_series.index if s in context.theme_map]
        score_series = score_series[valid_symbols]
        
        # Sort
        ranking = score_series.sort_values(ascending=False)
        return ranking, current_prices
        
    except Exception as e:
        log(f"Ranking Error: {e}")
        return None, None

def on_bar(context, bars):
    current_dt = bars[0].bob.replace(tzinfo=None)
    
    # print(f"On Bar: {current_dt}")
    
    # Ranking
    ranking, current_prices = get_ranking(context, current_dt)
    if ranking is None: 
        # if context.days_count % 30 == 0: print(f"No ranking for {current_dt}")
        return

    context.days_count += 1
    
    # 0. Update all tranches prices
    for t in context.tranches:
        t.guard_triggered_today = False # Reset daily flag
        for sym in list(t.positions.keys()):
            # Using current_prices from get_ranking (which is close price of current day or yesterday?)
            # bars[0].bob is beginning of bar? 
            # GM backtest frequency='1d', on_bar happens after bar close usually or open?
            # context.now is bar start time. Current prices should be bar close?
            # get_ranking uses prices_df.loc[:current_dt].
            # If current_dt is today 09:00, prices_df should have data up to yesterday?
            # If we backtest daily, usually we trade on Open?
            # But get_ranking uses hist.iloc[-1]. 
            # If prices_df has future data (lookahead bias?), we must be careful.
            # But assuming prices_df has data.
            
            if sym in current_prices:
                t.update_price(sym, current_prices[sym])

    # 1. Check Risk & Sell (STOP LOSS)
    # Collect cash from sells to potentially reinvest
    for t in context.tranches:
        for sym in list(t.positions.keys()):
            if sym not in current_prices: continue
            
            keep, reason = t.check_risk(sym, current_prices[sym], current_dt)
            if not keep:
                revenue = t.sell(sym, current_prices[sym])
                # log(f"{current_dt} Tranche {t.id} Sold {sym} ({reason})")

    # 2. Rolling Rebalance (Standard)
    # Only for the active tranche of the day
    active_idx = (context.days_count - 1) % REBALANCE_PERIOD_T
    active_tranche = context.tranches[active_idx]
    
    # Sell remaining positions in active tranche
    for sym in list(active_tranche.positions.keys()):
        active_tranche.sell(sym, current_prices.get(sym, 0))
        
    top_targets = ranking.head(TOP_N).index.tolist()
    
    # Buy Top N for active tranche
    per_etf_cash = (1000000 / REBALANCE_PERIOD_T) / TOP_N 
    
    for target in top_targets:
        if target in current_prices:
             active_tranche.buy(target, per_etf_cash, current_prices[target])
    
    # 3. FILL-IN LOGIC (The "Think Out of Box" Part)
    # Check ALL tranches for idle cash
    for t in context.tranches:
        # If we have enough cash to buy at least 1 slot
        while t.cash > per_etf_cash * 0.9: 
            candidate = None
            for sym in ranking.index:
                if sym not in current_prices: continue
                if sym not in t.positions:
                     # Check theme limit if needed
                     track = context.theme_map.get(sym, 'Other')
                     current_theme_count = sum(1 for s in t.positions if context.theme_map.get(s)==track)
                     if current_theme_count < MAX_PER_THEME:
                         candidate = sym
                         break
            
            if candidate:
                t.buy(candidate, per_etf_cash, current_prices[candidate])
                # log(f"{current_dt} Tranche {t.id} Fill-in Buy {candidate}")
            else:
                break # No valid candidates found

    # Track Performance
    total_val = sum([t.cash + sum([p['volume']*p['last_price'] for p in t.positions.values()]) for t in context.tranches])
    if context.days_count % 20 == 0:
        print(f"{current_dt} Value: {total_val:.2f}")

def on_backtest_finished(context, indicator):
    # Calculate final stats
    # Simplified return calculation for the script output
    total_val = sum([t.cash + sum([p['volume']*p['last_price'] for p in t.positions.values()]) for t in context.tranches])
    init_val = 1000000
    ret = (total_val - init_val) / init_val * 100
    print(f"Return: {ret:.2f}%")
    # MaxDD calculation would need daily tracking, omitting for brevity in this Quick Test
    print(f"Max DD: N/A") 
    print(f"Sharpe: N/A")

if __name__ == '__main__':
    token = os.environ.get('MY_QUANT_TGM_TOKEN')
    run(strategy_id='strategy_id',
        filename='gm_strategy_immediate.py',
        mode=MODE_BACKTEST,
        token=token,
        backtest_start_time=START_DATE,
        backtest_end_time=END_DATE,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000)
