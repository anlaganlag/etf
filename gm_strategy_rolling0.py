from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from dotenv import load_dotenv

from config import config
from src.data_fetcher import DataFetcher

load_dotenv()

# --- Simplified Rolling Strategy Config ---
# 1. 简化为: 简单打分 + 止盈止损空仓 + 滚动分仓(Tranche) + 主题分散
# 2. 去掉: 主题加分, 休息机制, 强势品种过滤, 大盘过滤
TOP_N = 5
STOP_LOSS = 0.20
TRAILING_TRIGGER = 0.10
TRAILING_DROP = 0.05
MIN_SCORE = 20
REBALANCE_PERIOD_T = 14
STATE_FILE = "rolling_state_simple.json"

START_DATE='2021-12-03 09:00:00'
END_DATE='2026-01-23 16:00:00'

class Tranche:
    def __init__(self, t_id, initial_cash=0):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {} # {symbol: shares}
        # Record for Guard: {symbol: {'entry_price': x, 'high_price': y}}
        self.pos_records = {} 
        self.total_value = initial_cash
        self.guard_triggered_today = False 

    def to_dict(self):
        return {
            "id": self.id,
            "cash": self.cash,
            "holdings": self.holdings,
            "pos_records": self.pos_records,
            "total_value": self.total_value
        }

    @staticmethod
    def from_dict(d):
        t = Tranche(d["id"], d["cash"])
        t.holdings = d["holdings"]
        t.pos_records = d["pos_records"]
        t.total_value = d["total_value"]
        return t

    def update_value(self, price_map):
        val = self.cash
        current_symbols = list(self.holdings.keys())
        for sym in current_symbols:
            if sym in price_map:
                price = price_map[sym]
                val += self.holdings[sym] * price
                
                # Update Guard High Price
                if sym in self.pos_records:
                    self.pos_records[sym]['high_price'] = max(self.pos_records[sym]['high_price'], price)
        self.total_value = val

    def check_guard(self, price_map):
        to_sell = []
        is_tp = False # 是否触发了移动止盈
        for sym, rec in self.pos_records.items():
            if sym not in self.holdings: continue
            curr_price = price_map.get(sym, 0)
            if curr_price <= 0: continue

            entry = rec['entry_price']
            high = rec['high_price']
            
            # 止损
            if curr_price < entry * (1 - STOP_LOSS):
                to_sell.append(sym)
                continue
            
            # 移动止盈
            if high > entry * (1 + TRAILING_TRIGGER):
                if curr_price < high * (1 - TRAILING_DROP):
                    to_sell.append(sym)
                    is_tp = True
        return to_sell, is_tp

    def sell(self, symbol, price):
        if symbol in self.holdings:
            shares = self.holdings[symbol]
            revenue = shares * price
            self.cash += revenue
            del self.holdings[symbol]
            if symbol in self.pos_records:
                del self.pos_records[symbol]

    def buy(self, symbol, cash_allocated, price):
        if price <= 0: return
        shares = int(cash_allocated / price / 100) * 100
        cost = shares * price
        if shares > 0 and self.cash >= cost:
            self.cash -= cost
            self.holdings[symbol] = self.holdings.get(symbol, 0) + shares
            # Init or Reset Guard Record
            self.pos_records[symbol] = {'entry_price': price, 'high_price': price}

class RollingPortfolioManager:
    def __init__(self, context):
        self.context = context
        self.tranches = []
        self.params = {
            "T": REBALANCE_PERIOD_T, 
            "top_n": TOP_N
        }
        self.initialized = False
        self.state_path = os.path.join(config.BASE_DIR, STATE_FILE)
        
    def load_state(self):
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, 'r') as f:
                    data = json.load(f)
                    self.params = data.get("params", self.params)
                    self.initialized = data.get("initialized", False)
                    # Reconstruct Tranches
                    t_data = data.get("tranches", [])
                    self.tranches = []
                    for d in t_data:
                        self.tranches.append(Tranche.from_dict(d))
                print(f"Loaded Rolling State from {self.state_path}. {len(self.tranches)} tranches.")
                return True
            except Exception as e:
                print(f"Failed to load state: {e}")
        return False
        
    def save_state(self):
        data = {
            "params": self.params,
            "initialized": self.initialized,
            "tranches": [t.to_dict() for t in self.tranches]
        }
        with open(self.state_path, 'w') as f:
            json.dump(data, f, indent=2)

    def initialize_tranches(self, total_cash):
        if self.initialized and self.tranches: return
        
        T = self.params["T"]
        share = total_cash / T
        self.tranches = [Tranche(i, share) for i in range(T)]
        self.initialized = True
        print(f"Initialized {T} tranches with {share:.2f} each.")
        self.save_state()

    def get_market_exposure(self, context, total_scores):
        # 简化: 永远满仓
        return 1.0

def init(context):
    print(f"Initializing Simplified Rolling Strategy (T={REBALANCE_PERIOD_T}, TopN={TOP_N})")
    
    context.rpm = RollingPortfolioManager(context)
    
    # Data Loading
    context.whitelist = set()
    context.theme_map = {}
    
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    # Pre-build Price Matrix for Ranking (Daily Data)
    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            if code.startswith('sh'): code = 'SHSE.' + code[2:]
            elif code.startswith('sz'): code = 'SZSE.' + code[2:]
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except: pass
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"Data matrix built: {context.prices_df.shape[1]} symbols.")

    if context.mode == MODE_BACKTEST:
        if os.path.exists(context.rpm.state_path): os.remove(context.rpm.state_path)
    else:
        context.rpm.load_state()

    context.days_count = 0
    # No optimize needed, but keep subscription just in case
    subscribe(symbols='SZSE.399006', frequency='1d')

def get_ranking(context, current_dt):
    # Standard V6 Logic (Simplified)
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 251: return None, None

    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    base_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        ranks = rets.rank(ascending=False, method='min')
        base_scores += (ranks <= threshold) * pts
    
    valid_base = base_scores[base_scores.index.isin(context.whitelist)]
    
    # 简化: 去掉主题加分 FRUIT_THEME_BOOST
    final_scores = valid_base

    valid_final = final_scores[final_scores >= MIN_SCORE]
    if valid_final.empty: return None, base_scores
    
    r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0, index=history_prices.columns)
    df = pd.DataFrame({'score': valid_final, 'r20': r20[valid_final.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid_final.index]})
    return df.sort_values(by=['score', 'r20'], ascending=False), base_scores

def on_bar(context, bars):
    current_dt = context.now.replace(tzinfo=None)
    context.days_count += 1
    
    # 0. Initialize Tranches if needed
    if not context.rpm.initialized:
        if hasattr(context.account().cash, 'available'):
            cash = context.account().cash.available
        else:
            cash = context.account().cash.nav
        context.rpm.initialize_tranches(cash)

    # 1. Get Today's Prices
    if current_dt not in context.prices_df.index:
        real_dt_idx = context.prices_df.index.searchsorted(current_dt)
        if real_dt_idx >= len(context.prices_df): return
        today_prices = context.prices_df.iloc[real_dt_idx]
    else:
        today_prices = context.prices_df.loc[current_dt]
        
    price_map = today_prices.to_dict()

    # 2. Get Ranking & Exposure (Always 1.0)
    ranking_df, total_scores = get_ranking(context, current_dt)
    exposure = context.rpm.get_market_exposure(context, total_scores) # 1.0
    
    # 3. Update Tranches & Guard
    for tranche in context.rpm.tranches:
        # Update Value
        tranche.update_value(price_map)
        
        # Check Guard (SL / TP)
        to_sell_list, is_tp = tranche.check_guard(price_map)
        if to_sell_list:
            tranche.guard_triggered_today = True
            print(f"Tranche {tranche.id} Guard Triggered (TP={is_tp}). Sold: {to_sell_list}")
            for sym in to_sell_list:
                tranche.sell(sym, price_map.get(sym, 0))
        else:
            tranche.guard_triggered_today = False

    # 4. Rolling Rebalance
    rebalance_idx = (context.days_count - 1) % context.rpm.params["T"]
    active_tranche = context.rpm.tranches[rebalance_idx]
    
    # Sell All (Standard Rolling Logic)
    for sym in list(active_tranche.holdings.keys()):
        price = price_map.get(sym, 0)
        if price > 0: active_tranche.sell(sym, price)
    
    # Buy New
    # Conditions: 
    # 1. Ranking exists
    # 2. Not triggered guard today (prevent intraday re-entry after SL/TP)
    # 3. No rest days check (removed)
    if ranking_df is not None and not active_tranche.guard_triggered_today:
        targets = []
        seen = set()
        # Theme Dispersion: Max 1 per theme
        for code, row in ranking_df.iterrows():
            if row['theme'] not in seen:
                targets.append(code)
                seen.add(row['theme'])
            if len(targets) >= context.rpm.params["top_n"]: break
        
        if targets:
            invest_amt = active_tranche.cash * exposure
            per_amt = invest_amt / len(targets)
            for sym in targets:
                price = price_map.get(sym, 0)
                if price > 0: active_tranche.buy(sym, per_amt, price)
    
    active_tranche.update_value(price_map)

    # 5. Sync to Broker
    global_target_holdings = {}
    for t in context.rpm.tranches:
        for sym, shares in t.holdings.items():
            global_target_holdings[sym] = global_target_holdings.get(sym, 0) + shares
            
    real_positions = context.account().positions()
    real_holdings = {p['symbol']: p['amount'] for p in real_positions}
    
    # Sell diffs
    for sym in real_holdings:
        tgt = global_target_holdings.get(sym, 0)
        if real_holdings[sym] > tgt:
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)
    
    # Buy diffs
    for sym, tgt in global_target_holdings.items():
        curr = real_holdings.get(sym, 0)
        if curr < tgt:
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)

    # 6. Save State
    context.rpm.save_state()

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print(f"SIMPLE ROLLING STRATEGY (T={context.rpm.params['T']})")
    print("="*50)
    print(f"Cumulative Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Max Drawdown: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {indicator.get('sharp_ratio', 0):.2f}")
    print("="*50)

if __name__ == '__main__':
    run(strategy_id='d6d71d85-fb4c-11f0-99de-00ffda9d6e63', 
        filename='gm_strategy_rolling0.py',
        mode=MODE_BACKTEST,
        token=os.getenv('MY_QUANT_TGM_TOKEN'),
        backtest_start_time=START_DATE,
        backtest_end_time=END_DATE,
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)
