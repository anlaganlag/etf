# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
from datetime import datetime
from dotenv import load_dotenv

from config import config
from src.data_fetcher import DataFetcher

load_dotenv()

# --- Config ---
# You can switch between 'Standard' (Top 10) or 'Aggressive' (Top 3) here
TOP_N = 3  
STOP_LOSS = 0.20
TRAILING_TRIGGER = 0.10
TRAILING_DROP = 0.05
MIN_SCORE = 150
REBALANCE_PERIOD_T = 14
FRUIT_THEME_BOOST = True

class Tranche:
    def __init__(self, t_id, initial_cash):
        self.id = t_id
        self.cash = initial_cash
        self.holdings = {} # {symbol: shares}
        # Record for Guard: {symbol: {'entry_price': x, 'high_price': y}}
        self.pos_records = {} 
        self.total_value = initial_cash

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
        # Return list of symbols to sell
        to_sell = []
        for sym, rec in self.pos_records.items():
            if sym not in self.holdings: continue
            curr_price = price_map.get(sym, 0)
            if curr_price <= 0: continue

            entry = rec['entry_price']
            high = rec['high_price']
            
            # Stop Loss
            if curr_price < entry * (1 - STOP_LOSS):
                to_sell.append(sym)
                continue
            
            # Trailing Profit
            if high > entry * (1 + TRAILING_TRIGGER):
                if curr_price < high * (1 - TRAILING_DROP):
                    to_sell.append(sym)
        return to_sell

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
            
            # Init Guard Record or update average cost (simplified to new entry for rolling logic)
            # In strictly independent tranche, this is a new entry or adding to existing
            # If adding, usually weighted average. For simplicity here, if we already hold it, 
            # we keep the OLD entry price (conservative) or update? 
            # Let's keep strict tranche logic: if we hold it, we usually don't buy "more" in rebalance unless selected again?
            # Actually rebalance sells all then buys. checking logic below.
            
            if symbol not in self.pos_records:
                self.pos_records[symbol] = {'entry_price': price, 'high_price': price}
            # If we already held it (e.g. didn't sell in rebalance?), we might average down/up
            # But the rebalance logic usually is "Sell All -> Buy New". 
            # If we keep holding the SAME stock, we should logically reset? 
            # Let's assume standard rebalance: clear non-selected, buy selected.
            # If selected is same as held, we usually re-weight. 
            pass

def init(context):
    print(f"Initializing Rolling Backtest with T={REBALANCE_PERIOD_T}, TopN={TOP_N}")
    
    context.T = REBALANCE_PERIOD_T
    context.top_n = TOP_N
    context.tranches = []
    
    # Init Data
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

    # Pre-build Price Matrix
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
    print(f"Global matrix built: {context.prices_df.shape[1]} symbols.")

    context.days_count = 0
    context.initialized_tranches = False
    
    subscribe(symbols='SZSE.399006', frequency='1d')

def get_ranking(context, current_dt):
    # Same V6.0 Logic
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
    
    # Theme Boost
    if FRUIT_THEME_BOOST:
        strong_etfs = valid_base[valid_base >= 150]
        theme_counts = {}
        for code in strong_etfs.index:
            t = context.theme_map.get(code, 'Unknown')
            theme_counts[t] = theme_counts.get(t, 0) + 1
        strong_themes = {t for t, count in theme_counts.items() if count >= 3}
        final_scores = valid_base.copy()
        for code in final_scores.index:
            if context.theme_map.get(code, 'Unknown') in strong_themes:
                final_scores[code] += 50
    else:
        final_scores = valid_base

    valid_final = final_scores[final_scores >= MIN_SCORE]
    if valid_final.empty: return None, base_scores
    
    r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0, index=history_prices.columns)
    df = pd.DataFrame({'score': valid_final, 'r20': r20[valid_final.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid_final.index]})
    return df.sort_values(by=['score', 'r20'], ascending=False), base_scores

def on_bar(context, bars):
    # Initialize Tranches on first day
    if not context.initialized_tranches:
        # account().cash returns a structure. use .available or .nav?
        # In backtest mode, cash.available represents usable.
        total_cash = context.account().cash.available
        share_per_tranche = total_cash / context.T
        for i in range(context.T):
            context.tranches.append(Tranche(i, share_per_tranche))
        context.initialized_tranches = True
        print(f"Initialized {context.T} tranches with {share_per_tranche:.2f} each.")

    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    # Get current prices for all valid symbols in universe (optimize: ensure we have prices for held + universe)
    # For backtest speed, we use prices_df logic.
    # Note: gm 'bars' only contains subscribed symbols. We need to fetch from prices_df.
    
    if current_dt not in context.prices_df.index:
        # Try finding nearest date or just return if data missing
        real_dt_idx = context.prices_df.index.searchsorted(current_dt)
        if real_dt_idx >= len(context.prices_df): return # No data yet
        # Ensure we are looking at valid day
        # Actually in gm backtest, on_bar is triggered by 399006, so date should exist approx
        # We'll use the last available row in prices_df <= current_dt
        pass
    
    # Slice prices for today
    today_prices = context.prices_df[context.prices_df.index <= current_dt].iloc[-1]
    price_map = today_prices.to_dict()
    
    # 1. Update Tranche Values & Run Guard
    for tranche in context.tranches:
        # A. Mark to Market
        tranche.update_value(price_map)
        
        # B. Check Guard
        to_sell_list = tranche.check_guard(price_map)
        for sym in to_sell_list:
            price = price_map.get(sym, 0)
            if price > 0:
                tranche.sell(sym, price)
                # print(f"  [Tranche {tranche.id}] Guard Sell {sym}")

    # 2. Rebalance Specific Tranche
    # Tranche ID to rebalance today: (days_count - 1) % T
    rebalance_idx = (context.days_count - 1) % context.T
    active_tranche = context.tranches[rebalance_idx]
    
    # print(f"--- Rebalancing Tranche {rebalance_idx} ---")
    
    # Ranking
    ranking_df, total_scores = get_ranking(context, current_dt)
    
    # Market Timing (simple)
    strong_market_count = (total_scores >= 150).sum() if total_scores is not None else 0
    market_exposure = 1.0 # Default full
    if strong_market_count < 5:
        market_exposure = 0.3 # Graded exposure
    
    # Check if we should rebalance? logic says "Every T days completely rebalance"
    # Step A: Sell All in this tranche
    held_syms = list(active_tranche.holdings.keys())
    for sym in held_syms:
        price = price_map.get(sym, 0)
        if price > 0: active_tranche.sell(sym, price)
    
    # Step B: Select New
    target_list = []
    if ranking_df is not None:
        seen = set()
        for code, row in ranking_df.iterrows():
            if row['theme'] not in seen:
                target_list.append(code)
                seen.add(row['theme'])
            if len(target_list) >= context.top_n: break
    
    # Step C: Buy New
    if target_list:
        # Allocating Cash
        # Exposure check: if market is weak, we hold cash in tranche
        avail_cash = active_tranche.cash
        invest_cash = avail_cash * market_exposure
        per_etf_cash = invest_cash / len(target_list)
        
        for sym in target_list:
            price = price_map.get(sym, 0)
            if price > 0:
                active_tranche.buy(sym, per_etf_cash, price)
                
    active_tranche.update_value(price_map)

    # 3. Sync to Global Account
    # Aggregate all tranches
    global_target_holdings = {}
    total_virtual_value = sum([t.total_value for t in context.tranches])
    
    for t in context.tranches:
        for sym, shares in t.holdings.items():
            global_target_holdings[sym] = global_target_holdings.get(sym, 0) + shares
            
    # Execute Diffs
    # To avoid tiny diffs, we calculate target percent for GM or just target shares?
    # order_target_volume is safer if we know exact shares.
    
    # However, existing positions in GM
    real_positions = context.account().positions()
    real_holdings = {p['symbol']: p['amount'] for p in real_positions}
    
    # 1. Sell what we don't need
    for sym in real_holdings:
        tgt = global_target_holdings.get(sym, 0)
        if real_holdings[sym] > tgt:
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)
    
    # 2. Buy what we need
    for sym, tgt in global_target_holdings.items():
        curr = real_holdings.get(sym, 0)
        if curr < tgt:
            order_target_volume(symbol=sym, volume=tgt, order_type=OrderType_Market, position_side=PositionSide_Long)

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print(f"ROLLING BACKTEST RESULT (T={context.T}, TopN={context.top_n})")
    print("="*50)
    print(f"Cumulative Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Max Drawdown: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {indicator.get('sharp_ratio', 0):.2f}")
    print("="*50)

if __name__ == '__main__':
    run(strategy_id='rolling_test_strategy', 
        filename='gm_rolling_test.py',
        mode=MODE_BACKTEST,
        token=os.getenv('MY_QUANT_TGM_TOKEN'),
        backtest_start_time='2024-09-01 09:00:00',
        backtest_end_time='2026-01-23 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)
