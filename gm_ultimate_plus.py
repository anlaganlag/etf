# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import project config
from config import config
from src.data_fetcher import DataFetcher

load_dotenv()

def init(context):
    # 1. Strategy Parameters
    context.T = 14                  
    context.top_n = 10              
    context.min_score = 150         
    context.target_percent = 0.98   
    context.days_count = 0
    context.last_rebalance_day = -1
    
    # --- Guard Parameters ---
    context.stop_loss = 0.20        # 20% Hard Stop Loss (Optimized)
    context.trailing_trigger = 0.10 # Start trailing stop after 10% profit
    context.trailing_pnl_drop = 0.05 # Exit if price drops 5% from peak
    
    # Store position info: {symbol: {'entry_price': p, 'high_price': p}}
    context.pos_records = {}

    # 2. Load Whitelist & Theme Map
    # Use the Excel file to filter and de-duplicate by sector
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns:
        df_excel['theme'] = df_excel['etf_name']
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    # 3. Build Global Price Matrix for fast ranking
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    df_all_etfs = fetcher.get_all_etfs()
    all_codes = df_all_etfs['etf_code'].tolist()
    
    price_data = {}
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except: pass
    
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"Global matrix built: {context.prices_df.shape[1]} symbols.")

    # 4. Subscribe
    subscribe(symbols='SHSE.000300', frequency='1d')

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    # --- 1. Daily Guard Check ---
    positions = context.account().positions()
    current_holding_count = 0
    for pos in positions:
        if pos['amount'] == 0: continue
        current_holding_count += 1
        
        symbol = pos['symbol']
        curr_price = pos['price']
        
        if symbol not in context.pos_records or context.pos_records[symbol]['entry_price'] == 0:
            context.pos_records[symbol] = {
                'entry_price': pos['vwap'],
                'high_price': curr_price
            }
        
        rec = context.pos_records[symbol]
        rec['high_price'] = max(rec['high_price'], curr_price)
        
        entry = rec['entry_price']
        high = rec['high_price']
        
        # Stop Loss
        if curr_price < entry * (1 - context.stop_loss):
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            print(f"  !!! [Stop Loss] {symbol} hit -20%. Selling.")
            current_holding_count -= 1
            continue
            
        # Trailing Profit
        if high > entry * (1 + context.trailing_trigger):
            if curr_price < high * (1 - context.trailing_pnl_drop):
                order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
                print(f"  $$$ [Trailing Profit] {symbol} locked. Selling.")
                current_holding_count -= 1
                continue

    # --- 2. Regular Rebalance ---
    is_rebalance_day = (context.days_count - 1) % context.T == 0
    if is_rebalance_day:
        rebalance_ultimate(context, current_dt)
        context.last_rebalance_day = context.days_count
    else:
        # --- Fruit 1: Dynamic Repopulation ---
        # If slots are empty before next rebalance, fill them.
        if current_holding_count < context.top_n:
            fill_empty_slots(context, current_dt, current_holding_count)

def get_ranking_df(context, current_dt):
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 251: return None, None

    # Optimized 5P Scoring (1, 3, 5, 10, 20)
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    total_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        ranks = rets.rank(ascending=False, method='min')
        total_scores += (ranks <= threshold) * pts
    
    r20 = (history_prices.iloc[-1] / history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0, index=history_prices.columns)
    
    valid_scores = total_scores[total_scores.index.isin(context.whitelist)]
    valid_scores = valid_scores[valid_scores >= context.min_score]
    
    if valid_scores.empty:
        return None, total_scores
        
    ranking_df = pd.DataFrame({
        'score': valid_scores.values,
        'r20': r20[valid_scores.index].values,
        'theme': [context.theme_map.get(c, 'Unknown') for c in valid_scores.index]
    }, index=valid_scores.index)
    return ranking_df.sort_values(by=['score', 'r20'], ascending=False), total_scores

def fill_empty_slots(context, current_dt, current_holding_count):
    # Only try to fill if market is strong (using simple check similar to rebalance)
    ranking_df, total_scores = get_ranking_df(context, current_dt)
    if ranking_df is None: return
    
    # Market Check
    strong_market_count = (total_scores >= 150).sum()
    if strong_market_count < 5: return # Don't fill in weak markets

    positions = context.account().positions()
    held_symbols = [p['symbol'] for p in positions if p['amount'] > 0]
    held_themes = {context.theme_map.get(s, 'Unknown') for s in held_symbols}
    
    empty_slots = context.top_n - len(held_symbols)
    if empty_slots <= 0: return
    
    new_candidates = []
    for code, row in ranking_df.iterrows():
        if code not in held_symbols and row['theme'] not in held_themes:
            new_candidates.append(code)
            held_themes.add(row['theme'])
            if len(new_candidates) >= empty_slots:
                break
    
    if new_candidates:
        print(f"  >>> [Fruit 1] Filling {len(new_candidates)} empty slots on Day {context.days_count}")
        # Weighting: for simplicity in dynamic fill, we use equal weight for the new slot
        # A more advanced version would rebalance everything, but that increases turnover.
        # Here we just buy the new ones to fill the 1/N target.
        weight = context.target_percent / context.top_n
        for sym in new_candidates:
            order_target_percent(symbol=sym, percent=weight, order_type=OrderType_Market, position_side=PositionSide_Long)

def rebalance_ultimate(context, current_dt):
    print(f"\n--- [{current_dt.strftime('%Y-%m-%d')}] Scheduled Rebalancing (Day {context.days_count}) ---")
    ranking_df, total_scores = get_ranking_df(context, current_dt)
    
    strong_market_count = (total_scores >= 150).sum()
    market_exposure = 0.3 if strong_market_count < 5 else 1.0

    if ranking_df is None:
        target_dict = {}
    else:
        seen_themes = set()
        selected_etfs = []
        for code, row in ranking_df.iterrows():
            if row['theme'] not in seen_themes:
                selected_etfs.append((code, row['score']))
                seen_themes.add(row['theme'])
            if len(selected_etfs) >= context.top_n:
                break
        
        total_target_score = sum([s for c, s in selected_etfs])
        target_dict = {}
        for code, score in selected_etfs:
            base_weight = 1.0 / len(selected_etfs)
            avg_score = total_target_score / len(selected_etfs)
            adj_factor = 1.0 + (score / avg_score - 1.0) * 0.5 
            target_dict[code] = base_weight * adj_factor
    
    positions = context.account().positions()
    current_symbols = [p['symbol'] for p in positions if p['amount'] > 0]
    context.pos_records = {s: info for s, info in context.pos_records.items() if s in current_symbols}
    
    for sym in current_symbols:
        if sym not in target_dict:
            order_target_percent(symbol=sym, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            
    if target_dict:
        total_weight_sum = sum(target_dict.values())
        for sym, weight in target_dict.items():
            normalized_weight = (weight / total_weight_sum) * context.target_percent * market_exposure
            order_target_percent(symbol=sym, percent=normalized_weight, order_type=OrderType_Market, position_side=PositionSide_Long)

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print("ULTIMATE PLUS STRATEGY (Fruit 1: Dynamic Repopulation)")
    print("="*50)
    print(f"Cumulative Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Annual Return: {indicator.get('pnl_ratio_annual', 0)*100:.2f}%")
    print(f"Max Drawdown: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {indicator.get('sharp_ratio', 0):.2f}")
    print("="*50)

if __name__ == '__main__':
    run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', 
        filename='gm_ultimate_plus.py',
        mode=MODE_BACKTEST,
        token=os.getenv('MY_QUANT_TGM_TOKEN'),
        backtest_start_time='2024-09-01 09:00:00',
        backtest_end_time='2026-01-23 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000)
