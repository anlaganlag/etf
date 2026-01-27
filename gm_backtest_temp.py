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
    # 1. Strategy Parameters (v6.0 Theme Boost)
    context.T = 14                  
    context.top_n = 2          # Top 3 (Aggressive) = ~195% Return. Set to 10 for Standard (~112%)
    context.min_score = 150        
    context.target_percent = 0.98   
    context.days_count = 0
    
    # --- Guard Parameters ---
    context.stop_loss = 0.20        
    context.trailing_trigger = 0.10 
    context.trailing_pnl_drop = 0.05 
    
    # Store position info
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

    # Create reverse map: theme -> [etf_codes]
    context.theme_to_codes = {}
    for code, theme in context.theme_map.items():
        if theme not in context.theme_to_codes:
            context.theme_to_codes[theme] = []
        context.theme_to_codes[theme].append(code)

    # 3. Build Global Price Matrix for fast ranking
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # Ensure cache is populated (in case DataFetcher doesn't do it automatically in this context)
    # But usually DataFetcher reads from cache. We just need to load all CSVs.
    
    price_data = {}
    # Scan directory for all CSVs to build the universe
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
             if code.startswith('sh'): code = 'SHSE.' + code[2:]
             elif code.startswith('sz'): code = 'SZSE.' + code[2:]
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            if not df.empty:
                price_data[code] = df.set_index('日期')['收盘']
        except: pass
    
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"Global matrix built: {context.prices_df.shape[1]} symbols.")

    subscribe(symbols='SHSE.000300', frequency='1d')

def get_ranking_with_theme_boost(context, current_dt):
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 251: return None, None

    # Base 5P Scoring (1, 3, 5, 10, 20)
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    base_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        ranks = rets.rank(ascending=False, method='min')
        base_scores += (ranks <= threshold) * pts
    
    # Filter whitelist
    valid_base = base_scores[base_scores.index.isin(context.whitelist)]
    
    # --- Fruit 3: Theme Boost Logic ---
    # 1. Identity strong themes (at least 3 ETFs with base score >= 150)
    strong_etfs = valid_base[valid_base >= 150]
    theme_counts = {}
    for code in strong_etfs.index:
        t = context.theme_map.get(code, 'Unknown')
        theme_counts[t] = theme_counts.get(t, 0) + 1
    
    strong_themes = {t for t, count in theme_counts.items() if count >= 3}
    
    # 2. Apply Boost
    final_scores = valid_base.copy()
    for code in final_scores.index:
        if context.theme_map.get(code, 'Unknown') in strong_themes:
            final_scores[code] += 50
            
    # filter by final score threshold
    valid_final = final_scores[final_scores >= context.min_score]
    if valid_final.empty: return None, base_scores
    
    r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0, index=history_prices.columns)
    
    df = pd.DataFrame({
        'score': valid_final.values,
        'r20': r20[valid_final.index].values,
        'theme': [context.theme_map.get(c, 'Unknown') for c in valid_final.index]
    }, index=valid_final.index)
    
    return df.sort_values(by=['score', 'r20'], ascending=False), base_scores

def fill_empty_slots(context, current_dt):
    ranking_df, total_scores = get_ranking_with_theme_boost(context, current_dt)
    if ranking_df is None: return

    # Market Check (using base scores to stay consistent)
    strong_market_count = (total_scores >= 150).sum()
    if strong_market_count < 5: return 

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
        print(f"  >>> [Fruit 3] Theme Boost Fill: {new_candidates} on Day {context.days_count}")
        weight = context.target_percent / context.top_n
        for sym in new_candidates:
            order_target_percent(symbol=sym, percent=weight, order_type=OrderType_Market, position_side=PositionSide_Long)

def rebalance_ultimate(context, current_dt):
    print(f"\n--- [{current_dt.strftime('%Y-%m-%d')}] Scheduled Rebalancing (Day {context.days_count}) ---")
    ranking_df, total_scores = get_ranking_with_theme_boost(context, current_dt)
    
    strong_market_count = (total_scores >= 150).sum()
    market_exposure = 0.3 if strong_market_count < 5 else 1.0

    target_dict = {}
    if ranking_df is not None:
        seen_themes = set()
        selected_etfs = []
        for code, row in ranking_df.iterrows():
            if row['theme'] not in seen_themes:
                selected_etfs.append((code, row['score']))
                seen_themes.add(row['theme'])
            if len(selected_etfs) >= context.top_n:
                break
        
        if selected_etfs:
            total_target_score = sum([s for c, s in selected_etfs])
            for code, score in selected_etfs:
                base_weight = 1.0 / len(selected_etfs)
                avg_score = total_target_score / len(selected_etfs)
                adj_factor = 1.0 + (score / avg_score - 1.0) * 0.5 
                target_dict[code] = base_weight * adj_factor
    
    positions = context.account().positions()
    current_symbols = [p['symbol'] for p in positions if p['amount'] > 0]
    
    # Critical Fix: Clean up pos_records to remove stale data from old trades
    # This prevents re-entries from inheriting old entry prices and triggering immediate stops
    context.pos_records = {s: context.pos_records[s] for s in current_symbols if s in context.pos_records}
    
    for sym in current_symbols:
        if sym not in target_dict:
            order_target_percent(symbol=sym, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            if sym in context.pos_records: del context.pos_records[sym]
            
    if target_dict:
        total_w = sum(target_dict.values())
        for sym, w in target_dict.items():
            order_target_percent(symbol=sym, percent=(w/total_w) * context.target_percent * market_exposure, order_type=OrderType_Market, position_side=PositionSide_Long)

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    # 1. Guard
    positions = context.account().positions()
    current_holding_count = 0
    for pos in positions:
        if pos['amount'] == 0: continue
        current_holding_count += 1
        symbol, curr_price = pos['symbol'], pos['price']
        if symbol not in context.pos_records or context.pos_records[symbol]['entry_price'] == 0:
            context.pos_records[symbol] = {'entry_price': pos['vwap'], 'high_price': curr_price}
        rec = context.pos_records[symbol]
        rec['high_price'] = max(rec['high_price'], curr_price)
        entry, high = rec['entry_price'], rec['high_price']
        if curr_price < entry * (1 - context.stop_loss):
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            current_holding_count -= 1
            print(f"  !!! [Stop Loss] {symbol}")
            continue
        if high > entry * (1 + context.trailing_trigger):
            if curr_price < high * (1 - context.trailing_pnl_drop):
                order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
                current_holding_count -= 1
                print(f"  $$$ [Trailing Profit] {symbol}")
                continue

    # 2. Rebalance / Filling
    if (context.days_count - 1) % context.T == 0:
        rebalance_ultimate(context, current_dt)
    else:
        if current_holding_count < context.top_n:
            fill_empty_slots(context, current_dt)

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print("ULTIMATE THEME BOOST (GM BACKTEST REPRO)")
    print("="*50)
    print(f"Cumulative Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Max Drawdown: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {indicator.get('sharp_ratio', 0):.2f}")
    print("="*50)

if __name__ == '__main__':
    run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', 
        filename='gm_backtest_temp.py',
        mode=MODE_BACKTEST,
        token=os.getenv('MY_QUANT_TGM_TOKEN'),
        backtest_start_time='2024-09-01 09:00:00',
        backtest_end_time='2026-01-27 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)
