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
    # 1. Aggressive Elite Parameters
    context.T = 10                  # Faster cycle to catch heat
    context.top_n = 3               # Extremely concentrated
    context.buffer_n = 5            # Tight buffer
    context.min_score = 150         
    context.exit_score = 30         # Very loose: don't sell unless momentum is dead
    context.target_percent = 0.99   # More exposure
    context.days_count = 0
    
    # 2. Load Whitelist
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', '主题': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    context.whitelist = set(df_excel['etf_code'])
    print(f"Loaded whitelist: {len(context.whitelist)} ETFs")

    # 3. Build Global Price Matrix from Cache
    print("Building Global Price Matrix from data_cache...")
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
    print(f"Global matrix built: {context.prices_df.shape[1]} symbols, {context.prices_df.shape[0]} dates.")

    # 4. Subscribe
    subscribe(symbols='SHSE.000300', frequency='1d')

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    # --- Feature A: Daily Momentum Guard ---
    # Every day, check if existing holdings have completely lost momentum
    positions = context.account().positions()
    current_symbols = [p['symbol'] for p in positions if p['amount'] > 0]
    
    if current_symbols:
        # Quick ranking check
        scores = get_current_scores(context, current_dt, current_symbols)
        for sym in current_symbols:
            # If a holding's score drops too low, don't wait for T days, sell immediately
            if sym in scores and scores[sym] < context.exit_score:
                order_target_percent(symbol=sym, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
                print(f"  !!! [Momentum Guard] {sym} score dropped to {scores[sym]}. Exit immediately.")
    
    # --- Feature B: Regular Elite Rebalancing ---
    if (context.days_count - 1) % context.T == 0:
        rebalance_elite(context, current_dt)

def get_current_scores(context, current_dt, symbols_to_check):
    """Calculates scores for specific symbols up to current_dt"""
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 251: return {}
    
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    
    # We need to rank against the whole market to get score
    # To speed up daily check, we only do this locally
    total_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        ranks = rets.rank(ascending=False, method='min')
        total_scores += (ranks <= threshold) * pts
    
    return total_scores[total_scores.index.isin(symbols_to_check)].to_dict()

def rebalance_elite(context, current_dt):
    current_date_str = current_dt.strftime('%Y-%m-%d')
    print(f"\n--- [{current_date_str}] Elite Rebalancing (Day {context.days_count}) ---")
    
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 21: return

    # 1. Global Ranking
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    total_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        if len(history_prices) > p:
            rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
            ranks = rets.rank(ascending=False, method='min')
            total_scores += (ranks <= threshold) * pts
    
    r20 = (history_prices.iloc[-1] / history_prices.iloc[-21] - 1) if len(history_prices) > 20 else pd.Series(0.0, index=history_prices.columns)
    
    # 2. Filter Whitelist + Entry Threshold
    valid_whitelist = total_scores[total_scores.index.isin(context.whitelist)]
    
    # 3. Buffer Ranking Logic
    # Get current holdings
    positions = context.account().positions()
    current_holdings = [p['symbol'] for p in positions if p['amount'] > 0]
    
    # Candidates for buying (Strict Top N)
    rank_df = pd.DataFrame({'score': valid_whitelist, 'r20': r20[valid_whitelist.index]})
    sorted_candidates = rank_df[rank_df['score'] >= context.min_score].sort_values(by=['score', 'r20'], ascending=False)
    
    top_picks = sorted_candidates.head(context.top_n).index.tolist()
    buffer_picks = sorted_candidates.head(context.buffer_n).index.tolist()
    
    final_targets = []
    # A. Prioritize Current Strong Holdings (If in Buffer N)
    for sym in current_holdings:
        if sym in buffer_picks:
            final_targets.append(sym)
            print(f"  [Hold] {sym} remains strong (Top {context.buffer_n}), keeping.")
            
    # B. Fill remaining spots with Top Picks
    for sym in top_picks:
        if len(final_targets) >= context.top_n:
            break
        if sym not in final_targets:
            final_targets.append(sym)
            print(f"  [New] {sym} is an elite performer, adding to portfolio.")
            
    # 4. Execution
    # Sell non-targets
    for sym in current_holdings:
        if sym not in final_targets:
            order_target_percent(symbol=sym, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            print(f"  >>> Selling: {sym}")
            
    # Buy targets
    if final_targets:
        weight = context.target_percent / len(final_targets)
        for sym in final_targets:
            order_target_percent(symbol=sym, percent=weight, order_type=OrderType_Market, position_side=PositionSide_Long)
            if sym not in current_holdings:
                print(f"  <<< Buying: {sym} (Weight: {weight*100:.1f}%)")
    else:
        print("  [Wait] No high-momentum ETFs found. Scaling back to cash.")

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print("AGGRESSIVE TOP 3 STRATEGY (T=10 + Loose Guard)")
    print("="*50)
    print(f"Cumulative Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Annual Return: {indicator.get('pnl_ratio_annual', 0)*100:.2f}%")
    print(f"Max Drawdown: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {indicator.get('sharp_ratio', 0):.2f}")
    print("="*50)

if __name__ == '__main__':
    TGM_TOKEN = os.getenv('MY_QUANT_TGM_TOKEN')
    
    run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', 
        filename='gm_elite_t14.py',
        mode=MODE_BACKTEST,
        token=TGM_TOKEN,
        backtest_start_time='2024-09-01 09:00:00',
        backtest_end_time='2026-01-23 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
