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
    
    # 2. Load Whitelist
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', '主题': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    context.whitelist = set(df_excel['etf_code'])
    print(f"Loaded whitelist: {len(context.whitelist)} ETFs")

    # 3. Build Global Price Matrix from Cache (Matching replicate logic)
    print("Building Global Price Matrix from data_cache...")
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    df_all_etfs = fetcher.get_all_etfs()
    all_codes = df_all_etfs['etf_code'].tolist()
    
    price_data = {}
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                # Use faster loading
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
    
    if (context.days_count - 1) % context.T == 0:
        current_dt = context.now.replace(tzinfo=None)
        rebalance(context, current_dt)

def rebalance(context, current_dt):
    current_date_str = current_dt.strftime('%Y-%m-%d')
    print(f"\n--- [{current_date_str}] Rebalancing (Day {context.days_count}) ---")
    
    # Slice prices up to current date
    # Use asof or simply mask
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    
    if len(history_prices) < 20:
        print("Not enough history for ranking.")
        return

    # Global Ranking Logic
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    
    total_scores = pd.Series(0.0, index=history_prices.columns)
    
    # Calculate ranks for all 
    for p, pts in periods_rule.items():
        if len(history_prices) > p:
            rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
            ranks = rets.rank(ascending=False, method='min')
            total_scores += (ranks <= threshold) * pts
    
    # Tie-breaker R20
    r20 = (history_prices.iloc[-1] / history_prices.iloc[-21] - 1) if len(history_prices) > 20 else pd.Series(0.0, index=history_prices.columns)
    
    # Filter by whitelist and min_score
    valid_scores = total_scores[total_scores.index.isin(context.whitelist)]
    valid_scores = valid_scores[valid_scores >= context.min_score]
    
    if valid_scores.empty:
        print("  No candidates meet the score threshold.")
        target_symbols = []
    else:
        # Sort
        # We can build a df to sort easily
        ranking_df = pd.DataFrame({
            'score': valid_scores,
            'r20': r20[valid_scores.index]
        })
        sorted_df = ranking_df.sort_values(by=['score', 'r20'], ascending=False)
        target_symbols = sorted_df.head(context.top_n).index.tolist()
    
    print(f"  Top Picks: {target_symbols}")
    
    # Execution
    positions = context.account().positions()
    current_symbols = [p['symbol'] for p in positions if p['amount'] > 0]
    
    # Sell non-targets
    for sym in current_symbols:
        if sym not in target_symbols:
            order_target_percent(symbol=sym, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            print(f"  >>> Selling: {sym}")
            
    # Buy targets
    if target_symbols:
        weight = context.target_percent / len(target_symbols)
        for sym in target_symbols:
            order_target_percent(symbol=sym, percent=weight, order_type=OrderType_Market, position_side=PositionSide_Long)
            if sym not in current_symbols:
                print(f"  <<< Buying: {sym}")
    else:
        # Avoid holding old stuff if nothing is strong
        if current_symbols:
            for sym in current_symbols:
                order_target_percent(symbol=sym, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print("Replication Strategy (GM Script - Cached Data)")
    print("="*50)
    print(f"Cumulative Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Annual Return: {indicator.get('pnl_ratio_annual', 0)*100:.2f}%")
    print(f"Max Drawdown: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {indicator.get('sharp_ratio', 0):.2f}")
    print("="*50)

if __name__ == '__main__':
    TGM_TOKEN = os.getenv('MY_QUANT_TGM_TOKEN')
    
    run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', 
        filename='gm_replicate_t14.py',
        mode=MODE_BACKTEST,
        token=TGM_TOKEN,
        backtest_start_time='2024-09-01 09:00:00',
        backtest_end_time='2026-01-23 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
