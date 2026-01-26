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
from src.etf_ranker import EtfRanker

load_dotenv()

def init(context):
    # 1. Ultimate Strategy Parameters
    context.T = 14                  
    context.top_n = 10              # Max slots
    context.min_score = 150         
    context.target_percent = 0.98   
    context.days_count = 0
    
    # 2. Load Whitelist & Theme Map
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    # If theme is still missing, use etf_name or Unknown
    if 'theme' not in df_excel.columns:
        df_excel['theme'] = df_excel['etf_name']
    context.whitelist = set(df_excel['etf_code'])
    
    # Pre-map themes for sector de-duplication
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()
    print(f"Loaded whitelist: {len(context.whitelist)} ETFs with theme mapping.")

    # 3. Build Global Price Matrix
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
    if (context.days_count - 1) % context.T == 0:
        current_dt = context.now.replace(tzinfo=None)
        rebalance_ultimate(context, current_dt)

def rebalance_ultimate(context, current_dt):
    current_date_str = current_dt.strftime('%Y-%m-%d')
    print(f"\n--- [{current_date_str}] Ultimate Rebalancing (Day {context.days_count}) ---")
    
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 251: return

    # 1. Global Ranking
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    total_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        ranks = rets.rank(ascending=False, method='min')
        total_scores += (ranks <= threshold) * pts
    
    r20 = (history_prices.iloc[-1] / history_prices.iloc[-21] - 1) if len(history_prices) > 20 else pd.Series(0.0, index=history_prices.columns)
    
    # 2. Market Regime Filter (择时开关)
    # If less than 3 symbols in the market have score > 150, reduce exposure
    strong_market_count = (total_scores >= 150).sum()
    if strong_market_count < 5:
        print(f"  [Market Guard] Weak market (only {strong_market_count} strong ETFs). Reducing exposure to 30%.")
        market_exposure = 0.3
    else:
        market_exposure = 1.0

    # 3. Filter & Sort Whitelist
    valid_scores = total_scores[total_scores.index.isin(context.whitelist)]
    valid_scores = valid_scores[valid_scores >= context.min_score]
    
    if valid_scores.empty:
        print("  No strong candidates.")
        target_dict = {}
    else:
        # Optimization: use pre-calculated theme mapping safely
        themes = [context.theme_map.get(c, 'Unknown') for c in valid_scores.index]
        ranking_df = pd.DataFrame({
            'score': valid_scores.values,
            'r20': r20[valid_scores.index].values,
            'theme': themes
        }, index=valid_scores.index)
        # Sort by score then r20
        sorted_df = ranking_df.sort_values(by=['score', 'r20'], ascending=False)
        
        # 4. Sector De-duplication (行业去重)
        # Limit 1 ETF per theme to maximize diversification
        seen_themes = set()
        selected_etfs = []
        for code, row in sorted_df.iterrows():
            theme = row['theme']
            if theme not in seen_themes:
                selected_etfs.append((code, row['score']))
                seen_themes.add(theme)
            if len(selected_etfs) >= context.top_n:
                break
        
        # 5. Tiered Loading (分级仓位)
        # Higher score = slightly higher weight
        total_target_score = sum([s for c, s in selected_etfs])
        target_dict = {}
        for code, score in selected_etfs:
            # Simple tiered weight: relative to score but capped for safety
            # Base weight = 1/N, adjusted by score relativity
            base_weight = 1.0 / len(selected_etfs)
            # Give +/- 20% weight adjustment based on score
            avg_score = total_target_score / len(selected_etfs)
            adj_factor = 1.0 + (score / avg_score - 1.0) * 0.5 
            # Final effective weight within the portfolio
            target_dict[code] = base_weight * adj_factor
    
    # 6. Execution
    positions = context.account().positions()
    current_symbols = [p['symbol'] for p in positions if p['amount'] > 0]
    
    # Sell non-targets
    for sym in current_symbols:
        if sym not in target_dict:
            order_target_percent(symbol=sym, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            print(f"  >>> Selling: {sym}")
            
    # Buy targets with Market Exposure applied
    total_alloc = 0
    if target_dict:
        total_weight_sum = sum(target_dict.values())
        for sym, weight in target_dict.items():
            # Apply market exposure and normalize
            normalized_weight = (weight / total_weight_sum) * context.target_percent * market_exposure
            order_target_percent(symbol=sym, percent=normalized_weight, order_type=OrderType_Market, position_side=PositionSide_Long)
            total_alloc += normalized_weight
            if sym not in current_symbols:
                print(f"  <<< Buying: {sym} (Weight: {normalized_weight*100:.1f}%, Score: {ranking_df.loc[sym, 'score']})")

    print(f"  Total Portfolio Exposure: {total_alloc*100:.1f}%")

def on_backtest_finished(context, indicator):
    print("\n" + "="*50)
    print("ULTIMATE STRATEGY (Sector De-dupe + Tiered Weight + Market Guard)")
    print("="*50)
    print(f"Cumulative Return: {indicator.get('pnl_ratio', 0)*100:.2f}%")
    print(f"Annual Return: {indicator.get('pnl_ratio_annual', 0)*100:.2f}%")
    print(f"Max Drawdown: {indicator.get('max_drawdown', 0)*100:.2f}%")
    print(f"Sharpe Ratio: {indicator.get('sharp_ratio', 0):.2f}")
    print("="*50)

if __name__ == '__main__':
    TGM_TOKEN = os.getenv('MY_QUANT_TGM_TOKEN')
    
    run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', 
        filename='gm_ultimate_t14.py',
        mode=MODE_BACKTEST,
        token=TGM_TOKEN,
        backtest_start_time='2024-09-01 09:00:00',
        backtest_end_time='2026-01-23 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=1000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001,
        backtest_match_mode=1)
