# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

load_dotenv()

# Configurations to test
SCORING_CONFIGS = {
    '5p': {1: 100, 3: 70, 5: 50, 10: 30, 20: 20},
    '8p': {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
}

def init(context):
    config_key = os.getenv('SCORING_CONFIG', '8p')
    context.periods_rule = SCORING_CONFIGS[config_key]
    context.s_name = config_key
    
    context.T = 14
    context.top_n = 10
    context.min_score = 0 
    context.target_percent = 0.98
    context.days_count = 0
    context.stop_loss = 0.20
    context.trailing_trigger = 0.10
    context.trailing_pnl_drop = 0.05
    context.pos_records = {}

    # Build price matrix manually to avoid heavy history() calls in loop
    if not hasattr(init, 'prices_df'):
        price_data = {}
        # Use a subset if possible, but for ranking we need the whole pool
        files = [f for f in os.listdir(DATA_CACHE_DIR) if f.endswith('.csv') and ('SHSE' in f or 'SZSE' in f)]
        for f in files:
            code = f.replace('_', '.').replace('.csv', '')
            try:
                df = pd.read_csv(os.path.join(DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
                price_data[code] = df.set_index('日期')['收盘']
            except: pass
        init.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    context.prices_df = init.prices_df

    excel_path = os.path.join(BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    df_excel = df_excel.rename(columns={'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'})
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    subscribe(symbols='SHSE.000300', frequency='1d')

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    # Simple Guard
    positions = context.account().positions()
    for pos in positions:
        if pos['amount'] == 0: continue
        symbol = pos['symbol']
        curr_price = pos['price']
        if symbol not in context.pos_records:
            context.pos_records[symbol] = {'entry_price': pos['vwap'], 'high_price': curr_price}
        rec = context.pos_records[symbol]
        rec['high_price'] = max(rec['high_price'], curr_price)
        if curr_price < rec['entry_price'] * 0.8:
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            continue
        if rec['high_price'] > rec['entry_price'] * 1.1:
            if curr_price < rec['high_price'] * 0.95:
                order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)

    # Rebalance
    if (context.days_count - 1) % context.T == 0:
        history_prices = context.prices_df[context.prices_df.index <= current_dt]
        if len(history_prices) < 251: return
        
        scores = pd.Series(0.0, index=history_prices.columns)
        for p, pts in context.periods_rule.items():
            if len(history_prices) > p:
                rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
                scores += (rets.rank(ascending=False, method='min') <= 15) * pts
        
        valid = scores[scores.index.isin(context.whitelist)]
        if not valid.empty:
            r20 = (history_prices.iloc[-1] / history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0)
            df = pd.DataFrame({'score': valid, 'r20': r20[valid.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid.index]})
            df = df.sort_values(['score', 'r20'], ascending=False)
            
            selected = []
            seen = set()
            for code, row in df.iterrows():
                if row['theme'] not in seen:
                    selected.append((code, row['score']))
                    seen.add(row['theme'])
                if len(selected) >= context.top_n: break
            
            if selected:
                total_s = sum([x[1] for x in selected])
                avg_s = total_s / len(selected) if len(selected) > 0 else 1
                target_dict = {c: (1.0/len(selected)) * (1.0 + (s/avg_s - 1.0)*0.5) for c, s in selected}
                
                curr_pos = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
                for s in curr_pos:
                    if s not in target_dict: order_target_percent(symbol=s, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
                for c, w in target_dict.items():
                    order_target_percent(symbol=c, percent=w * context.target_percent, order_type=OrderType_Market, position_side=PositionSide_Long)

def on_backtest_finished(context, indicator):
    print(f"FINAL_RESULT|{os.getenv('TIMEFRAME')}|{context.s_name}|{indicator.get('pnl_ratio',0)*100:.2f}|{indicator.get('max_drawdown',0)*100:.2f}|{indicator.get('sharp_ratio',0):.2f}")

if __name__ == '__main__':
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    
    # Timeframes relative to 2026-01-23
    end_date = '2026-01-23 16:00:00'
    timeframes = {
        '3M': '2025-10-23 09:00:00',
        '6M': '2025-07-23 09:00:00',
        '12M': '2025-01-23 09:00:00',
        'Full': '2024-09-01 09:00:00'
    }
    
    print("Starting Multi-Timeframe Comparison: 5P vs 8P")
    print("-" * 60)
    
    for tf_name, start_date in timeframes.items():
        os.environ['TIMEFRAME'] = tf_name
        for config_key in ['5p', '8p']:
            os.environ['SCORING_CONFIG'] = config_key
            print(f">>> Running {tf_name} | {config_key}...")
            run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63',
                filename='test_5p_vs_8p_batch.py',
                mode=MODE_BACKTEST,
                token=token,
                backtest_start_time=start_date,
                backtest_end_time=end_date,
                backtest_adjust=ADJUST_PREV,
                backtest_initial_cash=1000000,
                backtest_commission_ratio=0.0001,
                backtest_slippage_ratio=0.0001)
