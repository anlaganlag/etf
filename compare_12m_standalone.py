# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Hardcoded project paths to avoid import issues in MyQuant's constrained environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_CACHE_DIR = os.path.join(BASE_DIR, "data_cache")

load_dotenv()

def init(context):
    # Strategy mode passed from environment
    context.s_mode = os.getenv('STRATEGY_MODE', 'Standard')
    context.T = 14
    context.top_n = 10
    context.min_score = 150
    context.target_percent = 0.98
    context.days_count = 0
    context.pos_records = {}
    
    # 1. Build Global Price Matrix
    all_codes = []
    # Simplified list fetching
    cache_files = [f for f in os.listdir(DATA_CACHE_DIR) if f.endswith('.csv') and 'SHSE' in f or 'SZSE' in f]
    price_data = {}
    for f in cache_files:
        code = f.replace('_', '.').replace('.csv', '')
        try:
            df = pd.read_csv(os.path.join(DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except: pass
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()

    # 2. Load Whitelist
    excel_path = os.path.join(BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    subscribe(symbols='SHSE.000300', frequency='1d')

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    # Guard logic
    if context.s_mode == 'Guarded':
        positions = context.account().positions()
        for pos in positions:
            if pos['amount'] == 0: continue
            symbol = pos['symbol']
            curr_price = pos['price']
            if symbol not in context.pos_records:
                context.pos_records[symbol] = {'entry_price': pos['vwap'], 'high_price': curr_price}
            rec = context.pos_records[symbol]
            rec['high_price'] = max(rec['high_price'], curr_price)
            if curr_price < rec['entry_price'] * 0.8: # -20% SL
                order_target_percent(symbol=symbol, percent=0)
                continue
            if rec['high_price'] > rec['entry_price'] * 1.1: # 10% Trig
                if curr_price < rec['high_price'] * 0.95: # 5% Drop
                    order_target_percent(symbol=symbol, percent=0)

    # Rebalance logic
    if (context.days_count - 1) % context.T == 0:
        history_prices = context.prices_df[context.prices_df.index <= current_dt]
        if len(history_prices) < 251: return
        
        # Ranking
        periods = {1:100, 3:70, 5:50, 10:30, 20:20, 60:15, 120:10, 250:5}
        scores = pd.Series(0.0, index=history_prices.columns)
        for p, pts in periods.items():
            if len(history_prices) > p:
                rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
                scores += (rets.rank(ascending=False, method='min') <= 15) * pts
        
        valid = scores[scores.index.isin(context.whitelist)]
        valid = valid[valid >= context.min_score]
        
        if not valid.empty:
            r20 = (history_prices.iloc[-1] / history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0)
            df = pd.DataFrame({'score': valid, 'r20': r20[valid.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid.index]})
            df = df.sort_values(['score', 'r20'], ascending=False)
            
            dedupe = (context.s_mode in ['Ultimate', 'Guarded'])
            selected = []
            seen = set()
            for code, row in df.iterrows():
                if not dedupe or row['theme'] not in seen:
                    selected.append((code, row['score']))
                    seen.add(row['theme'])
                if len(selected) >= context.top_n: break
            
            if selected:
                # Weights
                target_dict = {}
                if context.s_mode == 'Standard':
                    w = 1.0 / len(selected)
                    target_dict = {c: w for c, s in selected}
                else: # Tiered
                    sum_s = sum([x[1] for x in selected])
                    avg_s = sum_s / len(selected)
                    target_dict = {c: (1.0/len(selected)) * (1.0 + (s/avg_s - 1.0)*0.5) for c, s in selected}
                
                exposure = 0.3 if (scores >= 150).sum() < 5 else 1.0
                curr_pos = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
                for s in curr_pos:
                    if s not in target_dict: order_target_percent(symbol=s, percent=0)
                for c, w in target_dict.items():
                    order_target_percent(symbol=c, percent=w * context.target_percent * exposure)

def on_backtest_finished(context, indicator):
    print(f"RESULT|{context.s_mode}|{indicator.get('pnl_ratio',0)*100:.2f}|{indicator.get('max_drawdown',0)*100:.2f}")

if __name__ == '__main__':
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    for m in ['Standard', 'Ultimate', 'Guarded']:
        print(f"--- Running {m} ---")
        os.environ['STRATEGY_MODE'] = m
        run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63',
            filename=os.path.abspath(__file__),
            mode=MODE_BACKTEST,
            token=token,
            backtest_start_time='2025-01-26 09:00:00',
            backtest_end_time='2026-01-23 16:00:00',
            backtest_adjust=ADJUST_PREV,
            backtest_initial_cash=1000000)
