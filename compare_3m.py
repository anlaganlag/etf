# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv

# Import project config
from config import config
from src.data_fetcher import DataFetcher

load_dotenv()

def get_shared_prices(context):
    print("Building Price Matrix...")
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    df_all = fetcher.get_all_etfs()
    all_codes = df_all['etf_code'].tolist()
    price_data = {}
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
                price_data[code] = df.set_index('日期')['收盘']
            except: pass
    context.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    if 'theme' not in df_excel.columns: df_excel['theme'] = df_excel['etf_name']
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

def global_rank(history_prices):
    periods = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    total_scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods.items():
        if len(history_prices) > p:
            rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
            total_scores += (rets.rank(ascending=False, method='min') <= 15) * pts
    return total_scores

def init(context):
    mode = os.getenv('TEST_MODE', 'Standard')
    context.mode = mode
    context.T = 14
    context.top_n = 10
    context.min_score = 150
    context.target_percent = 0.98
    context.days_count = 0
    get_shared_prices(context)
    
    if mode == 'Guarded':
        context.stop_loss = 0.20
        context.trailing_trigger = 0.10
        context.trailing_pnl_drop = 0.05
        context.pos_records = {}
        
    subscribe(symbols='SHSE.000300', frequency='1d')

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    # Guard Check (Only for Guarded Mode)
    if context.mode == 'Guarded':
        positions = context.account().positions()
        for pos in positions:
            if pos['amount'] == 0: continue
            sym = pos['symbol']; curr_price = pos['price']
            if sym not in context.pos_records: context.pos_records[sym] = {'entry': pos['vwap'], 'high': curr_price}
            rec = context.pos_records[sym]
            rec['high'] = max(rec['high'], curr_price)
            if curr_price < rec['entry']*(1-context.stop_loss) or (rec['high']>rec['entry']*(1+context.trailing_trigger) and curr_price < rec['high']*(1-context.trailing_pnl_drop)):
                order_target_percent(symbol=sym, percent=0)

    # Rebalance
    if (context.days_count - 1) % context.T == 0:
        history_prices = context.prices_df[context.prices_df.index <= current_dt]
        if len(history_prices) < 251: return
        total_scores = global_rank(history_prices)
        r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0)
        
        valid = total_scores[total_scores.index.isin(context.whitelist)]
        valid = valid[valid >= context.min_score]
        
        if valid.empty:
            target_dict = {}
        else:
            df = pd.DataFrame({'score': valid.values, 'r20': r20[valid.index].values, 'theme': [context.theme_map.get(c, 'Unknown') for c in valid.index]}, index=valid.index).sort_values(['score', 'r20'], ascending=False)
            
            selected = []
            if context.mode in ['Ultimate', 'Guarded']:
                # De-dupe
                seen = set()
                for code, row in df.iterrows():
                    if row['theme'] not in seen:
                        selected.append((code, row['score']))
                        seen.add(row['theme'])
                    if len(selected) >= context.top_n: break
            else:
                # Standard
                for code, row in df.head(context.top_n).iterrows():
                    selected.append((code, row['score']))

            # Weights
            total_s = sum([s for c,s in selected])
            if context.mode in ['Ultimate', 'Guarded']:
                target_dict = {c: (1.0/len(selected)) * (1.0 + (s/(total_s/len(selected)) - 1.0)*0.5) for c, s in selected}
            else:
                target_dict = {c: 1.0/len(selected) for c, s in selected}
        
        # Execute
        exposure = 0.3 if (total_scores >= 150).sum() < 5 else 1.0
        curr_pos = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
        for s in curr_pos:
            if s not in target_dict: order_target_percent(symbol=s, percent=0)
        for s, w in target_dict.items():
            order_target_percent(symbol=s, percent=w * context.target_percent * exposure)

def on_backtest_finished(context, indicator):
    print(f"\nFINISH_MARKER | {context.mode} | PnL: {indicator.get('pnl_ratio',0)*100:.2f}% | MDD: {indicator.get('max_drawdown',0)*100:.2f}% | Sharpe: {indicator.get('sharp_ratio',0):.2f}")

if __name__ == '__main__':
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    for m in ['Standard', 'Ultimate', 'Guarded']:
        print(f"\n>>> Running Mode: {m}")
        os.environ['TEST_MODE'] = m
        run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63',
            filename=os.path.abspath(__file__),
            mode=MODE_BACKTEST,
            token=token,
            backtest_start_time='2025-10-25 09:00:00',
            backtest_end_time='2026-01-23 16:00:00',
            backtest_adjust=ADJUST_PREV,
            backtest_initial_cash=1000000,
            backtest_commission_ratio=0.0001,
            backtest_slippage_ratio=0.0001)
