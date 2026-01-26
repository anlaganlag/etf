# coding=utf-8
from __future__ import print_function, absolute_import
from gm.api import *
import pandas as pd
import numpy as np
import os
import sys
from dotenv import load_dotenv

# Import project config
from config import config
from src.data_fetcher import DataFetcher

load_dotenv()

def init(context):
    # Retrieve stop_loss from sys.argv because MyQuant run() is a bit picky about user_params in some versions
    # Use environment variable as a safer communication channel between loop and backtest
    context.stop_loss = float(os.getenv('TEST_SL_VALUE', 0.10))
    
    context.T = 14
    context.top_n = 10
    context.min_score = 150
    context.target_percent = 0.98
    context.trailing_trigger = 0.10
    context.trailing_pnl_drop = 0.05
    context.days_count = 0
    context.pos_records = {}

    # Build Global Price Matrix
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

    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    df_excel = df_excel.rename(columns={'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'})
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    subscribe(symbols='SHSE.000300', frequency='1d')

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    positions = context.account().positions()
    for pos in positions:
        if pos['amount'] == 0: continue
        symbol = pos['symbol']
        curr_price = pos['price']
        if symbol not in context.pos_records:
            context.pos_records[symbol] = {'entry_price': pos['vwap'], 'high_price': curr_price}
        rec = context.pos_records[symbol]
        rec['high_price'] = max(rec['high_price'], curr_price)
        
        if curr_price < rec['entry_price'] * (1 - context.stop_loss):
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            continue
        if rec['high_price'] > rec['entry_price'] * (1 + context.trailing_trigger):
            if curr_price < rec['high_price'] * (1 - context.trailing_pnl_drop):
                order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)

    if (context.days_count - 1) % context.T == 0:
        history_prices = context.prices_df[context.prices_df.index <= current_dt]
        if len(history_prices) < 251: return
        periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
        total_scores = pd.Series(0.0, index=history_prices.columns)
        for p, pts in periods_rule.items():
            rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
            ranks = rets.rank(ascending=False, method='min')
            total_scores += (ranks <= 15) * pts
        
        valid_scores = total_scores[total_scores.index.isin(context.whitelist)]
        valid_scores = valid_scores[valid_scores >= context.min_score]
        
        if not valid_scores.empty:
            df = pd.DataFrame({'score': valid_scores, 'r20': (history_prices.iloc[-1]/history_prices.iloc[-21]-1 if len(history_prices)>20 else 0) })
            df['theme'] = [context.theme_map.get(c, 'Unknown') for c in df.index]
            df = df.sort_values(['score', 'r20'], ascending=False)
            
            selected = []
            seen = set()
            for code, row in df.iterrows():
                if row['theme'] not in seen:
                    selected.append((code, row['score']))
                    seen.add(row['theme'])
                if len(selected) >= context.top_n: break
            
            total_s = sum([s for c,s in selected])
            curr_pos = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
            for s in curr_pos:
                if s not in [x[0] for x in selected]: order_target_percent(symbol=s, percent=0)
            
            exposure = 0.3 if (total_scores >= 150).sum() < 5 else 1.0
            for c, s in selected:
                weight = (1.0/len(selected)) * (1.0 + (s/(total_s/len(selected)) - 1.0)*0.5)
                order_target_percent(symbol=c, percent=weight * context.target_percent * exposure)

def on_backtest_finished(context, indicator):
    print(f"SL_RESULT: {context.stop_loss:.2f} | PnL: {indicator.get('pnl_ratio',0)*100:.2f}% | MDD: {indicator.get('max_drawdown',0)*100:.2f}% | Sharpe: {indicator.get('sharp_ratio',0):.2f}")

if __name__ == '__main__':
    TGM_TOKEN = os.getenv('MY_QUANT_TGM_TOKEN')
    for sl in [0.10, 0.15, 0.20, 0.25, 0.30]:
        print(f"\n>>> TESTING STOP LOSS: -{sl*100:.0f}%")
        os.environ['TEST_SL_VALUE'] = str(sl)
        run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', 
            filename=os.path.abspath(__file__),
            mode=MODE_BACKTEST,
            token=TGM_TOKEN,
            backtest_start_time='2024-09-01 09:00:00',
            backtest_end_time='2026-01-23 16:00:00',
            backtest_adjust=ADJUST_PREV,
            backtest_initial_cash=1000000,
            backtest_commission_ratio=0.0001,
            backtest_slippage_ratio=0.0001)
