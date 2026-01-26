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

def init(context):
    # Standard 176 version parameters
    context.T = 14
    context.top_n = 10
    context.min_score = 150
    context.target_percent = 0.98
    context.days_count = 0
    
    # 8-period rule
    context.periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    
    # Data loading
    if not hasattr(init, 'prices_df'):
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
        init.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    context.prices_df = init.prices_df

    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
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
    
    if (context.days_count - 1) % context.T == 0:
        history_prices = context.prices_df[context.prices_df.index <= current_dt]
        if len(history_prices) < 251: return

        threshold = 15
        total_scores = pd.Series(0.0, index=history_prices.columns)
        for p, pts in context.periods_rule.items():
            if len(history_prices) > p:
                rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
                total_scores += (rets.rank(ascending=False, method='min') <= threshold) * pts
        
        valid_scores = total_scores[total_scores.index.isin(context.whitelist)]
        valid_scores = valid_scores[valid_scores >= context.min_score]
        
        r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0)
        
        if valid_scores.empty:
            target_dict = {}
        else:
            ranking_df = pd.DataFrame({'score': valid_scores, 'r20': r20[valid_scores.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid_scores.index]})
            sorted_df = ranking_df.sort_values(by=['score', 'r20'], ascending=False)
            
            seen_themes = set()
            selected = []
            for code, row in sorted_df.iterrows():
                if row['theme'] not in seen_themes:
                    selected.append((code, row['score']))
                    seen_themes.add(row['theme'])
                if len(selected) >= context.top_n: break
            
            if not selected:
                target_dict = {}
            else:
                target_dict = {c: 1.0/len(selected) for c, s in selected}

        # Market timing (Simple) - Origin likely used 1.0
        exposure = 1.0

        curr_pos = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
        for s in curr_pos:
            if s not in target_dict: order_target_percent(symbol=s, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
        if target_dict:
            tw = sum(target_dict.values())
            for sym, w in target_dict.items():
                order_target_percent(symbol=sym, percent=(w/tw)*0.98*exposure, order_type=OrderType_Market, position_side=PositionSide_Long)

def on_backtest_finished(context, indicator):
    print(f"REPRO_RESULT|{indicator.get('pnl_ratio',0)*100:.2f}|{indicator.get('max_drawdown',0)*100:.2f}|{indicator.get('sharp_ratio',0):.2f}")

if __name__ == '__main__':
    run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', filename='reproduce_176.py', mode=MODE_BACKTEST, token=os.getenv('MY_QUANT_TGM_TOKEN'),
        backtest_start_time='2024-09-01 09:00:00', backtest_end_time='2026-01-23 16:00:00', backtest_adjust=ADJUST_PREV, backtest_initial_cash=1000000)
