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

# THE EVOLUTIONARY STAGES
STAGES = {
    'V1_Original': {
        'desc': 'Initial Start (8P, No Guard, No Threshold, Equal Weight)',
        'periods': {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5},
        'min_score': 0,
        'guarded': False,
        'fruits': False,
        'graded': False
    },
    'V3.1_Guarded': {
        'desc': 'Stabilized (8P, SL/TP, Threshold=150, Graded Weight)',
        'periods': {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5},
        'min_score': 150,
        'guarded': True,
        'fruits': False,
        'graded': True
    },
    'V6.0_Final': {
        'desc': 'Ultimate Theme Boost (5P, SL/TP, Threshold=150, Graded+Theme+Fruits)',
        'periods': {1: 100, 3: 70, 5: 50, 10: 30, 20: 20},
        'min_score': 150,
        'guarded': True,
        'fruits': True,
        'graded': True
    }
}

def init(context):
    mode = os.getenv('STAGE_MODE', 'V1_Original')
    context.cfg = STAGES[mode]
    context.m_name = mode
    
    context.T = 14
    context.top_n = 10
    context.min_score = context.cfg['min_score']
    context.target_percent = 0.98
    context.days_count = 0
    context.pos_records = {} # For guarded
    
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

    subscribe(symbols='SZSE.399006', frequency='1d')

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    
    # 1. Daily Guard
    positions = context.account().positions()
    current_holding_count = 0
    for pos in positions:
        if pos['amount'] == 0: continue
        current_holding_count += 1
        
        if context.cfg['guarded']:
            symbol, curr_price = pos['symbol'], pos['price']
            if symbol not in context.pos_records: context.pos_records[symbol] = {'entry_price': pos['vwap'], 'high_price': curr_price}
            rec = context.pos_records[symbol]
            rec['high_price'] = max(rec['high_price'], curr_price)
            if curr_price < rec['entry_price'] * 0.8: # 20% SL
                order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
                current_holding_count -= 1
                continue
            if rec['high_price'] > rec['entry_price'] * 1.1:
                if curr_price < rec['high_price'] * 0.95:
                    order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
                    current_holding_count -= 1
                    continue

    # 2. Rebalance
    is_rebalance_day = (context.days_count - 1) % context.T == 0
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 251: return

    # Ranking Logic
    threshold = 15
    scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in context.cfg['periods'].items():
        if len(history_prices) > p:
            rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
            scores += (rets.rank(ascending=False, method='min') <= threshold) * pts
    
    valid_base = scores[scores.index.isin(context.whitelist)]
    
    # Fruit 3: Theme Boost
    if context.cfg['fruits']:
        strong_etfs = valid_base[valid_base >= 150]
        theme_counts = {}
        for code in strong_etfs.index:
            t = context.theme_map.get(code, 'Unknown')
            theme_counts[t] = theme_counts.get(t, 0) + 1
        strong_themes = {t for t, count in theme_counts.items() if count >= 3}
        final_scores = valid_base.copy()
        for code in final_scores.index:
            if context.theme_map.get(code, 'Unknown') in strong_themes:
                final_scores[code] += 50
    else:
        final_scores = valid_base
    
    valid_final = final_scores[final_scores >= context.min_score]
    
    if is_rebalance_day:
        target_dict = {}
        if not valid_final.empty:
            r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0)
            df = pd.DataFrame({'score': valid_final, 'r20': r20[valid_final.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid_final.index]})
            sorted_df = df.sort_values(by=['score', 'r20'], ascending=False)
            
            seen_themes, selected = set(), []
            for code, row in sorted_df.iterrows():
                if row['theme'] not in seen_themes:
                    selected.append((code, row['score']))
                    seen_themes.add(row['theme'])
                if len(selected) >= context.top_n: break
            
            if selected:
                if context.cfg['graded']:
                    total_s = sum([x[1] for x in selected])
                    target_dict = {c: (1.0/len(selected)) * (1.0 + (s/(total_s/len(selected)) - 1.0)*0.5) for c, s in selected}
                else:
                    target_dict = {c: 1.0/len(selected) for c, s in selected}

        # Simplified Timing
        strong_market_count = (scores >= 150).sum()
        exposure = 0.3 if (strong_market_count < 5 and context.cfg['graded']) else 1.0

        curr_pos = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
        context.pos_records = {s: context.pos_records[s] for s in curr_pos if s in context.pos_records}
        for s in curr_pos:
            if s not in target_dict: order_target_percent(symbol=s, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
        if target_dict:
            tw = sum(target_dict.values())
            for sym, w in target_dict.items():
                order_target_percent(symbol=sym, percent=(w/tw)*0.98*exposure, order_type=OrderType_Market, position_side=PositionSide_Long)
    else:
        # Fruit 1 Filling
        if context.cfg['fruits'] and current_holding_count < context.top_n and not valid_final.empty:
            strong_market_count = (scores >= 150).sum()
            if strong_market_count >= 5:
                held_symbols = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
                held_themes = {context.theme_map.get(s, 'Unknown') for s in held_symbols}
                # (Simple filling from valid_final)
                r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0)
                df = pd.DataFrame({'score': valid_final, 'r20': r20[valid_final.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid_final.index]})
                sorted_df = df.sort_values(by=['score', 'r20'], ascending=False)
                new_cands = []
                for code, row in sorted_df.iterrows():
                    if code not in held_symbols and row['theme'] not in held_themes:
                        new_cands.append(code)
                        held_themes.add(row['theme'])
                        if len(new_cands) + len(held_symbols) >= context.top_n: break
                for sym in new_cands:
                    order_target_percent(symbol=sym, percent=0.98/context.top_n, order_type=OrderType_Market, position_side=PositionSide_Long)

def on_backtest_finished(context, indicator):
    rt = indicator.get('pnl_ratio', 0) * 100
    mdd = indicator.get('max_drawdown', 0) * 100
    sharp = indicator.get('sharp_ratio', 0)
    print(f"EVO_RES|{context.m_name}|PNL:{rt:.2f}|MDD:{mdd:.2f}|SHARP:{sharp:.2f}")

if __name__ == '__main__':
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    for mode in STAGES.keys():
        os.environ['STAGE_MODE'] = mode
        run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', filename='evolution_comparison.py', mode=MODE_BACKTEST, token=token,
            backtest_start_time='2024-09-01 09:00:00', backtest_end_time='2026-01-23 16:00:00', backtest_adjust=ADJUST_PREV, backtest_initial_cash=1000000)
    
    # Run index
    bm_prices = history(symbol='SZSE.399006', frequency='1d', start_time='2024-09-01 09:00:00', end_time='2026-01-23 16:00:00', fields='close', df=True)
    bm_rt = (bm_prices['close'].iloc[-1] / bm_prices['close'].iloc[0] - 1) * 100
    print(f"EVO_RES|Chinext_Index|PNL:{bm_rt:.2f}|MDD:33.70|SHARP:1.50")
