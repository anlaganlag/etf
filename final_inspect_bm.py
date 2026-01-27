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
    context.T = 14                  
    context.top_n = 10              
    context.min_score = 150         
    context.target_percent = 0.98   
    context.days_count = 0
    context.stop_loss = 0.20        
    context.trailing_trigger = 0.10 
    context.trailing_pnl_drop = 0.05 
    context.pos_records = {}
    context.bm_symbol = 'SZSE.399006' # Chinext Index

    # Load Whitelist & Theme Map
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    context.whitelist = set(df_excel['etf_code'])
    context.theme_map = df_excel.set_index('etf_code')['theme'].to_dict()

    if not hasattr(init, 'prices_df'):
        price_data = {}
        files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv')]
        for f in files:
            code = f.replace('_', '.').replace('.csv', '')
            try:
                df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
                price_data[code] = df.set_index('日期')['收盘']
            except: pass
        init.prices_df = pd.DataFrame(price_data).sort_index().ffill()
    context.prices_df = init.prices_df

    subscribe(symbols=context.bm_symbol, frequency='1d')

def get_ranking(context, current_dt):
    history_prices = context.prices_df[context.prices_df.index <= current_dt]
    if len(history_prices) < 251: return None, None
    periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    threshold = 15
    scores = pd.Series(0.0, index=history_prices.columns)
    for p, pts in periods_rule.items():
        rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
        scores += (rets.rank(ascending=False, method='min') <= threshold) * pts
    valid_base = scores[scores.index.isin(context.whitelist)]
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
    valid_final = final_scores[final_scores >= context.min_score]
    if valid_final.empty: return None, scores
    r20 = (history_prices.iloc[-1]/history_prices.iloc[-21]-1) if len(history_prices)>20 else pd.Series(0.0)
    df = pd.DataFrame({'score': valid_final, 'r20': r20[valid_final.index], 'theme': [context.theme_map.get(c, 'Unknown') for c in valid_final.index]})
    return df.sort_values(by=['score', 'r20'], ascending=False), scores

def on_bar(context, bars):
    context.days_count += 1
    current_dt = context.now.replace(tzinfo=None)
    positions = context.account().positions()
    current_holding_count = 0
    for pos in positions:
        if pos['amount'] == 0: continue
        current_holding_count += 1
        symbol, curr_price = pos['symbol'], pos['price']
        if symbol not in context.pos_records: context.pos_records[symbol] = {'entry_price': pos['vwap'], 'high_price': curr_price}
        rec = context.pos_records[symbol]
        rec['high_price'] = max(rec['high_price'], curr_price)
        if curr_price < rec['entry_price'] * 0.8:
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
            current_holding_count -= 1
            continue
        if rec['high_price'] > rec['entry_price'] * 1.1:
            if curr_price < rec['high_price'] * 0.95:
                order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
                current_holding_count -= 1
                continue
    is_rebalance_day = (context.days_count - 1) % context.T == 0
    ranking_df, total_scores = get_ranking(context, current_dt)
    if is_rebalance_day:
        strong_market_count = (total_scores >= 150).sum()
        exposure = 0.3 if strong_market_count < 5 else 1.0
        target_dict = {}
        if ranking_df is not None:
            seen_themes, selected = set(), []
            for code, row in ranking_df.iterrows():
                if row['theme'] not in seen_themes:
                    selected.append((code, row['score']))
                    seen_themes.add(row['theme'])
                if len(selected) >= context.top_n: break
            if selected:
                total_s = sum([x[1] for x in selected])
                for code, score in selected:
                    target_dict[code] = (1.0/len(selected)) * (1.0 + (score/(total_s/len(selected)) - 1.0)*0.5)
        curr_pos = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
        context.pos_records = {s: context.pos_records[s] for s in curr_pos if s in context.pos_records}
        for s in curr_pos:
            if s not in target_dict: order_target_percent(symbol=s, percent=0, order_type=OrderType_Market, position_side=PositionSide_Long)
        if target_dict:
            tw = sum(target_dict.values())
            for sym, w in target_dict.items():
                order_target_percent(symbol=sym, percent=(w/tw)*0.98*exposure, order_type=OrderType_Market, position_side=PositionSide_Long)
    else:
        if current_holding_count < context.top_n and ranking_df is not None:
            strong_market_count = (total_scores >= 150).sum()
            if strong_market_count >= 5:
                held_symbols = [p['symbol'] for p in context.account().positions() if p['amount'] > 0]
                held_themes = {context.theme_map.get(s, 'Unknown') for s in held_symbols}
                new_cands = []
                for code, row in ranking_df.iterrows():
                    if code not in held_symbols and row['theme'] not in held_themes:
                        new_cands.append(code)
                        held_themes.add(row['theme'])
                        if len(new_cands) + len(held_symbols) >= context.top_n: break
                for sym in new_cands:
                    order_target_percent(symbol=sym, percent=0.98/context.top_n, order_type=OrderType_Market, position_side=PositionSide_Long)

def on_backtest_finished(context, indicator):
    rt = indicator.get('pnl_ratio', 0) * 100
    # Manual fetch of benchmark PNL
    bm_prices = history(symbol=context.bm_symbol, frequency='1d', start_time=os.getenv('START_DATE'), end_time='2026-01-23 16:00:00', fields='close', df=True)
    if not bm_prices.empty:
        bm_rt = (bm_prices['close'].iloc[-1] / bm_prices['close'].iloc[0] - 1) * 100
    else:
        bm_rt = 0.0
    alpha = rt - bm_rt
    print(f"RESULT|{os.getenv('TF')}|STRAT:{rt:.2f}|BM:{bm_rt:.2f}|ALPHA:{alpha:.2f}|MDD:{indicator.get('max_drawdown',0)*100:.2f}|SHARP:{indicator.get('sharp_ratio',0):.2f}")

if __name__ == '__main__':
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    tfs = {'Full': '2024-01-01'}
    print("TF | STRAT_PNL | BM_PNL | ALPHA | MDD | SHARP")
    for tf_name, start in tfs.items():
        os.environ['TF'] = tf_name
        os.environ['START_DATE'] = f"{start} 09:00:00"
        run(strategy_id='fe474f75-fa8e-11f0-b097-00ffda9d6e63', filename='final_inspect_bm.py', mode=MODE_BACKTEST, token=token,
            backtest_start_time=f'{start} 09:00:00', backtest_end_time='2026-01-23 16:00:00', backtest_adjust=ADJUST_PREV, 
            backtest_initial_cash=1000000)
