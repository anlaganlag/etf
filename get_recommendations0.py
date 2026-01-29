from gm.api import *
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from config import config

load_dotenv()

# --- Sync params from gm_strategy_rolling0.py ---
TOP_N = 5
MIN_SCORE = 20
MAX_PER_THEME = 1
SCORING_METHOD = 'STEP'

def get_ranking(prices_df, whitelist, theme_map, current_dt):
    history = prices_df[prices_df.index <= current_dt]
    if len(history) < 251: 
        print(f"Not enough history: {len(history)} days")
        return None, None

    base_scores = pd.Series(0.0, index=history.columns)
    periods_rule = {1: 20, 3: 30, 5: 50, 10: 70, 20: 100}
    
    rets_dict = {}
    last_row = history.iloc[-1]
    
    for p, pts in periods_rule.items():
        rets = (last_row / history.iloc[-(p+1)]) - 1
        rets_dict[f'r{p}'] = rets
        ranks = rets.rank(ascending=False, method='min')
        
        if SCORING_METHOD == 'SMOOTH':
             decay = (30 - ranks) / 30
             decay = decay.clip(lower=0)
             base_scores += decay * pts
        else:
             base_scores += (ranks <= 15) * pts
    
    valid_scores = base_scores[base_scores.index.isin(whitelist)]
    valid_scores = valid_scores[valid_scores >= MIN_SCORE]
    
    if valid_scores.empty: return None, base_scores

    data_to_df = {
        'score': valid_scores, 
        'theme': [theme_map.get(c, 'Unknown') for c in valid_scores.index],
        'etf_code': valid_scores.index
    }
    
    for p in periods_rule.keys():
        data_to_df[f'r{p}'] = rets_dict[f'r{p}'][valid_scores.index]

    df = pd.DataFrame(data_to_df)
    sort_cols = ['score', 'r1', 'r3', 'r5', 'r10', 'r20', 'etf_code']
    asc_order = [False, False, False, False, False, False, True]
    
    return df.sort_values(by=sort_cols, ascending=asc_order), base_scores

def main():
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    if not token:
        print("Error: MY_QUANT_TGM_TOKEN not found in .env")
        return
    set_token(token)

    # 1. Load Whitelist & Theme Map
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    df_excel.columns = df_excel.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme'}
    df_excel = df_excel.rename(columns=rename_map)
    whitelist = set(df_excel['etf_code'])
    theme_map = df_excel.set_index('etf_code')['theme'].to_dict()
    name_map = df_excel.set_index('etf_code')['etf_name'].to_dict()

    # 2. Build Price Matrix from Cache
    print("Loading data from cache...")
    price_data = {}
    files = [f for f in os.listdir(config.DATA_CACHE_DIR) if f.endswith('.csv') and (f.startswith('sh') or f.startswith('sz'))]
    
    for f in files:
        code = f.replace('_', '.').replace('.csv', '')
        if '.' not in code:
            code = ('SHSE.' if code.startswith('sh') else 'SZSE.') + code[2:]
            
        if code not in whitelist: continue
            
        try:
            df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f), usecols=['日期', '收盘'])
            df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
            price_data[code] = df.set_index('日期')['收盘']
        except: pass
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    # 3. Get Real-time prices if possible to update today
    print("Fetching real-time quotes...")
    try:
        current_quotes = current(symbols=list(whitelist))
        if current_quotes:
            today_prices = {q['symbol']: q['price'] for q in current_quotes if q['price'] > 0}
            if today_prices:
                now_dt = datetime.now().replace(microsecond=0, second=0, minute=0, hour=15)
                # Append today's data as a temporary row
                temp_df = prices_df.copy()
                # Create a row with all columns, filled with NaNs initially
                new_row = pd.Series(index=temp_df.columns, dtype=float)
                for sym, price in today_prices.items():
                    if sym in new_row:
                        new_row[sym] = price
                temp_df.loc[now_dt] = new_row
                temp_df = temp_df.sort_index().ffill()
                prices_df = temp_df
                print(f"Updated with real-time data for {len(today_prices)} ETFs.")
    except Exception as e:
        print(f"Real-time update failed: {e}. Using cached data.")

    # 4. Get Ranking
    latest_dt = prices_df.index[-1]
    print(f"Calculating ranking for: {latest_dt}")
    
    ranking_df, _ = get_ranking(prices_df, whitelist, theme_map, latest_dt)
    
    if ranking_df is not None:
        print("\n" + "="*80)
        print(f"🚀 基于 gm_strategy_rolling0.py 的今日推荐 (TOP_N={TOP_N}, T={13})")
        print(f"数据截止日期: {latest_dt}")
        print("="*80)
        
        targets = []
        theme_count = {}
        
        # Display all filtered candidates first
        print(f"{'代码':<12} {'名称':<15} {'评分':<6} {'主题':<15} {'状态'}")
        print("-" * 80)
        
        for code, row in ranking_df.iterrows():
            theme = row['theme']
            name = name_map.get(code, "Unknown")
            is_selected = False
            
            if theme_count.get(theme, 0) < MAX_PER_THEME and len(targets) < TOP_N:
                targets.append((code, name, theme, row['score']))
                theme_count[theme] = theme_count.get(theme, 0) + 1
                is_selected = True
            
            status = "SELECTED" if is_selected else "SKIP"
            print(f"DEBUG: {code}, {name}, {row['score']}, {theme}, {status}")
            
            if len(ranking_df) > 30 and list(ranking_df.index).index(code) > 30:
                break

        output_lines = []
        output_lines.append("="*80)
        output_lines.append(f"🚀 基于 gm_strategy_rolling0.py 的今日推荐 (TOP_N={TOP_N}, T=13)")
        output_lines.append(f"数据截止日期: {latest_dt}")
        output_lines.append("="*80)
        
        for i, (code, name, theme, score) in enumerate(targets):
            output_lines.append(f"推荐 {i+1}: {code} | {name} | 主题: {theme} | 综合得分: {score}")
        output_lines.append("="*80)
        
        with open('today_recommendations.txt', 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        
        print('\n'.join(output_lines))
    else:
        print("无法生成排名数据，请检查缓存文件是否完整。")

if __name__ == "__main__":
    main()
