
import pandas as pd
import numpy as np
import os
from datetime import datetime
from config import config

# Mock standard ranking logic from gm_strategy_rolling.py
def get_analysis_for_period(start_date, end_date):
    # Load whitelist
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    whitelist = set(df_excel['symbol'].astype(str))
    theme_map = df_excel.set_index('symbol')['name_cleaned'].to_dict()

    # Load Price Data from Cache
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
    prices_df = pd.DataFrame(price_data).sort_index().ffill()

    target_range = prices_df[(prices_df.index >= start_date) & (prices_df.index <= end_date)].index
    
    results = []
    
    for current_dt in target_range:
        history_prices = prices_df[prices_df.index <= current_dt]
        if len(history_prices) < 251: continue

        periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
        threshold = 15
        base_scores = pd.Series(0.0, index=history_prices.columns)
        for p, pts in periods_rule.items():
            rets = (history_prices.iloc[-1] / history_prices.iloc[-(p+1)]) - 1
            ranks = rets.rank(ascending=False, method='min')
            base_scores += (ranks <= threshold) * pts
        
        valid_base = base_scores[base_scores.index.isin(whitelist)]
        strong_count = (valid_base >= 150).sum()
        exposure = 1.0 if strong_count >= 5 else 0.3
        
        # Get Top 5 themes and check their prices vs 10/8 high
        ranking = valid_base.sort_values(ascending=False).head(20)
        
        results.append({
            'date': current_dt.strftime('%Y-%m-%d'),
            'strong_count': strong_count,
            'exposure': exposure,
            'top_score': valid_base.max()
        })
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    df = get_analysis_for_period("2024-09-20", "2024-10-20")
    print(df.to_string())
