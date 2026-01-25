
import os
import sys
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_fetcher import DataFetcher
from config import config

def show_latest_holdings():
    output_str = "=== T14 Latest Holdings Analysis (2026-01-22) ===\n\n"
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Load Strong List and Details
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    
    # Map code to name/theme
    code_map = {}
    for _, row in df_strong.iterrows():
        code_map[row['symbol']] = {
            'name': row['sec_name'],
            'theme': row['主题'] if '主题' in row else row.get('name_cleaned', '')
        }
    
    strong_codes = set(df_strong['symbol'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Build Price Matrix
    print("[Data] Loading Prices...")
    price_data = {}
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except: pass
    
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    # 3. Calculate Scores for Target Date
    target_date = "2026-01-22"
    if pd.to_datetime(target_date) not in prices_df.index:
        output_str += f"Error: Date {target_date} not found in data.\n"
    else:
        print(f"[Analysis] Calculating scores for {target_date}...")
        
        periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
        threshold = 15
        idx_loc = prices_df.index.get_loc(target_date)
        
        # Vectorized way for target date
        final_scores = pd.Series(0.0, index=prices_df.columns)
        
        for p, pts in periods_rule.items():
            pcts = prices_df.iloc[idx_loc] / prices_df.iloc[idx_loc - p] - 1
            ranks = pcts.rank(ascending=False, method='min')
            final_scores += (ranks <= threshold) * pts
            
        r20 = prices_df.iloc[idx_loc] / prices_df.iloc[idx_loc - 20] - 1
        r20 = r20.fillna(-999)
        metric = final_scores * 10000 + r20
        
        valid_mask = (final_scores >= 150) & (pd.Series(metric.index).isin(strong_codes).values)
        valid_metric = metric[valid_mask]
        
        top_10 = valid_metric.sort_values(ascending=False).head(10)
        
        output_str += "=== TOP 10 HOLDINGS (2026-01-22) ===\n"
        output_str += f"{'Code':<12} | {'Name':<15} | {'Theme':<12} | {'Score':<6} | {'R20':<8}\n"
        output_str += "-" * 70 + "\n"
        
        for code, m_val in top_10.items():
            score = final_scores[code]
            r20_val = r20[code]
            info = code_map.get(code, {'name': 'Unknown', 'theme': ''})
            # Truncate text to fit
            name = (info['name'][:13] + '..') if len(info['name']) > 13 else info['name']
            
            output_str += f"{code:<12} | {name:<15} | {info['theme']:<12} | {int(score):<6} | {r20_val*100:5.1f}%\n"

    # Write to file
    out_path = os.path.join(config.DATA_OUTPUT_DIR, "latest_holdings.txt")
    with open(out_path, "w", encoding='utf-8') as f:
        f.write(output_str)
    
    print(f"Result saved to {out_path}")

if __name__ == "__main__":
    show_latest_holdings()
