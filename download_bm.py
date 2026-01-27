# coding=utf-8
from gm.api import *
import pandas as pd
import os
from dotenv import load_dotenv
from config import config

load_dotenv()

def download_benchmark():
    token = os.getenv('MY_QUANT_TGM_TOKEN')
    set_token(token)
    
    # Chinext Index
    symbol = 'SZSE.399006'
    print(f"Downloading {symbol}...")
    
    # Use config-defined start/end dates or just broad range
    df = history(symbol=symbol, frequency='1d', start_time='2020-01-01 09:00:00', end_time='2026-01-27 16:00:00', fields='eob,open,close', adjust=ADJUST_PREV, df=True)
    
    if df is not None and not df.empty:
        df = df.rename(columns={'eob': '日期', 'open': '开盘', 'close': '收盘'})
        df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
        
        # Save in two formats to be safe
        cache_path1 = os.path.join(config.DATA_CACHE_DIR, 'sz399006.csv')
        cache_path2 = os.path.join(config.DATA_CACHE_DIR, 'SZSE_399006.csv')
        cache_path3 = os.path.join(config.DATA_CACHE_DIR, 'sz_399006.csv')
        
        df.to_csv(cache_path1, index=False)
        df.to_csv(cache_path2, index=False)
        df.to_csv(cache_path3, index=False)
        print(f"Saved benchmark to {cache_path1}")
    else:
        print("Failed to download benchmark.")

if __name__ == '__main__':
    download_benchmark()
