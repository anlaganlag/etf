
import os
import pandas as pd
from src.data_fetcher import DataFetcher
from config import config
from datetime import datetime, timedelta

def refresh_curated_data():
    print("=== Refreshing Curated ETF Data (Adding Open Price) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # Load Strong List (Baseline)
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df = pd.read_excel(excel_path)
    codes = df['symbol'].str.strip().tolist()
    
    # ChiNext Benchmark
    codes.append('SZSE.159915')
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = "2023-01-01"
    
    success = 0
    for i, code in enumerate(codes):
        # Force re-fetch by checking if '开盘' exists in CSV or just re-fetching
        print(f"[{i+1}/{len(codes)}] Fetching {code}...")
        
        # We delete existing cache for these 150 to be safe
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            os.remove(cache_file)
            
        df_hist = fetcher.get_etf_daily_history(code, start_date, end_date)
        if not df_hist.empty and '开盘' in df_hist.columns:
            success += 1
            
    print(f"\nDone. Success: {success}/{len(codes)}")

if __name__ == "__main__":
    refresh_curated_data()
