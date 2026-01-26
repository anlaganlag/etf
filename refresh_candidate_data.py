
import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_fetcher import DataFetcher
from config import config

def refresh_candidates():
    print("=== Refreshing Data for Candidate ETFs ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # Load candidates
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    candidates = df_excel.rename(columns={'symbol': 'etf_code', 'sec_name': 'etf_name'})
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = "2023-01-01" # Fetch enough for logic
    
    codes = candidates['etf_code'].unique()
    print(f"Total candidates to refresh: {len(codes)}")
    
    success = 0
    for i, code in enumerate(codes):
        # Force refresh by removing 12-hour constraint? 
        # Actually DataFetcher checks file modification time. 
        # I'll just call get_etf_daily_history. 
        # If the file exists and is "fresh", it won't re-fetch.
        # But our files are NOT fresh (they only have data up to Sept 2024).
        # Wait, if the file was MODIFIED recently but contains old data? 
        # No, DataFetcher writes the new data to the file.
        
        df = fetcher.get_etf_daily_history(code, start_date, end_date)
        if df is not None and not df.empty:
            success += 1
            if success % 10 == 0:
                print(f"  [{i+1}/{len(codes)}] {code} fetched: {len(df)} rows")
        else:
            print(f"  [{i+1}/{len(codes)}] {code} FAILED")
            
    print(f"\nDone. Success: {success}/{len(codes)}")

if __name__ == "__main__":
    refresh_candidates()
