import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(r'd:\antigravity\old_etf\etf', 'src'))
from data_fetcher import DataFetcher
from config import config

def main():
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    start_date = "2020-01-01"
    end_date = "2026-01-27"
    
    # 1. Update Index
    print("Updating SHSE.000001 benchmark...")
    fetcher.get_etf_daily_history("SHSE.000001", start_date, end_date)
    
    # 2. Get All ETFs list
    print("Getting all ETF list...")
    etf_list_df = fetcher.get_all_etfs()
    all_etf_codes = etf_list_df['etf_code'].tolist()
    
    # 3. Load Whitelist
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_excel = pd.read_excel(excel_path)
    whitelist = df_excel['symbol'].tolist()
    
    # Combine (priority: whitelist, then top ETFs for ranking pool)
    # To keep it efficient, let's fetch whitelist + 300 more ETFs.
    target_pool = list(set(whitelist) | set(all_etf_codes[:400])) # 100 whitelist + 300 pool
    
    print(f"Total target sync: {len(target_pool)} ETFs")
    
    # Parallelize? No, MyQuant likes serial.
    for i, code in enumerate(target_pool):
        if i % 10 == 0:
            print(f"Progress: {i}/{len(target_pool)} - Current: {code}")
        try:
            fetcher.get_etf_daily_history(code, start_date, end_date)
        except Exception as e:
            print(f"Failed {code}: {e}")

if __name__ == "__main__":
    main()
