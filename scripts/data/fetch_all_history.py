import sys
import os
import pandas as pd
from datetime import datetime, timedelta
import time

# Add root to path so we can import config & src
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config
from src.data_fetcher import DataFetcher

def fetch_all():
    print("=== Starting Full ETF History Fetch (Data Warmup) ===")
    
    # 1. Init
    config.ensure_dirs()
    fetcher = DataFetcher(retry_count=3, retry_delay=1, cache_dir=config.DATA_CACHE_DIR)
    
    # 2. Get List
    print("Fetching ETF List...")
    all_etfs = fetcher.get_all_etfs()
    
    if all_etfs is None or all_etfs.empty:
        print("Failed to get ETF list.")
        return

    # Filter out inactive or tiny ETFs to save time?
    # Let's filter by turnover if available to skip zombie funds.
    # But for backtest we might want looser filter. Let's keep turnover > 100k
    if 'turnover' in all_etfs.columns:
        # Turnover is likely string or mixed. Clean it.
        all_etfs['turnover'] = pd.to_numeric(all_etfs['turnover'], errors='coerce').fillna(0)
        
        # Original count
        orig_count = len(all_etfs)
        # Filter: Turnover > 100,000 (100k CNY)
        # This is a safe filter to avoid wasting time on funds with 0 volume.
        all_etfs = all_etfs[all_etfs['turnover'] > 100000]
        print(f"Filtered {orig_count} -> {len(all_etfs)} ETFs (Turnover > 100k)")
    
    # 3. Fetch History Loop
    total = len(all_etfs)
    print(f"Fetching history for {total} ETFs (Past 6 months)...")
    
    end_date = datetime.now().strftime("%Y-%m-%d")
    # Fetch 400 days (approx 1 year + buffer)
    start_date = (datetime.now() - timedelta(days=400)).strftime("%Y-%m-%d")
    
    success_count = 0
    fail_count = 0
    
    start_time = time.time()
    
    for idx, (_, row) in enumerate(all_etfs.iterrows()):
        code = row['etf_code']
        name = row['etf_name']
        
        # Print progress every 10 items
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_speed = (idx + 1) / elapsed
            remaining = (total - idx - 1) / avg_speed if avg_speed > 0 else 0
            print(f"[{idx+1}/{total}] Fetching {code} ({name})... Est. remaining: {remaining/60:.1f} min")

        # The fetcher handles caching logic internally.
        # If file exists and is fresh, it returns quickly.
        df = fetcher.get_etf_daily_history(code, start_date, end_date)
        
        if df is not None and not df.empty and len(df) > 20:
            success_count += 1
        else:
            fail_count += 1
            # print(f"Warning: No data for {code} {name}")
            
    print("\n=== Fetch Complete ===")
    print(f"Success: {success_count}")
    print(f"Failed/Empty: {fail_count}")
    print(f"Data saved to directory: {fetcher.cache_dir}")

if __name__ == "__main__":
    fetch_all()
