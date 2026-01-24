
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.etf_ranker import EtfRanker
from src.data_fetcher import DataFetcher
from config import config

def run_backtest():
    print("=== Starting Backtest (2024-09-01 to Present) ===")
    
    # 1. Setup
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    ranker = EtfRanker(fetcher)
    
    candidate_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    if not os.path.exists(candidate_path):
        print("Candidate file not found")
        return
        
    df_candidates = pd.read_excel(candidate_path)
    df_candidates = df_candidates.rename(columns={
        'symbol': 'etf_code',
        'sec_name': 'etf_name',
        '主题': 'theme'
    })
    
    print(f"Candidate Universe: {len(df_candidates)} ETFs")

    # 2. Define Timeline
    start_date_str = "2024-09-01"
    end_date_str = datetime.now().strftime("%Y-%m-%d")
    
    # Get trading calendar (simplified: just use one main ETF's dates)
    # SHSE.510300 is usually liquid enough to represent the calendar
    print("Initializing Calendar...")
    ref_df = fetcher.get_etf_daily_history("SHSE.510300", start_date_str, end_date_str)
    if ref_df is None or ref_df.empty:
        print("Failed to fetch reference calendar")
        return
        
    # Filter dates >= 2024-09-01
    trading_days = ref_df['日期'].dt.strftime('%Y-%m-%d').tolist()
    trading_days = [d for d in trading_days if d >= start_date_str]
    
    print(f"Backtest Period: {len(trading_days)} trading days")
    
    # 3. Main Loop
    portfolio_value = 1.0
    daily_returns = []
    equity_curve = []
    
    # Cache all data first to avoid millions of IO ops?
    # For simplicity, we stick to Ranker logic which fetches on demand.
    # To speed up, we can modify Ranker or just accept it might take 5-10 mins.
    # We will iterate day by day.
    
    prev_top_10 = None
    
    for i, current_date in enumerate(trading_days[:-1]): # Last day has no T+1 return
        next_date = trading_days[i+1]
        
        print(f"Processing {current_date}...")
        
        # Ranker needs to see data UP TO current_date
        # We need to hack the ranker or pass a specific end_date_override?
        # The existing ranker uses:
        # end_date = datetime.now().strftime("%Y-%m-%d") (Hardcoded in select_top_etfs)
        # We MUST Modify EtfRanker to accept a reference date if we want to backtest properly without data leakage!
        # Wait, I didn't verify EtfRanker's date handling in the previous step carefully enough.
        # Let's check EtfRanker code again.
        pass

    # IMPORTANT: I need to modify EtfRanker to accept 'analysis_date' argument.
    # Otherwise it always looks at 'now' and back 730 days.
    # If I run it today for 2024-09-01, it will see 2024-10 data if I don't stop it.
    
    print("Cannot proceed without modifying EtfRanker to accept analysis_date.")

if __name__ == "__main__":
    run_backtest()
