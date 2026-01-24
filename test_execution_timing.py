
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_fetcher import DataFetcher
from config import config

def run_timing_comparison():
    print("=== Execution Timing Comparison (Close-Trade vs Open-Trade) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # Load Strong List
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    # 1. Load Data (Prices Matrix)
    print("[1/3] Loading OHLC Data Matrix...")
    price_close = {}
    price_open = {}
    
    start_load = "2023-01-01"
    
    for code in strong_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty and '开盘' in df.columns:
                    price_close[code] = df.set_index('日期')['收盘']
                    price_open[code] = df.set_index('日期')['开盘']
            except:
                pass
                
    close_df = pd.DataFrame(price_close).sort_index().ffill()
    open_df = pd.DataFrame(price_open).sort_index().ffill()
    print(f"  Matrix Size: {close_df.shape}")

    # 2. Scoring (Based on Close prices)
    print("\n[2/3] Calculating Scores (Global Ranking Logic)...")
    # Note: We need all market scoring to be true to 'Global Strength'.
    # But for timing comparison, if we just want to see the DIFF, 
    # we can use the pre-calculated scores if available, or just re-calc for the subset.
    # Actually, to be accurate, we must rank against ALL.
    # I will assume the SCORES are generated based on CLOSE prices (standard practice).
    
    # Fetch All Market Close prices for accurate ranking
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])
    all_close = {}
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty:
                    all_close[code] = df.set_index('日期')['收盘']
            except: pass
    all_close_df = pd.DataFrame(all_close).sort_index().ffill()
    
    # Re-calc Global Scores
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = 15
    total_scores = pd.DataFrame(0.0, index=all_close_df.index, columns=all_close_df.columns)
    for p, pts in scores_rule.items():
        ranks = all_close_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = all_close_df.pct_change(20).fillna(-999)
    print("  Scoring Complete.")

    # 3. Simulation
    print("\n[3/3] Running Simulations...")
    start_sim = "2024-11-01"
    valid_dates = close_df.index[close_df.index >= pd.to_datetime(start_sim)]
    T = 20
    min_score = 150
    
    curves = {
        'Close_Trade': [1.0],
        'Open_Trade': [1.0]
    }
    
    # Strategy State
    curr_holdings = []
    
    # SIMULATION WITH DAILY COMPOUNDING (To align with +149.9% result)
    val_close = 1.0
    val_open = 1.0
    
    holdings_close = []
    holdings_open = [] # This is what we are holding during the day (Open mode holds signal T at T+1)
    
    # We need to track which holdings generate returns for 'today' (i -> i+1)
    for i in range(len(valid_dates) - 1):
        date_t = valid_dates[i]
        date_next = valid_dates[i+1]
        
        # 1. Update Signals
        if i % T == 0:
            s = total_scores.loc[date_t]
            metric = (s * 10000 + r20_matrix.loc[date_t]).dropna()
            valid = s[s >= min_score].index
            metric = metric[metric.index.isin(valid)]
            new_picks = [c for c in metric.sort_values(ascending=False).index if c in strong_codes][:10]
            
            # Close Strategy: Swaps EXACTLY at Close T. 
            # So for i -> i+1, it holds NEW PICKS.
            holdings_close = new_picks
            
            # Open Strategy: Swaps at Open T+1.
            # So for Close(T) -> Open(T+1), it holds OLD PICKS (or cash if start).
            # From Open(T+1) -> Close(T+1), it holds NEW PICKS.
            pass
        
        # 2. Calculate Daily Returns
        
        # A. Close Strategy Return (Close_i -> Close_{i+1})
        if holdings_close:
            r_c = ((close_df.loc[date_next, holdings_close] - close_df.loc[date_t, holdings_close]) / close_df.loc[date_t, holdings_close]).mean()
            val_close *= (1 + r_c)
            
        # B. Open Strategy Return (This is the tricky one)
        # It's a rebalance day today (i % T == 0)?
        if i % T == 0:
            # We are switching from holdings_open (old) to new_picks.
            # Part 1: Close_i to Open_{i+1} using OLD holdings
            # (If i=0, holdings_open is empty)
            ret_overnight = 0.0
            if holdings_open:
                p_c0 = close_df.loc[date_t, holdings_open]
                p_o1 = open_df.loc[date_next, holdings_open]
                ret_overnight = ((p_o1 - p_c0) / p_c0).mean()
            
            # Part 2: Open_{i+1} to Close_{i+1} using NEW holdings
            ret_intraday = 0.0
            if new_picks:
                p_o1 = open_df.loc[date_next, new_picks]
                p_c1 = close_df.loc[date_next, new_picks]
                ret_intraday = ((p_c1 - p_o1) / p_o1).mean()
            
            # Combined return for Day i -> i+1
            # (1+r) = (Open/Close_old) * (Close/Open_new)
            val_open *= (1 + ret_overnight) * (1 + ret_intraday)
            
            # Record what we hold for subsequent "normal" days
            holdings_open = new_picks
        else:
            # Normal Day (NOT rebalance day)
            # We hold holdings_open from Close_i to Close_{i+1}
            if holdings_open:
                r_o = ((close_df.loc[date_next, holdings_open] - close_df.loc[date_t, holdings_open]) / close_df.loc[date_t, holdings_open]).mean()
                val_open *= (1 + r_o)
                
    print(f"\nFinal Aligned Results (Post-924, 11-01 to Present):")
    print(f"  Close-to-Close Trade (推薦): {(val_close-1)*100:>6.1f}%")
    print(f"  Open-to-Trade (次日開盤)   : {(val_open-1)*100:>6.1f}%")
    print(f"  Execution Alpha            : {((val_close/val_open)-1)*100:>6.2f}%")

    
    print(f"\nResults for Period {start_sim} to Now:")
    print(f"  Close-to-Close Trade: {(val_close-1)*100:>6.1f}%")
    print(f"  Open-to-Open Trade  : {(val_open-1)*100:>6.1f}%")
    print(f"  Execution Alpha     : {((val_close/val_open)-1)*100:>6.2f}% (Close over Open)")

if __name__ == "__main__":
    run_timing_comparison()
