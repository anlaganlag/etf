
import os
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

def run_unified_timing():
    print("=== Unified Execution Timing Test (DEFINITIVE) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Setup Universes
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    
    # Strong List (as used in 149.9% result)
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Build Multi-Column Matrix (Close for scoring, Open for execution test)
    print("\n[Data] Loading Prices...")
    close_data = {}
    open_data = {}
    start_load = "2023-01-01"
    
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty:
                    close_data[code] = df.set_index('日期')['收盘']
                    if '开盘' in df.columns:
                        open_data[code] = df.set_index('日期')['开盘']
            except: pass
            
    close_df = pd.DataFrame(close_data).sort_index().ffill()
    open_df = pd.DataFrame(open_data).sort_index().ffill()
    print(f"  Total All-Market ETFs: {close_df.shape[1]}")
    print(f"  Curated ETFs with Open: {len([c for c in strong_codes if c in open_df.columns])}")

    # 3. Vectorized Scoring (R1..R250)
    print("\n[Score] Calculating Scores...")
    # ENSURE ALL 8 PERIODS ARE USED
    periods = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    
    total_scores = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    for p, pts in periods.items():
        ranks = close_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = close_df.pct_change(20).fillna(-999)
    print("  Scoring Complete.")

    # 4. Simulation (2024-11-01 to Present)
    print("\n[Sim] Running Unified Comparison...")
    # Exact start to match previous high yield
    start_sim = "2024-11-01"
    # Filter dates
    sim_dates = close_df.index[close_df.index >= pd.to_datetime(start_sim)]
    
    T = 20
    min_score = 150
    
    val_close = 1.0
    val_open = 1.0
    
    hold_close = []
    hold_open = []
    
    for i in range(len(sim_dates) - 1):
        dt = sim_dates[i]
        dt_next = sim_dates[i+1]
        
        # A. Rebalance Check (Signal at Close of dt)
        if i % T == 0:
            s = total_scores.loc[dt]
            r = r20_matrix.loc[dt]
            metric = (s * 10000 + r).dropna()
            # Threshold filter
            valid = s[s >= min_score].index
            metric = metric[metric.index.isin(valid)]
            sorted_all = metric.sort_values(ascending=False).index
            new_picks = [c for c in sorted_all if c in strong_codes][:10]
            
            # Close Strategy: executes at dt Close.
            # So for dt -> dt_next return, it owns new picks.
            hold_close = new_picks
            
            # Open Strategy: executes at dt_next Open.
            # So for dt Close -> dt_next Open, it still owns old picks.
            pass
            
        # B. Calculate Gains for dt -> dt_next
        
        # 1. Close Mode (Compounded Daily) - EXACTLY as in test_strong_vs_weak.py
        if hold_close:
            c0 = close_df.loc[dt, hold_close]
            c1 = close_df.loc[dt_next, hold_close]
            # Handle possible NaNs if an ETF stopped trading
            day_ret = ((c1 - c0) / c0).dropna().mean()
            val_close *= (1 + day_ret)
            
        # 2. Open Mode (Corrected)
        # Part i: dt Close -> dt_next Open (Holding OLD)
        # Part ii: dt_next Open -> dt_next Close (Holding NEW)
        if i % T == 0:
            # Rebalance Switch
            ret_1 = 0.0
            if hold_open:
                # Part i: dt Close -> dt_next Open with OLD
                # (Existing holdings from previous period)
                c0 = close_df.loc[dt, hold_open]
                o1 = open_df.loc[dt_next, hold_open]
                ret_1 = ((o1 - c0) / c0).dropna().mean()
            
            # Part ii: dt_next Open -> dt_next Close with NEW
            ret_2 = 0.0
            if new_picks:
                o1 = open_df.loc[dt_next, new_picks]
                c1 = close_df.loc[dt_next, new_picks]
                ret_2 = ((c1 - o1) / o1).dropna().mean()
            
            val_open *= (1 + ret_1) * (1 + ret_2)
            # Update state for next non-rebalance days
            hold_open = new_picks
        else:
            # Normal Day
            if hold_open:
                c0 = close_df.loc[dt, hold_open]
                c1 = close_df.loc[dt_next, hold_open]
                day_ret = ((c1 - c0) / c0).dropna().mean()
                val_open *= (1 + day_ret)
                
    print(f"\nFinal Unified Results (2024-11-01起):")
    print(f"  Close-to-Close Trade (推薦): {(val_close-1)*100:>6.1f}%")
    print(f"  Open-to-Trade (次日開盤)   : {(val_open-1)*100:>6.1f}%")
    print(f"  Execution Alpha            : {((val_close/val_open)-1)*100:>6.2f}%")

if __name__ == "__main__":
    run_unified_timing()
