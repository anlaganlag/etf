
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_fetcher import DataFetcher
from src.etf_ranker import EtfRanker
from config import config

def run_fast_comparison():
    print("=== Starting Optimized Market-Wide Comparison (Vectorized) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    # We need ranker just for theme logic
    ranker = EtfRanker(fetcher)
    
    # A. Curated (Strong)
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_curated = pd.read_excel(excel_path)
    # Strip whitespace from columns
    df_curated.columns = df_curated.columns.str.strip()
    
    # Handle both old and new column names
    rename_map = {
        'symbol': 'etf_code', 
        'sec_name': 'etf_name', 
        '主题': 'theme',
        'name_cleaned': 'theme', 
        '最新成交金额': 'cum_amount',
        'cum_amount': 'cum_amount'
    }
    
    df_curated = df_curated.rename(columns=rename_map)
    curated_codes = set(df_curated['etf_code'])
    
    # A2. Curated (Weak)
    weak_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果-弱.xlsx")
    curated_weak_codes = set()
    if os.path.exists(weak_path):
        is_weak_loaded = True
        df_weak = pd.read_excel(weak_path)
        df_weak.columns = df_weak.columns.str.strip()
        df_weak = df_weak.rename(columns=rename_map)
        curated_weak_codes = set(df_weak['etf_code'])
        print(f"  Loaded Weak List: {len(curated_weak_codes)} ETFs")
    else:
        is_weak_loaded = False
        print("  Weak List not found.")

    # All Market
    df_all = fetcher.get_all_etfs()
    # Pre-compute themes map
    print("  Mapping themes...")
    etf_themes = {}
    for _, row in df_all.iterrows():
        code = row['etf_code']
        # If in curated, use curated theme, else guess
        if code in curated_codes:
            # Find theme in curated df
            res = df_curated[df_curated['etf_code'] == code]
            if not res.empty:
                t = res.iloc[0]['theme']
            else:
                t = ranker.get_theme_normalized(row['etf_name'])
        elif code in curated_weak_codes:
             res = df_weak[df_weak['etf_code'] == code]
             if not res.empty:
                 t = res.iloc[0]['theme']
             else:
                 t = ranker.get_theme_normalized(row['etf_name'])
        else:
            t = ranker.get_theme_normalized(row['etf_name'])
        etf_themes[code] = t
        
    all_codes = list(df_all['etf_code'])
    print(f"  Total ETFs: {len(all_codes)}")
    
    # 2. Build Price Matrix (Date x Ticker)
    print("\n[2/5] Building Price Matrix (Fast I/O)...")
    # We will iterate files and build a list of Series, then concat
    # Only reading '日期' and '收盘'
    
    price_data = {}
    
    start_load = "2022-09-01"
    
    count = 0
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                # Read specific columns for speed
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                # Filter start date
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty:
                    # Set index to Date
                    price_data[code] = df.set_index('日期')['收盘']
            except:
                pass
        count += 1
        if count % 200 == 0: print(f"  Processed {count}/{len(all_codes)}")
            
    print("  Concatenating...")
    prices_df = pd.DataFrame(price_data)
    prices_df = prices_df.sort_index()
    prices_df = prices_df.ffill() # Handle halts with ffill (or leave nan?)
    # If NaN, returns will be NaN or 0. For ranking, we want valid scores.
    
    print(f"  Price Matrix: {prices_df.shape} (Dates x ETFs)")
    
    # 3. Vectorized Scoring
    print("\n[3/5] Calculating Vectorized Scores...")
    # Config: 
    # {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = config.SECTOR_TOP_N_THRESHOLD # 15
    
    # Init Score Matrix with 0
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    
    for period, pts in scores_rule.items():
        # Calculate Returns: (Close_T - Close_{T-period}) / Close_{T-period}
        # pct_change(period)
        period_ret = prices_df.pct_change(period)
        
        # Rank row-wise: method='min' means ties get same low rank? 
        # We want top N. ascending=False.
        # rank 1 is highest return.
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        
        # Mask where rank <= threshold
        mask = (ranks <= threshold)
        
        # Add points
        total_scores = total_scores.add(mask * pts, fill_value=0)
    
    # Also need R20 for tie-breaking? Or just use raw score?
    # Ranker uses r20 as secondary sort.
    # We can fetch r20 matrix
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    
    print("  Scoring Complete.")
    
    # 4. Simulation Loop (Vectorized lookup)
    print("\n[4/5] Running Simulation...")
    
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    # Strategies
    # Track equity curves: Series aligned with valid_dates
    # Actually we just list of values
    curves = {
        'Curated_Strong': [1.0],
        'Curated_Weak': [1.0] if is_weak_loaded else [],
        'All_Limit1': [1.0],
        'All_NoLimit': [1.0],
        'CSI300': [1.0],
        'ChiNext': [1.0]
    }
    
    # Benchmark Data
    bm_codes = {'CSI300': 'SHSE.510300', 'ChiNext': 'SZSE.159915'}
    
    # Pre-calculating logic
    # We iterate T, T+1
    
    for i in range(len(valid_dates) - 1):
        date_t = valid_dates[i]
        date_next = valid_dates[i+1]
        
        # 1. Get Scores for Date T 
        if date_t not in total_scores.index: continue
        
        # Series of scores
        day_scores = total_scores.loc[date_t]
        # Series of r20 (secondary)
        day_r20 = r20_matrix.loc[date_t]
        
        # Combine to sort: score * 10000 + r20 (simple hack)
        # Assuming r20 usually < 100
        sort_metric = day_scores * 10000 + day_r20
        
        # Drop NaNs (Etfs not existing yet)
        sort_metric = sort_metric.dropna()
        sort_metric = sort_metric[sort_metric > -99999] # Filter dust
        
        # Sort descending
        sorted_codes = sort_metric.sort_values(ascending=False).index.tolist()
        
        # --- Strategy Selection ---
        
        # A. Curated (Strong)
        # Filter sorted_codes to only those in curated_codes
        curated_candidates = [c for c in sorted_codes if c in curated_codes]
        picks_curated = curated_candidates[:10]
        
        # A2. Curated (Weak)
        picks_weak = []
        if is_weak_loaded:
            weak_candidates = [c for c in sorted_codes if c in curated_weak_codes]
            picks_weak = weak_candidates[:10]
        
        # B. All No Limit
        picks_no_limit = sorted_codes[:10]
        
        # C. All Limit 1
        picks_limit = []
        seen_themes = set()
        for c in sorted_codes:
            if len(picks_limit) >= 10: break
            t = etf_themes.get(c, "Unknown")
            if t not in seen_themes:
                picks_limit.append(c)
                seen_themes.add(t)
                
        # --- Calculate Returns (Close T to Close Next) ---
        # (Price_Next - Price_T) / Price_T
        # We can look up in prices_df directly
        
        def calc_basket_ret(codes):
            if not codes: return 0.0
            p_t = prices_df.loc[date_t, codes]
            p_next = prices_df.loc[date_next, codes]
            
            # Returns
            # Handle zeros or nans
            rets = (p_next - p_t) / p_t
            return rets.mean() # Equal weight
        
        ret_curated = calc_basket_ret(picks_curated)
        ret_weak = calc_basket_ret(picks_weak)
        ret_no_limit = calc_basket_ret(picks_no_limit)
        ret_limit = calc_basket_ret(picks_limit)
        
        # Benchmarks
        def calc_bm_ret(name):
            code = bm_codes[name]
            if code not in prices_df.columns: return 0.0
            p0 = prices_df.at[date_t, code]
            p1 = prices_df.at[date_next, code]
            if pd.isna(p0) or pd.isna(p1) or p0 == 0: return 0.0
            return (p1 - p0) / p0
            
        ret_csi = calc_bm_ret('CSI300')
        ret_chi = calc_bm_ret('ChiNext')
        
        # Update Curves
        # Handle NaN returns
        def update(key, r):
            if key not in curves: return
            if pd.isna(r): r = 0.0
            curves[key].append(curves[key][-1] * (1 + r))
            
        update('Curated_Strong', ret_curated)
        update('Curated_Weak', ret_weak)
        update('All_Limit1', ret_limit)
        update('All_NoLimit', ret_no_limit)
        update('CSI300', ret_csi)
        update('ChiNext', ret_chi)
        
        if i % 50 == 0:
            print(f"  Simulated {date_t.date()}...")

    # 5. Output
    print("\n[5/5] Generating Report...")
    
    results = []
    
    def get_stats(vals):
        arr = np.array(vals)
        ret = (arr[-1] - 1.0) * 100
        peak = np.maximum.accumulate(arr)
        dd = np.min((arr - peak)/peak) * 100
        return ret, dd

    for name, vals in curves.items():
        arr = np.array(vals)
        ret_pct = (arr[-1] - 1.0) * 100
        peak = np.maximum.accumulate(arr)
        dd = np.min((arr - peak)/peak) * 100
        results.append({'Strategy': name, 'Return': ret_pct, 'MaxDD': dd})
        print(f"{name:<15} | {ret_pct:>6.2f}% | {dd:>6.2f}%")
        
    # Save CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(config.DATA_OUTPUT_DIR, "market_comparison_fast.csv"), index=False)
    
    # Save Plot
    plt.figure(figsize=(12, 6))
    
    # Adjust length: dates has N, curves has N+1 (init). Use last N points of curve?
    # Valid dates: [D0, D1, D2]. curve indices: 0(D0-pre), 1(D0-post/D1-pre), 2(D1-post/D2-post)
    # The curve[i] is value at END of date[i-1]?
    # Actually curve[0]=1.0 (Start of Day 0).
    # curve[1] = curve[0] * (1+ret0). ret0 is Day0->Day1 return. So curve[1] is Value approx at Day 1.
    # Curves correspond roughly to [Start, Day1, Day2... DayN].
    # Plot dates should be [Start, D1... DN].
    # For simplicity, we plot dates vs curves[1:] (alignment might be off by 1 day but trend is ok)
    
    plot_dates = valid_dates
    
    for name, vals in curves.items():
        if len(vals) <= 1: continue
        # use vals[1:] to match valid_dates length (337 vs 338?)
        # Let's just slice to min length
        min_len = min(len(vals)-1, len(plot_dates))
        y_vals = vals[1:min_len+1]
        x_vals = plot_dates[:min_len]
        
        ret, _ = get_stats(vals)
        plt.plot(x_vals, y_vals, label=f"{name} ({ret:.1f}%)")
        
    plt.title("Backtest Comparison: Curated vs All-Market (2024-09-01 ~ Now)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(config.CHART_OUTPUT_DIR, "market_comparison_fast.svg"))
    print(f"Chart saved to {config.CHART_OUTPUT_DIR}")

if __name__ == "__main__":
    run_fast_comparison()
