
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.etf_ranker import EtfRanker
from src.data_fetcher import DataFetcher
from config import config

def run_comparison():
    print("=== Starting Market-Wide Comparison (2024-09-01 to Present) ===")
    
    # 1. Setup
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    ranker = EtfRanker(fetcher)
    
    # 2. Prepare Universes
    print("\n[1/3] Preparing ETF Universes...")
    
    # A. Curated (Limit=1 implicit)
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_curated = pd.read_excel(excel_path).rename(columns={
        'symbol': 'etf_code', 'sec_name': 'etf_name', '主题': 'theme'
    })
    
    # B. All Market (Needs theme deduction)
    df_all = fetcher.get_all_etfs()
    # Add theme column using ranker's logic
    print(f"Annotating themes for {len(df_all)} ETFs...")
    df_all['theme'] = df_all['etf_name'].apply(ranker.get_theme_normalized)
    
    # 3. Cache Data
    # Assuming update_2year_data.py has run, we rely on cache.
    # We load cache into memory for speed (1360 files ~ small enough)
    
    data_cache = {}
    print("\n[2/3] Loading Data Cache into Memory...")
    # Load for ALL unique codes in both lists
    all_codes = set(df_curated['etf_code']).union(set(df_all['etf_code']))
    
    start_history_range = "2022-09-01"
    end_history_range = datetime.now().strftime("%Y-%m-%d")
    
    count = 0
    for code in all_codes:
        # Try direct file read to skip API overhead if possible, but fetcher handles it
        # We assume cache is populated.
        # We use a trick: directly read CSV if exists to speed up 1700 reads
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file)
                df['日期'] = pd.to_datetime(df['日期'])
                # Filter range in memory
                mask = (df['日期'] >= pd.to_datetime(start_history_range)) 
                data_cache[code] = df[mask].copy()
                count += 1
            except:
                pass
        
        if count % 200 == 0 and count > 0:
            print(f"Loaded {count} / {len(all_codes)}...")
            
    print(f"Loaded {len(data_cache)} history files.")

    # 4. Benchmarks
    benchmarks = {'CSI300': 'SHSE.510300', 'ChiNext': 'SZSE.159915'}
    bm_data = {}
    for k, v in benchmarks.items():
        if v in data_cache:
            bm_data[k] = data_cache[v]
        else:
            # Fallback fetch
            df = fetcher.get_etf_daily_history(v, start_history_range, end_history_range)
            if df is not None: bm_data[k] = df
    
    if 'CSI300' not in bm_data:
        print("Missing Benchmark Data. Aborting.")
        return

    # 5. Calendar
    start_date = "2024-09-01"
    ref_df = bm_data['CSI300'].set_index('日期')
    trading_days = ref_df.index[ref_df.index >= pd.to_datetime(start_date)].strftime('%Y-%m-%d').tolist()
    print(f"Backtest Days: {len(trading_days)}")

    # 6. Strategies
    strategies = {
        'Curated': {'universe': df_curated, 'limit_sector': True},
        'All_Limit1': {'universe': df_all, 'limit_sector': True},
        'All_NoLimit': {'universe': df_all, 'limit_sector': False}
    }
    
    # Store equity curves: name -> [1.0, ...]
    curves = {name: [1.0] for name in strategies}
    curves.update({name: [1.0] for name in benchmarks})
    
    # To properly implement 'limit_sector' for All market, we need to handle it in the ranker call
    # or post-process the ranker output. The current etf_ranker.select_top_etfs DOES NOT support limit.
    # We must implement the limit logic HERE or modify ranker.
    # For flexibility, let's implement the selection logic here using scores from ranker.
    # Wait, ranker.select_top_etfs returns Top 10 directly.
    # It sorts by score.
    # The 'All_Limit1' needs to filter duplicates BEFORE picking top 10.
    # The 'All_NoLimit' just picks top 10.
    
    # Modify Ranker to return ALL scored results if top_n is large?
    # Or better: We instantiate a special 'score_only' mode or just ask for top 9999.
    
    print("\n[3/3] Running Simulation Loop...")
    
    for i, current_date in enumerate(trading_days[:-1]):
        next_date = trading_days[i+1]
        
        if (i+1) % 10 == 0:
            print(f"Processing {current_date}...")
            
        # Optimization: Score ALL candidates once per day?
        # No, Curated vs All have different universes.
        # But Curated is a subset of All. 
        # For simplicity, calculate separately.
        
        for strat_name, strat_conf in strategies.items():
            univ = strat_conf['universe']
            limit = strat_conf['limit_sector']
            
            # Step 1: Score Universe
            # We ask for a somewhat large N to allow filtering
            scored_df = ranker.select_top_etfs(
                univ, top_n=200, reference_date=current_date, history_cache=data_cache
            )
            
            if scored_df.empty:
                # No holdings, return 0
                avg_ret = 0.0
            else:
                # Step 2: Apply Logic
                if limit:
                    # Filter: Take best per theme
                    # Assuming sorted by total_score desc
                    # drop_duplicates keeps first (highest score)
                    filtered = scored_df.drop_duplicates(subset=['theme'], keep='first')
                    selection = filtered.head(10)
                else:
                    selection = scored_df.head(10)
                
                codes = selection['etf_code'].tolist()
                
                # Step 3: Calculate Return
                rets = []
                for code in codes:
                    if code not in data_cache: continue
                    hist = data_cache[code]
                    
                    sub = hist[(hist['日期'] >= pd.to_datetime(current_date)) & (hist['日期'] <= pd.to_datetime(next_date))]
                    if len(sub) < 2: 
                        rets.append(0.0)
                        continue
                    
                    p0 = float(sub.iloc[0]['收盘'])
                    p1 = float(sub.iloc[-1]['收盘'])
                    if p0 > 0:
                        rets.append((p1 - p0)/p0)
                    else:
                        rets.append(0.0)
                
                if rets:
                    avg_ret = sum(rets) / len(rets)
                else:
                    avg_ret = 0.0
            
            # Update Curve
            curves[strat_name].append(curves[strat_name][-1] * (1 + avg_ret))
            
        # Update Benchmarks
        for bm_name, bm_df in bm_data.items():
             sub = bm_df[(bm_df['日期'] >= pd.to_datetime(current_date)) & (bm_df['日期'] <= pd.to_datetime(next_date))]
             if len(sub) >= 2:
                 p0 = float(sub.iloc[0]['收盘'])
                 p1 = float(sub.iloc[-1]['收盘'])
                 ret = (p1 - p0)/p0
             else:
                 ret = 0.0
             curves[bm_name].append(curves[bm_name][-1] * (1 + ret))

    # 7. Analysis & Output
    results = []
    dates = pd.to_datetime(trading_days)
    
    def get_stats(vals):
        arr = np.array(vals)
        ret = (arr[-1] - 1.0) * 100
        peak = np.maximum.accumulate(arr)
        dd = np.min((arr - peak)/peak) * 100
        return ret, dd

    print("\n=== Market-Wide Comparison Results ===")
    print(f"{'Strategy':<20} | Return | MaxDD")
    print("-" * 45)
    
    for name, vals in curves.items():
        ret, dd = get_stats(vals)
        print(f"{name:<20} | {ret:>6.1f}% | {dd:>6.1f}%")
        results.append({'Name': name, 'Return': ret, 'MaxDD': dd, 'FinalValue': vals[-1]})
        
    # Plot
    plt.figure(figsize=(12, 6))
    for name, vals in curves.items():
        # Adjust length: curves has N+1 points (init 1.0), dates has N points
        # Actually daily loop runs N-1 times (T->T+1).
        # Trading days: [D0, D1, D2]. Loop: D0->D1, D1->D2. Returns: 2. Curve: 3 points (Init, V1, V2).
        # We should plot against [D0, D1, D2].
        plt.plot(dates, vals, label=f"{name} ({get_stats(vals)[0]:.1f}%)")
        
    plt.title("Constraint Impact Analysis: Curated vs All-Market")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(os.path.join(config.CHART_OUTPUT_DIR, "market_comparison.svg"))
    
    # Save CSV
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(config.DATA_OUTPUT_DIR, "market_comparison_stats.csv"), index=False)
    print(f"\nSaved stats to {config.DATA_OUTPUT_DIR}")

if __name__ == "__main__":
    run_comparison()
