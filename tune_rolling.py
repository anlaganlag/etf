
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_fetcher import DataFetcher
from src.etf_ranker import EtfRanker
from config import config

def run_rolling_scan():
    print("=== Rolling Rebalance Scan (T=1..20) ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Load Universe
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_curated = pd.read_excel(excel_path)
    df_curated.columns = df_curated.columns.str.strip()
    rename_map = {
        'symbol': 'etf_code', 'sec_name': 'etf_name', 
        'name_cleaned': 'theme', '主题': 'theme'
    }
    df_curated = df_curated.rename(columns=rename_map)
    curated_codes = set(df_curated['etf_code'])
    
    # All Market for Global Ranking
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Build Price Matrix
    print("\n[Data] Building Price Matrix...")
    price_data = {}
    start_load = "2022-09-01"
    
    count = 0
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except:
                pass
        count += 1
        if count % 500 == 0: print(f"  Processed {count}...")
            
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    print(f"  Matrix: {prices_df.shape}")
    
    # 3. Vectorized Scoring
    print("\n[Score] Calculating Scores...")
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = config.SECTOR_TOP_N_THRESHOLD
    
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        period_ret = prices_df.pct_change(period)
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    
    # Pre-calc daily returns for fast simulation
    daily_rets_matrix = prices_df.pct_change(1).shift(-1) # T->T+1 return at index T
    
    print("  Scoring Complete.")
    
    # 4. Rolling Scan Loop
    print("\n[Sim] Scanning Rolling Windows (1..20)...")
    
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    results = [] # {Period, Return, MaxDD}
    
    # Helper to pick Top 10
    def pick_top10(date_t):
        if date_t not in total_scores.index: return []
        s = total_scores.loc[date_t]
        r = r20_matrix.loc[date_t]
        metric = s * 10000 + r
        # Fast filter
        metric = metric.dropna()
        metric = metric[metric > -99999]
        sorted_codes = metric.sort_values(ascending=False).index.tolist()
        
        # Filter Curated
        candidates = [c for c in sorted_codes if c in curated_codes]
        return candidates[:10]

    # Pre-compute ALL daily picks to save time?
    # No, picking 10 out of 1700 is fast enough for 300 days.
    # But we run T=1..20 simulations.
    # Actually, the "Pick" on Day D is deterministic. 
    # Let's pre-compute the "Daily Pick Portfolio Return" for every day D.
    # On Day D, if we rebalanced, what would be the return for D->D+1?
    
    print("  Pre-computing daily hypothetical strategy returns...")
    strategy_daily_rets = pd.Series(0.0, index=valid_dates[:-1])
    
    for i in range(len(valid_dates) - 1):
        date_t = valid_dates[i]
        codes = pick_top10(date_t)
        if not codes:
            ret = 0.0
        else:
            # Get mean return of these codes
            if date_t in daily_rets_matrix.index:
                rets = daily_rets_matrix.loc[date_t, codes]
                ret = rets.mean()
                if pd.isna(ret): ret = 0.0
            else:
                ret = 0.0
        strategy_daily_rets.iloc[i] = ret
        
    # Now we simulate Rolling Logic
    # For a rolling period T:
    # We have T sub-portfolios.
    # Sub-portfolio 'k' (where k in 0..T-1) updates its holdings dates: 0, T, 2T... + k
    # Its return on day 't' is determined by the holdings picked on the last rebalance date <= t.
    
    # Wait, the prompt says "Sell oldest, buy newest".
    # This effectively means:
    # On Day t, we hold T baskets selected on days t-1, t-2, ..., t-T.
    # (Assuming 1 day holding period for valid return).
    # Wait, if we hold for T days.
    # Basket bought on t-T is sold on t.
    # Basket bought on t-1 is held until t+T-1.
    # So on Day t (investment period t -> t+1), we hold funds invested on t, t-1, ... t-(T-1).
    # The return for day t is Average( Return of Basket(t), Return of Basket(t-1), ... )?
    # YES. Because we hold T equal portions.
    
    # BUT! Basket(t-k) generates return on day t based on the *daily* move of assets picked at t-k.
    # The assets picked at t-k might vary.
    # So we strictly need to track:
    # R_rolling(t) = 1/T * Sum( Return of Assets_Picked_At(t-k) for k in 0..T-1 )
    
    # This requires looking up "Assets Picked At X" and calculating their return on Day T.
    # My pre-computed `strategy_daily_rets` assumes we picked at T and hold for 1 day.
    # It DOES NOT tell me return of assets picked at T-5 held on Day T.
    
    # Optimization: We need efficient lookup of "Basket Return on Day D".
    # For T=20, this is heavy (20 lookups per day).
    # Total ops = 300 days * 20 periods * ~10 lookups. OK.
    
    # Let's cache the "Picks" for each day.
    picks_cache = {} # date -> list of codes
    for i in range(len(valid_dates) - 1):
        picks_cache[valid_dates[i]] = pick_top10(valid_dates[i])
        
    print("  Simulation Start...")
    
    for T in range(1, 21):
        # Calculate Rolling Equity Curve
        val = 1.0
        vals = [1.0]
        
        # Start from day T to allow full distinct portfolios? 
        # Or build up? Prompt says "T days ramp up".
        # We start simulation at index 0.
        # But we only have picked history starting at index 0.
        # So for first T-1 days, we are partially invested?
        # Let's assume on Day 0 we buy 1/Tth. Day 1 buy 1/Tth.
        # until Day T-1 we are full.
        
        # To simplify: We track the sum of T sub-strategies.
        # Sub-strategy k starts at day k.
        sub_strategies = [1.0] * T # Value of each sub-bucket (initially 1/T of total capital? No, let's track separately normalized)
        
        # Portfolio Value = Sum(sub_strategies) / T.
        
        agg_values = []
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i] # Today, we decide/hold
            
            # For each sub-strategy j (0..T-1)
            # Determine if it rebalances today.
            # It rebalances if (i - j) % T == 0.
            # If rebalance: holdings = picks_cache[date_t]
            # Else: holdings = holdings from previous
            
            # This requires state tracking for T portfolios.
            pass 
            
        # Refined Logic:
        # Instead of complex loop, just run T sub-simulations and sum them.
        # Sub-sim k: rebalances on day i where i%T == k.
        
        combined_curve = np.zeros(len(valid_dates))
        
        for k in range(T):
            # Run a single fixed-period simulation with offset k
            # Initial rebalance at day k
            sub_val = 1.0
            sub_vals = [1.0] # Length N
            
            curr_holdings = []
            if k == 0: # Rebalance on day 0
                curr_holdings = picks_cache.get(valid_dates[0], [])
            
            for i in range(len(valid_dates) - 1):
                date_t = valid_dates[i]
                
                # Check rebalance for NEXT period (i -> i+1)
                # Logic: Rebalance at close of day i if (i - k) % T == 0?
                # Case k=0: rebal at 0, T, 2T...
                # current holdings determine return for i->i+1
                
                # Calculate return first (based on prev holdings)
                if not curr_holdings:
                    day_ret = 0.0
                else:
                    # Look up return of curr_holdings at date_t
                    if date_t in daily_rets_matrix.index:
                        day_ret = daily_rets_matrix.loc[date_t, curr_holdings].mean()
                        if pd.isna(day_ret): day_ret = 0.0
                    else:
                        day_ret = 0.0
                
                sub_val *= (1 + day_ret)
                sub_vals.append(sub_val)
                
                # Update holdings for NEXT day
                # We rebalance at CLOSE of day i if (i - k) % T == 0?
                # Actually if we rebalance at 0, we hold 0->1, 1->2...
                # Next rebalance is at T.
                # So we update holdings if (i + 1 - k) % T == 0? 
                
                # Let's align:
                # k=0: Rebalance at i=0. Holds for 1..T. Rebalance at i=T.
                if (i + 1 - k) % T == 0:
                     # Rebalance at Close of Day i
                     # Pick new for tomorrow
                     # Valid since we have price data up to i
                     curr_holdings = picks_cache.get(valid_dates[i+1], []) if (i+1) < len(valid_dates) else []
            
            # Pad sub_vals to ensure correct length?
            # Iterated len(valid_dates)-1 times. sub_vals has N entries.
            # Add to combined
            combined_curve += np.array(sub_vals)
            
        # Average
        avg_curve = combined_curve / T
        
        # Stats
        total_ret = (avg_curve[-1] - 1.0) * 100
        peak = np.maximum.accumulate(avg_curve)
        dd = np.min((avg_curve - peak)/peak) * 100
        
        print(f"  Rolling T={T:<2}: Return={total_ret:>6.1f}%, MaxDD={dd:>6.1f}%")
        results.append({'Period': T, 'Return': total_ret, 'MaxDD': dd})

    # Save Results
    df_res = pd.DataFrame(results)
    df_res.to_csv(os.path.join(config.DATA_OUTPUT_DIR, "tuning_rolling.csv"), index=False)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(df_res['Period'], df_res['Return'], 'g-o', label='Rolling Return')
    plt.xlabel('Rolling Period T (Days)')
    plt.ylabel('Total Return (%)')
    plt.title('Performance of Rolling Rebalance Strategy')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(config.CHART_OUTPUT_DIR, "tuning_rolling.svg"))

if __name__ == "__main__":
    run_rolling_scan()
