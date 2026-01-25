
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from src.data_fetcher import DataFetcher
from config import config

# Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def run_comparison_scan():
    print("=== Scanning T=1 to T=20: Periodic vs Rolling ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Load Data
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_curated = pd.read_excel(excel_path)
    df_curated.columns = df_curated.columns.str.strip()
    df_curated = df_curated.rename(columns={'symbol': 'etf_code', 'sec_name': 'etf_name', '主题': 'theme'})
    curated_codes = set(df_curated['etf_code'])
    
    all_codes = list(fetcher.get_all_etfs()['etf_code'])
    print("[Data] Loading Prices...")
    price_data = {}
    start_load = "2023-01-01"
    
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except: pass
            
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    daily_rets = prices_df.pct_change(1).shift(-1) 
    
    # 2. Scores
    print("[Score] Calculating Scores...")
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = config.SECTOR_TOP_N_THRESHOLD
    
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        ranks = prices_df.pct_change(period).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    
    # 3. Simulation Settings
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    # Pre-compute Picks
    picks_cache = {}
    print("[Sim] Pre-computing daily picks...")
    for d in valid_dates:
        if d not in total_scores.index:
            picks_cache[d] = []
            continue
        s = total_scores.loc[d]
        r = r20_matrix.loc[d]
        metric = s * 10000 + r
        metric = metric.dropna()
        metric = metric[metric > -99999]
        sorted_codes = metric.sort_values(ascending=False).index
        top10 = [c for c in sorted_codes if c in curated_codes][:10]
        picks_cache[d] = top10

    results = []
    
    print("[Sim] Running Loop T=1..20...")
    for T in range(1, 21):
        # We need to simulate T tranches
        tranche_rets = []
        tranche_curves = []
        
        for k in range(T):
            # Tranche k
            curr_val = 1.0
            curve = [1.0]
            
            # Initial Holdings (Common start)
            curr_holdings = picks_cache.get(valid_dates[0], [])
            
            for i in range(len(valid_dates) - 1):
                date_t = valid_dates[i]
                
                # Rebalance check (Before calculating return)
                # Tranche k rebalances when (i - k) % T == 0 and i >= k
                if (i - k) % T == 0 and i >= k:
                    curr_holdings = picks_cache.get(date_t, [])

                # Daily Return
                if not curr_holdings:
                    r = 0.0
                else:
                    if date_t in daily_rets.index:
                        rr = daily_rets.loc[date_t, curr_holdings]
                        r = rr.mean()
                        if pd.isna(r): r = 0.0
                    else:
                        r = 0.0
                
                curr_val *= (1.0 + r)
                curve.append(curr_val)
                    
            tranche_curves.append(np.array(curve))
            tranche_rets.append((curr_val - 1.0) * 100)
            
        # 1. Periodic Return (Tranche 0)
        # This is the standard "Start on Day 0" strategy
        per_ret = tranche_rets[0]
        
        # 2. Rolling Return (Average of Curves)
        avg_curve = np.sum(tranche_curves, axis=0) / T
        roll_ret = (avg_curve[-1] - 1.0) * 100
        
        # Win Rate (How many tranches beat the Rolling avg? Interesting but optional)
        
        print(f"  T={T:2d}: Periodic={per_ret:>6.1f}% | Rolling={roll_ret:>6.1f}%")
        results.append({
            'T': T, 
            'Periodic': per_ret, 
            'Rolling': roll_ret
        })
        
    # 4. Plotting
    df_res = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 7))
    
    # Plot Periodic
    plt.plot(df_res['T'], df_res['Periodic'], marker='o', label='定期调仓 (Periodic)', linewidth=2, color='#1f77b4')
    
    # Plot Rolling
    plt.plot(df_res['T'], df_res['Rolling'], marker='s', label='滚动持仓 (Rolling)', linewidth=2, color='#ff7f0e')
    
    # Add labels
    for idx, row in df_res.iterrows():
        t_val = row['T']
        p_val = row['Periodic']
        r_val = row['Rolling']
        
        # Label formatting to avoid overlap
        # Only label if T is even or specific points to reduce clutter? 
        # Or just label all if T<=20
        
        offset_p = 3 if p_val > r_val else -10
        offset_r = 3 if r_val > p_val else -10
        
        plt.text(t_val, p_val + offset_p, f"{p_val:.1f}%", ha='center', fontsize=8, color='#1f77b4')
        plt.text(t_val, r_val + offset_r, f"{r_val:.1f}%", ha='center', fontsize=8, color='#ff7f0e')

    plt.title("定期调仓 vs 滚动持仓: 不同周期(T)下的收益对比 (参数已对齐)", fontsize=14, fontweight='bold')
    plt.xlabel("T 值 (持仓天数 / 调仓周期)", fontsize=12)
    plt.ylabel("累计收益率 (%)", fontsize=12)
    plt.xticks(range(1, 21))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=12)
    
    out_path = os.path.join(config.CHART_OUTPUT_DIR, "compare_t_periodic_vs_rolling.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Chart saved to {out_path}")
    
    # Save CSV
    csv_path = os.path.join(config.DATA_OUTPUT_DIR, "compare_t_periodic_vs_rolling.csv")
    df_res.to_csv(csv_path, index=False)

if __name__ == "__main__":
    run_comparison_scan()
