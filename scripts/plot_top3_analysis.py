
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_fetcher import DataFetcher
from config import config

def run_top3_analysis():
    print("=== Top 3 Holdings Analysis (T1-T20) vs ChiNext ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Load Universe
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_curated = pd.read_excel(excel_path)
    df_curated.columns = df_curated.columns.str.strip()
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    df_curated = df_curated.rename(columns=rename_map)
    curated_codes = set(df_curated['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Build Price Matrix & Load ChiNext
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
    
    # ChiNext
    chinext_code = 'SZSE.159915'
    chinext_df = prices_df[chinext_code] if chinext_code in prices_df.columns else None

    # 3. Scoring
    scores_rule = config.SECTOR_PERIOD_SCORES
    threshold = 15
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for period, pts in scores_rule.items():
        period_ret = prices_df.pct_change(period)
        ranks = period_ret.rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)
    
    # 4. Simulation Config
    start_sim = "2024-09-01"
    valid_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    # Calculate ChiNext Return
    chinext_ret = 0.0
    chinext_curve = []
    if chinext_df is not None:
        sub_bm = chinext_df[chinext_df.index >= pd.to_datetime(start_sim)]
        if not sub_bm.empty:
            chinext_ret = (sub_bm.iloc[-1] / sub_bm.iloc[0] - 1) * 100
            # Normalized curve
            chinext_curve = sub_bm / sub_bm.iloc[0]
            
    print(f"Benchmark (ChiNext): +{chinext_ret:.1f}%")
    
    # 5. Scanning T1-T20
    scan_results = [] # {T, Return}
    curves = {} # {T: Series}
    
    selected_periods = [1, 7, 14, 20] # For equity curve plot
    
    for T in range(1, 21):
        portfolio_value = 1.0
        portfolio_values = [1.0]
        current_holdings = []
        
        for i in range(len(valid_dates) - 1):
            date_t = valid_dates[i]
            
            if i % T == 0:
                if date_t in total_scores.index:
                    s = total_scores.loc[date_t]
                    r = r20_matrix.loc[date_t]
                    metric = s * 10000 + r
                    metric = metric.dropna()
                    
                    # No filter, just sorting
                    sorted_codes = metric.sort_values(ascending=False).index.tolist()
                    candidates = [c for c in sorted_codes if c in curated_codes]
                    current_holdings = candidates[:3] # TOP 3 ONLY
            
            day_ret = 0.0
            if current_holdings:
                if date_t in daily_rets.index:
                    day_ret = daily_rets.loc[date_t, current_holdings].mean()
                    if pd.isna(day_ret): day_ret = 0.0
            
            portfolio_value *= (1 + day_ret)
            portfolio_values.append(portfolio_value)
            
        arr = np.array(portfolio_values)
        total_ret = (arr[-1] - 1.0) * 100
        
        scan_results.append({'T': T, 'Return': total_ret})
        
        if T in selected_periods:
            curves[T] = pd.Series(portfolio_values, index=valid_dates)
            
    # 6. Plot 1: Bar Chart (Parameter Sensitivity)
    df_res = pd.DataFrame(scan_results)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_res['T'], df_res['Return'], color='skyblue', label='Strategy (Top 3)')
    
    # Highlight T14
    bars[13].set_color('red') # T=14 (index 13)
    
    plt.axhline(y=chinext_ret, color='gray', linestyle='--', linewidth=2, label=f'ChiNext (+{chinext_ret:.0f}%)')
    
    plt.xlabel('Holding Period (Days)')
    plt.ylabel('Total Return (%)')
    plt.title('Top 3 Holdings: T1-T20 Performance vs Benchmark')
    plt.xticks(df_res['T'])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    out_path1 = os.path.join(config.CHART_OUTPUT_DIR, "top3_t_scan.png")
    plt.savefig(out_path1)
    print(f"Saved param scan to {out_path1}")
    
    # 7. Plot 2: Equity Curves
    plt.figure(figsize=(12, 6))
    
    # Plot Benchmark
    if not chinext_curve.empty:
        plt.plot(chinext_curve.index, chinext_curve.values, color='gray', linestyle='--', label='ChiNext', alpha=0.6)
        
    # Plot Strategies
    colors = {1: 'orange', 7: 'green', 14: 'red', 20: 'blue'}
    for T in selected_periods:
        curve = curves[T]
        final_val = (curve.iloc[-1] - 1) * 100
        label = f'T={T} (+{final_val:.0f}%)'
        alpha = 1.0 if T == 14 else 0.6
        width = 2.5 if T == 14 else 1.5
        plt.plot(curve.index, curve.values, label=label, color=colors.get(T, 'black'), alpha=alpha, linewidth=width)
        
    plt.title('Equity Curves: Top 3 Holdings (Different Periods)')
    plt.ylabel('Normalized Equity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path2 = os.path.join(config.CHART_OUTPUT_DIR, "top3_equity_curves.png")
    plt.savefig(out_path2)
    print(f"Saved equity curves to {out_path2}")

if __name__ == "__main__":
    run_top3_analysis()
