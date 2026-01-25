
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.data_fetcher import DataFetcher
from config import config

# Chinese font support for Windows
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def generate_t14_visuals():
    print("=== Generating Definitive T=14 Strategy Visuals ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Setup Universes
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong.columns = df_strong.columns.str.strip()
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Build Multi-Column Matrix
    print("[Data] Loading Prices...")
    close_data = {}
    start_load = "2023-01-01"
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[df['日期'] >= pd.to_datetime(start_load)]
                if not df.empty:
                    close_data[code] = df.set_index('日期')['收盘']
            except: pass
            
    close_df = pd.DataFrame(close_data).sort_index().ffill()
    
    # ChiNext Benchmark
    chinext_code = 'SZSE.159915'
    chinext_prices = close_df[chinext_code] if chinext_code in close_df.columns else None

    # 3. Vectorized Scoring
    print("[Score] Calculating Global Scores...")
    periods = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    threshold = 15
    total_scores = pd.DataFrame(0.0, index=close_df.index, columns=close_df.columns)
    for p, pts in periods.items():
        ranks = close_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= threshold) * pts, fill_value=0)
    
    r20_matrix = close_df.pct_change(20).fillna(-999)

    # 4. Simulation (T=14, 2024-11-01 to Present)
    print("[Sim] Running T=14 Simulation...")
    start_sim = "2024-11-01"
    sim_dates = close_df.index[close_df.index >= pd.to_datetime(start_sim)]
    
    T = 14
    min_score = 150
    
    val = 1.0
    vals = [1.0]
    holdings = []
    
    segment_returns = []
    current_segment_start_val = 1.0
    
    for i in range(len(sim_dates) - 1):
        dt = sim_dates[i]
        dt_next = sim_dates[i+1]
        
        # Rebalance check
        if i % T == 0:
            # End of previous segment
            if i > 0:
                seg_ret = (val / current_segment_start_val - 1) * 100
                segment_returns.append(seg_ret)
                current_segment_start_val = val
                
            s = total_scores.loc[dt]
            r = r20_matrix.loc[dt]
            metric = (s * 10000 + r).dropna()
            valid = s[s >= min_score].index
            metric = metric[metric.index.isin(valid)]
            sorted_all = metric.sort_values(ascending=False).index
            holdings = [c for c in sorted_all if c in strong_codes][:10]
            
        # Daily return
        if not holdings:
            day_ret = 0.0
        else:
            c0 = close_df.loc[dt, holdings]
            c1 = close_df.loc[dt_next, holdings]
            day_ret = ((c1 - c0) / c0).dropna().mean()
            if pd.isna(day_ret): day_ret = 0.0
            
        val *= (1 + day_ret)
        vals.append(val)

    # Final segment
    if val != current_segment_start_val:
        segment_returns.append((val / current_segment_start_val - 1) * 100)

    # 5. Visuals
    print("[Plot] Generating Charts...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [2, 1]})
    
    # Subplot 1: Equity Curve
    strategy_curve = pd.Series(vals, index=sim_dates)
    ax1.plot(strategy_curve.index, strategy_curve.values, color='#e74c3c', linewidth=2, label=f'T=14 策略 (较初值: +{(val-1)*100:.1f}%)')
    
    if chinext_prices is not None:
        bm = chinext_prices.loc[sim_dates]
        bm = bm / bm.iloc[0]
        ax1.plot(bm.index, bm.values, color='#34495e', linestyle='--', alpha=0.7, label=f'创业板指 (基准: +{(bm.iloc[-1]-1)*100:.1f}%)')
    
    ax1.set_title("ETF 动量巅峰策略 (T=14) 净值曲线与回测表现", fontsize=16, fontweight='bold', pad=20)
    ax1.set_ylabel("累积净值 (起点=1.0)", fontsize=12)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    
    # Subplot 2: Segment Returns
    colors = ['#27ae60' if x >=0 else '#c0392b' for x in segment_returns]
    ax2.bar(range(len(segment_returns)), segment_returns, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # Add labels on top of bars
    for idx, r in enumerate(segment_returns):
        ax2.text(idx, r + (1 if r>=0 else -2), f'{r:.1f}%', ha='center', fontsize=9, fontweight='bold')
        
    ax2.set_title("每 14 个交易日调仓周期收益率 (%)", fontsize=14, pad=15)
    ax2.set_ylabel("收益率 (%)", fontsize=12)
    ax2.set_xlabel("调仓期序列", fontsize=12)
    ax2.axhline(0, color='black', linewidth=1)
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    chart_path = os.path.join(config.CHART_OUTPUT_DIR, "t14_final_performance.png")
    plt.savefig(chart_path, dpi=300)
    print(f"Chart saved to: {chart_path}")
    
    # Summary Info
    win_rate = (np.array(segment_returns) > 0).mean() * 100
    avg_seg = np.mean(segment_returns)
    print(f"\nStats:")
    print(f"  Total Return  : {(val-1)*100:.1f}%")
    print(f"  Segment WinRate: {win_rate:.1f}%")
    print(f"  Avg Seg Return: {avg_seg:.1f}%")

if __name__ == "__main__":
    generate_t14_visuals()
