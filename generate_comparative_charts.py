
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from src.data_fetcher import DataFetcher
from config import config

# Set Plot Style
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('bmh')

def generate_comparative_visuals():
    print("=== Generating Final T=14 vs T=20 Visualization ===")
    config.ensure_dirs()
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    
    # 1. Universes
    rename_map = {'symbol': 'etf_code', 'sec_name': 'etf_name', 'name_cleaned': 'theme', '主题': 'theme'}
    strong_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    df_strong = pd.read_excel(strong_path)
    df_strong = df_strong.rename(columns=rename_map)
    strong_codes = set(df_strong['etf_code'])
    
    df_all = fetcher.get_all_etfs()
    all_codes = list(df_all['etf_code'])

    # 2. Data
    print("[1/4] Loading Data Matrix...")
    price_data = {}
    start_load = "2023-01-01"
    for code in all_codes:
        cache_file = os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv")
        if os.path.exists(cache_file):
            try:
                df = pd.read_csv(cache_file, usecols=['日期', '收盘'])
                df['日期'] = pd.to_datetime(df['日期'])
                if not df.empty:
                    price_data[code] = df.set_index('日期')['收盘']
            except: pass
    prices_df = pd.DataFrame(price_data).sort_index().ffill()
    
    # Benchmark
    chinext_code = 'SZSE.159915'
    chinext_prices = prices_df[chinext_code]

    # 3. Scoring
    print("[2/4] Calculating Global Ranking Scores...")
    # Use the definitive 8-period system
    periods = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20, 60: 15, 120: 10, 250: 5}
    total_scores = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for p, pts in periods.items():
        ranks = prices_df.pct_change(p).rank(axis=1, ascending=False, method='min')
        total_scores = total_scores.add((ranks <= 15) * pts, fill_value=0)
    
    r20_matrix = prices_df.pct_change(20).fillna(-999)
    daily_rets = prices_df.pct_change(1).shift(-1)

    # 4. Simulation
    print("[3/4] Running Simulations (T=14 and T=20)...")
    start_sim = "2024-11-01"
    sim_dates = prices_df.index[prices_df.index >= pd.to_datetime(start_sim)]
    
    results = {}
    curves = {}
    all_seg_rets = {}
    
    for T in [14, 20]:
        val = 1.0
        vals = [1.0]
        holdings = []
        seg_rets = []
        seg_start_val = 1.0
        
        for i in range(len(sim_dates) - 1):
            dt = sim_dates[i]
            if i % T == 0:
                if i > 0:
                    seg_rets.append((val/seg_start_val - 1) * 100)
                    seg_start_val = val
                
                s = total_scores.loc[dt]
                valid = s[s >= 150].index
                metric = (s * 10000 + r20_matrix.loc[dt])[valid].dropna()
                holdings = [c for c in metric.sort_values(ascending=False).index if c in strong_codes][:10]
            
            if holdings:
                val *= (1 + daily_rets.loc[dt, holdings].mean())
            vals.append(val)
            
        if val != seg_start_val:
            seg_rets.append((val/seg_start_val - 1) * 100)
            
        curves[T] = vals
        all_seg_rets[T] = seg_rets
        results[T] = (val - 1.0) * 100

    # 5. Plotting
    print("[4/4] Generating Charts...")
    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 1, 1])
    
    # Subplot 1: Comparison Equity Curves
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(sim_dates, curves[14], color='#e74c3c', linewidth=2.5, label=f'T=14 最佳效率方案 (+{results[14]:.1f}%)')
    ax1.plot(sim_dates, curves[20], color='#2980b9', linewidth=2, linestyle='--', label=f'T=20 传统周期方案 (+{results[20]:.1f}%)')
    
    bm = chinext_prices.loc[sim_dates]
    bm = bm / bm.iloc[0]
    ax1.plot(sim_dates, bm, color='#34495e', alpha=0.6, label=f'创业板指基准 (+{(bm.iloc[-1]-1)*100:.1f}%)')
    
    ax1.set_title("ETF 动量巅峰策略：T=14 与 T=20 绩效对比", fontsize=18, fontweight='bold', pad=25)
    ax1.set_ylabel("累积净值 (起点 1.0)", fontsize=14)
    ax1.legend(loc='upper left', fontsize=12, frameon=True)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: T=14 Segment Returns
    ax2 = fig.add_subplot(gs[1])
    s14 = all_seg_rets[14]
    clrs = ['#27ae60' if x>=0 else '#c0392b' for x in s14]
    ax2.bar(range(len(s14)), s14, color=clrs, alpha=0.8, edgecolor='black')
    for i, r in enumerate(s14):
        ax2.text(i, r + (1 if r>=0 else -3), f'{r:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax2.set_title("T = 14 方案：每调仓周期收益率", fontsize=14)
    ax2.set_ylabel("收益率 (%)")
    ax2.axhline(0, color='black', linewidth=1)
    
    # Subplot 3: T=20 Segment Returns
    ax3 = fig.add_subplot(gs[2])
    s20 = all_seg_rets[20]
    clrs = ['#2980b9' if x>=0 else '#c0392b' for x in s20]
    ax3.bar(range(len(s20)), s20, color=clrs, alpha=0.8, edgecolor='black')
    for i, r in enumerate(s20):
        ax3.text(i, r + (1 if r>=0 else -3), f'{r:.1f}%', ha='center', fontsize=9, fontweight='bold')
    ax3.set_title("T = 20 方案：每调仓周期收益率", fontsize=14)
    ax3.set_ylabel("收益率 (%)")
    ax3.set_xlabel("调仓期序列")
    ax3.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    chart_path = os.path.join(config.CHART_OUTPUT_DIR, "final_strategy_comparison.png")
    plt.savefig(chart_path, dpi=300)
    print(f"Chart saved: {chart_path}")
    
    # Output markdown snippets
    print("\nStats Summary:")
    print(f"  T=14 Return: {results[14]:.1f}%")
    print(f"  T=20 Return: {results[20]:.1f}%")

if __name__ == "__main__":
    generate_comparative_visuals()
