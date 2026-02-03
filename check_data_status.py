import os
import pandas as pd
from datetime import datetime

cache_dir = r'd:\antigravity\old_etf\etf\data_cache'
files = [f for f in os.listdir(cache_dir) if f.endswith('.csv') and ('SHSE' in f or 'SZSE' in f)]

stats = []
for f in files:
    path = os.path.join(cache_dir, f)
    try:
        df = pd.read_csv(path)
        if df.empty or '日期' not in df.columns:
            continue
        df['日期'] = pd.to_datetime(df['日期']).dt.tz_localize(None)
        stats.append({
            'file': f,
            'count': len(df),
            'min_date': df['日期'].min(),
            'max_date': df['日期'].max()
        })
    except:
        pass

stats_df = pd.DataFrame(stats)
if not stats_df.empty:
    print(f"Total files: {len(files)}")
    print(f"Files with data: {len(stats_df)}")
    print(f"Latest max_date: {stats_df['max_date'].max()}")
    print(f"Files ending in Jan 2026: {len(stats_df[stats_df['max_date'] >= '2026-01-01'])}")
    print(f"Typical start_date mean: {stats_df['min_date'].mean()}")
    
    print("\n--- Samples ending ealier than 2025 ---")
    print(stats_df[stats_df['max_date'] < '2025-01-01'].head(10))
else:
    print("No valid CSV data found.")
