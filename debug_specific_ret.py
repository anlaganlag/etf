
import pandas as pd
import os
from config import config
from src.data_fetcher import DataFetcher

fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
# Load one of the ETFs from your holdings log (e.g. SHSE.515220)
code = "SHSE.515220"
df = pd.read_csv(os.path.join(config.DATA_CACHE_DIR, f"{code.replace('.', '_')}.csv"))
df['日期'] = pd.to_datetime(df['日期'])
df = df[df['日期'] >= pd.to_datetime("2024-09-01")].head(10)
print(f"History for {code}:")
print(df[['日期', '收盘']])
print("Pct Change:\n", df['收盘'].pct_change())
