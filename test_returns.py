
import pandas as pd
import numpy as np
from src.data_fetcher import DataFetcher
from config import config

fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
all_etfs = fetcher.get_all_etfs()
test_codes = all_etfs['etf_code'].head(5).tolist()

price_data = {}
for code in test_codes:
    df = fetcher.get_etf_daily_history(code, "2024-09-01", "2024-10-01")
    if not df.empty:
        price_data[code] = df.set_index('日期')['收盘']
df_p = pd.DataFrame(price_data)
print("Prices head:\n", df_p.head())
print("Returns head:\n", df_p.pct_change().head())
