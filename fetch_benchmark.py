import akshare as ak
import pandas as pd
from datetime import datetime, timedelta

def fetch_benchmark(index_code="000985"):
    print(f"Fetching benchmark data for {index_code}...")
    
    first_method_success = False
    try:
        # stock_zh_index_daily often stale for some indices
        df = ak.stock_zh_index_daily(symbol="sh000985")
        if df is not None and not df.empty:
            last_date = str(df.iloc[-1]['date'])
            print(f"First method date range: {df.iloc[0]['date']} to {last_date}")
            if "2024" in last_date or "2025" in last_date or "2026" in last_date:
                 print("Data is fresh.")
                 df.to_csv("benchmark_000985.csv", index=False)
                 return df
            else:
                 print("Data is stale (older than 2024). Trying alternatives...")
    except Exception as e:
        print(f"Fetch failed: {e}")
        
    print("Trying alternative: index_zh_a_hist...")
    try:
        # Requires '000985' usually
        df = ak.index_zh_a_hist(symbol="000985", period="daily", start_date="20230101", end_date=datetime.now().strftime("%Y%m%d"))
        if not df.empty:
            print(f"Success (Alt)! {len(df)} rows.")
            print(df.tail())
            df.to_csv("benchmark_000985.csv", index=False)
            return df
    except Exception as e:
        print(f"Alt Fetch failed: {e}")

if __name__ == "__main__":
    fetch_benchmark()
