import os
import pandas as pd
from src.data_fetcher import DataFetcher
from src.etf_ranker import EtfRanker
from config import config

def main():
    print("=== Starting A-Share ETF Selection System (Baostock Stable Mode) ===")
    
    # Ensure directories exist
    config.ensure_dirs()
    
    # 1. Init
    fetcher = DataFetcher(cache_dir=config.DATA_CACHE_DIR)
    etf_ranker = EtfRanker(fetcher)

    # 2. Get All ETFs
    print("\n--- Step 1: Fetching Market Data ---")
    all_etfs = fetcher.get_all_etfs()
    
    if all_etfs is None or all_etfs.empty:
        print("Failed to fetch ETF list.")
        return
    print(f"Total ETFs found: {len(all_etfs)}")

    # 3. Rank ETFs Directly
    print("\n--- Step 2: Scoring and Ranking ETFs ---")
    final_etfs = etf_ranker.select_top_etfs(all_etfs, top_n=10)

    if final_etfs.empty:
        print("No ETFs selected.")
        return

    print("\n=== Final Selection: Top 10 Strongest ETFs (One per Sector) ===")
    cols = ['etf_code', 'etf_name', 'theme', 'total_score', 'r20', 'r10', 'r5', 'latest_close']
    print(final_etfs[cols])
    
    # Save to file
    output_path = os.path.join(config.DATA_OUTPUT_DIR, "top_10_etfs.csv")
    final_etfs.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    print("\n--- Market Insight: Dominant Themes ---")
    themes = final_etfs['theme'].value_counts()
    print(themes)

if __name__ == "__main__":
    main()
