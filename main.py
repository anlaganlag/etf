from src.data_fetcher import DataFetcher
from src.etf_ranker import EtfRanker
import pandas as pd

def main():
    print("=== Starting A-Share ETF Selection System (Baostock Stable Mode) ===")
    
    # 1. Init
    fetcher = DataFetcher()
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
    final_etfs.to_csv("top_10_etfs.csv", index=False)
    print("\nResults saved to top_10_etfs.csv")
    
    print("\n--- Market Insight: Dominant Themes ---")
    themes = final_etfs['theme'].value_counts()
    print(themes)

if __name__ == "__main__":
    main()
