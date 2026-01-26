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

    # 2. Get Candidates from Excel
    print("\n--- Step 1: Loading Candidates from Excel ---")
    excel_path = os.path.join(config.BASE_DIR, "ETF合并筛选结果.xlsx")
    
    if not os.path.exists(excel_path):
        print(f"Error: {excel_path} not found.")
        return

    try:
        df_excel = pd.read_excel(excel_path)
        # Map columns: symbol -> etf_code, sec_name -> etf_name, 主题 -> theme
        candidate_etfs = df_excel.rename(columns={
            'symbol': 'etf_code',
            'sec_name': 'etf_name',
            '主题': 'theme'
        })
        print(f"Loaded {len(candidate_etfs)} candidates from Excel.")
    except Exception as e:
        print(f"Error reading Excel: {e}")
        return

    # 3. Rank ETFs Directly
    print("\n--- Step 2: Scoring and Ranking ETFs ---")
    final_etfs = etf_ranker.select_top_etfs(candidate_etfs, top_n=10)

    if final_etfs.empty:
        print("No ETFs selected.")
        return

    print("\n=== Final Selection: Top 10 Strongest ETFs ===")
    # 动态选择可用的列
    base_cols = ['etf_code', 'etf_name', 'theme', 'total_score']
    period_cols = [col for col in ['r250', 'r120', 'r60', 'r20', 'r10', 'r5', 'r1'] if col in final_etfs.columns]
    cols = base_cols + period_cols + ['latest_close']
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
