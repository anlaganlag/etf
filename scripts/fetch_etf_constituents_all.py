import os
import pandas as pd
from src.data_fetcher import DataFetcher
from tqdm import tqdm
import time

def main():
    fetcher = DataFetcher(cache_dir="data_cache")
    
    # 1. Get all ETFs
    print("Fetching ETF list...")
    etf_list = fetcher.get_all_etfs()
    
    if etf_list.empty:
        print("Error: Could not fetch ETF list.")
        return

    print(f"Found {len(etf_list)} ETFs. Starting to fetch constituents...")
    
    # 2. Iterate through ETFs and fetch constituents
    all_constituents = []
    
    # Ensure output directory exists
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Use tqdm for progress bar
    for index, row in tqdm(etf_list.iterrows(), total=len(etf_list), desc="Fetching ETF constituents"):
        etf_code = row['etf_code']
        etf_name = row['etf_name']
        
        try:
            df = fetcher.get_etf_constituents(etf_code)
            if df is not None and not df.empty:
                # Add ETF name for better readability in the final output
                df['etf_name'] = etf_name
                all_constituents.append(df)
            
            # Rate limiting / polite fetching
            # time.sleep(0.1) 
            
        except Exception as e:
            print(f"\nFailed for {etf_code} ({etf_name}): {e}")
            continue

    # 3. Combine and save
    if all_constituents:
        print("\nCombining results...")
        final_df = pd.concat(all_constituents, ignore_index=True)
        
        output_file = os.path.join(output_dir, "all_etf_constituents.csv")
        final_df.to_csv(output_file, index=False)
        
        print(f"Success! Saved constituents for {len(all_constituents)} ETFs to {output_file}")
        print(f"Total rows: {len(final_df)}")
    else:
        print("\nNo constituents were fetched.")

if __name__ == "__main__":
    main()
