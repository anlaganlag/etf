import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import config

def verify_correlation():
    # Codes from the "duplicate" top 10 result
    # Chemical: sh516570, sz159133, sh516120
    # Oil/Gas: sh561360, sh561570, sz159697, sz159588, sz159309
    
    target_groups = {
        "Chemical": ["sh516570", "sz159133", "sh516120"],
        "OilGas": ["sh561360", "sh561570", "sz159697", "sz159588", "sz159309"]
    }
    
    print("=== Verifying Correlation of Suspected Duplicates ===")
    
    for group_name, codes in target_groups.items():
        print(f"\nGroup: {group_name}")
        price_data = {}
        
        for code in codes:
            path = os.path.join(config.DATA_CACHE_DIR, f"{code}.csv")
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['日期'] = pd.to_datetime(df['日期'])
                df.set_index('日期', inplace=True)
                # Use last 60 days
                price_data[code] = df['收盘'].tail(60)
            else:
                print(f"Missing data for {code}")
        
        if not price_data: continue
        
        df_prices = pd.DataFrame(price_data)
        # Calculate daily returns
        df_rets = df_prices.pct_change().dropna()
        
        # Correlation Matrix
        corr_matrix = df_rets.corr()
        print(corr_matrix.round(4))
        
        # Check if > 0.95
        min_corr = corr_matrix.min().min()
        print(f"Minimum Correlation in group: {min_corr:.4f}")
        if min_corr > 0.9:
            print(">> CONFIRMED: Highly correlated. Safe to deduplicate.")
        else:
            print(">> WARNING: Low correlation found.")

if __name__ == "__main__":
    verify_correlation()
