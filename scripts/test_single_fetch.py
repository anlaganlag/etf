from gm.api import *
from src.data_fetcher import DataFetcher
import os
from dotenv import load_dotenv

load_dotenv()

def test_single_etf():
    fetcher = DataFetcher(cache_dir="data_cache")
    # Test with a very common ETF
    symbol = 'SHSE.510050' 
    print(f"Testing fetch for {symbol}...")
    df = fetcher.get_etf_constituents(symbol)
    
    if not df.empty:
        print("Success!")
        print(df.head())
    else:
        print("Failed: DataFrame is empty.")

if __name__ == "__main__":
    test_single_etf()
