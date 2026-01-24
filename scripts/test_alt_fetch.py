from gm.api import *
from src.data_fetcher import DataFetcher
import os
from dotenv import load_dotenv

load_dotenv()

def test_alternatives():
    fetcher = DataFetcher(cache_dir="data_cache")
    symbol = 'SHSE.510050'
    
    print(f"--- Testing fnd_get_portfolio for {symbol} ---")
    try:
        df_p = fnd_get_portfolio(symbol)
        if df_p is not None and not df_p.empty:
            print("Success with fnd_get_portfolio!")
            print(df_p.head())
        else:
            print(f"fnd_get_portfolio returned {type(df_p)} (empty: {getattr(df_p, 'empty', 'N/A')})")
    except Exception as e:
        print(f"fnd_get_portfolio error: {e}")

    print(f"\n--- Testing stk_get_index_constituents (via underlying index) ---")
    # 510050 tracks 000016
    index_symbol = 'SHSE.000016'
    try:
        constituents = stk_get_index_constituents(index_symbol)
        if constituents:
            print(f"Success with stk_get_index_constituents for {index_symbol}!")
            print(f"Found {len(constituents)} constituents.")
            print(constituents[:5])
        else:
            print(f"stk_get_index_constituents returned empty for {index_symbol}")
    except Exception as e:
        print(f"stk_get_index_constituents error: {e}")

if __name__ == "__main__":
    test_alternatives()
