"""
Test script to verify MyQuant data source integration
"""
from src.data_fetcher import DataFetcher
from datetime import datetime, timedelta

def test_etf_list():
    """Test fetching ETF list"""
    print("=" * 60)
    print("TEST 1: Fetching ETF List")
    print("=" * 60)

    fetcher = DataFetcher(cache_dir="data_cache")
    etf_list = fetcher.get_all_etfs()

    if etf_list.empty:
        print("❌ Failed to fetch ETF list")
        return False

    print(f"✓ Successfully fetched {len(etf_list)} ETFs")
    print("\nSample ETFs:")
    print(etf_list.head(10))
    print(f"\nColumns: {list(etf_list.columns)}")
    return True

def test_etf_history():
    """Test fetching ETF history data"""
    print("\n" + "=" * 60)
    print("TEST 2: Fetching ETF History Data")
    print("=" * 60)

    fetcher = DataFetcher(cache_dir="data_cache")

    # Test with a known ETF (沪深300ETF)
    test_symbol = "SHSE.510300"
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"\nFetching {test_symbol} from {start_date} to {end_date}")

    df = fetcher.get_etf_daily_history(test_symbol, start_date, end_date)

    if df.empty:
        print(f"❌ Failed to fetch history for {test_symbol}")
        return False

    print(f"✓ Successfully fetched {len(df)} records")
    print("\nSample data:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nLatest close: {df.iloc[-1]['收盘']:.2f}")
    return True

def main():
    """Run all tests"""
    print("\n🚀 Starting MyQuant Data Source Tests\n")

    results = []

    # Test 1: ETF List
    results.append(("ETF List", test_etf_list()))

    # Test 2: ETF History
    results.append(("ETF History", test_etf_history()))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n🎉 All tests passed! MyQuant data source is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Please check the error messages above.")

    return all_passed

if __name__ == "__main__":
    main()
