from gm.api import *
import pandas as pd
import time
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class DataFetcher:
    """
    Handles data fetching using MyQuant/掘金 API with File-based Caching.
    Configured for Serial Execution.
    """
    def __init__(self, retry_count: int = 3, retry_delay: int = 0.5, cache_dir: str = "data_cache"):
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.cache_dir = cache_dir

        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

        # Initialize MyQuant token
        token = os.getenv('MY_QUANT_TGM_TOKEN')
        if not token:
            raise ValueError("MY_QUANT_TGM_TOKEN not found in environment variables")
        set_token(token)
        print(f"MyQuant token initialized successfully")

    def _safe_fetch(self, func, *args, **kwargs):
        for attempt in range(self.retry_count):
            try:
                # Slight delay to prevent hard blocking
                time.sleep(0.05) 
                return func(*args, **kwargs)
            except Exception as e:
                # print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    time.sleep(self.retry_delay)
        return None

    def get_all_etfs(self) -> pd.DataFrame:
        """
        Fetch list of all ETFs from MyQuant. Cached explicitly.
        """
        today = datetime.now().strftime("%Y%m%d")
        cache_path = os.path.join(self.cache_dir, f"etf_list_{today}.csv")

        if os.path.exists(cache_path):
            print("Loading ETF list from cache...")
            return pd.read_csv(cache_path)

        print("Fetching ETF list via MyQuant...")
        try:
            # Get all securities including funds (sec_types=2)
            # Then filter for ETFs based on symbol patterns
            df = self._safe_fetch(
                get_instruments,
                exchanges='SZSE,SHSE',
                sec_types=[1, 2],  # 1=stocks, 2=funds (includes ETFs)
                fields='symbol,sec_name',
                df=True
            )

            if df is not None and not df.empty:
                # Filter for ETF codes: SHSE.51*/56*/58* or SZSE.15*
                mask = (
                    df['symbol'].str.match(r'SHSE\.51\d{4}') |
                    df['symbol'].str.match(r'SHSE\.56\d{4}') |
                    df['symbol'].str.match(r'SHSE\.58\d{4}') |
                    df['symbol'].str.match(r'SZSE\.15\d{4}')
                )
                df_etf = df[mask].copy()

                # Rename columns to match existing code structure
                df_etf = df_etf.rename(columns={
                    'symbol': 'etf_code',
                    'sec_name': 'etf_name'
                })

                # Save to cache
                df_etf.to_csv(cache_path, index=False)
                print(f"Fetched {len(df_etf)} ETFs from MyQuant")
                return df_etf
        except Exception as e:
            print(f"MyQuant ETF list fetch failed: {e}")
            return pd.DataFrame()

    def get_etf_daily_history(self, etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily history from MyQuant with caching.
        Store each ETF's history in a separate CSV: data_cache/SHSE.510300.csv

        Args:
            etf_code: MyQuant format symbol (e.g., 'SHSE.510300')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        # Sanitize filename (replace dots with underscores for cache file)
        cache_filename = etf_code.replace('.', '_')
        cache_file = os.path.join(self.cache_dir, f"{cache_filename}.csv")

        # 1. Try Load Cache
        df_cache = pd.DataFrame()
        if os.path.exists(cache_file):
            try:
                df_cache = pd.read_csv(cache_file)
                df_cache['日期'] = pd.to_datetime(df_cache['日期'])

                # Ensure timezone-naive datetime
                if df_cache['日期'].dt.tz is not None:
                    df_cache['日期'] = df_cache['日期'].dt.tz_localize(None)
            except:
                pass # corrupted cache?

        # 2. Determine if we need to fetch
        need_fetch = False

        if df_cache.empty:
            need_fetch = True
        else:
            # If cache is older than 12 hours, re-fetch
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mod_time).total_seconds() > 43200: # 12 hours
                need_fetch = True

        if need_fetch:
            try:
                df_new = self._safe_fetch(
                    history,
                    symbol=etf_code,
                    frequency='1d',  # Daily data
                    start_time=f'{start_date} 09:00:00',
                    end_time=f'{end_date} 16:00:00',
                    fields='eob,open,close',  # eob = end of bar, open = 开盘价, close = 收盘价
                    adjust=ADJUST_PREV,  # 前复权
                    df=True
                )

                if df_new is not None and not df_new.empty:
                    # Rename columns to match existing code structure
                    df_new = df_new.rename(columns={'eob': '日期', 'open': '开盘', 'close': '收盘'})
                    df_new['日期'] = pd.to_datetime(df_new['日期'])

                    # Remove timezone info to match existing code
                    if df_new['日期'].dt.tz is not None:
                        df_new['日期'] = df_new['日期'].dt.tz_localize(None)

                    df_new['开盘'] = pd.to_numeric(df_new['开盘'], errors='coerce')
                    df_new['收盘'] = pd.to_numeric(df_new['收盘'], errors='coerce')
                    df_new = df_new.sort_values('日期')

                    # Save full history to cache
                    df_new.to_csv(cache_file, index=False)
                    df_cache = df_new
            except Exception as e:
                print(f"Failed to fetch history for {etf_code}: {e}")

        # 3. Filter and Return
        if df_cache.empty:
            return pd.DataFrame()

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        mask = (df_cache['日期'] >= start_dt) & (df_cache['日期'] <= end_dt)
        return df_cache[mask].copy()
    
    def get_etf_constituents(self, etf_code: str) -> pd.DataFrame:
        """
        Fetch constituents of an ETF from MyQuant.
        """
        # Create cache directory for constituents if it doesn't exist
        constituents_cache_dir = os.path.join(self.cache_dir, "constituents")
        if not os.path.exists(constituents_cache_dir):
            os.makedirs(constituents_cache_dir)

        cache_filename = etf_code.replace('.', '_')
        today = datetime.now().strftime("%Y%m%d")
        cache_file = os.path.join(constituents_cache_dir, f"{cache_filename}_{today}.csv")

        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

        try:
            df = self._safe_fetch(fnd_get_etf_constituents, etf=etf_code)
            if df is not None and not df.empty:
                df.to_csv(cache_file, index=False)
                return df
            else:
                # Debug: print if result is empty or None
                print(f"DEBUG: fnd_get_etf_constituents for {etf_code} returned {type(df)} (empty: {getattr(df, 'empty', 'N/A')})")
        except Exception as e:
            print(f"Failed to fetch constituents for {etf_code}: {e}")
        
        return pd.DataFrame()

    def get_all_sectors(self):
        return pd.DataFrame()
