import akshare as ak
import pandas as pd
import time
import os
from datetime import datetime
from typing import Optional
import baostock as bs

class DataFetcher:
    """
    Handles data fetching using AkShare Sina interfaces with File-based Caching.
    Configured for Serial Execution.
    """
    def __init__(self, retry_count: int = 3, retry_delay: int = 0.5, cache_dir: str = "data_cache"):
        self.retry_count = retry_count
        self.retry_delay = retry_delay
        self.cache_dir = cache_dir
        
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)

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
        Fetch list of all ETFs. Cached explicitly.
        Prioritize Sina for names, fallback to Baostock (codes only) if Sina fails.
        """
        today = datetime.now().strftime("%Y%m%d")
        cache_path = os.path.join(self.cache_dir, f"etf_list_{today}.csv")
        
        if os.path.exists(cache_path):
            print("Loading ETF list from cache...")
            return pd.read_csv(cache_path)

        # 1. Try Sina
        print("Fetching ETF list via Sina...")
        try:
            df = self._safe_fetch(ak.fund_etf_category_sina, symbol="ETF基金")
            if df is not None and not df.empty:
                df = df.rename(columns={
                    '代码': 'etf_code',
                    '名称': 'etf_name',
                    '最新价': 'latest_price',
                    '成交额': 'turnover'
                })
                df.to_csv(cache_path, index=False)
                return df
        except Exception as e:
            print(f"Sina list fetch failed: {e}")

        # 2. Fallback to Baostock
        print("Fallback: Fetching ETF list via Baostock (Names will be missing)...")
        lg = bs.login()
        # Query yesterday (to ensure data ready)
        date = (datetime.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=date)
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        bs.logout()
            
        if not data_list:
            return pd.DataFrame()
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        # Filter for ETF prefixes
        mask = (
            df['code'].str.startswith('sh.51') | 
            df['code'].str.startswith('sh.56') |
            df['code'].str.startswith('sh.58') |
            df['code'].str.startswith('sz.15')
        )
        df_etf = df[mask].copy()
        df_etf = df_etf.rename(columns={'code': 'etf_code', 'code_name': 'etf_name'})
        
        # Baostock names are often empty for ETFs, fill with Unknown
        df_etf['etf_name'] = df_etf['etf_name'].replace('', 'Unknown ETF')
        
        # Normalize codes? Sina uses sz159995, Baostock uses sz.159995
        # Our Sina history fetcher expects Sina format (sz159995).
        # Need to strip dots from Baostock codes.
        df_etf['etf_code'] = df_etf['etf_code'].str.replace('.', '')
        
        # Save to cache
        df_etf.to_csv(cache_path, index=False)
        return df_etf

    def get_etf_daily_history(self, etf_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch daily history with caching.
        Store each ETF's history in a separate CSV: data_cache/pre_sz159995.csv
        """
        # Sina symbols sometimes have prefixes locally or not.
        cache_file = os.path.join(self.cache_dir, f"{etf_code}.csv")
        
        # 1. Try Load Cache
        df_cache = pd.DataFrame()
        if os.path.exists(cache_file):
            try:
                df_cache = pd.read_csv(cache_file)
                df_cache['日期'] = pd.to_datetime(df_cache['日期'])
            except:
                pass # corrupted cache?

        # 2. Determine if we need to fetch
        # If cache is empty, fetch all.
        # If cache exists, check if 'end_date' is covered.
        need_fetch = False
        fetch_start = start_date
        
        if df_cache.empty:
            need_fetch = True
        else:
            last_date = df_cache['日期'].max()
            end_dt = pd.to_datetime(end_date)
            # If last cached date is before requested end_date, we MIGHT need update.
            # But checking if today is trading day is complex.
            # Simple logic: If last cache date < today, try incremental fetch?
            # Sina history API usually returns ALL history or recent.
            # ak.fund_etf_hist_sina returns full history. So incremental is hard unless we slice.
            # But it's fast enough to just fetch full if needed.
            
            # Let's say if cache is older than 18 hours, re-fetch?
            file_mod_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if (datetime.now() - file_mod_time).total_seconds() > 43200: # 12 hours
                need_fetch = True
            else:
                # Cache is fresh enough
                pass

        if need_fetch:
            df_new = self._safe_fetch(ak.fund_etf_hist_sina, symbol=etf_code)
            
            if df_new is not None and not df_new.empty:
                df_new = df_new.rename(columns={'date': '日期', 'close': '收盘'})
                df_new['日期'] = pd.to_datetime(df_new['日期'])
                df_new['收盘'] = pd.to_numeric(df_new['收盘'], errors='coerce')
                df_new = df_new.sort_values('日期')
                
                # Save full history to cache
                df_new.to_csv(cache_file, index=False)
                df_cache = df_new

        # 3. Filter and Return
        if df_cache.empty:
            return pd.DataFrame()

        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        mask = (df_cache['日期'] >= start_dt) & (df_cache['日期'] <= end_dt)
        return df_cache[mask].copy()
    
    def get_all_sectors(self):
        return pd.DataFrame()
