import pandas as pd
from datetime import datetime, timedelta
import time
from .data_fetcher import DataFetcher
import sys
import os

# Ensure we can import config from root if running as module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import config
except ImportError:
    # Fallback or mock if config not found (but it should be there)
    class ConfigMock:
        SECTOR_TOP_N_THRESHOLD = 15
        SECTOR_PERIOD_SCORES = {1:100, 3:70, 5:50, 10:30, 20:20}
    config = ConfigMock()

class SectorRanker:
    def __init__(self, data_fetcher: DataFetcher):
        self.fetcher = data_fetcher
        self.scores = config.SECTOR_PERIOD_SCORES
        self.threshold = config.SECTOR_TOP_N_THRESHOLD

    def get_ranked_sectors(self, top_n: int = 30) -> pd.DataFrame:
        """
        Fetches all sectors, calculates scores, and returns the top N.
        """
        print("Fetching sector list...")
        sectors_df = self.fetcher.get_all_sectors()
        if sectors_df is None or sectors_df.empty:
            print("Failed to fetch sector list.")
            return pd.DataFrame()

        # Iterate and calculate returns
        # Optimization: We need efficient history fetching. 
        # For now, we loop.
        
        sector_returns = []
        
        # We need enough history for R20. Let's get ~40 days to be safe with value days.
        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=60)).strftime("%Y%m%d")

        print(f"Processing {len(sectors_df)} sectors...")
        results = []
        
        for idx, (index, row) in enumerate(sectors_df.iterrows()):
            sector_name = row['板块名称']
            sector_code = row['板块代码']
            
            if (idx + 1) % 10 == 0:
                print(f"Progress: {idx + 1}/{len(sectors_df)} sectors processed...")

            # Fetch history using code
            hist_df = self.fetcher.get_sector_daily_history(sector_code, start_date, end_date)
            
            # Rate limiting
            time.sleep(0.1)
            
            if hist_df is None or hist_df.empty or len(hist_df) < 21:
                continue

            # Ensure sorting
            hist_df = hist_df.sort_values('日期').reset_index(drop=True)
            
            # efinance '收盘' is usually numeric
            current_price = float(hist_df.iloc[-1]['收盘'])
            
            def get_ret(days):
                if len(hist_df) > days:
                    prev_price = float(hist_df.iloc[-(days+1)]['收盘'])
                    if prev_price == 0: return 0
                    return (current_price - prev_price) / prev_price * 100
                return -999

            r1 = get_ret(1)
            r3 = get_ret(3)
            r5 = get_ret(5)
            r10 = get_ret(10)
            r20 = get_ret(20)

            results.append({
                'sector_name': sector_name,
                'sector_code': sector_code,
                'r1': r1,
                'r3': r3,
                'r5': r5,
                'r10': r10,
                'r20': r20,
                'total_score': 0
            })

        df_res = pd.DataFrame(results)
        
        if df_res.empty:
            return df_res

        # Calculate Scores
        # For each period, find top N and add score
        for period, score in self.scores.items():
            col = f'r{period}'
            if col in df_res.columns:
                # Sort by return desc
                sorted_indices = df_res[col].sort_values(ascending=False).index
                # Top N get score
                top_indices = sorted_indices[:self.threshold]
                df_res.loc[top_indices, 'total_score'] += score

        # Final Sort
        df_final = df_res.sort_values('total_score', ascending=False).head(top_n)
        return df_final
