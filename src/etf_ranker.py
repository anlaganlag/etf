import pandas as pd
from datetime import datetime, timedelta
import time
from .data_fetcher import DataFetcher
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from config import config
except ImportError:
    class ConfigMock:
        SECTOR_PERIOD_SCORES = {1:100, 3:70, 5:50, 10:30, 20:20}
    config = ConfigMock()

class EtfRanker:
    def __init__(self, data_fetcher: DataFetcher):
        self.fetcher = data_fetcher
        self.scores = config.SECTOR_PERIOD_SCORES

    def get_theme_normalized(self, name: str) -> str:
        """Robust theme extraction for meaningful grouping"""
        if not name: return "Unknown"
        name = name.lower()
        # Priority keywords for grouping
        keywords = ["芯片", "半导体", "人工智能", "ai", "红利", "银行", "机器人", "光伏", "白酒", "医药", "医疗", "军工", "新能源", "券商", "证券", "黄金", "纳斯达克", "标普", "信创", "软件", "房地产", "中药"]
        for k in keywords:
            if k in name: return k
        # Fallback
        theme = name.replace("etf", "").replace("基金", "").replace("增强", "").replace("指数", "")
        for word in ["中证", "沪深", "上证", "深证", "科创", "创业板", "港股通", "300", "500", "1000", "50", "100"]:
            theme = theme.replace(word.lower(), "")
        return theme.strip() if theme.strip() else "宽基"

    def _calculate_max_correlation(self, target_df, selected_histories):
        """Check max correlation of target_df against selected_histories"""
        if not selected_histories:
            return 0.0
        
        max_corr = 0.0
        # Align lengths. Join on index (Date).
        target_close = target_df['收盘']
        
        for pool_df in selected_histories:
            # Inner join on Date index
            combined = pd.concat([target_close, pool_df['收盘']], axis=1, join='inner')
            if len(combined) < 20: # Not enough overlapping data
                continue
                
            corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            if abs(corr) > max_corr:
                max_corr = abs(corr)
                
        return max_corr

    def select_top_etfs(self, candidate_etfs: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Ranks ETFs using serial processing and applies sector limits using Correlation.
        """
        if candidate_etfs.empty:
            return pd.DataFrame()

        print(f"Ranking {len(candidate_etfs)} ETFs (Serial Mode)...")

        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
        
        results = []
        
        # 1. Filter out known non-ETFs if name is empty
        if 'etf_name' in candidate_etfs.columns:
             candidate_etfs = candidate_etfs[candidate_etfs['etf_name'] != '']
        
        print(f"ETFs after name filter: {len(candidate_etfs)}")
        total_count = len(candidate_etfs)
        
        candidate_histories = {} 
        
        for idx, (_, row) in enumerate(candidate_etfs.iterrows()):
            code = row['etf_code']
            name = row['etf_name']
            
            if (idx + 1) % 50 == 0:
                print(f"Progress: {idx + 1}/{total_count}...")

            hist_df = self.fetcher.get_etf_daily_history(code, start_date, end_date)
            
            if hist_df is None or hist_df.empty or len(hist_df) < 10:
                 continue
            
            # Ensure Date index for alignment
            if '日期' in hist_df.columns:
                hist_df = hist_df.set_index('日期')

            current_price = hist_df.iloc[-1]['收盘']

            def get_ret(days):
                if len(hist_df) > days:
                    prev_rec = hist_df.iloc[-(days+1)]
                    prev_price = float(prev_rec['收盘'])
                    if prev_price == 0: return 0
                    return (current_price - prev_price) / prev_price * 100
                return -999

            theme = self.get_theme_normalized(name)
            
            # Store history for later correlation check
            candidate_histories[code] = hist_df

            results.append({
                'etf_code': code,
                'etf_name': name,
                'theme': theme,
                'r1': get_ret(1),
                'r3': get_ret(3),
                'r5': get_ret(5),
                'r10': get_ret(10),
                'r20': get_ret(20),
                'total_score': 0,
                'latest_close': current_price
            })
        
        df_res = pd.DataFrame(results)
        if df_res.empty:
            return df_res

        # Scoring Logic
        threshold = max(20, int(len(df_res) * 0.1)) 
        for period, score in self.scores.items():
            col = f'r{period}'
            if col in df_res.columns:
                top_indices = df_res[col].sort_values(ascending=False).index[:threshold]
                df_res.loc[top_indices, 'total_score'] += score

        # Sort by score
        # Secondary sort by r5 (short term) or r20 (long term)? 
        # r5 breaks ties better for momentum.
        df_sorted = df_res.sort_values(['total_score', 'r5'], ascending=False)
        
        # Selection with Correlation Deduplication
        final_list = []
        selected_histories = []
        
        # Strict correlation to avoid duplicates
        CORR_THRESHOLD = 0.85 
        sector_limit = getattr(config, 'ETF_SECTOR_LIMIT', 999)
        
        print(f"\nSelecting Top ETFs with Correlation Check (Threshold {CORR_THRESHOLD})...")
        
        for _, row in df_sorted.iterrows():
            if len(final_list) >= top_n:
                break
            
            code = row['etf_code']
            theme = row['theme']
            hist = candidate_histories.get(code)
            
            # 1. Correlation Check
            max_corr = self._calculate_max_correlation(hist, selected_histories)
            if max_corr > CORR_THRESHOLD:
                print(f"Skipping {row['etf_name']} ({row['theme']}) - Correlation: {max_corr:.2f}")
                continue
            
            # 2. Keyword/Theme Check via config limit
            same_theme_count = sum(1 for item in final_list if item['theme'] == theme)
            if same_theme_count >= sector_limit:
                 continue

            final_list.append(row)
            selected_histories.append(hist)
        
        return pd.DataFrame(final_list)
