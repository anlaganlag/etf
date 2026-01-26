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

    def select_top_etfs(self, candidate_etfs: pd.DataFrame, top_n: int = 10, lookback_days: int = 730, reference_date: str = None, history_cache: dict = None, min_score: int = 0) -> pd.DataFrame:
        """
        Ranks ETFs based on config scores. 
        
        Args:
            reference_date: If provided (YYYY-MM-DD), score based on data available up to this date.
            history_cache: Optional dict {code: full_history_df} to speed up backtesting.
            min_score: Minimum total_score required to be selected (timing gate).
        """
        if candidate_etfs.empty:
            return pd.DataFrame()

        # Determine actual data range needed
        if reference_date:
            end_date_str = reference_date
            end_dt = datetime.strptime(reference_date, "%Y-%m-%d")
            start_date_str = (end_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        else:
            end_date_str = datetime.now().strftime("%Y-%m-%d")
            end_dt = datetime.now()
            start_date_str = (end_dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        
        results = []
        
        for idx, (_, row) in enumerate(candidate_etfs.iterrows()):
            code = row['etf_code']
            # Use provided name/theme if available, else empty
            name = row.get('etf_name', '')
            theme = row.get('theme', '')
            
            # Fetch history
            if history_cache and code in history_cache:
                full_hist = history_cache[code]
                if full_hist.empty:
                    continue
                # Slice in memory
                mask = (full_hist['日期'] >= pd.to_datetime(start_date_str)) & (full_hist['日期'] <= pd.to_datetime(end_date_str))
                hist_df = full_hist[mask].copy()
            else:
                hist_df = self.fetcher.get_etf_daily_history(code, start_date_str, end_date_str)
            
            if hist_df is None or hist_df.empty or len(hist_df) < 10:
                 continue
            
            # Ensure Date index for calculations if needed, or just use iloc
            # DataFetcher returns '日期' and '收盘' columns sorted by date
            
            current_price = hist_df.iloc[-1]['收盘']

            def get_ret(days):
                if len(hist_df) > days:
                    prev_rec = hist_df.iloc[-(days+1)]
                    prev_price = float(prev_rec['收盘'])
                    if prev_price == 0: return 0
                    return (current_price - prev_price) / prev_price * 100
                return -999

            result = {
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
            }

            # Add longer periods if available
            if len(hist_df) > 60:
                result['r60'] = get_ret(60)
            if len(hist_df) > 120:
                result['r120'] = get_ret(120)
            if len(hist_df) > 250:
                result['r250'] = get_ret(250)

            results.append(result)
        
        df_res = pd.DataFrame(results)
        if df_res.empty:
            return df_res

        # Scoring Logic
        # Use config.SECTOR_TOP_N_THRESHOLD (default 15)
        threshold = getattr(config, 'SECTOR_TOP_N_THRESHOLD', 15)
        
        for period, score in self.scores.items():
            col = f'r{period}'
            if col in df_res.columns:
                # Get Top N indices
                top_indices = df_res[col].sort_values(ascending=False).index[:threshold]
                df_res.loc[top_indices, 'total_score'] += score

        # Filter by min_score (The "Timing Gate")
        if min_score > 0:
            df_res = df_res[df_res['total_score'] >= min_score]

        if df_res.empty:
            return pd.DataFrame()

        # Sort by total_score desc, then r20 (momentum) desc
        df_sorted = df_res.sort_values(['total_score', 'r20'], ascending=False)
        
        return df_sorted.head(top_n)
    def rank_global_strength(self, all_etf_history: pd.DataFrame, whitelist: set, top_n: int = 10, min_score: int = 150) -> pd.DataFrame:
        """
        PERFORMS TRUE GLOBAL RANKING (as per documentation).
        
        Args:
            all_etf_history: DataFrame (Date x Code) of CLOSE prices for ALL ETFs.
            whitelist: set of codes (Curated List) to filter from.
            top_n: max holdings.
            min_score: entry gate.
        """
        if all_etf_history.empty: return pd.DataFrame()
        
        # 1. Calculate Multi-Period Returns for ALL
        # periods = {1:100, 3:70, 5:50, 10:30, 20:20, 60:15, 120:10, 250:5}
        periods = self.scores
        threshold = getattr(config, 'SECTOR_TOP_N_THRESHOLD', 15)
        
        total_scores = pd.Series(0.0, index=all_etf_history.columns)
        
        # We assume input df has enough data. Ranks are based on the LAST row vs N-days-ago.
        for p, pts in periods.items():
            if len(all_etf_history) > p:
                ret = (all_etf_history.iloc[-1] / all_etf_history.iloc[-(p+1)] - 1)
                ranks = ret.rank(ascending=False, method='min')
                total_scores += (ranks <= threshold) * pts
        
        # 2. Add R20 as tie-breaker
        r20 = (all_etf_history.iloc[-1] / all_etf_history.iloc[-21] - 1) if len(all_etf_history) > 20 else pd.Series(0, index=all_etf_history.columns)
        
        # 3. Filter and Sort
        df_res = pd.DataFrame({
            'etf_code': all_etf_history.columns,
            'total_score': total_scores,
            'r20_mom': r20
        }).set_index('etf_code')
        
        # Whitelist filtering (Only buying curated ones)
        df_res = df_res[df_res.index.isin(whitelist)]
        
        # Score Threshold filtering (The "择时" logic)
        df_res = df_res[df_res['total_score'] >= min_score]
        
        # Final Sort
        df_sorted = df_res.sort_values(['total_score', 'r20_mom'], ascending=False)
        return df_sorted.head(top_n)
