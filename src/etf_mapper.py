import pandas as pd
from .data_fetcher import DataFetcher

class EtfMapper:
    def __init__(self, data_fetcher: DataFetcher):
        self.fetcher = data_fetcher
        self.all_etfs = None

    def get_candidate_etfs(self, strong_sectors: list) -> pd.DataFrame:
        """
        Finds ETFs related to the provided list of strong sector names.
        """
        if self.all_etfs is None:
            print("Fetching all ETF spot data...")
            self.all_etfs = self.fetcher.get_all_etfs()
        
        if self.all_etfs is None or self.all_etfs.empty:
            print("Failed to fetch ETF list.")
            return pd.DataFrame()

        candidates = []
        
        # known synonyms or simple cleaning could go here
        # For now, strict substring match
        
        print(f"Mapping {len(strong_sectors)} strong sectors to ETFs...")

        for sector in strong_sectors:
            # Simple keyword match
            # sector might be '半导体'
            # Match ETFs with '半导体' in name
            matches = self.all_etfs[self.all_etfs['名称'].str.contains(sector, na=False)]
            
            for _, match in matches.iterrows():
                candidates.append({
                    'etf_code': match['代码'],
                    'etf_name': match['名称'],
                    'related_sector': sector,
                    # We can carry over current stats from spot data if needed
                    'latest_price': match.get('最新价'),
                    'turnover': match.get('成交额')
                })

        # Remove duplicates (an ETF might match multiple sectors if words overlap)
        df_candidates = pd.DataFrame(candidates)
        if not df_candidates.empty:
            df_candidates = df_candidates.drop_duplicates(subset=['etf_code'])
        
        return df_candidates
