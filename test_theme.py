from src.etf_ranker import EtfRanker
import pandas as pd

class MockFetcher:
    def get_etf_daily_history(self, *args): return None

ranker = EtfRanker(MockFetcher())
names = ["广发中证光伏龙头30ETF", "永赢国证通用航空产业ETF", "鹏华中证国防ETF", "富国中证大数据产业ETF"]
for name in names:
    theme = ranker.get_theme_normalized(name)
    print(f"Name: {name} -> Theme: {theme}")
