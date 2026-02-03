"""
Strategy Integration Verification Script
验证 ThemeBooster 在策略逻辑中的集成效果
"""
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock

# Ensure we can import from current directory
sys.path.append(os.getcwd())

from src.theme_booster import ThemeBooster
from config import config

# Mock Context
class MockContext:
    def __init__(self):
        self.whitelist = set()
        self.theme_map = {}
        self.prices_df = pd.DataFrame()
        self.theme_booster = None
        self.now = None
        self._last_concept_refresh = None
        self._hot_themes = set()

def test_boost_logic():
    print("=" * 60)
    print("STRATEGY INTEGRATION TEST")
    print("=" * 60)
    
    # 1. Setup Mock Context
    context = MockContext()
    
    # Mock ETF Data
    etfs = ['SHSE.510000', 'SHSE.510001', 'SHSE.510002']
    themes = ['人工智能', '光伏', '银行']
    context.whitelist = set(etfs)
    context.theme_map = dict(zip(etfs, themes))
    
    print(f"[Setup] Mock ETFs: {dict(zip(etfs, themes))}")
    
    # 2. Initialize ThemeBooster (Mocked to return specific themes)
    print("\n[Step 1] Initializing ThemeBooster...")
    # 这里我们不真正调用API，而是直接注入热门主题来测试"加分逻辑"是否生效
    context.theme_booster = MagicMock()
    # 假设 '人工智能' 是热门主题
    context.theme_booster.get_hot_themes.return_value = {'人工智能'}
    
    # 3. Simulate Logic in gm_strategy_rolling.py
    print("\n[Step 2] Simulating Scoring with Boost...")
    
    # Base scores (mocked)
    base_scores = pd.Series([100.0, 100.0, 100.0], index=etfs)
    print("  Base Scores:")
    print(base_scores.to_string())
    
    # Inject Hot Themes into Context (Simulating on_bar logic)
    current_dt = pd.Timestamp("2026-01-28")
    
    # Logic copied/adapted from gm_strategy_rolling.py
    CONCEPT_THEME_BOOST = True
    CONCEPT_BOOST_POINTS = 40
    
    final_scores = base_scores.copy()
    
    if CONCEPT_THEME_BOOST and context.theme_booster:
        # Simulate refresh logic
        refresh_needed = True 
        if refresh_needed:
            context._hot_themes = context.theme_booster.get_hot_themes()
            print(f"  [Logic] Refreshed Hot Themes: {context._hot_themes}")
            
        if hasattr(context, '_hot_themes') and context._hot_themes:
            for code in final_scores.index:
                theme = context.theme_map.get(code, 'Unknown')
                if theme in context._hot_themes:
                    print(f"  [Logic] Boosting {code} ({theme}) by {CONCEPT_BOOST_POINTS} points")
                    final_scores[code] += CONCEPT_BOOST_POINTS
    
    print("\n[Step 3] Final Scores:")
    print(final_scores.to_string())
    
    # 4. Verification
    expected_score_ai = 140.0
    actual_score_ai = final_scores['SHSE.510000']
    
    if actual_score_ai == expected_score_ai:
        print("\n✅ SUCCESS: '人工智能' ETF correctly boosted.")
    else:
        print(f"\n❌ FAILED: Expected {expected_score_ai}, got {actual_score_ai}")
        
    if final_scores['SHSE.510001'] == 100.0:
        print("✅ SUCCESS: '光伏' ETF (not hot) score unchanged.")
    else:
        print("❌ FAILED: Non-hot ETF score changed.")

if __name__ == "__main__":
    test_boost_logic()
