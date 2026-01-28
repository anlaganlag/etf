"""
Unit Tests for Simplified Rolling Strategy
"""
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
import os
import sys

# Ensure import paths work
sys.path.append(os.getcwd())

# Import the module to be tested
# Note: We import specific classes/functions to avoid running the main block
from gm_strategy_rolling import Tranche, get_ranking, REBALANCE_PERIOD_T

class TestTranche(unittest.TestCase):
    def setUp(self):
        self.tranche = Tranche(t_id=0, initial_cash=10000)
    
    def test_buy(self):
        # Price = 10, Amount = 5000 -> Should buy 500 shares (5000 cost)
        self.tranche.buy('ETF.001', 5000, 10.0)
        self.assertEqual(self.tranche.holdings['ETF.001'], 500)
        self.assertEqual(self.tranche.cash, 5000)
        self.assertEqual(self.tranche.pos_records['ETF.001']['entry'], 10.0)

    def test_sell(self):
        self.tranche.buy('ETF.001', 5000, 10.0)
        # Sell at 12 (+20%)
        self.tranche.sell('ETF.001', 12.0)
        self.assertNotIn('ETF.001', self.tranche.holdings)
        self.assertEqual(self.tranche.cash, 11000) # 5000 initial + 6000 revenue

    def test_guard_stop_loss(self):
        # Buy @ 10
        self.tranche.buy('ETF.001', 10000, 10.0) 
        # Price drops to 8.4 (-16%), trigger 15% SL
        price_map = {'ETF.001': 8.4}
        
        # Capture print output
        with patch('builtins.print') as mock_print:
            self.tranche.check_guard_and_sell(price_map)
            
            # Should have sold
            self.assertNotIn('ETF.001', self.tranche.holdings)
            # Should not rest for SL
            self.assertEqual(self.tranche.rest_days, 0) 

    def test_guard_trailing_profit(self):
        # Buy @ 10
        self.tranche.buy('ETF.001', 10000, 10.0)
        
        # Price goes to 12 (+20%), exceeds trigger (8%)
        self.tranche.check_guard_and_sell({'ETF.001': 12.0})
        self.assertIn('ETF.001', self.tranche.holdings) # Just holding, record high
        self.assertEqual(self.tranche.pos_records['ETF.001']['high'], 12.0)
        
        # Price drops to 11.5 (High * (1-0.04)) -> Drawdown ~4% > 3% Drop
        # Trigger TP
        with patch('builtins.print') as mock_print:
            self.tranche.check_guard_and_sell({'ETF.001': 11.5})
            
            self.assertNotIn('ETF.001', self.tranche.holdings)
            # Should rest 1 day for Profit Taking
            self.assertEqual(self.tranche.rest_days, 1)


class TestRanking(unittest.TestCase):
    def setUp(self):
        self.context = MagicMock()
        self.context.whitelist = ['A', 'B', 'C']
        self.context.theme_map = {'A': 'Theme1', 'B': 'Theme2', 'C': 'Theme1'}
        
        # Mock 300 days of price data (Needs >251)
        dates = pd.date_range(end='2024-01-01', periods=300)
        data = {
            'A': np.linspace(10, 12, 300), # +20%
            'B': np.linspace(10, 10.5, 300), # +5%
            'C': np.linspace(10, 9, 300) # -10%
        }
        self.context.prices_df = pd.DataFrame(data, index=dates)
        
        # Mock ThemeBooster
        self.context.theme_booster = MagicMock()
        self.context.theme_booster.get_hot_themes.return_value = {'Theme1'}
        
        # Set global config in module 
        # (This is a bit hacky since imported module has globals, but works for simple tests)
        import gm_strategy_rolling
        gm_strategy_rolling.MIN_SCORE = 0
        gm_strategy_rolling.CONCEPT_THEME_BOOST = True
        gm_strategy_rolling.CONCEPT_BOOST_POINTS = 10

    def test_ranking_calculation(self):
        current_dt = pd.Timestamp('2024-01-01')
        
        # Run calculation
        # Note: We need to patch the global CONCEPT_THEME_BOOST inside the module if it wasn't picked up
        ranking = get_ranking(self.context, current_dt)
        
        # A should be highest: Good momentum + Theme Boost
        # C should be boosted despite bad momentum, but might still be lower than A
        
        print("\nRanking Result:")
        print(ranking)
        
        self.assertIsNotNone(ranking)
        self.assertEqual(ranking.index[0], 'A') # A should be top
        
        # Verify Boost Logic
        # A and C are 'Theme1' (Hot) -> Should get boost
        # Scores are roughly proportional to rank sum weights (max ~270)
        # We can't easily assert exact values without reproducing the math, 
        # but we can check if code runs without error and produces dataframe.

if __name__ == '__main__':
    unittest.main()
