
import pandas as pd
import numpy as np

try:
    df = pd.read_csv('output/data/market_comparison_fast.csv')
    # This csv only has summary stats? No, wait, backtest_compare_fast.py saves summary to CSV?
    # Let's check code of backtest_compare_fast.py
    # Lines 338: df_res.to_csv(..., "market_comparison_fast.csv") -> This saves summary.
    # I don't have the daily time series for the comparison strategies saved!
    # Ah, I only saved valid_dates vs vals in the PLOT, but didn't save the time series to CSV in `backtest_compare_fast.py`.
    # I only printed it.
    
    # Re-running is fast. I will re-run `backtest_compare_fast.py` BUT modify it to save the daily values first?
    # Or just use the summary total return to calc CAGR.
    # Total Return is known: 114.55%
    # Start: 2024-09-01. End: 2026-01-24. 
    # Days = 510. Years = 510/365.25 = 1.396.
    
    years = 1.396
    
    print("Strategy | Total Return | CAGR")
    for idx, row in df.iterrows():
        name = row['Strategy']
        total_ret = row['Return'] # in %
        final_val = 1 + total_ret/100
        cagr = (final_val ** (1/years)) - 1
        print(f"{name:<15} | {total_ret:>6.2f}% | {cagr*100:.2f}%")
        
except Exception as e:
    print(f"Error: {e}")
