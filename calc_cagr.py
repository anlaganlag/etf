
import pandas as pd
import numpy as np
from datetime import datetime

# Load previous results
try:
    df_res = pd.read_csv('output/data/backtest_result.csv')
    df_res['Date'] = pd.to_datetime(df_res['Date'])
    
    start_date = df_res['Date'].iloc[0]
    end_date = df_res['Date'].iloc[-1]
    days = (end_date - start_date).days
    years = days / 365.25
    
    print(f"Period: {start_date.date()} ~ {end_date.date()} ({days} days, {years:.2f} years)")
    
    def calc_cagr(series):
        total_ret = series.iloc[-1] / series.iloc[0] - 1
        cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
        return total_ret, cagr

    print("\nStrategy Performance:")
    for col in df_res.columns:
        if col in ['Date', 'Holdings']: continue
        if col not in df_res.columns: continue
        
        # Ensure numeric
        vals = pd.to_numeric(df_res[col], errors='coerce').fillna(1.0)
        
        ret, cagr = calc_cagr(vals)
        print(f"{col:<15} | Total: {ret*100:.1f}% | CAGR: {cagr*100:.1f}%")

except Exception as e:
    print(f"Error: {e}")
