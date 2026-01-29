from gm.api import *
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from dotenv import load_dotenv
from config import config
import sys

load_dotenv()

# --- Search Ranges ---
TOP_N_RANGE = [4, 5, 6]
T_RANGE = [12, 13, 14]

def run_single_backtest(top_n, t_period):
    with open('gm_strategy_rolling0.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace('TOP_N = 5', f'TOP_N = {top_n}')
    content = content.replace('REBALANCE_PERIOD_T = 13', f'REBALANCE_PERIOD_T = {t_period}')
    
    temp_file = f'temp_bt_{top_n}_{t_period}.py'
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(content)
        
    try:
        res = run(strategy_id='d6d71d85-fb4c-11f0-99de-00ffda9d6e63', 
                  filename=temp_file, 
                  mode=MODE_BACKTEST,
                  token=os.getenv('MY_QUANT_TGM_TOKEN'), 
                  backtest_start_time='2021-12-03 09:00:00', 
                  backtest_end_time='2026-01-23 16:00:00',
                  backtest_adjust=ADJUST_PREV, 
                  backtest_initial_cash=1000000)
        
        if res:
            ind = res[0]
            ret = ind.get('pnl_ratio', 0) * 100
            mdd = ind.get('max_drawdown', 0) * 100
            shp = ind.get('sharp_ratio', 0)
            return ret, mdd, shp
    except Exception as e:
        print(f"Error running {top_n}, {t_period}: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists('rolling_state_simple.json'):
            os.remove('rolling_state_simple.json')
            
    return None

def main():
    results = {}
    
    for n in TOP_N_RANGE:
        for t in T_RANGE:
            print(f"Testing N={n}, T={t}...")
            res = run_single_backtest(n, t)
            if res:
                results[(n, t)] = res
                print(f"OK: {res[0]:.2f}% / {res[1]:.2f}% / {res[2]:.2f}")

    # Output Matrices
    print("\n=== NEIGHBORHOOD RETURN MATRIX (%) ===")
    header = "      " + "".join([f"T={t:<8}" for t in T_RANGE])
    print(header)
    for n in TOP_N_RANGE:
        row = f"N={n:<4} " + "".join([f"{results.get((n, t), (0,0,0))[0]:<10.2f}" for t in T_RANGE])
        print(row)

if __name__ == "__main__":
    main()
