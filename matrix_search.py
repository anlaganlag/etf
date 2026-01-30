from gm.api import *
import pandas as pd
import numpy as np
import os
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from config import config
import sys

load_dotenv()

# --- Search Ranges ---
TOP_N_RANGE = [5,6,7,8,9,10]
T_RANGE = [10,11,12,13,14,15]

# --- Test Period (Bull Market) ---
TEST_START = '2024-09-01 09:00:00'
TEST_END = '2026-01-23 16:00:00'

def run_single_backtest(top_n, t_period):
    with open('gm_strategy_rolling0.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Robust regex replacement - matches any existing value
    content = re.sub(r'TOP_N\s*=\s*\d+', f'TOP_N = {top_n}', content)
    content = re.sub(r'REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {t_period}', content)
    
    # Write to temp file
    temp_file = f'temp_bt_{top_n}_{t_period}.py'
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        res = run(strategy_id='0137c2ac-fd82-11f0-ae68-00ffda9d6e63', 
                  filename=temp_file, 
                  mode=MODE_BACKTEST,
                  token=os.getenv('MY_QUANT_TGM_TOKEN'), 
                  backtest_start_time=TEST_START, 
                  backtest_end_time=TEST_END,
                  backtest_adjust=ADJUST_PREV, 
                  backtest_initial_cash=1000000)
        
        # sys.stdout = original_stdout
        
        if res:
            # res is a list of dicts/account objects
            ind = res[0] # The account indicator
            ret = ind.get('pnl_ratio', 0) * 100
            mdd = ind.get('max_drawdown', 0) * 100
            shp = ind.get('sharp_ratio', 0)
            return ret, mdd, shp
    except Exception as e:
        # sys.stdout = original_stdout
        print(f"Error running {top_n}, {t_period}: {e}")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists('rolling_state_simple.json'):
            os.remove('rolling_state_simple.json')
            
    return None

def main():
    results = {}
    
    print(f"{'TOP_N':<6} {'T':<6} {'Return':<10} {'MaxDD':<10} {'Sharpe':<10}")
    
    # We'll save results as we go to avoid losing them
    save_file = 'neighborhood_results.json'
    if os.path.exists(save_file):
        with open(save_file, 'r') as f:
            data = json.load(f)
            # Reconstruct from dict keys which are strings "n,t"
            for k, v in data.items():
                n_str, t_str = k.split(',')
                results[(int(n_str), int(t_str))] = v

    for n in TOP_N_RANGE:
        for t in T_RANGE:
            if (n, t) in results:
                res = results[(n, t)]
                print(f"{n:<6} {t:<6} {res[0]:<10.2f} {res[1]:<10.2f} {res[2]:<10.2f} (Cached)")
                continue
                
            print(f"Testing N={n}, T={t}...", end=' ', flush=True)
            res = run_single_backtest(n, t)
            if res:
                results[(n, t)] = res
                print(f"Done: {res[0]:.2f}% / {res[1]:.2f}% / {res[2]:.2f}")
                
                # Save progress
                serializable_results = {f"{k[0]},{k[1]}": v for k, v in results.items()}
                with open(save_file, 'w') as f:
                    json.dump(serializable_results, f)
            else:
                print("Failed")

    # Output Matrices
    print("\n=== RETURN MATRIX (%) ===")
    header = "      " + "".join([f"T={t:<8}" for t in T_RANGE])
    print(header)
    for n in TOP_N_RANGE:
        row = f"N={n:<4} " + "".join([f"{results.get((n, t), (0,0,0))[0]:<10.2f}" for t in T_RANGE])
        print(row)

    print("\n=== MAX DRAWDOWN MATRIX (%) ===")
    print(header)
    for n in TOP_N_RANGE:
        row = f"N={n:<4} " + "".join([f"{results.get((n, t), (0,0,0))[1]:<10.2f}" for t in T_RANGE])
        print(row)

    print("\n=== SHARPE MATRIX ===")
    print(header)
    for n in TOP_N_RANGE:
        row = f"N={n:<4} " + "".join([f"{results.get((n, t), (0,0,0))[2]:<10.2f}" for t in T_RANGE])
        print(row)

if __name__ == "__main__":
    main()
