
import subprocess
import os
import itertools
import pandas as pd
import re
import queue
import threading

# Configuration
SCRIPT_PATH = "gm_strategy_rolling0.py"
STATE_FILE = "rolling_state_simple.json"
OUTPUT_FILE = "exploration_results_v2.csv"

# Search Space
# 1. Baseline: T=10, SL=0.20, Trig=0.15 (Current)
# 2. High Trend (A): Trig=0.25/0.30/0.40, Drop=0.08, SL=0.20
# 3. Fast Band (B): T=8, SL=0.10, Trig=0.10, Drop=0.03
# 4. Balanced: T=10, SL=0.15, Trig=0.20

params_grid = {
    'OPT_TOP_N': [4], # Keep constant for now
    'OPT_T': [8, 10], 
    'OPT_SL': [0.10, 0.15, 0.20],
    'OPT_TRIG': [0.10, 0.20, 0.25], 
    'OPT_DROP': [0.03, 0.05, 0.08]
}

# Generate Combinations
keys = list(params_grid.keys())
values = list(params_grid.values())
combinations = list(itertools.product(*values))

# Specific Interest Combinations (Pruning the search space for speed)
# Only keep logical combinations (e.g. Trig > Drop, Trigger > 0.05)
filtered_combs = []
for c in combinations:
    d = dict(zip(keys, c))
    # Logic Checks
    if d['OPT_TRIG'] <= d['OPT_DROP']: continue # Invalid
    
    # Filter for Strategy A (Trend) logic: Wide TP needs Wide Drop
    if d['OPT_TRIG'] >= 0.20 and d['OPT_DROP'] < 0.05: continue
    
    # Filter for Strategy B (Fast) logic: Tight TP needs Tight SL/Drop
    if d['OPT_TRIG'] <= 0.15 and d['OPT_DROP'] > 0.05: continue
    
    filtered_combs.append(d)

print(f"Generated {len(filtered_combs)} valid parameter sets.")

results = []

def run_strategy(params):
    # 1. Clean State
    if os.path.exists(STATE_FILE):
        try: os.remove(STATE_FILE)
        except: pass
        
    # 2. Set Env
    env = os.environ.copy()
    for k, v in params.items():
        env[k] = str(v)
    
    # 3. Run
    # print(f"👉 Running: {params}")
    try:
        # Run with timeout to prevent hang
        result = subprocess.run(
            ['python', SCRIPT_PATH], 
            env=env, 
            capture_output=True, 
            text=True, 
            encoding='utf-8', 
            cwd=os.getcwd()
        )
        output = result.stdout
        
        # 4. Parse Result
        # Look for "Return: 31.75%" etc
        ret_match = re.search(r'Return:\s*(-?\d+\.?\d*)%', output)
        dd_match = re.search(r'Max DD:\s*(-?\d+\.?\d*)%', output)
        sharpe_match = re.search(r'Sharpe:\s*(-?\d+\.?\d*)', output)
        
        ret = float(ret_match.group(1)) if ret_match else -999
        dd = float(dd_match.group(1)) if dd_match else 999
        sharpe = float(sharpe_match.group(1)) if sharpe_match else -999
        
        res = params.copy()
        res.update({
            'Return': ret,
            'MaxDD': dd,
            'Sharpe': sharpe
        })
        print(f"✅ {params} -> Ret: {ret}%, DD: {dd}%, Sharpe: {sharpe}")
        return res
        
    except Exception as e:
        print(f"❌ Error running {params}: {e}")
        return None

# Execution (Sequential for safety, as GM SDK might have singleton issues)
for i, params in enumerate(filtered_combs):
    print(f"[{i+1}/{len(filtered_combs)}] Testing...")
    res = run_strategy(params)
    if res:
        results.append(res)
        # Save intermediate
        pd.DataFrame(results).sort_values('Return', ascending=False).to_csv(OUTPUT_FILE, index=False)

print("\n=== TOP 5 RESULTS ===")
pd.DataFrame(results).sort_values('Return', ascending=False).head(5).to_markdown()
