import subprocess
import os
import re
import pandas as pd
import json
import itertools

# Grid Search Space for "Inverse Middle"
# Hypothesis: Score = w1*R1 + wMid*(R3+R5) + w20*R20
# wMid should be negative to reward pullbacks (or punish extensions)

R1_WEIGHTS = [50, 100, 150]
MID_WEIGHTS = [-10, -30, -50, -70] # R3 and R5 will both get this weight
R20_WEIGHTS = [100, 150, 200]

# Output File
RESULTS_FILE = "optimization_results_inverse.csv"

def parse_output(output_str):
    res = {'Return': 0.0, 'MaxDD': 0.0, 'Sharpe': 0.0}
    try:
        if "=== SIMULATED REPORT" not in output_str:
            return res
        sim_section = output_str.split("=== SIMULATED REPORT")[-1]
        
        ret_match = re.search(r"Return:\s*([\d\.\-]+)%", sim_section)
        dd_match = re.search(r"Max DD:\s*([\d\.\-]+)%", sim_section)
        sharpe_match = re.search(r"Sharpe:\s*([\d\.\-]+)", sim_section)
        
        if ret_match: res['Return'] = float(ret_match.group(1))
        if dd_match: res['MaxDD'] = float(dd_match.group(1))
        if sharpe_match: res['Sharpe'] = float(sharpe_match.group(1))
    except Exception as e:
        print(f"Error parsing output: {e}")
    return res

results = []
print("Starting Inverse Middle Optimization...")

total_runs = len(R1_WEIGHTS) * len(MID_WEIGHTS) * len(R20_WEIGHTS)
curr_run = 0

for r1, mid, r20 in itertools.product(R1_WEIGHTS, MID_WEIGHTS, R20_WEIGHTS):
    curr_run += 1
    
    # Construct weights dict
    # R10 is set to 0 to separate mid-term from long-term
    weights = {1: r1, 3: mid, 5: mid, 10: 0, 20: r20}
    name = f"Inverse_{r1}_{mid}_{r20}"
    
    print(f"[{curr_run}/{total_runs}] Running {name}: {weights}...")
    
    env = os.environ.copy()
    env['GM_SCORING_WEIGHTS'] = json.dumps(weights)
    env['GM_TOP_N'] = "3"
    env['GM_REBALANCE_T'] = "10"
    
    try:
        process = subprocess.run(
            ['python', 'gm_strategy_rolling0.py'], 
            env=env, 
            capture_output=True, 
            text=True,
            cwd=os.getcwd()
        )
        
        if process.returncode != 0:
            print(f"Error running {name}: {process.stderr[-200:]}")
            metric = {'Return': -999, 'MaxDD': 0, 'Sharpe': 0}
        else:
            metric = parse_output(process.stdout)
            
        print(f"  Result: Return={metric['Return']}%, Sharpe={metric['Sharpe']}")
        
        results.append({
            'Config': name,
            'R1': r1,
            'Mid': mid,
            'R20': r20,
            'Weights': str(weights),
            'Return': metric['Return'],
            'Sharpe': metric['Sharpe']
        })
        
    except Exception as e:
        print(f"Exception running {name}: {e}")

df = pd.DataFrame(results)
df.sort_values(by='Return', ascending=False, inplace=True)
df.to_csv(RESULTS_FILE, index=False)

print("\n=== Top 5 Configurations ===")
print(df.head())
print(f"\nResults saved to {RESULTS_FILE}")
