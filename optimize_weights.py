import subprocess
import os
import re
import pandas as pd
import json

# Define Archetypes
WEIGHT_CONFIGS = {
    "Long-Term (Baseline)": {1: 20, 3: 30, 5: 50, 10: 70, 20: 100},
    "Short-Term (Burst)":   {1: 100, 3: 70, 5: 50, 10: 30, 20: 10},
    "Balanced":             {1: 50, 3: 50, 5: 50, 10: 50, 20: 50},
    "Super-Long":           {1: 0, 3: 0, 5: 20, 10: 50, 20: 100},
    "Super-Short":          {1: 100, 3: 50, 5: 20, 10: 0, 20: 0},
    "Step-Up":              {1: 10, 3: 20, 5: 40, 10: 80, 20: 160}, # Steep long-term
    "Step-Down":            {1: 160, 3: 80, 5: 40, 10: 20, 20: 10}, # Steep short-term
    "Barbell":              {1: 100, 3: 0, 5: 0, 10: 0, 20: 100},   # R1 + R20 (Breakout + Trend)
    "Pure R20":             {1: 0, 3: 0, 5: 0, 10: 0, 20: 100},     # Only Long Term
    "Pure R1":              {1: 100, 3: 0, 5: 0, 10: 0, 20: 0},     # Only Short Term
    "Inverse Middle":       {1: 50, 3: -20, 5: -20, 10: 0, 20: 100},# Punish mid-term noise
}

# Output File
RESULTS_FILE = "optimization_results_weights.csv"

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
print("Starting Weight Optimization...")

for name, weights in WEIGHT_CONFIGS.items():
    print(f"Running Config: {name}...")
    
    env = os.environ.copy()
    env['GM_SCORING_WEIGHTS'] = json.dumps(weights)
    
    # Ensure optimal params from previous step are used
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
            'Config Name': name,
            'Weights': str(weights),
            'Return': metric['Return'],
            'MaxDD': metric['MaxDD'],
            'Sharpe': metric['Sharpe']
        })
        
    except Exception as e:
        print(f"Exception running {name}: {e}")

df = pd.DataFrame(results)
df.sort_values(by='Return', ascending=False, inplace=True)
df.to_csv(RESULTS_FILE, index=False)

print("\n=== Best Configurations ===")
print(df)
print(f"\nResults saved to {RESULTS_FILE}")
