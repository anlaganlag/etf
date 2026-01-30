import subprocess
import os
import re
import pandas as pd
import itertools

# Search Space
TOP_N_LIST = [1, 2, 3, 4, 6]
REBALANCE_T_LIST = [5, 7, 10, 13, 15]

# Output File
RESULTS_FILE = "optimization_results_matrix.csv"

def parse_output(output_str):
    """Extracts Return, Max DD, Sharpe from the simulation output."""
    res = {'Return': 0.0, 'MaxDD': 0.0, 'Sharpe': 0.0}
    
    # Needs to match: 
    # === SIMULATED REPORT (T-CLOSE EXECUTION / LIVE PROXY) ===
    # Return: 26.54%
    # Max DD: -19.88%
    # Sharpe: 0.42
    
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

print(f"Starting Matrix Optimization...")
print(f"N: {TOP_N_LIST}")
print(f"T: {REBALANCE_T_LIST}")

total_runs = len(TOP_N_LIST) * len(REBALANCE_T_LIST)
curr_run = 0

for n, t in itertools.product(TOP_N_LIST, REBALANCE_T_LIST):
    curr_run += 1
    print(f"[{curr_run}/{total_runs}] Running N={n}, T={t}...")
    
    env = os.environ.copy()
    env['GM_TOP_N'] = str(n)
    env['GM_REBALANCE_T'] = str(t)
    
    try:
        # Run the strategy
        # Capture stdout
        process = subprocess.run(
            ['python', 'gm_strategy_rolling0.py'], 
            env=env, 
            capture_output=True, 
            text=True,
            cwd=os.getcwd()
        )
        
        if process.returncode != 0:
            print(f"Error running N={n}, T={t}: {process.stderr[-200:]}")
            metric = {'Return': -999, 'MaxDD': 0, 'Sharpe': 0}
        else:
            metric = parse_output(process.stdout)
            
        print(f"  Result: Return={metric['Return']}%, Sharpe={metric['Sharpe']}")
        
        results.append({
            'TOP_N': n,
            'REBALANCE_T': t,
            'Return': metric['Return'],
            'MaxDD': metric['MaxDD'],
            'Sharpe': metric['Sharpe']
        })
        
    except Exception as e:
        print(f"Exception running N={n}, T={t}: {e}")

# Save results
df = pd.DataFrame(results)
df.sort_values(by='Return', ascending=False, inplace=True)
df.to_csv(RESULTS_FILE, index=False)

print("\n=== Top 5 Configurations ===")
print(df.head())
print(f"\nResults saved to {RESULTS_FILE}")
