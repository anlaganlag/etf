import subprocess
import os
import re
import pandas as pd
import sys
import time

# Define parameter ranges to test
top_n_range = [2, 3, 4, 5, 8]
rebalance_t_range = [5, 10, 13, 15, 20]

results = []

script_path = 'gm_strategy_rolling0.py'
cwd = os.path.dirname(os.path.abspath(__file__))

# Speed up by using recent data (e.g. 2023 onwards)
start_date = '2023-01-01 09:00:00'

print(f"Starting optimization for {script_path}...")
print(f"Period: {start_date} to Default End")
print(f"Testing TOP_N: {top_n_range}")
print(f"Testing REBALANCE_PERIOD_T: {rebalance_t_range}")
print("-" * 60)

start_time = time.time()
count = 0
total = len(top_n_range) * len(rebalance_t_range)

for n in top_n_range:
    for t in rebalance_t_range:
        count += 1
        print(f"[{count}/{total}] Running TOP_N={n}, T={t}...")
        sys.stdout.flush()
        
        env = os.environ.copy()
        env['GM_TOP_N'] = str(n)
        env['GM_REBALANCE_T'] = str(t)
        env['GM_START_DATE'] = start_date
        
        cmd = [sys.executable, script_path]
        
        try:
            # Run and capture output
            result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
            output = result.stdout
            
            # Check for errors
            if result.returncode != 0:
                print(f"Error running (N={n}, T={t}): Return Code {result.returncode}")
                # print(result.stderr[-300:])
                continue

            # Parse output
            sim_report_match = re.search(r"=== SIMULATED REPORT.*?Return:\s*([-+]?\d*\.?\d+)%.*?Max DD:\s*([-+]?\d*\.?\d+)%.*?Sharpe:\s*([-+]?\d*\.?\d+)", output, re.DOTALL)
            
            if sim_report_match:
                ret = float(sim_report_match.group(1))
                dd = float(sim_report_match.group(2))
                sharpe = float(sim_report_match.group(3))
                
                results.append({
                    'TOP_N': n,
                    'T': t,
                    'Return': ret,
                    'MaxDD': dd,
                    'Sharpe': sharpe
                })
                print(f"   -> Return: {ret}%, MaxDD: {dd}%, Sharpe: {sharpe}")
            else:
                std_report_match = re.search(r"=== GM STANDARD REPORT.*?Return:\s*([-+]?\d*\.?\d+)%.*?Max DD:\s*([-+]?\d*\.?\d+)%.*?Sharpe:\s*([-+]?\d*\.?\d+)", output, re.DOTALL)
                if std_report_match:
                    ret = float(std_report_match.group(1))
                    dd = float(std_report_match.group(2))
                    sharpe = float(std_report_match.group(3))
                    results.append({
                        'TOP_N': n,
                        'T': t,
                        'Return': ret,
                        'MaxDD': dd,
                        'Sharpe': sharpe,
                        'Note': 'Standard Report'
                    })
                    print(f"   -> Return: {ret}%, MaxDD: {dd}%, Sharpe: {sharpe} (Std)")
                else:
                    print(f"   -> Failed to parse results.")

        except Exception as e:
            print(f"   -> Exception: {e}")

elapsed = time.time() - start_time
print("-" * 60)
print(f"Optimization finished in {elapsed/60:.2f} minutes.")

# Create DataFrame and sort
df = pd.DataFrame(results)
if not df.empty:
    df_sorted = df.sort_values(by='Return', ascending=False)
    
    print("\nOptimization Results (Sorted by Return):")
    print(df_sorted.to_markdown(index=False))
    
    best = df_sorted.iloc[0]
    print(f"\nBest Parameters: TOP_N={best['TOP_N']}, T={best['T']}")
    
    csv_path = 'optimization_results.csv'
    df_sorted.to_csv(os.path.join(cwd, csv_path), index=False)
else:
    print("No results found.")
