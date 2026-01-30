import subprocess
import os
import re
import pandas as pd
import sys
import time

# Expanded Targeted Search
top_n_range = [3, 4, 5,6,7,8,9,10]
rebalance_t_range = [5, 6, 7, 8, 9, 10, 11, 12, 13]

results = []
script_path = 'gm_strategy_rolling0.py'
cwd = os.path.dirname(os.path.abspath(__file__))

# Mode: 'RECENT' or 'FULL'
mode = os.environ.get('OPT_MODE', 'RECENT')
if mode == 'FULL':
    start_date = '2021-12-03 09:00:00'
else:
    start_date = '2023-01-01 09:00:00'

print(f"Starting High-Res Optimization for {script_path}...")
print(f"Mode: {mode} | Period: {start_date} to Default End")
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
            result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
            output = result.stdout
            
            sim_ret, sim_dd, sim_sharpe = 0, 0, 0
            sim_match = re.search(r"=== SIMULATED REPORT.*?Return:\s*([-+]?\d*\.?\d+)%.*?Max DD:\s*([-+]?\d*\.?\d+)%.*?Sharpe:\s*([-+]?\d*\.?\d+)", output, re.DOTALL)
            if sim_match:
                sim_ret = float(sim_match.group(1))
                sim_dd = float(sim_match.group(2))
                sim_sharpe = float(sim_match.group(3))
            
            std_ret, std_dd, std_sharpe = 0, 0, 0
            std_match = re.search(r"=== GM STANDARD REPORT.*?Return:\s*([-+]?\d*\.?\d+)%.*?Max DD:\s*([-+]?\d*\.?\d+)%.*?Sharpe:\s*([-+]?\d*\.?\d+)", output, re.DOTALL)
            if std_match:
                std_ret = float(std_match.group(1))
                std_dd = float(std_match.group(2))
                std_sharpe = float(std_match.group(3))

            print(f"   -> Std(T+1): {std_ret}% | Sim(T0): {sim_ret}% | Sharpe: {std_sharpe}")
            
            results.append({
                'TOP_N': n,
                'T': t,
                'Std_Return': std_ret,
                'Std_MaxDD': std_dd,
                'Std_Sharpe': std_sharpe,
                'Sim_Return': sim_ret,
                'Sim_MaxDD': sim_dd,
                'Sim_Sharpe': sim_sharpe
            })

        except Exception as e:
            print(f"   -> Exception: {e}")

elapsed = time.time() - start_time
print("-" * 60)
print(f"Optimization finished in {elapsed/60:.2f} minutes.")

df = pd.DataFrame(results)
if not df.empty:
    df_sorted = df.sort_values(by='Std_Return', ascending=False)
    print(f"\nOptimization Results ({mode}):")
    print(df_sorted[['TOP_N', 'T', 'Std_Return', 'Std_MaxDD', 'Sim_Return', 'Sim_MaxDD']].to_markdown(index=False))
    
    csv_path = f'optimization_results_{mode.lower()}.csv'
    df_sorted.to_csv(os.path.join(cwd, csv_path), index=False)
else:
    print("No results found.")
