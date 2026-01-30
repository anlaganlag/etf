import subprocess
import os
import re
import pandas as pd
import sys
import time

# 3D Grid Search for Risk Parameters
# Current Best: SL=0.20, Trigger=0.10, Drop=0.05

# 1. Stop Loss: Include 0.10 for Bear Market protection
sl_range = [0.10, 0.15, 0.20]

# 2. Trailing Trigger: [0.10, 0.15, 0.20]
# Use 999 to disable trailing stop (pure stop loss)
trigger_range = [0.10, 0.15, 0.20]

# 3. Trailing Drop: [0.05, 0.08, 0.10]
drop_range = [0.05, 0.08]

results = []
script_path = 'gm_strategy_rolling0.py'
cwd = os.path.dirname(os.path.abspath(__file__))

# Mode: 'RECENT' or 'FULL'
mode = os.environ.get('OPT_MODE', 'RECENT') 
if mode == 'FULL':
    start_date = '2021-12-03 09:00:00'
else:
    start_date = '2023-01-01 09:00:00'

# Inherit N=3, T=11 from previous optimization
top_n = '3'
rebalance_t = '11'

print(f"Starting Risk Optimization for {script_path}...")
print(f"Mode: {mode} | Period: {start_date} | N={top_n}, T={rebalance_t}")
print(f"SL: {sl_range}")
print(f"Trigger: {trigger_range}")
print(f"Drop: {drop_range}")
print("-" * 60)

start_time = time.time()
count = 0
total = len(sl_range) * len(trigger_range) * len(drop_range)

for sl in sl_range:
    for trig in trigger_range:
        for drop in drop_range:
            count += 1
            print(f"[{count}/{total}] SL={sl}, Trig={trig}, Drop={drop}...", end="\r")
            
            env = os.environ.copy()
            env['GM_TOP_N'] = top_n
            env['GM_REBALANCE_T'] = rebalance_t
            env['GM_START_DATE'] = start_date
            
            env['GM_STOP_LOSS'] = str(sl)
            env['GM_TRAILING_TRIGGER'] = str(trig)
            env['GM_TRAILING_DROP'] = str(drop)
            
            cmd = [sys.executable, script_path]
            
            try:
                result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
                output = result.stdout
                
                # Parse Standard Report (T+1)
                std_ret, std_dd, std_sharpe = 0, 0, 0
                std_match = re.search(r"=== GM STANDARD REPORT.*?Return:\s*([-+]?\d*\.?\d+)%.*?Max DD:\s*([-+]?\d*\.?\d+)%.*?Sharpe:\s*([-+]?\d*\.?\d+)", output, re.DOTALL)
                if std_match:
                    std_ret = float(std_match.group(1))
                    std_dd = float(std_match.group(2))
                    std_sharpe = float(std_match.group(3))
                    
                    results.append({
                        'SL': sl,
                        'Trig': trig,
                        'Drop': drop,
                        'Return': std_ret,
                        'MaxDD': std_dd,
                        'Sharpe': std_sharpe
                    })
                    print(f"[{count}/{total}] SL={sl}, Trig={trig}, Drop={drop} => Ret: {std_ret}%, DD: {std_dd}%, Sharpe: {std_sharpe}")
                else:
                    print(f"[{count}/{total}] SL={sl}, Trig={trig}, Drop={drop} => Failed to parse.")

            except Exception as e:
                print(f"   -> Exception: {e}")

elapsed = time.time() - start_time
print("-" * 60)
print(f"Risk Optimization finished in {elapsed/60:.2f} minutes.")

df = pd.DataFrame(results)
if not df.empty:
    # Sort by Sharpe Ratio (Risk Adjusted Return) as primary metric
    df_sorted = df.sort_values(by='Sharpe', ascending=False)
    print(f"\nRisk Optimization Results (Sorted by Sharpe):")
    print(df_sorted.head(10).to_markdown(index=False))
    
    # Also show max return
    print(f"\nTop Returns:")
    print(df_sorted.sort_values(by='Return', ascending=False).head(5).to_markdown(index=False))
    
    csv_path = f'risk_optimization_results_{mode.lower()}.csv'
    df_sorted.to_csv(os.path.join(cwd, csv_path), index=False)
else:
    print("No results found.")
