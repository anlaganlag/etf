import subprocess
import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Expanded range to test Structural Hypothesis
# Anchor: R20 = 150 (Trend)
# Question 1: Is R5 (Structure) Negative (Mean Reversion) or Positive (Trend Follow)?
# Question 2: Is R3 (Sentiment) Noise or Signal?

R1_RANGE = [30, 50, 70]
R3_RANGE = [-70, -30, 0]
R5_RANGE = [-70, -30, 0, 30] # Added positive test
# R20 fixed at 150

RESULTS_FILE = "weight_optimization_results.csv"
MAX_PARALLEL = 3

def run_backtest(r1, r3, r5):
    env = os.environ.copy()
    env["OPT_W_R1"] = str(r1)
    env["OPT_W_R3"] = str(r3)
    env["OPT_W_R5"] = str(r5)
    env["OPT_W_R20"] = "150" # Fixed
    
    # Ensure risk params are optimal
    env["OPT_STOP_LOSS"] = "0.30"
    env["OPT_TRAILING_TRIGGER"] = "0.15"
    env["OPT_TRAILING_DROP"] = "0.03"
    
    env["GM_MODE"] = "BACKTEST"
    env["GRID_SEARCH"] = "True"
    
    try:
        result = subprocess.run(
            ["python", "gm_strategy_rolling0.py"],
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8', 
            timeout=600 
        )
        
        output = result.stdout
        
        ret_match = re.search(r"Return: ([\d.-]+)%", output)
        mdd_match = re.search(r"Max DD: ([\d.-]+)%", output)
        sharpe_match = re.search(r"Sharpe: ([\d.-]+)", output)
        
        if ret_match and mdd_match and sharpe_match:
            ret = float(ret_match.group(1))
            mdd = float(mdd_match.group(1))
            sharpe = float(sharpe_match.group(1))
            return {"r1": r1, "r3": r3, "r5": r5, "return": ret, "mdd": mdd, "sharpe": sharpe}
        else:
            return None
    except:
        return None

def main():
    configs = []
    for r1 in R1_RANGE:
        for r3 in R3_RANGE:
            for r5 in R5_RANGE:
                configs.append((r1, r3, r5))

    results = []
    if os.path.exists(RESULTS_FILE):
        try:
            existing_df = pd.read_csv(RESULTS_FILE)
            results = existing_df.to_dict('records')
        except: pass

    completed_configs = set([(r['r1'], r['r3'], r['r5']) for r in results])
    to_run = [c for c in configs if c not in completed_configs]

    print(f"Total configs: {len(configs)}, To run: {len(to_run)}")
    
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
        future_to_config = {executor.submit(run_backtest, r1, r3, r5): (r1, r3, r5) for (r1, r3, r5) in to_run}
        
        count = 0
        for future in as_completed(future_to_config):
            count += 1
            r1, r3, r5 = future_to_config[future]
            res = future.result()
            
            if res:
                results.append(res)
                print(f"[{count}/{len(to_run)}] R1={r1}, R3={r3}, R5={r5} -> Ret: {res['return']}%, Shp: {res['sharpe']}")
                pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
            else:
                print(f"[{count}/{len(to_run)}] R1={r1}, R3={r3}, R5={r5} -> FAILED")

    if results:
        df = pd.DataFrame(results)
        best_ret = df.sort_values(by="return", ascending=False).iloc[0]
        
        print("\n--- WEIGHT OPTIMIZATION FINISHED ---")
        print(f"Best Return: {best_ret['return']}% (R1={best_ret['r1']}, R3={best_ret['r3']}, R5={best_ret['r5']})")

if __name__ == "__main__":
    main()
