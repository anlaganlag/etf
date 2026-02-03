import subprocess
import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Dynamic Gate K-Sigma Calculation
# Testing K range from 1.5 to 3.0
# Hypothesis: -8% corresponded roughly to 2.0-2.5 sigma for standard ETFs.
K_RANGE = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]

RESULTS_FILE = "k_sigma_results.csv"

def run_backtest(k):
    env = os.environ.copy()
    env["OPT_R5_K"] = str(k)
    
    # Ensure all other params are locked to best
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
            return {"k": k, "return": ret, "mdd": mdd, "sharpe": sharpe}
        else:
            return None
    except:
        return None

def main():
    print(f"Testing Dynamic Gate K-Sigma: {K_RANGE}")
    
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_k = {executor.submit(run_backtest, k): k for k in K_RANGE}
        
        for future in as_completed(future_to_k):
            k = future_to_k[future]
            res = future.result()
            
            if res:
                results.append(res)
                print(f"K={k} -> Ret: {res['return']}%, Mdd: {res['mdd']}%, Shp: {res['sharpe']}")
                pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
            else:
                print(f"K={k} -> FAILED")

    if results:
        df = pd.DataFrame(results)
        best_ret = df.sort_values(by="return", ascending=False).iloc[0]
        
        print("\n--- K OPTIMIZATION FINISHED ---")
        print(f"Best Return: {best_ret['return']}% (K={best_ret['k']})")

if __name__ == "__main__":
    main()
