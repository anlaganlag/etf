import subprocess
import os
import re
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Structural Gate Testing
# Testing if R5 Gate (e.g., > -0.05) improves result vs Baseline (Gate OFF / -1.0)
GATE_RANGE = [-1.0, -0.03, -0.05, -0.08, -0.10]
# -1.0 = Control (Current Best 45.2%)
# -0.03 = Strict
# -0.05 = Normal
# -0.08 = Loose
# -0.10 = Very Loose (Only stops crashes)

RESULTS_FILE = "structure_gate_results.csv"

def run_backtest(gate):
    env = os.environ.copy()
    env["OPT_R5_GATE"] = str(gate)
    
    # Ensure current best params are locked
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
            return {"gate": gate, "return": ret, "mdd": mdd, "sharpe": sharpe}
        else:
            return None
    except:
        return None

def main():
    print(f"Testing Structural Gates: {GATE_RANGE}")
    
    results = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_gate = {executor.submit(run_backtest, g): g for g in GATE_RANGE}
        
        for future in as_completed(future_to_gate):
            gate = future_to_gate[future]
            res = future.result()
            
            if res:
                results.append(res)
                print(f"Gate={gate} -> Ret: {res['return']}%, Mdd: {res['mdd']}%, Shp: {res['sharpe']}")
                pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
            else:
                print(f"Gate={gate} -> FAILED")

    if results:
        df = pd.DataFrame(results)
        best_ret = df.sort_values(by="return", ascending=False).iloc[0]
        
        print("\n--- GATE OPTIMIZATION FINISHED ---")
        print(f"Best Return: {best_ret['return']}% (Gate={best_ret['gate']})")

if __name__ == "__main__":
    main()
