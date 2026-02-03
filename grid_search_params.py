import subprocess
import os
import re
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parameter ranges (Optimized for first quick pass)
STOP_LOSS_RANGE = [0.10, 0.20, 0.30]
TRAILING_TRIGGER_RANGE = [0.10, 0.15, 0.20]
TRAILING_DROP_RANGE = [0.03, 0.05, 0.08]

RESULTS_FILE = "risk_grid_search_results.csv"
MAX_PARALLEL = 3 # Run 3 backtests in parallel

def run_backtest(sl, trig, drop):
    env = os.environ.copy()
    env["OPT_STOP_LOSS"] = str(sl)
    env["OPT_TRAILING_TRIGGER"] = str(trig)
    env["OPT_TRAILING_DROP"] = str(drop)
    env["GM_MODE"] = "BACKTEST"
    env["GRID_SEARCH"] = "True"
    
    # print(f"Starting: SL={sl}, Trig={trig}, Drop={drop}")
    
    try:
        # Run the strategy script and capture output
        result = subprocess.run(
            ["python", "gm_strategy_rolling0.py"],
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8', 
            timeout=600 
        )
        
        output = result.stdout
        
        # Regex to extract results
        ret_match = re.search(r"Return: ([\d.-]+)%", output)
        mdd_match = re.search(r"Max DD: ([\d.-]+)%", output)
        sharpe_match = re.search(r"Sharpe: ([\d.-]+)", output)
        
        if ret_match and mdd_match and sharpe_match:
            ret = float(ret_match.group(1))
            mdd = float(mdd_match.group(1))
            sharpe = float(sharpe_match.group(1))
            return {"sl": sl, "trig": trig, "drop": drop, "return": ret, "mdd": mdd, "sharpe": sharpe}
        else:
            return None
            
    except Exception as e:
        return None

def main():
    configs = []
    for sl in STOP_LOSS_RANGE:
        for trig in TRAILING_TRIGGER_RANGE:
            for drop in TRAILING_DROP_RANGE:
                configs.append((sl, trig, drop))

    results = []
    if os.path.exists(RESULTS_FILE):
        try:
            existing_df = pd.read_csv(RESULTS_FILE)
            if all(col in existing_df.columns for col in ['sl', 'trig', 'drop']):
                results = existing_df.to_dict('records')
        except: pass

    completed_configs = set([(r['sl'], r['trig'], r['drop']) for r in results])
    to_run = [c for c in configs if c not in completed_configs]

    print(f"Total configs: {len(configs)}, To run: {len(to_run)}")
    
    if not to_run:
        print("All configs already completed.")
    else:
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL) as executor:
            future_to_config = {executor.submit(run_backtest, sl, trig, drop): (sl, trig, drop) for (sl, trig, drop) in to_run}
            
            count = 0
            for future in as_completed(future_to_config):
                count += 1
                sl, trig, drop = future_to_config[future]
                res = future.result()
                if res:
                    results.append(res)
                    print(f"[{count}/{len(to_run)}] SL={sl}, Trig={trig}, Drop={drop} -> Return: {res['return']}%, Sharpe: {res['sharpe']}")
                    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)
                else:
                    print(f"[{count}/{len(to_run)}] SL={sl}, Trig={trig}, Drop={drop} -> FAILED")

    if results:
        df = pd.DataFrame(results)
        best_return = df.sort_values(by="return", ascending=False).iloc[0]
        best_sharpe = df.sort_values(by="sharpe", ascending=False).iloc[0]
        
        print("\n--- GRID SEARCH FINISHED ---")
        print(f"Best Return: {best_return['return']}% (SL={best_return['sl']}, Trig={best_return['trig']}, Drop={best_return['drop']}, Sharpe={best_return['sharpe']})")
        print(f"Best Sharpe: {best_sharpe['sharpe']} (SL={best_sharpe['sl']}, Trig={best_sharpe['trig']}, Drop={best_sharpe['drop']}, Return={best_sharpe['return']}%)")

if __name__ == "__main__":
    main()
