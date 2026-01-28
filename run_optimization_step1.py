
import os
import re
import subprocess
import pandas as pd
import time
import concurrent.futures
from datetime import datetime

# === STEP 1 参数网格 ===
TOP_N_LIST = [3, 4, 5, 6, 7, 8, 9, 10]
T_LIST = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
MAX_WORKERS = 6

SOURCE_FILE = 'gm_strategy_rolling0.py'
RESULT_FILE = 'optimization_step1_b27.csv'

def prepare_strategy_file(top_n, t_period):
    """创建并返回临时且唯一的策略文件路径"""
    filename = f'gm_strategy_rolling0_opt_{top_n}_{t_period}.py'
    state_file = f'rolling_state_simple_{top_n}_{t_period}.json'
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换参数
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {top_n}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {t_period}', content, flags=re.MULTILINE)
    
    # 替换 State File 为唯一文件
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    
    # 【修复】替换 run() 中的 filename 参数
    content = re.sub(r"filename\s*=\s*['\"]gm_strategy_rolling0\.py['\"]", f"filename='{filename}'", content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return filename, state_file

def run_task(args):
    top_n, t_period = args
    strategy_file, state_file = prepare_strategy_file(top_n, t_period)
    
    try:
        # print(f"Starts: TOP_N={top_n}, T={t_period}")
        result = subprocess.run(
            ['python', strategy_file], 
            capture_output=True, 
            text=True,
            encoding='utf-8',
            env=os.environ.copy()
        )
        output = result.stdout
        
        # Parse
        ret_match = re.search(r'Return:\s*([\d\.\-]+)%', output)
        dd_match = re.search(r'Max DD:\s*([\d\.\-]+)%', output)
        sharpe_match = re.search(r'Sharpe:\s*([\d\.\-]+)', output)
        
        ret = float(ret_match.group(1)) if ret_match else 0.0
        dd = float(dd_match.group(1)) if dd_match else 0.0
        sharpe = float(sharpe_match.group(1)) if sharpe_match else 0.0
        calmar = ret / dd if dd > 0 else 0
        
        # Cleanup
        if os.path.exists(strategy_file): os.remove(strategy_file)
        if os.path.exists(state_file): os.remove(state_file)
        if os.path.exists(state_file + '.tmp'): os.remove(state_file + '.tmp') # cleanup temp if any
        
        return {
            'TOP_N': top_n,
            'T': t_period,
            'Return': ret,
            'MaxDD': dd,
            'Sharpe': sharpe,
            'Calmar': calmar,
            'Status': 'OK'
        }
    except Exception as e:
        return {
            'TOP_N': top_n,
            'T': t_period,
            'Return': 0,
            'MaxDD': 0,
            'Sharpe': 0,
            'Calmar': 0,
            'Status': str(e)
        }

def main():
    print(f"=== 开始参数优化 Step 1 (Parallel={MAX_WORKERS}) ===")
    tasks = []
    for top_n in TOP_N_LIST:
        for t_val in T_LIST:
            tasks.append((top_n, t_val))
    
    total = len(tasks)
    finished = 0
    results = []
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in tasks}
        
        print(f"Submitted {total} tasks. Waiting for results...")
        
        for future in concurrent.futures.as_completed(future_to_task):
            finished += 1
            data = future.result()
            results.append(data)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / finished
            eta = avg_time * (total - finished)
            
            print(f"[{finished}/{total}] T={data['T']} N={data['TOP_N']} -> R:{data['Return']}% DD:{data['MaxDD']}% S:{data['Sharpe']} (ETA: {eta/60:.1f}m)")

    # Save
    df = pd.DataFrame(results)
    df.to_csv(RESULT_FILE, index=False)
    
    print("\n=== 优化完成 ===")
    print(df.sort_values(by='Return', ascending=False).head(10).to_string(index=False))

if __name__ == '__main__':
    main()
