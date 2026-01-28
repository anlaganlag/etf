"""
牛市区间参数优化
时间段: 2024-09-01 至 2026-01-27 (约16个月)

测试最关键的参数组合:
Step 1: TOP_N × T
Step 2: 风控参数 (基于Step 1最优)
"""

import os
import re
import subprocess
import pandas as pd
import time
import concurrent.futures

# === 牛市时间段 ===
START_DATE = '2024-09-01 09:00:00'
END_DATE = '2026-01-27 16:00:00'

# === Step 1 参数网格 ===
TOP_N_LIST = [3, 4, 5, 6, 7, 8]
T_LIST = [6, 8, 10, 12, 13, 14, 16]

# === Step 2 风控参数 ===
STOP_LOSS_LIST = [0.05, 0.08, 0.10, 0.12]
TRAILING_TRIGGER_LIST = [0.06, 0.08, 0.10, 0.15]
TRAILING_DROP_LIST = [0.02, 0.03, 0.04]

MAX_WORKERS = 6
SOURCE_FILE = 'gm_strategy_rolling0.py'

def prepare_file_step1(top_n, t_period):
    filename = f'gm_bull_s1_{top_n}_{t_period}.py'
    state_file = f'bull_state_s1_{top_n}_{t_period}.json'
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {top_n}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {t_period}', content, flags=re.MULTILINE)
    content = re.sub(r"^START_DATE\s*=\s*['\"].*['\"]", f"START_DATE='{START_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r"^END_DATE\s*=\s*['\"].*['\"]", f"END_DATE='{END_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    content = re.sub(r"filename\s*=\s*['\"]gm_strategy_rolling0\.py['\"]", f"filename='{filename}'", content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return filename, state_file

def prepare_file_step2(top_n, t_period, sl, tt, td):
    sl_str = str(sl).replace('.', '')
    tt_str = str(tt).replace('.', '')
    td_str = str(td).replace('.', '')
    filename = f'gm_bull_s2_{sl_str}_{tt_str}_{td_str}.py'
    state_file = f'bull_state_s2_{sl_str}_{tt_str}_{td_str}.json'
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {top_n}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {t_period}', content, flags=re.MULTILINE)
    content = re.sub(r'^STOP_LOSS\s*=\s*[\d\.]+', f'STOP_LOSS = {sl}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_TRIGGER\s*=\s*[\d\.]+', f'TRAILING_TRIGGER = {tt}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_DROP\s*=\s*[\d\.]+', f'TRAILING_DROP = {td}', content, flags=re.MULTILINE)
    content = re.sub(r"^START_DATE\s*=\s*['\"].*['\"]", f"START_DATE='{START_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r"^END_DATE\s*=\s*['\"].*['\"]", f"END_DATE='{END_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    content = re.sub(r"filename\s*=\s*['\"]gm_strategy_rolling0\.py['\"]", f"filename='{filename}'", content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    return filename, state_file

def run_backtest(strategy_file, state_file):
    try:
        result = subprocess.run(['python', strategy_file], capture_output=True, text=True, encoding='utf-8', env=os.environ.copy())
        output = result.stdout
        
        ret_match = re.search(r'Return:\s*([\d\.\-]+)%', output)
        dd_match = re.search(r'Max DD:\s*([\d\.\-]+)%', output)
        sharpe_match = re.search(r'Sharpe:\s*([\d\.\-]+)', output)
        
        ret = float(ret_match.group(1)) if ret_match else 0.0
        dd = float(dd_match.group(1)) if dd_match else 0.0
        sharpe = float(sharpe_match.group(1)) if sharpe_match else 0.0
        calmar = ret / dd if dd > 0 else 0
        
        if os.path.exists(strategy_file): os.remove(strategy_file)
        if os.path.exists(state_file): os.remove(state_file)
        
        return ret, dd, sharpe, calmar
    except Exception as e:
        if os.path.exists(strategy_file): os.remove(strategy_file)
        if os.path.exists(state_file): os.remove(state_file)
        return 0, 0, 0, 0

def run_step1_task(args):
    top_n, t = args
    f, s = prepare_file_step1(top_n, t)
    ret, dd, sharpe, calmar = run_backtest(f, s)
    return {'TOP_N': top_n, 'T': t, 'Return': ret, 'MaxDD': dd, 'Sharpe': sharpe, 'Calmar': calmar}

def run_step2_task(args):
    top_n, t, sl, tt, td = args
    f, s = prepare_file_step2(top_n, t, sl, tt, td)
    ret, dd, sharpe, calmar = run_backtest(f, s)
    return {'STOP_LOSS': sl, 'TRAILING_TRIGGER': tt, 'TRAILING_DROP': td, 'Return': ret, 'MaxDD': dd, 'Sharpe': sharpe, 'Calmar': calmar}

def main():
    print(f"=== 牛市区间参数优化 ===")
    print(f"时间段: {START_DATE} 至 {END_DATE}")
    
    # === Step 1 ===
    print(f"\n--- Step 1: TOP_N × T ---")
    tasks1 = [(n, t) for n in TOP_N_LIST for t in T_LIST]
    results1 = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_step1_task, task): task for task in tasks1}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            data = future.result()
            results1.append(data)
            print(f"[{i+1}/{len(tasks1)}] N={data['TOP_N']} T={data['T']} -> R:{data['Return']}% DD:{data['MaxDD']}%")
    
    df1 = pd.DataFrame(results1).sort_values(by='Calmar', ascending=False)
    df1.to_csv('bull_optimization_step1.csv', index=False)
    
    print("\n🏆 Step 1 Top 5 (by Calmar):")
    print(df1.head(5).to_string(index=False))
    
    # 获取Step 1最优参数
    best = df1.iloc[0]
    best_top_n = int(best['TOP_N'])
    best_t = int(best['T'])
    print(f"\n✅ Step 1 最优: TOP_N={best_top_n}, T={best_t}")
    
    # === Step 2 ===
    print(f"\n--- Step 2: 风控参数 (基于 N={best_top_n}, T={best_t}) ---")
    tasks2 = [(best_top_n, best_t, sl, tt, td) for sl in STOP_LOSS_LIST for tt in TRAILING_TRIGGER_LIST for td in TRAILING_DROP_LIST if tt > td]
    results2 = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(run_step2_task, task): task for task in tasks2}
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            data = future.result()
            results2.append(data)
            print(f"[{i+1}/{len(tasks2)}] SL={data['STOP_LOSS']} TT={data['TRAILING_TRIGGER']} TD={data['TRAILING_DROP']} -> R:{data['Return']}%")
    
    df2 = pd.DataFrame(results2).sort_values(by='Calmar', ascending=False)
    df2.to_csv('bull_optimization_step2.csv', index=False)
    
    print("\n🏆 Step 2 Top 5 (by Calmar):")
    print(df2.head(5).to_string(index=False))
    
    # 最终最优
    final = df2.iloc[0]
    print(f"\n" + "="*50)
    print(f"🎯 牛市区间最优参数:")
    print(f"   TOP_N = {best_top_n}")
    print(f"   REBALANCE_PERIOD_T = {best_t}")
    print(f"   STOP_LOSS = {final['STOP_LOSS']}")
    print(f"   TRAILING_TRIGGER = {final['TRAILING_TRIGGER']}")
    print(f"   TRAILING_DROP = {final['TRAILING_DROP']}")
    print(f"\n   预期收益: {final['Return']}%")
    print(f"   最大回撤: {final['MaxDD']}%")
    print(f"   夏普比率: {final['Sharpe']}")
    print(f"   收益回撤比: {final['Calmar']:.2f}")
    print("="*50)

if __name__ == '__main__':
    main()
