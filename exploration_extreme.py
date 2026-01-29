"""
极致收益探索脚本
在进攻型配置(满仓+短期优先)基础上，测试:
1. 持仓数量 N = [3, 4, 5]
2. 轮动周期 T = [8, 10, 12, 14]
"""

import os
import re
import subprocess
import pandas as pd
from rich.console import Console
from rich.table import Table
import time

console = Console()

START_DATE = '2024-09-01 09:00:00'
END_DATE = '2026-01-27 16:00:00'

# 进攻型基准配置
BASE_CONFIG_AGGRESSIVE = {
    'STOP_LOSS': 0.05,
    'TRAILING_TRIGGER': 0.06,
    'TRAILING_DROP': 0.02,
    'DYNAMIC_POSITION': False,         # 满仓
    'MAX_PER_THEME': 1,
    'SCORING_WEIGHTS': 'SHORT_TERM'    # 短期优先
}

# 测试网格
N_LIST = [3, 4, 5]
T_LIST = [8, 10, 12, 14]

SOURCE_FILE = 'gm_strategy_rolling0.py'

def create_variant(n, t):
    safe_name = f"gm_extreme_n{n}_t{t}"
    filename = f"{safe_name}.py"
    state_file = f"{safe_name}.json"
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Apply Base Aggressive Config
    content = re.sub(r'^DYNAMIC_POSITION\s*=\s*(True|False)', 'DYNAMIC_POSITION = False', content, flags=re.MULTILINE)
    
    # Short Term Weights: replacing periods_rule
    # periods_rule = {1: 20, 3: 30, 5: 50, 10: 70, 20: 100} -> {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    new_rule = "periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}"
    content = re.sub(r'periods_rule\s*=\s*{.*?}', new_rule, content, flags=re.DOTALL)
    
    # 2. Apply N and T
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {n}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {t}', content, flags=re.MULTILINE)

    # 3. Update common setup
    content = re.sub(r"^START_DATE\s*=\s*['\"].*['\"]", f"START_DATE='{START_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r"^END_DATE\s*=\s*['\"].*['\"]", f"END_DATE='{END_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    content = re.sub(r"filename\s*=\s*['\"].*?['\"]", f"filename='{filename}'", content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return filename, state_file

def run_test(filename, state_file, n, t):
    try:
        if os.path.exists(state_file): os.remove(state_file)
        
        result = subprocess.run(['python', filename], capture_output=True, text=True, encoding='utf-8', env=os.environ.copy())
        output = result.stdout
        
        ret_match = re.search(r'Return:\s*([\d\.\-]+)%', output)
        dd_match = re.search(r'Max DD:\s*([\d\.\-]+)%', output)
        
        ret = float(ret_match.group(1)) if ret_match else 0.0
        dd = float(dd_match.group(1)) if dd_match else 0.0
        calmar = ret / dd if dd > 0 else 0
        
        return {'N': n, 'T': t, '收益率': ret, '最大回撤': dd, 'Calmar': calmar}
    except Exception as e:
        return {'N': n, 'T': t, '收益率': 0, '最大回撤': 0, 'Calmar': 0}
    finally:
        if os.path.exists(filename): os.remove(filename)
        if os.path.exists(state_file): os.remove(state_file)

def main():
    print(f"🚀 极致收益探索 (进攻模式)")
    print(f"时间段: {START_DATE} 至 {END_DATE}")
    print(f"固定配置: 满仓, 短期评分优先, SL=5%\n")
    
    results = []
    total = len(N_LIST) * len(T_LIST)
    count = 0
    
    for n in N_LIST:
        for t in T_LIST:
            count += 1
            print(f"[{count}/{total}] 测试: N={n}, T={t} ... ", end="", flush=True)
            fname, sfile = create_variant(n, t)
            res = run_test(fname, sfile, n, t)
            results.append(res)
            print(f"R: {res['收益率']}% / DD: {res['最大回撤']}%")
            
    # Display Results
    df = pd.DataFrame(results)
    df.to_csv('extreme_exploration_results.csv', index=False)
    
    print("\n=== 结果矩阵 ===")
    pivot = df.pivot(index='N', columns='T', values='收益率')
    print(pivot)
    
    best = df.loc[df['收益率'].idxmax()]
    print(f"\n🏆 最终冠军: N={int(best['N'])}, T={int(best['T'])}")
    print(f"💰 收益率: {best['收益率']}%")
    print(f"📉 最大回撤: {best['最大回撤']}%")

if __name__ == '__main__':
    main()
