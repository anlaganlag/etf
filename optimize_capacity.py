from __future__ import print_function, absolute_import
import pandas as pd
import numpy as np
import os
import sys
import subprocess
import re
from datetime import datetime

# === 配置参数范围 ===
# 针对大资金客户，测试 Top 3-6，以及更宽松的轮动周期 10-20
PARAM_GRID = {
    'TOP_N': [3, 4, 5, 6],
    'REBALANCE_PERIOD_T': [10, 15, 20]
}

# 保持防守型风控参数不变
ENV_VARS_BASE = {
    'GM_STOP_LOSS': '0.15',
    'GM_TRAILING_TRIGGER': '0.20',
    'GM_TRAILING_DROP': '0.05',
    # 使用全周期数据进行压力测试
    'GM_START_DATE': '2021-12-03 09:00:00',
    'GM_END_DATE': '2026-01-23 16:00:00'
}

RESULTS_FILE = 'capacity_optimization_results.csv'

def run_backtest(top_n, t_period):
    env = os.environ.copy()
    env.update(ENV_VARS_BASE)
    env['GM_TOP_N'] = str(top_n)
    env['GM_REBALANCE_T'] = str(t_period)
    
    cmd = [sys.executable, 'd:\\antigravity\\127\\etf\\gm_strategy_rolling0.py']
    
    print(f"Running: Top_N={top_n}, T={t_period} ...", end="", flush=True)
    
    try:
        # 运行回测并捕获输出
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd='d:\\antigravity\\127\\etf')
        output = result.stdout
        
        # 解析结果
        # === GM STANDARD REPORT (T+1 EXECUTION) ===
        # Return: 48.98%
        # Max DD: 32.17%
        # Sharpe: 0.75
        
        ret_match = re.search(r"Return:\s*([\d\.\-]+)%", output)
        dd_match = re.search(r"Max DD:\s*([\d\.\-]+)%", output)
        sharpe_match = re.search(r"Sharpe:\s*([\d\.\-]+)", output)
        
        ret = float(ret_match.group(1)) if ret_match else 0.0
        dd = float(dd_match.group(1)) if dd_match else 0.0
        sharpe = float(sharpe_match.group(1)) if sharpe_match else 0.0
        
        print(f" Done -> Ret: {ret}%, DD: {dd}%, Sharpe: {sharpe}")
        return ret, dd, sharpe
        
    except Exception as e:
        print(f" Error: {e}")
        return 0.0, 0.0, 0.0

def main():
    results = []
    
    # 遍历所有组合
    for t in PARAM_GRID['REBALANCE_PERIOD_T']:
        for n in PARAM_GRID['TOP_N']:
            ret, dd, sharpe = run_backtest(n, t)
            results.append({
                'T': t,
                'Top_N': n,
                'Return': ret,
                'MaxDD': dd,
                'Sharpe': sharpe,
                'Capacity_Score': t * n  # 极其粗略的容量打分 (越分散容量越大)
            })
    
    # 保存结果
    df = pd.DataFrame(results)
    df = df.sort_values(by=['Return', 'Sharpe'], ascending=False)
    print("\n=== 大资金容量优化结果 (按收益排序) ===")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_FILE, index=False)

if __name__ == "__main__":
    main()
