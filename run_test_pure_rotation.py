"""
纯轮动 vs 止盈止损 对比测试
纯轮动 = 禁用止盈止损，持仓至换仓日
"""

import os
import re
import subprocess
import pandas as pd

START_DATE = '2024-09-01 09:00:00'
END_DATE = '2026-01-27 16:00:00'

# 测试不同的 TOP_N 和 T 组合
CONFIGS = [
    # 名称, N, T, SL, TT, TD
    ("当前最优(N=8,T=10)", 8, 10, 0.05, 0.06, 0.02),
    ("纯轮动(N=8,T=10)", 8, 10, 0.99, 0.99, 0.99),
    ("纯轮动(N=5,T=13)", 5, 13, 0.99, 0.99, 0.99),
    ("纯轮动(N=6,T=10)", 6, 10, 0.99, 0.99, 0.99),
    ("纯轮动(N=4,T=8)", 4, 8, 0.99, 0.99, 0.99),  # 更激进
    ("纯轮动(N=3,T=5)", 3, 5, 0.99, 0.99, 0.99),  # 极端集中
]

SOURCE_FILE = 'gm_strategy_rolling0.py'

def prepare_and_run(config):
    name, n, t, sl, tt, td = config
    safe_name = name.replace('(', '_').replace(')', '_').replace(',', '_').replace('=', '')
    filename = f'gm_pure_{safe_name}.py'
    state_file = f'state_pure_{safe_name}.json'
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {n}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {t}', content, flags=re.MULTILINE)
    content = re.sub(r'^STOP_LOSS\s*=\s*[\d\.]+', f'STOP_LOSS = {sl}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_TRIGGER\s*=\s*[\d\.]+', f'TRAILING_TRIGGER = {tt}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_DROP\s*=\s*[\d\.]+', f'TRAILING_DROP = {td}', content, flags=re.MULTILINE)
    content = re.sub(r"^START_DATE\s*=\s*['\"].*['\"]", f"START_DATE='{START_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r"^END_DATE\s*=\s*['\"].*['\"]", f"END_DATE='{END_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    content = re.sub(r"filename\s*=\s*['\"]gm_strategy_rolling0\.py['\"]", f"filename='{filename}'", content)
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    
    try:
        result = subprocess.run(['python', filename], capture_output=True, text=True, encoding='utf-8', env=os.environ.copy())
        output = result.stdout
        
        ret_match = re.search(r'Return:\s*([\d\.\-]+)%', output)
        dd_match = re.search(r'Max DD:\s*([\d\.\-]+)%', output)
        sharpe_match = re.search(r'Sharpe:\s*([\d\.\-]+)', output)
        
        ret = float(ret_match.group(1)) if ret_match else 0.0
        dd = float(dd_match.group(1)) if dd_match else 0.0
        sharpe = float(sharpe_match.group(1)) if sharpe_match else 0.0
        
        if os.path.exists(filename): os.remove(filename)
        if os.path.exists(state_file): os.remove(state_file)
        
        return {'配置': name, 'N': n, 'T': t, '收益率': ret, '最大回撤': dd, '夏普': sharpe, 'Calmar': ret/dd if dd>0 else 0}
    except Exception as e:
        if os.path.exists(filename): os.remove(filename)
        if os.path.exists(state_file): os.remove(state_file)
        return {'配置': name, 'N': n, 'T': t, '收益率': 0, '最大回撤': 0, '夏普': 0, 'Calmar': 0}

def main():
    print(f"=== 纯轮动 vs 止盈止损 对比测试 ===")
    print(f"时间段: {START_DATE} 至 {END_DATE}")
    print(f"💡 创业板基准: ~111.5%\n")
    
    results = []
    for i, config in enumerate(CONFIGS):
        print(f"[{i+1}/{len(CONFIGS)}] 测试: {config[0]} ...", end="", flush=True)
        data = prepare_and_run(config)
        results.append(data)
        print(f" 收益:{data['收益率']}% 回撤:{data['最大回撤']}%")
    
    df = pd.DataFrame(results)
    df.to_csv('pure_rotation_comparison.csv', index=False)
    
    print("\n=== 结果对比 ===")
    print(df.to_string(index=False))
    
    # 找出最佳
    best_return = df.loc[df['收益率'].idxmax()]
    best_calmar = df.loc[df['Calmar'].idxmax()]
    
    print(f"\n🏆 收益最高: {best_return['配置']}, 收益={best_return['收益率']}%")
    print(f"🎯 Calmar最优: {best_calmar['配置']}, Calmar={best_calmar['Calmar']:.2f}")

if __name__ == '__main__':
    main()
