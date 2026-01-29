"""
牛市放松风控参数测试
对比: 紧止损 vs 宽止损
"""

import os
import re
import subprocess
import pandas as pd
import concurrent.futures

START_DATE = '2024-09-01 09:00:00'
END_DATE = '2026-01-27 16:00:00'

# 固定基础参数 (牛市最优)
BASE_TOP_N = 8
BASE_T = 10

# 测试参数
CONFIGS = [
    # 描述, SL, TT, TD
    ("紧止损(当前)", 0.05, 0.06, 0.02),
    ("中等放松", 0.08, 0.10, 0.04),
    ("宽松A", 0.10, 0.15, 0.05),
    ("宽松B", 0.10, 0.20, 0.06),
    ("宽松C", 0.12, 0.15, 0.05),
    ("宽松D", 0.12, 0.20, 0.06),
    ("极宽松", 0.15, 0.25, 0.08),
    ("无止盈止损", 0.99, 0.99, 0.99),  # 实际上不触发
]

SOURCE_FILE = 'gm_strategy_rolling0.py'

def prepare_and_run(config):
    name, sl, tt, td = config
    filename = f'gm_test_relax_{name}.py'
    state_file = f'state_relax_{name}.json'
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {BASE_TOP_N}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {BASE_T}', content, flags=re.MULTILINE)
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
        
        return {'配置': name, 'SL': sl, 'TT': tt, 'TD': td, '收益率': ret, '最大回撤': dd, '夏普': sharpe, 'Calmar': ret/dd if dd>0 else 0}
    except Exception as e:
        if os.path.exists(filename): os.remove(filename)
        if os.path.exists(state_file): os.remove(state_file)
        return {'配置': name, 'SL': sl, 'TT': tt, 'TD': td, '收益率': 0, '最大回撤': 0, '夏普': 0, 'Calmar': 0}

def main():
    print(f"=== 放松风控参数测试 ===")
    print(f"时间段: {START_DATE} 至 {END_DATE}")
    print(f"基础参数: TOP_N={BASE_TOP_N}, T={BASE_T}\n")
    
    results = []
    for i, config in enumerate(CONFIGS):
        print(f"[{i+1}/{len(CONFIGS)}] 测试: {config[0]} ...", end="", flush=True)
        data = prepare_and_run(config)
        results.append(data)
        print(f" 收益:{data['收益率']}% 回撤:{data['最大回撤']}%")
    
    df = pd.DataFrame(results)
    df.to_csv('relaxed_risk_comparison.csv', index=False)
    
    print("\n=== 结果对比 ===")
    print(df.to_string(index=False))
    
    print("\n💡 创业板基准收益约 111.5%")

if __name__ == '__main__':
    main()
