"""
延长换仓周期测试
目标: 减少交易频率，提高Beta露出
"""

import os
import re
import subprocess
import pandas as pd

START_DATE = '2024-09-01 09:00:00'
END_DATE = '2026-01-27 16:00:00'

# 固定参数
BASE_TOP_N = 8
BASE_SL = 0.05
BASE_TT = 0.06
BASE_TD = 0.02

# 测试周期
T_LIST = [10, 14, 18, 20, 22, 25, 28, 30]

SOURCE_FILE = 'gm_strategy_rolling0.py'

def prepare_and_run(t_period):
    filename = f'gm_test_t_{t_period}.py'
    state_file = f'state_t_{t_period}.json'
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {BASE_TOP_N}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {t_period}', content, flags=re.MULTILINE)
    content = re.sub(r'^STOP_LOSS\s*=\s*[\d\.]+', f'STOP_LOSS = {BASE_SL}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_TRIGGER\s*=\s*[\d\.]+', f'TRAILING_TRIGGER = {BASE_TT}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_DROP\s*=\s*[\d\.]+', f'TRAILING_DROP = {BASE_TD}', content, flags=re.MULTILINE)
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
        
        return {'T': t_period, '收益率': ret, '最大回撤': dd, '夏普': sharpe, 'Calmar': ret/dd if dd>0 else 0}
    except Exception as e:
        if os.path.exists(filename): os.remove(filename)
        if os.path.exists(state_file): os.remove(state_file)
        return {'T': t_period, '收益率': 0, '最大回撤': 0, '夏普': 0, 'Calmar': 0}

def main():
    print(f"=== 延长换仓周期测试 ===")
    print(f"时间段: {START_DATE} 至 {END_DATE}")
    print(f"固定参数: TOP_N={BASE_TOP_N}, SL={BASE_SL}, TT={BASE_TT}, TD={BASE_TD}")
    print(f"测试周期: {T_LIST}\n")
    
    results = []
    for i, t in enumerate(T_LIST):
        print(f"[{i+1}/{len(T_LIST)}] 测试 T={t} ...", end="", flush=True)
        data = prepare_and_run(t)
        results.append(data)
        print(f" 收益:{data['收益率']}% 回撤:{data['最大回撤']}% Calmar:{data['Calmar']:.2f}")
    
    df = pd.DataFrame(results)
    df.to_csv('extended_period_comparison.csv', index=False)
    
    print("\n=== 结果对比 ===")
    print(df.to_string(index=False))
    print("\n💡 创业板基准收益约 111.5%")
    
    # 分析趋势
    best = df.loc[df['收益率'].idxmax()]
    print(f"\n🏆 收益最高: T={int(best['T'])}, 收益={best['收益率']}%, 回撤={best['最大回撤']}%")

if __name__ == '__main__':
    main()
