"""
Low Hanging Fruit 探索脚本
在143%冠军配置(N=5, T=10, 满仓, 短期)基础上，测试简单参数改动
Fruit 1: 放宽止损 SL=8%
Fruit 2: 贪婪止盈 Trigger=15%
Fruit 3: 超短爆发 权重=R3/R5
Fruit 4: 组合(宽止损+贪婪)
"""

import os
import re
import subprocess
import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

START_DATE = '2024-09-01 09:00:00'
END_DATE = '2026-01-27 16:00:00'

# 冠军基准配置
CHAMPION_CONFIG = {
    'TOP_N': 5,
    'REBALANCE_PERIOD_T': 10,
    'STOP_LOSS': 0.05,
    'TRAILING_TRIGGER': 0.06,
    'TRAILING_DROP': 0.02,
    'DYNAMIC_POSITION': False,
    'MAX_PER_THEME': 1,
    'SCORING_WEIGHTS': 'SHORT_TERM' # {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
}

TEST_CASES = [
    {'name': '0.冠军基准', 'changes': {}},
    {'name': '1.宽止损(SL=8%)', 'changes': {'STOP_LOSS': 0.08}},
    {'name': '2.宽止损(SL=10%)', 'changes': {'STOP_LOSS': 0.10}},
    {'name': '3.贪婪止盈(Trig=15%)', 'changes': {'TRAILING_TRIGGER': 0.15}},
    {'name': '4.贪婪止盈(Trig=20%)', 'changes': {'TRAILING_TRIGGER': 0.20}},
    {'name': '5.超短爆发(R3/R5核心)', 'changes': {'SCORING_WEIGHTS': 'SUPER_SHORT'}},
    {'name': '6.组合果实(SL=8+Trig=15)', 'changes': {'STOP_LOSS': 0.08, 'TRAILING_TRIGGER': 0.15}},
]

SOURCE_FILE = 'gm_strategy_rolling0.py'

def create_variant(name, changes):
    safe_name = f"gm_fruit_{name.split('.')[0]}"
    filename = f"{safe_name}.py"
    state_file = f"{safe_name}.json"
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply Champion Config FIRST
    content = re.sub(r'^TOP_N\s*=\s*\d+', 'TOP_N = 5', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', 'REBALANCE_PERIOD_T = 10', content, flags=re.MULTILINE)
    content = re.sub(r'^DYNAMIC_POSITION\s*=\s*(True|False)', 'DYNAMIC_POSITION = False', content, flags=re.MULTILINE)
    
    # Base Short Term Rule: periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}
    short_term_rule = "periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}"
    content = re.sub(r'periods_rule\s*=\s*{.*?}', short_term_rule, content, flags=re.DOTALL)

    # Apply Specific Changes
    for key, val in changes.items():
        if key == 'SCORING_WEIGHTS':
            if val == 'SUPER_SHORT':
                # periods_rule = {1: 20, 3: 100, 5: 80, 10: 0, 20: 0}
                new_rule = "periods_rule = {1: 20, 3: 100, 5: 80, 10: 0, 20: 0}"
                content = re.sub(r'periods_rule\s*=\s*{.*?}', new_rule, content, flags=re.DOTALL)
        else:
            pattern = f"^{key}\s*=\s*.*"
            replacement = f"{key} = {val}"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
    
    # Update common setup
    content = re.sub(r"^START_DATE\s*=\s*['\"].*['\"]", f"START_DATE='{START_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r"^END_DATE\s*=\s*['\"].*['\"]", f"END_DATE='{END_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    content = re.sub(r"filename\s*=\s*['\"].*?['\"]", f"filename='{filename}'", content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return filename, state_file

def run_test(filename, state_file, name):
    try:
        if os.path.exists(state_file): os.remove(state_file)
        
        result = subprocess.run(['python', filename], capture_output=True, text=True, encoding='utf-8', env=os.environ.copy())
        output = result.stdout
        
        ret_match = re.search(r'Return:\s*([\d\.\-]+)%', output)
        dd_match = re.search(r'Max DD:\s*([\d\.\-]+)%', output)
        
        ret = float(ret_match.group(1)) if ret_match else 0.0
        dd = float(dd_match.group(1)) if dd_match else 0.0
        calmar = ret / dd if dd > 0 else 0
        
        return {'测试项': name, '收益率': ret, '最大回撤': dd, 'Calmar': calmar}
    except Exception as e:
        return {'测试项': name, '收益率': 0, '最大回撤': 0, 'Calmar': 0}
    finally:
        if os.path.exists(filename): os.remove(filename)
        if os.path.exists(state_file): os.remove(state_file)

def main():
    print(f"🚀 Low Hanging Fruit 探索")
    print(f"基准: N=5, T=10, 满仓, SL=5%, Trig=6%")
    
    results = []
    for i, case in enumerate(TEST_CASES):
        print(f"[{i+1}/{len(TEST_CASES)}] 测试: {case['name']} ... ", end="", flush=True)
        fname, sfile = create_variant(case['name'], case['changes'])
        res = run_test(fname, sfile, case['name'])
        results.append(res)
        print(f"R: {res['收益率']}% / DD: {res['最大回撤']}%")
        
    df = pd.DataFrame(results)
    
    table = Table(title="果实采摘结果")
    table.add_column("变体", justify="left")
    table.add_column("收益率", justify="right", style="green")
    table.add_column("回撤", justify="right", style="red")
    table.add_column("提升", justify="right")
    
    base_ret = results[0]['收益率']
    for r in results:
        diff = r['收益率'] - base_ret
        style = "green" if diff > 0 else "red" if diff < 0 else "white"
        table.add_row(r['测试项'], f"{r['收益率']}%", f"{r['最大回撤']}%", f"[{style}]{diff:+.2f}%[/{style}]")
    
    console.print(table)
    
    best = df.loc[df['收益率'].idxmax()]
    print(f"\n🏆 最佳变体: {best['测试项']} ({best['收益率']}%)")

if __name__ == '__main__':
    main()
