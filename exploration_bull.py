"""
牛市超额收益探索脚本
验证5个方向：
1. 集中持仓 (TOP_N=5)
2. 加大仓位 (DYNAMIC_POSITION=False)
3. 追逐热点 (短期动量权重高)
4. 放开板限 (MAX_PER_THEME=0)
5. 加快轮动 (T=5)
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

# 基准参数
BASE_CONFIG = {
    'TOP_N': 8,
    'REBALANCE_PERIOD_T': 10,
    'STOP_LOSS': 0.05,
    'TRAILING_TRIGGER': 0.06,
    'TRAILING_DROP': 0.02,
    'DYNAMIC_POSITION': True,
    'MAX_PER_THEME': 1,
    'SCORING_METHOD': 'SMOOTH' # 保持默认平滑
    # SCORING_WEIGHTS 默认为长期优先 {1:20, ... 20:100}
}

# 测试用例
TEST_CASES = [
    {'name': '0.基准策略', 'changes': {}},
    {'name': '1.集中持仓(N=5)', 'changes': {'TOP_N': 5}},
    {'name': '2.加大仓位(满仓)', 'changes': {'DYNAMIC_POSITION': False}},
    {'name': '3.追逐热点(短期权重)', 'changes': {'SCORING_WEIGHTS': 'SHORT_TERM'}},
    {'name': '4.放开板限(Theme=0)', 'changes': {'MAX_PER_THEME': 0}},
    {'name': '5.加快轮动(T=5)', 'changes': {'REBALANCE_PERIOD_T': 5}},
    # 组合拳
    {'name': '6.组合拳(集中+满仓+短期)', 'changes': {
        'TOP_N': 5, 
        'DYNAMIC_POSITION': False, 
        'SCORING_WEIGHTS': 'SHORT_TERM'
    }},
    {'name': '7.极致进攻(全开)', 'changes': {
        'TOP_N': 5,
        'DYNAMIC_POSITION': False,
        'SCORING_WEIGHTS': 'SHORT_TERM',
        'MAX_PER_THEME': 0,
        'REBALANCE_PERIOD_T': 5
    }}
]

SOURCE_FILE = 'gm_strategy_rolling0.py'

def create_variant(name, changes):
    safe_name = f"gm_test_{name.split('.')[0]}_{name.split('.')[1].split('(')[0]}"
    filename = f"{safe_name}.py"
    state_file = f"{safe_name}.json"
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Apply changes
    for key, val in changes.items():
        if key == 'SCORING_WEIGHTS':
            if val == 'SHORT_TERM':
                # Replace periods_rule dict in get_ranking
                # Finding the line: periods_rule = {1: 20, 3: 30, 5: 50, 10: 70, 20: 100}
                new_rule = "periods_rule = {1: 100, 3: 70, 5: 50, 10: 30, 20: 20}"
                content = re.sub(r'periods_rule\s*=\s*{.*?}', new_rule, content)
        else:
            # Simple variable replacement
            pattern = f"^{key}\s*=\s*.*"
            replacement = f"{key} = {val}"
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)

    # Update common config
    content = re.sub(r"^START_DATE\s*=\s*['\"].*['\"]", f"START_DATE='{START_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r"^END_DATE\s*=\s*['\"].*['\"]", f"END_DATE='{END_DATE}'", content, flags=re.MULTILINE)
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    
    # Update run() call filename
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
    print(f"🚀 牛市策略探索开始")
    print(f"时间段: {START_DATE} 至 {END_DATE}")
    print(f"基准: T=10, N=8, SL=5%, 长期优先, 动态仓位, 有板限\n")
    
    results = []
    
    # Process sequentially
    total = len(TEST_CASES)
    for i, case in enumerate(TEST_CASES):
        print(f"[{i+1}/{total}] 测试: {case['name']} ... ", end="", flush=True)
        fname, sfile = create_variant(case['name'], case['changes'])
        res = run_test(fname, sfile, case['name'])
        results.append(res)
        print(f"R: {res['收益率']}% / DD: {res['最大回撤']}%")
        
    # Display Table
    table = Table(title="牛市探索结果对比")
    table.add_column("测试项", justify="left", style="cyan")
    table.add_column("收益率", justify="right", style="green")
    table.add_column("最大回撤", justify="right", style="red")
    table.add_column("Calmar", justify="right", style="yellow")
    table.add_column("对比基准", justify="right")
    
    base_ret = results[0]['收益率']
    
    for r in results:
        diff = r['收益率'] - base_ret
        diff_str = f"{diff:+.2f}%" if r['测试项'] != '0.基准策略' else "-"
        style = "green" if diff > 0 else "red" if diff < 0 else "white"
        
        table.add_row(
            r['测试项'],
            f"{r['收益率']:.2f}%",
            f"{r['最大回撤']:.2f}%",
            f"{r['Calmar']:.2f}",
            f"[{style}]{diff_str}[/{style}]"
        )
        
    console.print(table)
    
    # Simple Analysis
    df = pd.DataFrame(results)
    best = df.loc[df['收益率'].idxmax()]
    print(f"\n🏆 收益冠军: {best['测试项']} (收益 {best['收益率']}%)")
    
    # Determine meaningful factors
    print("\n📝 因子分析:")
    factors = [
        ('集中持仓', df[df['测试项'].str.contains('集中')]['收益率'].mean() - base_ret),
        ('加大仓位', df[df['测试项'].str.contains('加大')]['收益率'].mean() - base_ret),
        ('追逐热点', df[df['测试项'].str.contains('追逐')]['收益率'].mean() - base_ret),
        ('放开板限', df[df['测试项'].str.contains('放开')]['收益率'].mean() - base_ret),
        ('加快轮动', df[df['测试项'].str.contains('加快')]['收益率'].mean() - base_ret)
    ]
    for f, impact in factors:
        print(f"- {f}: {'📈' if impact>0 else '📉'} {impact:+.2f}%")

if __name__ == '__main__':
    main()
