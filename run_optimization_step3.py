"""
Step 3: 筛选参数优化
基准参数 (来自Step 1+2最佳结果):
  - TOP_N = 5
  - REBALANCE_PERIOD_T = 13
  - STOP_LOSS = 0.05
  - TRAILING_TRIGGER = 0.06
  - TRAILING_DROP = 0.02

测试参数:
  - MIN_SCORE: 10, 15, 20, 25, 30, 35, 40
  - MAX_PER_THEME: 0 (不限), 1, 2, 3
"""

import os
import re
import subprocess
import pandas as pd
import time
import concurrent.futures
from datetime import datetime

# === 基准参数 (来自Step 1+2) ===
BASE_TOP_N = 5
BASE_T = 13
BASE_STOP_LOSS = 0.05
BASE_TRAILING_TRIGGER = 0.06
BASE_TRAILING_DROP = 0.02

# === Step 3 参数网格 ===
MIN_SCORE_LIST = [10, 15, 20, 25, 30, 35, 40]
MAX_PER_THEME_LIST = [0, 1, 2, 3]  # 0 = 不限制

MAX_WORKERS = 6

SOURCE_FILE = 'gm_strategy_rolling0.py'
RESULT_FILE = 'optimization_step3_filter.csv'

def prepare_strategy_file(min_score, max_per_theme):
    """创建并返回临时且唯一的策略文件路径"""
    filename = f'gm_opt_s3_{min_score}_{max_per_theme}.py'
    state_file = f'rolling_state_s3_{min_score}_{max_per_theme}.json'
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换基准参数
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {BASE_TOP_N}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {BASE_T}', content, flags=re.MULTILINE)
    content = re.sub(r'^STOP_LOSS\s*=\s*[\d\.]+', f'STOP_LOSS = {BASE_STOP_LOSS}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_TRIGGER\s*=\s*[\d\.]+', f'TRAILING_TRIGGER = {BASE_TRAILING_TRIGGER}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_DROP\s*=\s*[\d\.]+', f'TRAILING_DROP = {BASE_TRAILING_DROP}', content, flags=re.MULTILINE)
    
    # 替换筛选参数
    content = re.sub(r'^MIN_SCORE\s*=\s*\d+', f'MIN_SCORE = {min_score}', content, flags=re.MULTILINE)
    content = re.sub(r'^MAX_PER_THEME\s*=\s*\d+', f'MAX_PER_THEME = {max_per_theme}', content, flags=re.MULTILINE)
    
    # 替换 State File 为唯一文件
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    
    # 替换 run() 中的 filename 参数
    content = re.sub(r"filename\s*=\s*['\"]gm_strategy_rolling0\.py['\"]", f"filename='{filename}'", content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return filename, state_file

def run_task(args):
    min_score, max_per_theme = args
    strategy_file, state_file = prepare_strategy_file(min_score, max_per_theme)
    
    try:
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
        if os.path.exists(state_file + '.tmp'): os.remove(state_file + '.tmp')
        
        return {
            'MIN_SCORE': min_score,
            'MAX_PER_THEME': max_per_theme,
            'Return': ret,
            'MaxDD': dd,
            'Sharpe': sharpe,
            'Calmar': calmar,
            'Status': 'OK'
        }
    except Exception as e:
        # Cleanup on error
        if os.path.exists(strategy_file): os.remove(strategy_file)
        if os.path.exists(state_file): os.remove(state_file)
        return {
            'MIN_SCORE': min_score,
            'MAX_PER_THEME': max_per_theme,
            'Return': 0,
            'MaxDD': 0,
            'Sharpe': 0,
            'Calmar': 0,
            'Status': str(e)
        }

def main():
    print(f"=== 开始参数优化 Step 3: 筛选参数 (Parallel={MAX_WORKERS}) ===")
    print(f"基准参数: TOP_N={BASE_TOP_N}, T={BASE_T}, SL={BASE_STOP_LOSS}, TT={BASE_TRAILING_TRIGGER}, TD={BASE_TRAILING_DROP}")
    print(f"测试: MIN_SCORE={MIN_SCORE_LIST}")
    print(f"      MAX_PER_THEME={MAX_PER_THEME_LIST}")
    
    tasks = []
    for ms in MIN_SCORE_LIST:
        for mpt in MAX_PER_THEME_LIST:
            tasks.append((ms, mpt))
    
    total = len(tasks)
    finished = 0
    results = []
    
    print(f"\n总组合数: {total}")
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_task = {executor.submit(run_task, task): task for task in tasks}
        
        print(f"Submitted {total} tasks. Waiting for results...\n")
        
        for future in concurrent.futures.as_completed(future_to_task):
            finished += 1
            data = future.result()
            results.append(data)
            
            elapsed = time.time() - start_time
            avg_time = elapsed / finished
            eta = avg_time * (total - finished)
            
            mpt_display = "无限制" if data['MAX_PER_THEME'] == 0 else data['MAX_PER_THEME']
            print(f"[{finished}/{total}] MS={data['MIN_SCORE']} MPT={mpt_display} "
                  f"-> R:{data['Return']}% DD:{data['MaxDD']}% S:{data['Sharpe']:.2f} (ETA: {eta/60:.1f}m)")

    # Save
    df = pd.DataFrame(results)
    df = df.sort_values(by='Calmar', ascending=False)
    df.to_csv(RESULT_FILE, index=False)
    
    duration = time.time() - start_time
    
    print(f"\n=== Step 3 优化完成 ===")
    print(f"耗时: {duration/60:.1f} 分钟")
    print(f"结果已保存至: {RESULT_FILE}")
    
    # 打印最优结果
    if not df.empty:
        print("\n🏆 按收益回撤比 (Calmar) 排序 Top 10:")
        print(df.head(10).to_string(index=False))
        
        print("\n📈 按收益率 (Return) 排序 Top 5:")
        print(df.sort_values(by='Return', ascending=False).head(5).to_string(index=False))

if __name__ == '__main__':
    main()
