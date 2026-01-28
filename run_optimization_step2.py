"""
Step 2: 风控参数优化
基准参数: TOP_N=5, T=13 (来自Step 1最佳结果)
测试参数:
  - STOP_LOSS: 0.05, 0.08, 0.10, 0.12, 0.15
  - TRAILING_TRIGGER: 0.06, 0.08, 0.10, 0.12, 0.15, 0.20
  - TRAILING_DROP: 0.02, 0.03, 0.04, 0.05
"""

import os
import re
import subprocess
import pandas as pd
import time
import concurrent.futures
from datetime import datetime

# === 基准参数 (来自Step 1) ===
BASE_TOP_N = 5
BASE_T = 13

# === Step 2 参数网格 ===
STOP_LOSS_LIST = [0.05, 0.08, 0.10, 0.12, 0.15]
TRAILING_TRIGGER_LIST = [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
TRAILING_DROP_LIST = [0.02, 0.03, 0.04, 0.05]

MAX_WORKERS = 6

SOURCE_FILE = 'gm_strategy_rolling0.py'
RESULT_FILE = 'optimization_step2_risk.csv'

def prepare_strategy_file(stop_loss, trailing_trigger, trailing_drop):
    """创建并返回临时且唯一的策略文件路径"""
    # 使用参数值创建唯一文件名 (去掉小数点)
    sl_str = str(stop_loss).replace('.', '')
    tt_str = str(trailing_trigger).replace('.', '')
    td_str = str(trailing_drop).replace('.', '')
    
    filename = f'gm_opt_s2_{sl_str}_{tt_str}_{td_str}.py'
    state_file = f'rolling_state_s2_{sl_str}_{tt_str}_{td_str}.json'
    
    with open(SOURCE_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换基准参数 (确保使用Step 1最佳)
    content = re.sub(r'^TOP_N\s*=\s*\d+', f'TOP_N = {BASE_TOP_N}', content, flags=re.MULTILINE)
    content = re.sub(r'^REBALANCE_PERIOD_T\s*=\s*\d+', f'REBALANCE_PERIOD_T = {BASE_T}', content, flags=re.MULTILINE)
    
    # 替换风控参数
    content = re.sub(r'^STOP_LOSS\s*=\s*[\d\.]+', f'STOP_LOSS = {stop_loss}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_TRIGGER\s*=\s*[\d\.]+', f'TRAILING_TRIGGER = {trailing_trigger}', content, flags=re.MULTILINE)
    content = re.sub(r'^TRAILING_DROP\s*=\s*[\d\.]+', f'TRAILING_DROP = {trailing_drop}', content, flags=re.MULTILINE)
    
    # 替换 State File 为唯一文件
    content = re.sub(r'STATE_FILE\s*=\s*".*"', f'STATE_FILE = "{state_file}"', content, flags=re.MULTILINE)
    
    # 替换 run() 中的 filename 参数
    content = re.sub(r"filename\s*=\s*['\"]gm_strategy_rolling0\.py['\"]", f"filename='{filename}'", content)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
        
    return filename, state_file

def run_task(args):
    stop_loss, trailing_trigger, trailing_drop = args
    strategy_file, state_file = prepare_strategy_file(stop_loss, trailing_trigger, trailing_drop)
    
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
            'STOP_LOSS': stop_loss,
            'TRAILING_TRIGGER': trailing_trigger,
            'TRAILING_DROP': trailing_drop,
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
            'STOP_LOSS': stop_loss,
            'TRAILING_TRIGGER': trailing_trigger,
            'TRAILING_DROP': trailing_drop,
            'Return': 0,
            'MaxDD': 0,
            'Sharpe': 0,
            'Calmar': 0,
            'Status': str(e)
        }

def main():
    print(f"=== 开始参数优化 Step 2: 风控参数 (Parallel={MAX_WORKERS}) ===")
    print(f"基准参数: TOP_N={BASE_TOP_N}, T={BASE_T}")
    print(f"测试: STOP_LOSS={STOP_LOSS_LIST}")
    print(f"      TRAILING_TRIGGER={TRAILING_TRIGGER_LIST}")
    print(f"      TRAILING_DROP={TRAILING_DROP_LIST}")
    
    tasks = []
    for sl in STOP_LOSS_LIST:
        for tt in TRAILING_TRIGGER_LIST:
            for td in TRAILING_DROP_LIST:
                # 逻辑约束: 止盈触发必须 > 止盈回撤
                if tt > td:
                    tasks.append((sl, tt, td))
    
    total = len(tasks)
    finished = 0
    results = []
    
    print(f"\n有效组合数: {total} (已跳过不合理参数)")
    
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
            
            print(f"[{finished}/{total}] SL={data['STOP_LOSS']} TT={data['TRAILING_TRIGGER']} TD={data['TRAILING_DROP']} "
                  f"-> R:{data['Return']}% DD:{data['MaxDD']}% S:{data['Sharpe']:.2f} (ETA: {eta/60:.1f}m)")

    # Save
    df = pd.DataFrame(results)
    df = df.sort_values(by='Calmar', ascending=False)
    df.to_csv(RESULT_FILE, index=False)
    
    duration = time.time() - start_time
    
    print(f"\n=== Step 2 优化完成 ===")
    print(f"耗时: {duration/60:.1f} 分钟")
    print(f"结果已保存至: {RESULT_FILE}")
    
    # 打印最优结果
    if not df.empty:
        print("\n🏆 按收益回撤比 (Calmar) 排序 Top 10:")
        print(df.head(10).to_string(index=False))
        
        print("\n📈 按收益率 (Return) 排序 Top 5:")
        print(df.sort_values(by='Return', ascending=False).head(5).to_string(index=False))
        
        print("\n🛡️ 按最低回撤 (MaxDD) 排序 Top 5:")
        print(df.sort_values(by='MaxDD', ascending=True).head(5).to_string(index=False))

if __name__ == '__main__':
    main()
