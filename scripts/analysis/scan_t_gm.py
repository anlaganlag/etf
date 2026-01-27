
import os
import re
import subprocess
import pandas as pd

def run_gm_backtest_with_t(t_value):
    # Read gm_backtest.py
    with open('gm_backtest.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Modify context.T
    new_content = re.sub(r'context\.T\s*=\s*\d+', f'context.T = {t_value}', content)
    # Modify filename in run() call
    new_content = re.sub(r"filename\s*=\s*'gm_backtest\.py'", "filename='gm_backtest_temp.py'", new_content)
    
    with open('gm_backtest_temp.py', 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"Running gm_backtest with T={t_value}...")
    result = subprocess.run(['python', 'gm_backtest_temp.py'], capture_output=True, text=True, encoding='utf-8')
    
    # Extract return from output
    # Output format: Cumulative Return: 200.09%
    match = re.search(r'Cumulative Return:\s*([\d\.]+)%', result.stdout)
    if match:
        ret = float(match.group(1))
        # Also extract Max Drawdown and Sharpe Ratio if available
        dd_match = re.search(r'Max Drawdown:\s*([\d\.]+)%', result.stdout)
        sharpe_match = re.search(r'Sharpe Ratio:\s*([\d\.\-]+)', result.stdout)
        
        dd = float(dd_match.group(1)) if dd_match else 0
        sharpe = float(sharpe_match.group(1)) if sharpe_match else 0
        
        print(f"  T={t_value} -> Return: {ret}%, MaxDD: {dd}%, Sharpe: {sharpe}")
        return {'T': t_value, 'Return': ret, 'MaxDD': dd, 'Sharpe': sharpe}
    else:
        print(f"  T={t_value} -> FAILED to extract return")
        print(result.stdout)
        return {'T': t_value, 'Return': None}

def main():
    results = []
    # Test T from 1 to 14
    for t in range(1, 15):
        res = run_gm_backtest_with_t(t)
        results.append(res)
    
    df = pd.DataFrame(results)
    df.to_csv('output/data/t_scan_gm_results.csv', index=False)
    print("\nScan Results:")
    print(df)

if __name__ == '__main__':
    main()
