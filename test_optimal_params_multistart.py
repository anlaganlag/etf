# coding=utf-8
"""测试最优参数T=14, top_n=2在不同起点的表现"""
import sys
from parameter_optimization import ParamOptimizer
import pandas as pd
from datetime import timedelta

optimizer = ParamOptimizer()

# 测试T=14, top_n=2在不同起点的表现
start_dates = pd.date_range('2024-09-01', '2025-10-01', freq='30D')
results = []

print("测试最优参数 T=14, top_n=2 在不同起点的表现\n")
print("="*60)

for start_dt in start_dates:
    result = optimizer.simulate_strategy(
        T=14,
        top_n=2,
        start_date=start_dt,
        end_date=pd.Timestamp('2026-01-23')
    )
    if result:
        results.append({
            'Start': start_dt.strftime('%Y-%m-%d'),
            'Return': f"{result['total_return']:.1f}%",
            'MaxDD': f"{result['max_drawdown']:.1f}%",
            'Sharpe': f"{result['sharpe']:.2f}"
        })

df = pd.DataFrame(results)
print(df.to_string(index=False))
print("="*60)
print(f"\n收益率标准差: {pd.to_numeric(df['Return'].str.rstrip('%')).std():.1f}%")
print(f"最好收益: {df['Return'].iloc[0]}")
print(f"最差收益: {df['Return'].iloc[-1]}")
