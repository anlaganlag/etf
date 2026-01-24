
import pandas as pd

# Load tuning data
df = pd.read_csv('output/data/tuning_holding_period.csv')

# Filter T10 to T30
df_sub = df[(df['Period'] >= 10) & (df['Period'] <= 30)]

# ChiNext Return (Benchmark)
# Hardcoded from previous run: 121.43%
chinext_ret = 121.43

print(f"| 持仓天数 (T) | 策略总收益 | 相对创业板 (+121.4%) | 评价 |")
print(f"| :--- | :--- | :--- | :--- |")

for _, row in df_sub.iterrows():
    t = int(row['Period'])
    ret = row['Return']
    diff = ret - chinext_ret
    
    # Simple evaluation
    if diff > 5: eval_str = "🏆 跑赢"
    elif diff > -5: eval_str = "🤝 持平"
    else: eval_str = "📉 跑输"
    
    print(f"| T={t} | {ret:.1f}% | {diff:+.1f}% | {eval_str} |")
