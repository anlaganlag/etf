import pandas as pd

# 加载数据
df = pd.read_csv("holding_period_comparison_corrected.csv")
df['T_Value'] = df['Period'].str.extract(r'T(\d+)').astype(int)

print("="*60)
print("T1-T20 持有期策略分析报告")
print("="*60)

print("\n数据概览:")
print(f"策略数量: {len(df)}")
print("测试期间: 2024.10.09 - 2026.01.22")

# 最佳策略
best_score = df.loc[df['Score'].idxmax()]
best_return = df.loc[df['Return'].idxmax()]
best_risk = df.loc[df['MaxDD'].idxmax()]

print("\n=== 最佳策略 ===")
print("最佳综合: T{} (得分: {:.3f})".format(best_score['T_Value'], best_score['Score']))
print("最高收益: T{} ({:.2f}%)".format(best_return['T_Value'], best_return['Return']))
print("最低风险: T{} (回撤: {:.2f}%)".format(best_risk['T_Value'], best_risk['MaxDD']))

# 统计分析
profitable = len(df[df['Return'] > 0])
print("\n=== 统计分析 ===")
print("盈利策略: {}/20 ({:.1f}%)".format(profitable, profitable/20*100))
print("平均收益率: {:.2f}%".format(df['Return'].mean()))
print("平均最大回撤: {:.2f}%".format(df['MaxDD'].mean()))

# 趋势分析
corr_return = df['T_Value'].corr(df['Return'])
corr_risk = df['T_Value'].corr(df['MaxDD'])
print("\n=== 趋势分析 ===")
print("持有期与收益相关性: {:.3f}".format(corr_return))
print("持有期与风险相关性: {:.3f}".format(corr_risk))

print("\n=== 详细数据 ===")
print("T | 收益率% | 回撤% | 得分")
print("-" * 30)
for _, row in df.iterrows():
    print("{} | {:>8.2f} | {:>6.2f} | {:.3f}".format(
        row['T_Value'], row['Return'], row['MaxDD'], row['Score']
    ))

print("\n=== 投资建议 ===")
print("1. 推荐T10策略 (综合表现最佳)")
print("2. 保守投资者选择T8-T12")
print("3. 激进投资者选择T18")
print("4. 避免超短线策略T1-T3")