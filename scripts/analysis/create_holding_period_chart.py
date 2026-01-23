import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
df = pd.read_csv("holding_period_comparison_corrected.csv")
df['T_Value'] = df['Period'].str.extract(r'T(\d+)').astype(int)
df = df.sort_values('T_Value')

# 创建综合图表
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('T1-T20 持有期策略收益与风险对比分析', fontsize=16, fontweight='bold')

# 1. 收益与回撤双轴图
x = df['T_Value']
y1 = df['Return']
y2 = df['MaxDD']

line1 = ax1.plot(x, y1, 'o-', linewidth=2, markersize=6, label='收益率 (%)', color='#2E86AB')
ax1.set_ylabel('收益率 (%)', color='#2E86AB', fontsize=12)
ax1.tick_params(axis='y', labelcolor='#2E86AB')
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
line2 = ax1_twin.plot(x, y2, 's-', linewidth=2, markersize=6, label='最大回撤 (%)', color='#F24236')
ax1_twin.set_ylabel('最大回撤 (%)', color='#F24236', fontsize=12)
ax1_twin.tick_params(axis='y', labelcolor='#F24236')

ax1.set_title('收益率 vs 最大回撤趋势', fontsize=14, fontweight='bold')
ax1.set_xlabel('持有期 (交易日)')

# 2. 风险调整得分柱状图
bars = ax2.bar(df['Period'], df['Score'], color='#4CAF50', alpha=0.7)
ax2.set_title('风险调整得分排名', fontsize=14, fontweight='bold')
ax2.set_ylabel('得分')
ax2.grid(True, alpha=0.3)
ax2.tick_params(axis='x', rotation=45)

# 高亮最佳策略
best_idx = df['Score'].idxmax()
bars[df.index.get_loc(best_idx)].set_color('#FF6B35')
bars[df.index.get_loc(best_idx)].set_alpha(0.9)

# 3. 收益分布饼图
profitable = len(df[df['Return'] > 0])
unprofitable = len(df[df['Return'] <= 0])

labels = ['盈利策略', '亏损策略']
sizes = [profitable, unprofitable]
colors = ['#4CAF50', '#F44336']

ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
ax3.set_title('策略收益分布', fontsize=14, fontweight='bold')

# 4. 散点图：收益vs风险
scatter = ax4.scatter(df['MaxDD'], df['Return'], c=df['T_Value'],
                     s=df['Score']*50+50, alpha=0.7, cmap='viridis')

# 添加标签
for i, row in df.iterrows():
    ax4.annotate(f'T{row["T_Value"]}', (row['MaxDD'], row['Return']),
                xytext=(5, 5), textcoords='offset points', fontsize=9)

ax4.set_xlabel('最大回撤 (%)')
ax4.set_ylabel('收益率 (%)')
ax4.set_title('收益效率散点图', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)

plt.colorbar(scatter, ax=ax4, label='持有期 (天)')

plt.tight_layout()
plt.savefig('t1_t20_holding_period_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("图表已保存为: t1_t20_holding_period_analysis.png")
print("\n关键发现:")
print(f"- 最佳策略: T{df.loc[df['Score'].idxmax(), 'T_Value']} (得分: {df['Score'].max():.3f})")
print(f"- 最高收益: T{df.loc[df['Return'].idxmax(), 'T_Value']} ({df['Return'].max():.2f}%)")
print(f"- 最低风险: T{df.loc[df['MaxDD'].idxmax(), 'T_Value']} (回撤: {df['MaxDD'].max():.2f}%)")
print(f"- 盈利策略: {profitable}/20 ({profitable/20*100:.1f}%)")