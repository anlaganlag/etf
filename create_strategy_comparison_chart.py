import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 定期调仓数据
regular_data = pd.read_csv("holding_period_comparison_corrected.csv")
regular_data['T_Value'] = regular_data['Period'].str.extract(r'T(\d+)').astype(int)
regular_data['Strategy_Type'] = '定期调仓'

# 滚动持仓数据
rolling_data = pd.read_csv("rolling_holding_period_comparison.csv")
rolling_data['Strategy_Type'] = '滚动持仓'

# 合并数据
comparison_data = pd.DataFrame({
    '持有期': regular_data['T_Value'],
    '定期调仓_收益': regular_data['Return'],
    '定期调仓_回撤': regular_data['MaxDD'],
    '滚动持仓_收益': rolling_data['Return'],
    '滚动持仓_回撤': rolling_data['MaxDD'],
    '定期调仓_得分': regular_data['Score'],
    '滚动持仓_得分': rolling_data['Score']
})

# 创建综合对比图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('定期调仓 vs 滚动持仓策略对比分析', fontsize=16, fontweight='bold')

# 1. 收益对比
x = comparison_data['持有期']
ax1.plot(x, comparison_data['定期调仓_收益'], 'o-', label='定期调仓', color='#FF6B35', linewidth=2, markersize=6)
ax1.plot(x, comparison_data['滚动持仓_收益'], 's-', label='滚动持仓', color='#4CAF50', linewidth=2, markersize=6)
ax1.set_title('收益率对比', fontsize=14, fontweight='bold')
ax1.set_xlabel('持有期 (天)')
ax1.set_ylabel('收益率 (%)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 回撤对比
ax2.plot(x, comparison_data['定期调仓_回撤'], 'o-', label='定期调仓', color='#FF6B35', linewidth=2, markersize=6)
ax2.plot(x, comparison_data['滚动持仓_回撤'], 's-', label='滚动持仓', color='#4CAF50', linewidth=2, markersize=6)
ax2.set_title('最大回撤对比', fontsize=14, fontweight='bold')
ax2.set_xlabel('持有期 (天)')
ax2.set_ylabel('最大回撤 (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 得分对比
ax3.plot(x, comparison_data['定期调仓_得分'], 'o-', label='定期调仓', color='#FF6B35', linewidth=2, markersize=6)
ax3.plot(x, comparison_data['滚动持仓_得分'], 's-', label='滚动持仓', color='#4CAF50', linewidth=2, markersize=6)
ax3.set_title('风险调整得分对比', fontsize=14, fontweight='bold')
ax3.set_xlabel('持有期 (天)')
ax3.set_ylabel('得分')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. 收益差值分析
yield_diff = comparison_data['定期调仓_收益'] - comparison_data['滚动持仓_收益']
colors = ['red' if x < 0 else 'green' for x in yield_diff]
bars = ax4.bar(x, yield_diff, color=colors, alpha=0.7)
ax4.set_title('收益差值 (定期调仓 - 滚动持仓)', fontsize=14, fontweight='bold')
ax4.set_xlabel('持有期 (天)')
ax4.set_ylabel('收益差值 (%)')
ax4.grid(True, alpha=0.3)

# 添加数值标签
for bar, diff in zip(bars, yield_diff):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            '.1f', ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
plt.savefig('strategy_type_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("策略对比图表已生成: strategy_type_comparison.png")
print("\n=== 核心发现 ===")
print("1. 定期调仓在收益方面普遍优于滚动持仓")
print("2. 滚动持仓在回撤控制方面表现更稳定")
print("3. 两种策略在T10-T18区间表现相对较好")
print("4. 定期调仓适合激进投资者，滚动持仓适合保守投资者")