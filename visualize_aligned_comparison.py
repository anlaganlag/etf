import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_csv("aligned_strategy_comparison.csv")

# 转换数据格式用于绘图
p_data = df[df['Strategy'] == 'Periodic']
r_data = df[df['Strategy'] == 'Rolling']

# 设置中文字体 (尝试几种常见的)
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(12, 7))

plt.plot(p_data['T'], p_data['Return'], marker='o', label='定期调仓 (Periodic)', linewidth=2, color='#1f77b4')
plt.plot(r_data['T'], r_data['Return'], marker='s', label='滚动持仓 (Rolling)', linewidth=2, color='#ff7f0e')

plt.title('定期调仓 vs 滚动持仓: 不同T值下的收益率对比 (已对齐参数)', fontsize=14)
plt.xlabel('T 值 (持仓天数 / 调仓周期)', fontsize=12)
plt.ylabel('累计收益率 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 在点上标注数值
for x, y in zip(p_data['T'], p_data['Return']):
    plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='#1f77b4')

for x, y in zip(r_data['T'], r_data['Return']):
    plt.annotate(f"{y:.1f}%", (x, y), textcoords="offset points", xytext=(0,-15), ha='center', fontsize=9, color='#ff7f0e')

plt.tight_layout()
plt.savefig("strategy_t_comparison.png", dpi=300)
print("图表已保存至 strategy_t_comparison.png")
