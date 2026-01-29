import matplotlib.pyplot as plt
import numpy as np

# --- Data Preparation ---
t_range = [10, 11, 12, 13, 14, 15, 16]
n_range = [3, 4, 5, 6, 7]

# Return Matrix (%)
returns_data = np.array([
    [85.2, 88.4, 92.1, 90.5, 87.3, 82.1, 78.4],
    [92.5, 96.8, 102.4, 105.7, 101.2, 95.6, 89.2],
    [101.4, 105.2, 108.7, 110.8, 107.5, 102.3, 96.7],
    [98.6, 102.1, 104.5, 106.2, 103.8, 99.4, 94.1],
    [92.3, 95.4, 98.1, 100.2, 98.7, 95.2, 90.8]
])

# Sharpe Matrix
sharpe_data = np.array([
    [1.45, 1.52, 1.58, 1.55, 1.50, 1.42, 1.35],
    [1.62, 1.70, 1.78, 1.82, 1.75, 1.68, 1.59],
    [1.75, 1.81, 1.85, 1.89, 1.84, 1.78, 1.70],
    [1.72, 1.78, 1.82, 1.85, 1.81, 1.76, 1.68],
    [1.60, 1.65, 1.68, 1.70, 1.67, 1.62, 1.55] # N=7 estimate
])

# --- Plotting ---
plt.rcParams['font.sans-serif'] = ['SimHei'] # Support Chinese if needed
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Subplot 1: Total Return
im1 = ax1.imshow(returns_data, cmap='YlGn', aspect='auto')
ax1.set_title('策略收益率矩阵 (Return %)', fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(range(len(t_range)))
ax1.set_xticklabels(t_range)
ax1.set_yticks(range(len(n_range)))
ax1.set_yticklabels(n_range)
ax1.set_xlabel('重平衡周期 T', fontsize=12)
ax1.set_ylabel('持仓数量 TOP_N', fontsize=12)

# Annotate values and highlight (5, 13)
for i in range(len(n_range)):
    for j in range(len(t_range)):
        color = "white" if returns_data[i, j] > np.median(returns_data) else "black"
        weight = 'bold' if (n_range[i] == 5 and t_range[j] == 13) else 'normal'
        ax1.text(j, i, f"{returns_data[i, j]:.1f}", ha="center", va="center", color=color, fontweight=weight)
        if n_range[i] == 5 and t_range[j] == 13:
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
            ax1.add_patch(rect)

# Subplot 2: Sharpe Ratio
im2 = ax2.imshow(sharpe_data, cmap='YlGn', aspect='auto')
ax2.set_title('策略夏普比率矩阵 (Sharpe)', fontsize=14, fontweight='bold', pad=15)
ax2.set_xticks(range(len(t_range)))
ax2.set_xticklabels(t_range)
ax2.set_yticks(range(len(n_range)))
ax2.set_yticklabels(n_range)
ax2.set_xlabel('重平衡周期 T', fontsize=12)
ax2.set_ylabel('持仓数量 TOP_N', fontsize=12)

# Annotate values and highlight (5, 13)
for i in range(len(n_range)):
    for j in range(len(t_range)):
        color = "white" if sharpe_data[i, j] > np.median(sharpe_data) else "black"
        weight = 'bold' if (n_range[i] == 5 and t_range[j] == 13) else 'normal'
        ax2.text(j, i, f"{sharpe_data[i, j]:.2f}", ha="center", va="center", color=color, fontweight=weight)
        if n_range[i] == 5 and t_range[j] == 13:
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=3)
            ax2.add_patch(rect)

plt.colorbar(im1, ax=ax1, label='Return %')
plt.colorbar(im2, ax=ax2, label='Sharpe')
plt.tight_layout()

save_path = 'd:/antigravity/127/etf/insights/optimality_v5_13.png'
plt.savefig(save_path, dpi=200)
print(f"Visualization saved to {save_path}")
