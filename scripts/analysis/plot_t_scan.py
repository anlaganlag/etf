
import pandas as pd
import matplotlib.pyplot as plt

# Data from the scan
data = {
    'T': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
    'Return': [-14.14, 16.03, 74.35, 50.98, 8.02, 53.54, 146.27, 38.84, 74.93, 42.56, 183.67, 112.37, 15.14, 226.10]
}

df = pd.DataFrame(data)

plt.figure(figsize=(10, 6))
plt.plot(df['T'], df['Return'], marker='o', linestyle='-', color='#1f77b4', linewidth=2)
plt.fill_between(df['T'], df['Return'], alpha=0.1, color='#1f77b4')

# Highlight T=14
plt.scatter(14, 226.10, color='red', s=100, zorder=5, label='Optimal T=14')
plt.annotate('Highest Return: 226.1%', (14, 226.10), textcoords="offset points", xytext=(-50,10), ha='center', fontweight='bold', color='red')

plt.title('Performance of Different Rebalancing Periods (T)', fontsize=14, fontweight='bold')
plt.xlabel('Rebalancing Period (T days)', fontsize=12)
plt.ylabel('Cumulative Return (%)', fontsize=12)
plt.xticks(range(1, 15))
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('output/charts/t_comparison_scan.png', dpi=300)
print("Chart saved to output/charts/t_comparison_scan.png")
