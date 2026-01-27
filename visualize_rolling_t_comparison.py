# coding=utf-8
"""
可视化Rolling策略T值对比
"""
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import config

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    """加载短期和长期对比数据"""
    # 短期数据
    short_data = {
        'T': [6, 7, 8, 10, 12, 14],
        'Return': [25.75, 21.32, 20.52, 18.97, 14.88, 14.15],
        'Drawdown': [-10.26, -11.76, -12.18, -12.26, -11.40, -10.63],
        'Sharpe': [1.24, 1.07, 1.05, 1.02, 0.88, 0.88]
    }

    # 长期数据
    long_data = {
        'T': [6, 7, 8, 10, 12, 14],
        'Return': [62.45, 48.41, 59.66, 69.03, 61.59, 58.31],
        'Annualized': [12.90, 10.37, 12.41, 14.02, 12.75, 12.17],
        'Drawdown': [-24.64, -28.04, -27.21, -27.90, -28.37, -28.75],
        'Sharpe': [0.67, 0.57, 0.66, 0.74, 0.70, 0.68],
        'Risk_Adj': [2.53, 1.73, 2.19, 2.47, 2.17, 2.03]
    }

    return pd.DataFrame(short_data), pd.DataFrame(long_data)

def create_comparison_charts():
    """创建完整对比图表"""
    short_df, long_df = load_data()

    # 创建大图表
    fig = plt.figure(figsize=(18, 12))

    # 1. 短期收益对比
    ax1 = plt.subplot(2, 3, 1)
    bars = ax1.bar(short_df['T'], short_df['Return'],
                   color=['#ff4444' if x == 6 else '#666666' for x in short_df['T']],
                   edgecolor='black', linewidth=1.5)
    ax1.set_title('短期收益对比 (2024-09至今)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('T值 (调仓周期)', fontsize=12)
    ax1.set_ylabel('累计收益率 (%)', fontsize=12)
    ax1.grid(axis='y', alpha=0.3)
    ax1.axhline(y=short_df['Return'].mean(), color='red', linestyle='--', alpha=0.5, label='平均值')

    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, short_df['Return'])):
        ax1.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.legend()

    # 2. 长期收益对比
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(long_df['T'], long_df['Return'],
                   color=['#ff4444' if x == 10 else '#666666' for x in long_df['T']],
                   edgecolor='black', linewidth=1.5)
    ax2.set_title('长期收益对比 (2021-12至今)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('T值 (调仓周期)', fontsize=12)
    ax2.set_ylabel('累计收益率 (%)', fontsize=12)
    ax2.grid(axis='y', alpha=0.3)
    ax2.axhline(y=long_df['Return'].mean(), color='red', linestyle='--', alpha=0.5, label='平均值')

    for i, (bar, val) in enumerate(zip(bars, long_df['Return'])):
        ax2.text(bar.get_x() + bar.get_width()/2, val + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.legend()

    # 3. 回撤对比 (长期)
    ax3 = plt.subplot(2, 3, 3)
    bars = ax3.bar(long_df['T'], long_df['Drawdown'],
                   color=['#44ff44' if x == 6 else '#666666' for x in long_df['T']],
                   edgecolor='black', linewidth=1.5)
    ax3.set_title('最大回撤对比 (长期)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('T值', fontsize=12)
    ax3.set_ylabel('最大回撤 (%)', fontsize=12)
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(y=long_df['Drawdown'].mean(), color='red', linestyle='--', alpha=0.5, label='平均值')

    for i, (bar, val) in enumerate(zip(bars, long_df['Drawdown'])):
        ax3.text(bar.get_x() + bar.get_width()/2, val - 0.5,
                f'{val:.1f}%', ha='center', va='top', fontsize=10, fontweight='bold')
    ax3.legend()

    # 4. 夏普比率对比
    ax4 = plt.subplot(2, 3, 4)
    x = range(len(long_df))
    line1 = ax4.plot(x, short_df['Sharpe'], 'o-', linewidth=2, markersize=8,
                     label='短期 (2024-09至今)', color='#ff6b6b')
    line2 = ax4.plot(x, long_df['Sharpe'], 's-', linewidth=2, markersize=8,
                     label='长期 (2021-12至今)', color='#4ecdc4')
    ax4.set_title('夏普比率对比', fontsize=14, fontweight='bold')
    ax4.set_xlabel('T值', fontsize=12)
    ax4.set_ylabel('夏普比率', fontsize=12)
    ax4.set_xticks(x)
    ax4.set_xticklabels(long_df['T'])
    ax4.grid(alpha=0.3)
    ax4.legend()

    # 5. 风险调整比 (长期)
    ax5 = plt.subplot(2, 3, 5)
    bars = ax5.bar(long_df['T'], long_df['Risk_Adj'],
                   color=['#ff4444' if x == 6 else '#4444ff' if x == 10 else '#666666'
                          for x in long_df['T']],
                   edgecolor='black', linewidth=1.5)
    ax5.set_title('风险调整比 (收益/回撤)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('T值', fontsize=12)
    ax5.set_ylabel('风险调整比', fontsize=12)
    ax5.grid(axis='y', alpha=0.3)
    ax5.axhline(y=long_df['Risk_Adj'].mean(), color='red', linestyle='--', alpha=0.5, label='平均值')

    for i, (bar, val) in enumerate(zip(bars, long_df['Risk_Adj'])):
        ax5.text(bar.get_x() + bar.get_width()/2, val + 0.05,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax5.legend()

    # 6. 年化收益 vs 回撤散点图
    ax6 = plt.subplot(2, 3, 6)
    scatter = ax6.scatter(abs(long_df['Drawdown']), long_df['Annualized'],
                         s=200, c=long_df['T'], cmap='RdYlGn_r',
                         edgecolor='black', linewidth=2, alpha=0.7)

    # 添加T值标签
    for i, row in long_df.iterrows():
        ax6.annotate(f'T={row["T"]}',
                    xy=(abs(row['Drawdown']), row['Annualized']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=11, fontweight='bold')

    ax6.set_title('风险-收益矩阵 (长期)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('最大回撤 (绝对值 %)', fontsize=12)
    ax6.set_ylabel('年化收益率 (%)', fontsize=12)
    ax6.grid(alpha=0.3)

    # 添加最优区域标注
    ax6.axhline(y=long_df['Annualized'].mean(), color='gray', linestyle='--', alpha=0.3)
    ax6.axvline(x=abs(long_df['Drawdown']).mean(), color='gray', linestyle='--', alpha=0.3)
    ax6.text(23, 14.5, '理想区域\n(高收益,低回撤)', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.colorbar(scatter, ax=ax6, label='T值')

    plt.tight_layout()

    # 保存图表
    output_path = os.path.join(config.BASE_DIR, 'output', 'charts', 'rolling_t_comprehensive_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存至: {output_path}")
    plt.close()

def create_summary_table():
    """创建汇总表格图像"""
    short_df, long_df = load_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 短期表格
    ax1.axis('tight')
    ax1.axis('off')

    short_table_data = []
    for _, row in short_df.iterrows():
        short_table_data.append([
            f"T={row['T']}",
            f"{row['Return']:.2f}%",
            f"{row['Drawdown']:.2f}%",
            f"{row['Sharpe']:.2f}"
        ])

    table1 = ax1.table(cellText=short_table_data,
                      colLabels=['T值', '累计收益', '最大回撤', '夏普比率'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.2, 0.25, 0.25, 0.25])
    table1.auto_set_font_size(False)
    table1.set_fontsize(12)
    table1.scale(1, 2.5)

    # 高亮最优行
    for i in range(len(short_df)):
        if short_df.iloc[i]['T'] == 6:
            for j in range(4):
                table1[(i+1, j)].set_facecolor('#ffcccc')

    ax1.set_title('短期表现 (2024-09至今)', fontsize=16, fontweight='bold', pad=20)

    # 长期表格
    ax2.axis('tight')
    ax2.axis('off')

    long_table_data = []
    for _, row in long_df.iterrows():
        long_table_data.append([
            f"T={row['T']}",
            f"{row['Return']:.2f}%",
            f"{row['Annualized']:.2f}%",
            f"{row['Drawdown']:.2f}%",
            f"{row['Sharpe']:.2f}",
            f"{row['Risk_Adj']:.2f}"
        ])

    table2 = ax2.table(cellText=long_table_data,
                      colLabels=['T值', '累计收益', '年化收益', '最大回撤', '夏普比率', '风险调整比'],
                      cellLoc='center',
                      loc='center',
                      colWidths=[0.12, 0.18, 0.18, 0.18, 0.16, 0.18])
    table2.auto_set_font_size(False)
    table2.set_fontsize(11)
    table2.scale(1, 2.5)

    # 高亮最优行
    for i in range(len(long_df)):
        if long_df.iloc[i]['T'] == 10:
            for j in range(6):
                table2[(i+1, j)].set_facecolor('#ccffcc')
        elif long_df.iloc[i]['T'] == 6:
            for j in range(6):
                table2[(i+1, j)].set_facecolor('#ffffcc')

    ax2.set_title('长期表现 (2021-12至今)', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = os.path.join(config.BASE_DIR, 'output', 'charts', 'rolling_t_summary_table.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"汇总表格已保存至: {output_path}")
    plt.close()

if __name__ == '__main__':
    print("生成Rolling策略T值对比可视化...")
    create_comparison_charts()
    create_summary_table()
    print("\n所有图表生成完成！")
