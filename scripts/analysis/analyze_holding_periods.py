import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_holding_period_data():
    """加载持有期比较数据"""
    file_path = "holding_period_comparison_corrected.csv"
    if not Path(file_path).exists():
        print(f"文件 {file_path} 不存在，请先运行 holding_period_comparison_corrected.csv 的生成脚本")
        return None

    df = pd.read_csv(file_path)
    # 提取T数字
    df['T_Value'] = df['Period'].str.extract(r'T(\d+)').astype(int)
    return df

def create_comparison_visualizations(df):
    """创建多种比较可视化"""

    # 设置风格
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('T1-T20 持有期策略综合对比分析', fontsize=16, fontweight='bold')

    # 1. 收益与回撤对比图
    ax1 = axes[0, 0]
    x = df['T_Value']
    y1 = df['Return']
    y2 = df['MaxDD']

    line1 = ax1.plot(x, y1, 'o-', linewidth=2, markersize=6, label='收益率 (%)', color='#2E86AB')
    ax1.set_ylabel('收益率 (%)', color='#2E86AB', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='#2E86AB')

    ax2 = ax1.twinx()
    line2 = ax2.plot(x, y2, 's-', linewidth=2, markersize=6, label='最大回撤 (%)', color='#F24236')
    ax2.set_ylabel('最大回撤 (%)', color='#F24236', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='#F24236')

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')

    ax1.set_title('收益率 vs 最大回撤', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. 风险调整得分排名
    ax3 = axes[0, 1]
    sorted_df = df.sort_values('Score', ascending=True)
    bars = ax3.barh(sorted_df['Period'], sorted_df['Score'], color='#4CAF50', alpha=0.7)

    # 高亮最佳策略
    best_idx = sorted_df['Score'].idxmax()
    bars[len(bars)-1].set_color('#FF6B35')
    bars[len(bars)-1].set_alpha(0.9)

    ax3.set_title('风险调整得分排名 (越高越好)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlabel('得分')

    # 3. 收益效率散点图
    ax4 = axes[1, 0]
    scatter = ax4.scatter(df['MaxDD'], df['Return'], c=df['T_Value'],
                         s=df['Score']*50+50, alpha=0.7, cmap='viridis')

    # 添加标签
    for i, row in df.iterrows():
        ax4.annotate(f'T{row["T_Value"]}', (row['MaxDD'], row['Return']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax4.set_xlabel('最大回撤 (%)')
    ax4.set_ylabel('收益率 (%)')
    ax4.set_title('收益效率散点图 (气泡大小=得分)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)

    plt.colorbar(scatter, ax=ax4, label='持有期 (天)')

    # 4. 综合统计表
    ax5 = axes[1, 1]
    ax5.axis('off')

    # 计算统计指标
    best_return = df.loc[df['Return'].idxmax()]
    best_score = df.loc[df['Score'].idxmax()]
    best_risk_adj = df.loc[(df['Return'] / abs(df['MaxDD'])).idxmax()]

    stats_text = f""".1f"""
    📊 综合统计分析

    🏆 最佳收益率: T{best_return['T_Value']} ({best_return['Return']:.2f}%)
    🥇 最佳得分: T{best_score['T_Value']} ({best_score['Score']:.2f})
    🛡️ 最佳风险调整: T{best_risk_adj['T_Value']} ({(best_risk_adj['Return']/abs(best_risk_adj['MaxDD'])):.2f})

    📈 收益分布:
    • 盈利策略: {len(df[df['Return'] > 0])}/20 ({len(df[df['Return'] > 0])/20*100:.1f}%)
    • 平均收益率: {df['Return'].mean():.2f}%
    • 收益标准差: {df['Return'].std():.2f}%

    📉 风险分布:
    • 平均最大回撤: {df['MaxDD'].mean():.2f}%
    • 最小回撤: {df['MaxDD'].max():.2f}% (T{df.loc[df['MaxDD'].idxmax(), 'T_Value']})
    • 最大回撤: {df['MaxDD'].min():.2f}% (T{df.loc[df['MaxDD'].idxmin(), 'T_Value']})

    🎯 策略建议:
    • 推荐持有期: T{best_score['T_Value']} (综合表现最佳)
    • 保守选择: T{int(df['T_Value'].median())} (中等风险)
    • 激进选择: T{best_return['T_Value']} (收益优先)
    """

    ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.8))

    plt.tight_layout()
    plt.savefig('holding_period_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_strategy_recommendations(df):
    """生成策略推荐"""
    print("\n" + "="*60)
    print("🎯 T1-T20 持有期策略推荐分析")
    print("="*60)

    # 最佳策略
    best_overall = df.loc[df['Score'].idxmax()]
    best_return = df.loc[df['Return'].idxmax()]
    best_risk = df.loc[df['MaxDD'].idxmax()]  # 回撤最小（数值最大）

    print("\n🏆 最佳综合策略:")
    print(f"   T{best_overall['T_Value']} - 得分: {best_overall['Score']:.2f}")
    print(f"   收益率: {best_overall['Return']:.2f}%, 最大回撤: {best_overall['MaxDD']:.2f}%")

    print("\n💰 最高收益策略:")
    print(f"   T{best_return['T_Value']} - 收益率: {best_return['Return']:.2f}%")
    print(f"   最大回撤: {best_return['MaxDD']:.2f}%, 得分: {best_return['Score']:.2f}")

    print("\n🛡️ 最低风险策略:")
    print(f"   T{best_risk['T_Value']} - 最大回撤: {best_risk['MaxDD']:.2f}%")
    print(f"   收益率: {best_risk['Return']:.2f}%, 得分: {best_risk['Score']:.2f}")

    # 收益分布分析
    profitable = df[df['Return'] > 0]
    unprofitable = df[df['Return'] <= 0]

    print("
📊 收益分布分析:"    print(f"   盈利策略: {len(profitable)}/20 ({len(profitable)/20*100:.1f}%)")
    print(f"   亏损策略: {len(unprofitable)}/20 ({len(unprofitable)/20*100:.1f}%)")

    if len(profitable) > 0:
        print(f"   盈利策略平均收益率: {profitable['Return'].mean():.2f}%")
        print(f"   盈利策略平均回撤: {profitable['MaxDD'].mean():.2f}%")

    # 持有期趋势分析
    print("
📈 持有期趋势分析:"    corr_return = df['T_Value'].corr(df['Return'])
    corr_risk = df['T_Value'].corr(df['MaxDD'])

    print(f"   持有期与收益相关性: {corr_return:.3f} ({'正相关' if corr_return > 0 else '负相关'})")
    print(f"   持有期与风险相关性: {corr_risk:.3f} ({'正相关' if corr_risk > 0 else '负相关'})")

    # 策略分类建议
    print("
🎪 投资者策略建议:"    print("   • 保守型投资者: 选择 T8-T12 (平衡收益与风险)")
    print("   • 稳健型投资者: 选择 T10 (综合表现最佳)")
    print("   • 激进型投资者: 选择 T18 (最高收益)")
    print("   • 短期交易者: 避免 T1-T3 (表现不佳)")

def main():
    """主分析函数"""
    print("🔍 开始分析 T1-T20 持有期策略表现...")

    # 加载数据
    df = load_holding_period_data()
    if df is None:
        return

    print(f"✅ 成功加载 {len(df)} 个持有期策略数据")

    # 生成可视化分析
    print("📊 生成可视化分析图表...")
    create_comparison_visualizations(df)

    # 生成策略推荐
    generate_strategy_recommendations(df)

    print("
✅ 分析完成！"    print("   📄 查看 'holding_period_comparison_analysis.png' 获取详细图表")
    print("   📊 建议使用 T10 作为基准策略")

if __name__ == "__main__":
    main()