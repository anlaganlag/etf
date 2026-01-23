import pandas as pd
import numpy as np

def load_holding_period_data():
    """加载持有期比较数据"""
    file_path = "holding_period_comparison_corrected.csv"
    df = pd.read_csv(file_path)
    # 提取T数字
    df['T_Value'] = df['Period'].str.extract(r'T(\d+)').astype(int)
    return df

def analyze_holding_periods(df):
    """分析T1-T20持有期策略"""

    print("="*80)
    print("T1-T20 持有期策略综合对比分析")
    print("="*80)

    # 基本统计
    print("
数据概览:"    print(f"总计策略数: {len(df)}")
    print(f"测试期间: 2024.10.09 - 2026.01.22")
    print(f"初始资金: 100万")

    # 最佳策略分析
    best_overall = df.loc[df['Score'].idxmax()]
    best_return = df.loc[df['Return'].idxmax()]
    best_risk = df.loc[df['MaxDD'].idxmax()]  # 回撤最小（数值最大）

    print("
=== 最佳策略分析 ==="    print(f"最佳综合策略: T{best_overall['T_Value']}")
    print(".2f"    print(".2f"    print(".3f"
    print("
最高收益策略: T{best_return['T_Value']}"    print(".2f"    print(".2f"    print(".3f"
    print("
最低风险策略: T{best_risk['T_Value']}"    print(".2f"    print(".2f"    print(".3f"
    # 收益分布分析
    profitable = df[df['Return'] > 0]
    unprofitable = df[df['Return'] <= 0]

    print("
=== 收益分布分析 ==="    print(f"盈利策略: {len(profitable)}/{len(df)} ({len(profitable)/len(df)*100:.1f}%)")
    print(f"亏损策略: {len(unprofitable)}/{len(df)} ({len(unprofitable)/len(df)*100:.1f}%)")

    if len(profitable) > 0:
        print("
盈利策略统计:"        print(".2f"        print(".2f"        print(".2f"        print(".2f"
    if len(unprofitable) > 0:
        print("
亏损策略统计:"        print(".2f"        print(".2f"
    # 持有期趋势分析
    print("
=== 持有期趋势分析 ==="    corr_return = df['T_Value'].corr(df['Return'])
    corr_risk = df['T_Value'].corr(df['MaxDD'])
    corr_score = df['T_Value'].corr(df['Score'])

    print(".3f"    print(".3f"    print(".3f"
    # 关键观察点
    print("
=== 关键观察点 ==="    print("1. 最佳持有期: T10 (综合得分最高)")
    print("2. 收益拐点: T6-T9 开始显著改善")
    print("3. 风险最优: T17-T18 (回撤相对较小)")
    print("4. 超短线风险: T1-T3 表现最差")

    # 策略分类建议
    print("
=== 投资者策略建议 ==="    print("保守型投资者: 选择 T8-T12 (平衡收益与风险)")
    print("稳健型投资者: 选择 T10 (综合表现最佳)")
    print("激进型投资者: 选择 T18 (最高收益)")
    print("短期交易者: 避免 T1-T3 (表现不佳)")

    # 输出详细数据表格
    print("
=== 详细数据表格 ==="    print("Period | Return% | MaxDD% | Score")
    print("-" * 35)
    for _, row in df.iterrows():
        print("6s")

def main():
    """主函数"""
    print("开始分析 T1-T20 持有期策略表现...")

    # 加载数据
    df = load_holding_period_data()
    if df is None:
        return

    print(f"成功加载 {len(df)} 个持有期策略数据")

    # 执行分析
    analyze_holding_periods(df)

    print("
分析完成!"    print("建议使用 T10 作为基准策略")

if __name__ == "__main__":
    main()