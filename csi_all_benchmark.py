#!/usr/bin/env python3
"""中证全指(000985)基准对比 - 最简版"""

# ===== 配置区域（请修改这里的数据） =====
CSI_ALL_START = 4661.71  # 2024-10-09 中证全指收盘点位（真实数据）
CSI_ALL_END = 6298.06    # 2026-01-22 中证全指收盘点位（真实数据）
TRADING_DAYS = 316      # 测试期间交易日数量
T14_RETURN = 38.81      # T14策略年化收益率
# ========================================

# 计算中证全指收益
total_return = (CSI_ALL_END - CSI_ALL_START) / CSI_ALL_START * 100
ann_return = ((1 + total_return/100) ** (252 / TRADING_DAYS) - 1) * 100

# 打印结果
print("="*70)
print("中证全指(000985)基准对比分析")
print("="*70)

print(f"\n中证全指指数:")
print(f"  起始点位 (2024-10-09): {CSI_ALL_START:>10.2f}")
print(f"  结束点位 (2026-01-22): {CSI_ALL_END:>10.2f}")
print(f"  总收益率:              {total_return:>10.2f}%")
print(f"  年化收益率:            {ann_return:>10.2f}%")

print(f"\n📊 T14策略 vs 中证全指:")
print(f"  T14年化收益:           {T14_RETURN:>10.2f}%")
print(f"  中证全指年化:          {ann_return:>10.2f}%")
print(f"  超额收益:              {T14_RETURN - ann_return:>10.2f}%")
print(f"  超额倍数:              {T14_RETURN / ann_return:>10.2f}x")

# 与其他基准对比
print("\n"+"="*70)
print("完整基准对比表")
print("="*70)

benchmarks = {
    "沪深300": 13.85,
    "中证500": 33.74,
    "中证全指": ann_return
}

print(f"\n{'基准':<12} {'年化收益':<12} {'vs T14超额':<12} {'超额倍数':<12}")
print("-"*70)
for name, ret in benchmarks.items():
    excess = T14_RETURN - ret
    ratio = T14_RETURN / ret
    print(f"{name:<12} {ret:>10.2f}% {excess:>10.2f}% {ratio:>10.2f}x")

print("\n" + "="*70)
print("💡 使用说明:")
print("   1. 修改脚本开头的 CSI_ALL_START 和 CSI_ALL_END")
print("   2. 运行: python csi_all_benchmark.py")
print("="*70)
