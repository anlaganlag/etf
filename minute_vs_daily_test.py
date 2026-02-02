"""
分钟级 vs 日线级回测对比实验
目标：验证分钟级数据是否能提升策略表现
"""
from gm.api import *
import pandas as pd
import numpy as np

# === 实验设计 ===
# 测试1：日线止损（当前方案）
# 测试2：分钟级止损（每分钟检查）
# 测试3：分钟级止损 + 择时优化（避开盘初盘尾）

# 对比指标：
# - 年化收益
# - 最大回撤
# - 夏普比率
# - 交易次数
# - 胜率

class MinuteStopLoss:
    """分钟级止损策略"""
    def __init__(self, stop_loss_pct=0.15):
        self.stop_loss_pct = stop_loss_pct
        self.positions = {}  # {symbol: entry_price}
        self.stop_count = 0

    def check_stop(self, symbol, current_price):
        """每分钟检查止损"""
        if symbol not in self.positions:
            return False

        entry = self.positions[symbol]
        if current_price < entry * (1 - self.stop_loss_pct):
            self.stop_count += 1
            return True
        return False

    def add_position(self, symbol, entry_price):
        self.positions[symbol] = entry_price

    def remove_position(self, symbol):
        if symbol in self.positions:
            del self.positions[symbol]

class DailyStopLoss:
    """日线级止损策略（当前方案）"""
    def __init__(self, stop_loss_pct=0.15):
        self.stop_loss_pct = stop_loss_pct
        self.positions = {}
        self.stop_count = 0

    def check_stop(self, symbol, close_price):
        """仅在收盘时检查"""
        if symbol not in self.positions:
            return False

        entry = self.positions[symbol]
        if close_price < entry * (1 - self.stop_loss_pct):
            self.stop_count += 1
            return True
        return False

def backtest_comparison():
    """
    运行对比回测
    """
    results = {
        'daily': {},
        'minute': {},
        'minute_smart': {}
    }

    # TODO: 实现三个版本的回测
    # 1. 日线版本（现有逻辑）
    # 2. 分钟版本（每分钟检查止损）
    # 3. 分钟智能版本（避开盘初盘尾 + 分钟止损）

    return results

def analyze_results(results):
    """
    分析三个版本的差异
    """
    print("=" * 70)
    print("分钟级 vs 日线级回测对比")
    print("=" * 70)

    for name, metrics in results.items():
        print(f"\n【{name}】")
        print(f"  年化收益: {metrics.get('annual_return', 0):.2%}")
        print(f"  最大回撤: {metrics.get('max_dd', 0):.2%}")
        print(f"  夏普比率: {metrics.get('sharpe', 0):.2f}")
        print(f"  交易次数: {metrics.get('trade_count', 0)}")
        print(f"  止损次数: {metrics.get('stop_count', 0)}")

    # 计算增量价值
    daily_sharpe = results['daily'].get('sharpe', 0)
    minute_sharpe = results['minute'].get('sharpe', 0)

    improvement = (minute_sharpe - daily_sharpe) / daily_sharpe * 100 if daily_sharpe > 0 else 0

    print("\n" + "=" * 70)
    print(f"分钟级相比日线级的夏普改进: {improvement:+.1f}%")

    if improvement > 10:
        print("✓ 分钟级有显著价值，建议采用")
    elif improvement > 5:
        print("⚠️ 分钟级有一定价值，但考虑实现复杂度")
    else:
        print("✗ 分钟级价值有限，建议继续用日线级")
    print("=" * 70)

if __name__ == '__main__':
    # results = backtest_comparison()
    # analyze_results(results)

    # 示例输出（需要实际跑完回测）
    mock_results = {
        'daily': {
            'annual_return': 0.35,
            'max_dd': -0.18,
            'sharpe': 1.85,
            'trade_count': 48,
            'stop_count': 12
        },
        'minute': {
            'annual_return': 0.32,  # 可能略低（更多止损）
            'max_dd': -0.12,        # 回撤显著降低
            'sharpe': 1.95,         # 夏普略有提升
            'trade_count': 115,     # 交易次数暴增
            'stop_count': 67        # 止损次数增加
        },
        'minute_smart': {
            'annual_return': 0.37,  # 择时优化后可能更好
            'max_dd': -0.13,
            'sharpe': 2.05,
            'trade_count': 52,
            'stop_count': 15
        }
    }

    analyze_results(mock_results)
