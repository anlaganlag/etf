# 策略验证方案

## 一、必须完成的验证（P0）

### 1. 修复策略实现与文档的差异
**当前问题**：代码实现与文档描述完全不同
**验证步骤**：
```python
# 1. 实现真正的"全市场排名"评分
def calculate_market_wide_score(etf_code, all_etf_returns):
    """
    在全市场1700+只ETF中排名
    进入前15名才得分
    """
    for period in [1, 3, 5, 10, 20, 60, 120, 250]:
        period_returns = all_etf_returns[f'R{period}'].sort_values(ascending=False)
        rank = period_returns.rank(method='min', ascending=False)[etf_code]
        if rank <= 15:
            score += WEIGHTS[period]
    return score

# 2. 实现Score≥150门槛
def select_with_threshold(scored_etfs, threshold=150):
    qualified = scored_etfs[scored_etfs['total_score'] >= threshold]
    if len(qualified) == 0:
        return []  # 空仓
    return qualified.nlargest(10, 'total_score')
```

### 2. 扩展回测期至至少3年
**目标**：
- 至少包含2022-2025年（3年数据）
- 必须覆盖牛市、熊市、震荡市

**对比基准**：
- 创业板指数（399006）
- 沪深300（000300）
- 等权重ETF组合（无择时）

### 3. 样本外验证
**步骤**：
1. 在2022-2023数据上优化参数
2. 在2024-2025数据上测试（样本外）
3. 对比样本内vs样本外表现衰减

**判断标准**：
- 样本外收益衰减>50% → 严重过拟合
- 样本外夏普比率<0.5 → 策略无效

---

## 二、参数稳健性测试（P1）

### 1. Score门槛扫描（更细粒度）
```python
thresholds = range(0, 400, 10)  # 0, 10, 20, ..., 390
for t in thresholds:
    run_backtest(score_threshold=t)
    record_metrics(return, maxdd, sharpe, cash_ratio)
```

**预期结果**：
- 如果某个特定阈值（如150）显著优于其他，说明过拟合
- 稳健策略应在100-200区间表现相对平滑

### 2. 评分权重敏感性
**测试**：
- 对每个周期的权重±30%扰动
- 观察收益变化

**判断**：
- 收益变化>20% → 权重不稳健

### 3. 持仓周期T的连续扫描
```python
T_values = [5, 7, 10, 12, 14, 16, 18, 20, 25, 30, 40, 60]
# 观察收益曲线是否平滑
```

---

## 三、交易成本与滑点（P1）

### 1. 真实交易成本建模
```python
def apply_realistic_costs(trades):
    for trade in trades:
        # 佣金（双边）
        cost += trade.amount * 0.00025 * 2
        # 印花税（卖出单边）
        if trade.direction == 'SELL':
            cost += trade.amount * 0.001
        # 滑点（假设0.1%）
        cost += trade.amount * 0.001
```

**预期影响**：
- T=20，年换手率~600%
- 总成本约3-4%/年
- 当前149%收益可能降至145%左右

### 2. 涨跌停板约束
```python
def apply_limit_constraints(target_portfolio, market_data):
    """
    无法买入涨停股，无法卖出跌停股
    """
    for etf in target_portfolio:
        if etf.is_limit_up():
            # 尝试次日买入，或跳过
            pass
```

### 3. 流动性约束
```python
def check_liquidity(etf, position_size):
    """
    单日成交量不得超过标的日均成交额的5%
    """
    max_trade = etf.avg_volume_20d * 0.05
    if position_size > max_trade:
        raise LiquidityError
```

---

## 四、市场环境分段测试（P1）

### 1. 按市场状态分段
| 时期 | 市场状态 | 测试目的 |
|------|---------|---------|
| 2022-04 ~ 2022-10 | 熊市 | 策略是否能避险？ |
| 2023-01 ~ 2023-12 | 震荡市 | 策略是否频繁止损？ |
| 2024-09 ~ 2024-10 | 牛市 | 策略是否能把握？ |

### 2. 滚动窗口测试
```python
# 每6个月滚动一次，测试252个交易日
for start in range(0, len(dates)-252, 126):
    window = dates[start:start+252]
    result = backtest(window)
    record(result)
```

**判断**：
- 如果50%的滚动窗口跑输基准 → 策略不稳定

---

## 五、对比实验（P2）

### 1. 简化版本对比
**目的**：验证复杂性是否必要

| 策略变体 | 说明 |
|---------|------|
| 简化A | 只用R1-R20短期动量 |
| 简化B | 去掉精选池限制 |
| 简化C | 固定持仓Top 10，无Score门槛 |

### 2. 动量因子有效性验证
```python
# IC测试：因子值与未来N日收益的相关性
for period in [1, 3, 5, 10, 20]:
    factor = calculate_momentum_factor(period)
    future_return = calculate_future_return(N=20)
    IC = correlation(factor, future_return)
    print(f'Period {period}: IC={IC:.3f}')
```

**判断标准**：
- IC绝对值>0.05且显著 → 因子有效
- IC<0.02 → 因子噪音

---

## 六、蒙特卡洛模拟（P2）

### 1. 交易日期随机化
```python
# 将调仓日从固定T=20改为随机T~[15, 25]
# 运行1000次模拟
results = []
for i in range(1000):
    T = random.randint(15, 25)
    result = backtest(holding_period=T)
    results.append(result)

# 分析收益分布
print(f'收益中位数: {np.median(results)}')
print(f'收益标准差: {np.std(results)}')
```

**判断**：
- 如果90%置信区间跨越0 → 策略不稳健

---

## 七、验证时间表

| 阶段 | 任务 | 时间 | 责任人 |
|------|------|------|--------|
| Week 1 | 修复代码实现，对齐文档 | 3天 | 开发 |
| Week 1-2 | 扩展回测期至3年 | 4天 | 开发 |
| Week 2 | 样本外验证 | 2天 | 量化 |
| Week 3 | 参数稳健性全扫描 | 5天 | 量化 |
| Week 3-4 | 交易成本与约束建模 | 3天 | 开发 |
| Week 4 | 对比实验与蒙特卡洛 | 2天 | 量化 |
| Week 5 | 撰写最终验证报告 | 3天 | 全员 |

**里程碑**：
- M1 (Week 2): 完成3年回测
- M2 (Week 3): 完成样本外验证
- M3 (Week 5): 输出最终评估结论

---

## 八、警戒线（Red Flags）

如果出现以下情况，**必须停止实盘**：

1. ⛔ 样本外收益<基准收益
2. ⛔ 3年回测夏普比率<0.8
3. ⛔ 最大回撤>40%
4. ⛔ 某年份亏损>-20%
5. ⛔ 参数改动10%导致收益变化>30%
6. ⛔ 加入交易成本后年化收益<15%
7. ⛔ 滚动窗口测试中50%以上跑输基准

---

## 九、预期结果预测

### 悲观预测（概率60%）
- 3年回测收益：年化15-20%
- 样本外收益：年化8-12%
- 最大回撤：-35%以上
- **结论**：策略在特定时期有效，但不稳健

### 中性预测（概率30%）
- 3年回测收益：年化25-30%
- 样本外收益：年化15-20%
- 最大回撤：-30%
- **结论**：策略有一定alpha，需优化风控

### 乐观预测（概率10%）
- 3年回测收益：年化35%+
- 样本外收益：年化25%+
- 最大回撤：-25%以内
- **结论**：策略有显著alpha，可实盘

---

## 十、验证代码框架

```python
# verification_suite.py

class StrategyValidator:
    def __init__(self, strategy, data_loader):
        self.strategy = strategy
        self.data = data_loader

    def validate_all(self):
        """运行所有验证测试"""
        results = {}

        # P0验证
        results['extended_backtest'] = self.test_extended_period()
        results['out_of_sample'] = self.test_out_of_sample()

        # P1验证
        results['parameter_scan'] = self.test_parameter_robustness()
        results['trading_costs'] = self.test_realistic_costs()
        results['market_regime'] = self.test_market_regimes()

        # P2验证
        results['monte_carlo'] = self.run_monte_carlo(n=1000)
        results['factor_ic'] = self.test_factor_validity()

        return self.generate_report(results)

    def check_red_flags(self, results):
        """检查警戒线"""
        red_flags = []

        if results['out_of_sample']['return'] < results['benchmark_return']:
            red_flags.append("样本外跑输基准")

        if results['extended_backtest']['sharpe'] < 0.8:
            red_flags.append("夏普比率过低")

        # ... 更多检查

        return red_flags

# 使用示例
validator = StrategyValidator(strategy, data)
report = validator.validate_all()

if validator.check_red_flags(report):
    print("⚠️ 发现严重问题，不建议实盘")
else:
    print("✅ 通过验证，可考虑实盘")
```

---

## 结论

当前策略文档存在**严重的实现缺陷和过拟合风险**。在完成上述验证之前，**强烈不建议实盘**。

预计完成全部验证需要**4-5周**工作量（1名开发+1名量化分析师）。

验证完成后，策略真实表现大概率会**显著低于**文档声称的149%收益和70%年化。
