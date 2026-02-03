# Robust Volatility State Machine - System Design Document

**Version**: 1.0 (Final Robust)  
**Date**: 2026-02-03  
**Philosophy**: "Physics over Curve Fitting"

---

## 1. Core Philosophy (设计哲学)

本策略不再是一个寻找“最优参数”的黑箱，而是一台基于物理意义的**分层状态机 (Hierarchical State Machine)**。我们不再试图预测市场，而是通过测量**“结构的完整性”**来决定生存姿态。

### The "Ruler" (度量衡)
我们抛弃了传统的“即时波动率”，采用了 **Lagged Downside Volatility (滞后下行波动率)** 作为核心尺子。
*   **物理意义**：用平静时期的下跌习惯，去衡量当前的下跌幅度。
*   **优势**：避免了“崩盘时波动率飙升 → 阈值变宽 → 门禁失效”的自适应陷阱。

---

## 2. System Architecture (系统架构)

系统由两个解耦的层级组成，分别负责微观选股和宏观风控。

### Layer 1: Micro Gate (Stock Selection)
**目标**：剔除结构损坏的个股，保留因情绪错杀的潜力股。
*   **指标**：Individual Z-Score ($Z_i = R5 / \sigma_{down}$)
*   **阈值 (K_ENTRY)**: **1.6**
*   **逻辑**：
    *   如果 $Z_i > -1.6$：结构完整，下跌视为“噪音”或“错杀”，**买入**。
    *   如果 $Z_i < -1.6$：结构损坏，下跌视为“破位”，**剔除**。

### Layer 2: Macro Gate (Regime Detection)
**目标**：探测系统性崩盘（地震），保护本金。不可抗力时降仓。
*   **指标**：Broken Ratio ($BR = \frac{\text{Count}(Z_i < -2.5)}{N}$)
*   **阈值 (K_CRASH)**: **2.5** (严苛的崩盘标准)
*   **逻辑**：
    *   在这个层级，我们只关心有多少标的发生了**极度深跌 ($Z < -2.5$)**。
    *   这不是选股，这是在听“森林里树倒下的声音”。

### State Machine (状态机)
我们使用带滞回（Hysteresis）的状态机来控制总仓位：

| State | Condition (BR) | Risk Scaler | Description |
| :--- | :--- | :--- | :--- |
| **SAFE** | BR < 20% | **1.0** (100%) | 市场结构正常，满仓进攻。 |
| **CAUTION** | BR > 20% | **0.5** (50%) | >20% 标的极度深跌，系统性风险预警，半仓。 |
| **DANGER** | BR > 40% | **0.0** (0%) | >40% 标的崩溃，泥沙俱下，空仓避险。 |

*注：从 DANGER/CAUTION 恢复需要 BR 降至 30%/15% 以下（滞回区间），防止信号震荡。*

---

## 3. Key Parameters (核心参数)

| Parameter | Value | Rationale |
| :--- | :--- | :--- |
| **K_ENTRY** | **1.6** | 微观过滤。经 Plateau Test 验证，1.5~1.8 是最佳过滤区间，既能剔除弱势股，又不误杀回调股。 |
| **K_CRASH** | **2.5** | 宏观探测。必须比 Entry 更严格，确保只对“真正的雪崩”产生反应，避免震荡市频繁误触。 |
| **BR_Thresholds** | **20% / 40%** | 经验常数。20% 代表相关性飙升的起点，40% 代表全面崩盘。 |

---

## 4. Performance & Validation (性能验证)

*   **Net Return**: **44.29%** (vs 49.08% No-Gate)
    *   *代价*: ~5% 的收益作为“保险费”。
*   **Max Drawdown**: **29.91%**
    *   *特征*: 在 2022/2024 等大跌年份，Meta-Gate 成功触发（BR > 20%），将持仓降至 50% 或 0%，有效保护了本金。
*   **Trigger Frequency**: **~13%**
    *   系统仅在约 13% 的极端日子里开启防御，其余 87% 时间保持满仓。这证明了它是一台**“静默的地震仪”**。

---

## 5. Future Roadmap (未来路线)

1.  **Capital DD Gate (Manual)**: 如果实盘净值回撤触及 30%，人工介入暂停策略。（这就是我们移除的 Layer 3，但在实盘中应作为 SOP 存在）。
2.  **Smart Broken Ratio**: 未来可将 BR 升级为“加权 BR”，赋予权重股（如沪深300成分）更高的投票权。

---

**"The system thrives not because it predicts the future, but because it respects the structure of the present."**
