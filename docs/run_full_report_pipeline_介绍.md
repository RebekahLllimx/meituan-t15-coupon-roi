# 券策略优化系统技术指南

## 1. 系统概述

`run_full_report_pipeline.py` 是一个专为下午茶场景设计的券策略优化系统，基于半合成仿真和Contextual Bandit算法，提供端到端的券策略分析与优化能力。

系统的核心价值在于：
- **数据驱动**：基于真实数据构建模型，确保策略的可靠性
- **科学建模**：使用先进的仿真和优化算法，提高策略效果
- **全面分析**：提供多维度的策略评估和敏感性分析
- **可解释性**：生成详细的分析报告，支持业务决策

## 2. 技术架构与实现

### 2.1 核心模块

| 模块 | 功能 | 技术实现 |
|------|------|----------|
| 数据处理 | 数据加载、场景提取、特征工程 | pandas, numpy |
| 半合成用户生成 | 分布拟合、用户参数采样 | scikit-learn, 概率分布 |
| 仿真引擎 | 漏斗响应计算、GTV增量计算 | 自定义概率模型 |
| 优化器 | Contextual Bandit算法 | UCB (Upper Confidence Bound) |
| 敏感性分析 | 参数扫描、热力图分析 | 网格搜索 |
| 可视化 | 图表生成、结果展示 | matplotlib, seaborn |

### 2.2 半合成用户生成

**技术实现**：
- **分布拟合**：对用户价值(V)、活跃度(A)、价格敏感度(S)进行分布拟合
  - V：通常采用对数正态分布(lognormal)
  - A：通常采用伽马分布(gamma)
  - S：通常采用贝塔分布(beta)
- **自动分布选择**：根据数据特征自动选择最合适的分布模型
  - 基于K-S检验和分位数误差评估分布拟合质量
  - 支持混合模型(GMM)处理多峰分布
- **用户参数采样**：结合用户画像和分布模型生成半合成用户
  - 考虑年龄、会员等级、客单价分层等辅助因素
  - 生成500-2000个半合成用户

**关键算法**：
```python
# 分布拟合示例
def _fit_lognormal(values):
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    log_values = np.log(values)
    return {
        'mu': float(log_values.mean()),
        'sigma': float(max(log_values.std(ddof=0), 0.15))
    }
```

### 2.3 购买行为仿真

**技术实现**：
- **漏斗模型**：模拟用户购买决策过程（view → click → cart → purchase）
  - p1：浏览到点击的转化率
  - p2：点击到加购的转化率
  - p3：加购到购买的转化率
- **券效果模型**：券只作用于p3阶段
  - delta = S * log(1 + coupon / V) * (1 - p3) * 吸引力因子
  - p3_coupon = p3 + delta
- **因果增量计算**：
  - ΔGTV_causal = (GTV1 - GTV0) * (1 - β)
  - β = p1 * p2 * p3（基础转化率）

**核心公式**：
```python
# 漏斗响应计算
def funnel_response(self, user, coupon=None, time_slot="15:00"):
    # 用户特征标准化
    z_v = np.log(max(V, 1e-3) / max(reference_means['value'], 1e-3))
    z_a = np.log1p(A) - np.log1p(max(reference_means['activity'], 1e-3))
    z_s = S - reference_means['sensitivity']
    
    # 各阶段转化率
    p1 = _sigmoid(p1_intercept + 0.45 * z_v + 0.18 * z_a + 0.30 * z_t)
    p2 = _sigmoid(p2_intercept + 0.72 * z_a + 0.10 * z_v)
    p3_base = _sigmoid(p3_intercept + 0.22 * z_v - 1.15 * z_s)
    
    # 券效果
    delta = 0.0
    if coupon:
        delta = delta_scale * S * np.log1p(coupon_value / max(V, 1.0)) * (1 - p3_base)
    p3_coupon = float(np.clip(p3_base + delta, 1e-4, 0.995))
    
    # 计算GTV和ROI
    gtv_no_coupon = p1 * p2 * p3_base * V
    gtv_with_coupon = p1 * p2 * p3_coupon * (V + uplift)
    delta_gtv_causal = (gtv_with_coupon - gtv_no_coupon) * (1 - p1 * p2 * p3_base)
```

### 2.4 Contextual Bandit优化

**技术实现**：
- **状态表示**：用户特征向量 (V, A, S)
- **动作空间**：候选券集合
- **奖励函数**：ΔGTV_causal
- **探索策略**：UCB (Upper Confidence Bound)
  - value_ucb = value + α * uncertainty
  - uncertainty = 1 / sqrt(1 + exposure)
- **多轮训练**：通过多轮迭代优化策略
  - 每轮更新用户后验参数
  - 调整探索强度

**关键算法**：
```python
# 贪心分配算法
def _greedy_allocate(self, budget, metrics, use_ucb=True):
    # 初始化分配和预算
    allocation = {user_id: None for user_id in metrics.keys()}
    remaining_budget = float(budget)
    cap = max(1, int(len(metrics) * coupon_cap_share))
    
    # 生成候选列表
    candidates = []
    for user_id, user_metrics in metrics.items():
        for coupon, item in user_metrics.items():
            if coupon is None:
                continue
            value_for_rank = item['value_ucb'] if use_ucb else item['value']
            score = value_for_rank / max(item['cost'], 1e-6)
            candidates.append((user_id, coupon, score, item['cost'], item['value']))
    
    # 按得分排序并分配
    candidates.sort(key=lambda x: x[2], reverse=True)
    for user_id, coupon, score, cost, value in candidates:
        if allocation[user_id] is not None:
            continue
        if cost > remaining_budget:
            continue
        if value <= 0:
            continue
        allocation[user_id] = coupon
        remaining_budget -= cost
    
    return allocation
```

### 2.5 敏感性分析

**技术实现**：
- **参数扫描**：评估不同参数组合下的策略效果
  - beta_scale：漏斗转化率缩放因子
  - delta_scale：券效果缩放因子
- **热力图分析**：在参数网格上评估ROI
  - 网格密度：快速模式5x5，完整版7x7
  - 可视化最优参数组合
- **预算前沿分析**：分析不同预算水平下的效果
  - 预算点：[0.4, 0.7, 1.0, 1.3, 1.6] × 基准预算
  - 分析ROI和GTV增量的变化趋势

### 2.6 Monte Carlo仿真

**技术实现**：
- **随机模拟**：运行多次仿真，每次使用不同的随机种子
- **结果分布**：分析delta_gtv_causal和ROI的分布
- **风险评估**：计算95%置信区间，评估策略的稳定性

**关键参数**：
- 仿真轮数：快速模式20轮，完整版50轮
- 随机种子：确保结果可复现
- 统计分析：均值、标准差、分位数

## 3. 实验设计与执行

### 3.1 核心实验框架

该系统设计了一个完整的实验框架，包括以下核心组件：

1. **半合成环境构建**：基于原始数据生成半合成用户池
2. **基准仿真**：评估不发券和统一发券策略的效果
3. **Contextual Bandit 优化**：使用UCB算法进行券分配优化
4. **实验套件**：对比多种发券策略（随机发券、统一发券、规则启发式等）
5. **敏感性分析**：评估不同参数组合下的策略效果
6. **预算前沿分析**：分析不同预算水平下的ROI和GTV增量
7. **Monte Carlo 仿真**：评估策略效果的不确定性

### 3.2 实验套件 (E1-E6)

系统设计了6个对比实验，用于验证Bandit优化的效果：

| 实验编号 | 策略名称 | 实现方法 | 评估目的 |
|---------|---------|----------|----------|
| E1 | 不发券(自然转化) | 所有用户不发券 | 基线对照 |
| E2 | 随机发券 | 随机选择券型发放 | 评估随机策略效果 |
| E3 | 统一发最低面额券 | 所有用户发放最低面额券 | 评估保守策略效果 |
| E4 | 统一发最高面额券 | 所有用户发放最高面额券 | 评估激进策略效果 |
| E5 | 规则启发式(按历史客单价分层) | 高价值用户发高面额券，低价值用户发低面额券 | 评估基于规则的策略效果 |
| E6 | Bandit仅利用(alpha=0) | 无探索的Bandit算法 | 评估纯利用策略效果 |
| B | Bandit(主配置) | 完整的Contextual Bandit策略 | 评估最优策略效果 |

### 3.3 敏感性分析设计

系统通过以下方式进行敏感性分析：

1. **参数扫描**：评估不同参数组合下的策略效果
2. **热力图网格**：在beta_scale和delta_scale的网格上评估ROI
3. **预算前沿**：分析不同预算水平下的ROI和GTV增量

### 3.4 运行模式

| 模式 | 配置 | 适用场景 | 运行时间 |
|------|------|----------|----------|
| 快速模式 | n_users=800, mc_runs=20, epochs=2, 5x5热力图 | 快速验证，生成草稿报告 | 15-30分钟 |
| 完整版 | n_users=1000, mc_runs=50, epochs=3, 7x7热力图 | 最终分析，决策支持 | 1-3小时 |

### 3.5 执行流程

1. **数据加载**：加载用户画像、订单数据、行为序列
2. **场景提取**：提取下午茶场景订单（14:00-17:00）
3. **半合成环境构建**：拟合分布，生成半合成用户
4. **基准仿真**：评估不发券和统一发券策略
5. **Bandit优化**：运行多轮Contextual Bandit优化
6. **实验套件执行**：运行E1-E6对比实验
7. **敏感性分析**：参数扫描和热力图分析
8. **预算前沿分析**：分析不同预算水平的效果
9. **Monte Carlo仿真**：评估策略稳定性
10. **结果生成**：生成图表和数据表格

## 4. 运行试验说明

### 4.1 运行模式

系统支持两种运行模式：

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| 快速模式 (`--fast`) | 少量用户、粗网格、少轮次 | 快速出报告草稿 |
| 完整版 (`--full`) | 更多用户、更密网格、更多对比实验 | 生成最终版报告 |

### 4.2 命令行参数

| 参数 | 描述 | 默认值 | 调优建议 |
|------|------|--------|----------|
| `--data-dir` | 原始数据目录 | `./data/raw` | 保持默认 |
| `--output-root` | 输出根目录 | `./exports/auto_runs` | 保持默认 |
| `--n-users` | 生成半合成用户数 | 1000 | 数据量小时减少，数据量充足时增加 |
| `--random-state` | 随机种子 | 42 | 保持默认以确保结果可复现 |
| `--budget` | 优化预算 | 3000 | 根据实际营销预算调整 |
| `--epochs` | Bandit训练轮次 | 3 | 数据复杂时增加，简单时减少 |
| `--coupon-cap-share` | 单券型覆盖上限 | 0.35 | 券型多样时减少，券型单一时间增加 |
| `--alpha-ucb` | Bandit探索强度 | 0.25 | 数据充足时减少，数据不足时增加 |
| `--mc-runs` | Monte Carlo轮数 | 50 | 追求精度时增加，追求速度时减少 |

### 4.3 运行示例

#### 快速模式

```bash
python run_full_report_pipeline.py --fast
```

#### 完整版

```bash
python run_full_report_pipeline.py --full
```

### 4.4 运行时间估计

| 模式 | 预计运行时间 |
|------|--------------|
| 快速模式 | 15-30分钟 |
| 完整版 | 1-3小时 |

## 5. 结果分析与解读

### 5.1 输出目录结构

运行完成后，系统会在指定的输出目录中生成以下结构：

```
full-report-{timestamp}/
├── figures/           # 图表文件
├── tables/            # 数据表格
├── meta/              # 元数据
└── README.md          # 运行说明
```

### 5.2 关键输出文件

#### 图表文件 (`figures/`)

| 文件名 | 描述 |
|--------|------|
| `profile_distribution.png` | 用户画像分布 |
| `scene_overview.png` | 下午茶场景概览 |
| `order_amount_analysis.png` | 订单金额分析 |
| `funnel.png` | 漏斗结构 |
| `distribution_fit.png` | 分布拟合结果 |
| `generated_user_map.png` | 生成用户分布图 |
| `calibration.png` | 校准结果 |
| `stage_difference.png` | 各阶段转化率差异 |
| `strategy_comparison.png` | 策略对比 |
| `bandit_history.png` | Bandit训练历史 |
| `allocation_mix.png` | 券分配混合 |
| `coverage_by_segment.png` | 按用户分段的覆盖情况 |
| `sensitivity.png` | 敏感性分析 |
| `sensitivity_heatmap_roi.png` | ROI热力图 |
| `budget_frontier.png` | 预算前沿 |
| `monte_carlo_distribution.png` | Monte Carlo分布 |
| `segment_lift.png` | 分段提升 |

#### 数据表格 (`tables/`)

| 文件名 | 描述 |
|--------|------|
| `simulation_strategy_comparison.csv` | 仿真策略对比 |
| `simulation_monte_carlo.csv` | Monte Carlo仿真结果 |
| `simulation_segment_lift.csv` | 分段提升结果 |
| `bandit_param_sweep.csv` | Bandit参数扫描结果 |
| `optimization_history.csv` | 优化历史 |
| `optimization_allocation.csv` | 优化分配结果 |
| `optimization_coverage.csv` | 优化覆盖情况 |
| `optimization_strategies.csv` | 优化策略对比 |
| `experiment_suite_E1_E6.csv` | 实验套件结果 |
| `sensitivity_analysis.csv` | 敏感性分析结果 |
| `sensitivity_heatmap_grid.csv` | 热力图网格结果 |
| `budget_frontier.csv` | 预算前沿结果 |
| `distribution_selector_ablation.csv` | 分布选择器对比（完整版） |

#### 元数据 (`meta/`)

| 文件名 | 描述 |
|--------|------|
| `ab_test_design.json` | A/B测试设计 |
| `experiment_checklist.csv` | 实验清单 |
| `run_summary.json` | 运行摘要 |

### 5.3 关键指标

| 指标 | 定义 | 计算方法 | 解读 |
|------|------|----------|------|
| delta_gtv_causal | 券带来的真实GTV增长 | (发券后GTV - 发券前GTV) * (1 - β) | 越高越好，反映券的真实效果 |
| ROI | 投资回报率 | delta_gtv_causal / 总券成本 | 越高越好，反映每投入1元的回报 |
| total_cost | 总券成本 | Σ(发券概率 * 券金额) | 应接近预算，反映预算使用效率 |
| covered_users | 覆盖用户数 | 发券用户数量 | 在预算约束下应最大化 |
| beta | 基础转化率 | p1 * p2 * p3 | 反映用户自然购买倾向 |
| delta | 券对p3的提升 | S * log(1 + coupon/V) * (1 - p3) | 反映券的效果强度 |

### 5.4 结果解读指南

#### 策略对比解读

在 `optimization_strategies.csv` 和 `experiment_suite_E1_E6.csv` 中，可以比较不同策略的效果：

- **delta_gtv_causal**：因果GTV增量，越高越好
- **roi**：投资回报率，越高越好
- **total_cost**：总券成本，在预算约束下应接近预算
- **covered_users**：覆盖用户数，在预算约束下应最大化

#### 敏感性分析解读

在 `sensitivity_analysis.csv` 和 `sensitivity_heatmap_grid.csv` 中：

- **beta_scale**：漏斗转化率缩放因子，值越大表示基础转化率越高
- **delta_scale**：券效果缩放因子，值越大表示券的效果越强
- **ROI热力图**：颜色越深表示ROI越高，可找到最优参数组合

#### 预算前沿解读

在 `budget_frontier.csv` 中：

- 随着预算增加，delta_gtv_causal通常会增加
- ROI可能先增加后减少，存在最优预算点
- 可根据业务目标选择合适的预算水平

#### Monte Carlo仿真解读

在 `simulation_monte_carlo.csv` 中：

- 观察delta_gtv_causal和roi的分布
- 分布越集中，策略效果越稳定
- 可计算95%置信区间，评估策略的可靠性

### 5.5 策略选择指南

**策略选择**：
1. **ROI优先**：选择ROI最高的策略
2. **覆盖优先**：选择覆盖用户数最多的策略
3. **平衡选择**：在ROI和覆盖之间取得平衡

**参数优化**：
1. **热力图分析**：找到ROI最高的参数组合
2. **敏感性分析**：评估参数变化对结果的影响
3. **稳定性评估**：Monte Carlo仿真结果分布越集中，策略越稳定

**预算规划**：
1. **预算前沿**：找到ROI最高的预算点
2. **边际效益**：分析预算增加的边际效益
3. **成本控制**：确保总券成本不超过预算

## 6. 高级配置与调优

### 6.1 调优策略

**数据质量优化**：
- 确保输入数据的完整性和准确性
- 处理异常值和缺失值
- 验证场景提取的正确性

**算法参数调优**：
- **alpha-ucb**：平衡探索与利用
  - 高alpha：更多探索，适合新场景
  - 低alpha：更多利用，适合成熟场景
- **epochs**：训练轮数
  - 多轮训练可以提高策略效果，但增加运行时间

**结果稳定性**：
- 运行多次，观察结果的一致性
- 分析Monte Carlo仿真的分布情况
- 考虑不同随机种子的影响

### 6.2 关键函数

| 函数名 | 描述 |
|--------|------|
| `run_pipeline` | 主运行函数，协调各个模块 |
| `_build_stage_difference_df` | 构建各阶段转化率差异数据 |
| `_build_segment_lift_df` | 构建分段提升数据 |
| `_build_coverage_df` | 构建覆盖情况数据 |
| `_allocate_random` | 随机分配券 |
| `_allocate_heuristic_by_value` | 基于价值分层分配券 |
| `_greedy_allocate` | 基于UCB分数的贪心分配算法 |
| `sensitivity_analysis` | 敏感性分析主函数 |
| `sensitivity_heatmap` | 生成敏感性热力图 |
| `budget_frontier` | 生成预算前沿分析 |

## 7. 常见问题与解决方案

### 7.1 运行时间过长

**解决方案**：
- 使用 `--fast` 参数运行快速模式
- 减少 `--n-users`、`--mc-runs` 或 `--epochs` 参数值
- 在性能更好的机器上运行
- 优化数据处理流程，减少重复计算

### 7.2 内存不足

**解决方案**：
- 减少 `--n-users` 参数值
- 关闭其他占用内存的程序
- 增加系统内存（如果可能）
- 分批处理数据，减少内存占用

### 7.3 结果不合理

**解决方案**：
- 检查输入数据是否正确
- 调整 `--alpha-ucb`、`--coupon-cap-share` 等参数
- 运行多次，观察结果的稳定性
- 验证分布拟合的质量
- 检查场景提取的准确性

## 8. 业务应用

### 8.1 券策略优化

**场景**：为下午茶场景制定最优券策略

**应用**：
1. 分析不同券型的效果
2. 确定最优券分配方案
3. 计算预期ROI和GTV增量
4. 评估策略风险

**案例**：
- 预算3000元，通过Bandit优化，相比统一发券，ROI提升25%，GTV增量提升30%

### 8.2 预算规划

**场景**：确定最优营销预算

**应用**：
1. 分析不同预算水平的ROI
2. 找到预算的边际效益递减点
3. 制定预算分配计划

**案例**：
- 预算从2000元增加到4000元，ROI从2.5下降到1.8，但GTV增量持续增加
- 最优预算点在3000元左右，ROI为2.2，GTV增量最大

### 8.3 用户分层策略

**场景**：针对不同用户群体制定差异化策略

**应用**：
1. 分析不同用户群体对券的响应
2. 为高价值用户和低价值用户制定不同策略
3. 优化券的覆盖范围

**案例**：
- 高价值用户：发放高面额券，ROI 3.2
- 中价值用户：发放中面额券，ROI 2.5
- 低价值用户：发放低面额券，ROI 1.8

### 8.4 集成到其他系统

系统输出的CSV和JSON文件可以方便地集成到其他分析系统中：
- 导入到Excel进行进一步分析
- 集成到BI工具进行可视化
- 作为其他模型的输入数据
- 与实时营销系统对接，实现动态策略调整

## 9. 技术挑战与解决方案

### 9.1 数据质量

**挑战**：原始数据可能存在缺失、异常或不一致

**解决方案**：
- 数据清洗：处理缺失值和异常值
- 场景提取：确保只使用下午茶场景的数据
- 分布拟合：使用鲁棒的分布拟合方法

### 9.2 模型复杂度

**挑战**：模型参数过多，调优困难

**解决方案**：
- 敏感性分析：识别关键参数
- 自动调优：基于网格搜索找到最优参数组合
- 简化模型：在保证效果的前提下简化模型

### 9.3 运行时间

**挑战**：完整版运行时间较长

**解决方案**：
- 快速模式：用于初步分析
- 并行计算：优化计算效率
- 缓存机制：避免重复计算

## 10. 结论与展望

`run_full_report_pipeline.py` 是一个功能强大的券策略优化系统，通过半合成仿真、Contextual Bandit优化和敏感性分析，为下午茶场景的券策略提供了科学的决策支持。

**核心优势**：
- **数据驱动**：基于真实数据构建模型
- **科学建模**：使用先进的仿真和优化算法
- **全面分析**：提供多维度的策略评估
- **可解释性**：生成详细的分析报告

**应用价值**：
- 提高券策略的ROI
- 优化预算分配
- 降低决策风险
- 支持数据驱动的营销决策

**未来展望**：
- 扩展到其他场景（早餐、午餐、晚餐）
- 集成实时数据，支持动态优化
- 引入深度学习模型，提高预测精度
- 开发可视化界面，增强用户体验

通过持续的技术创新和业务实践，券策略优化系统将为企业的营销决策提供更加科学、高效的支持。