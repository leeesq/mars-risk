# API 与业务口径

## 数据和目标变量

- 底层分箱器、selector、模型策略：`X, y`。
- 高层画像、评估、监控：`df, target`。
- target 校验不要替用户做复杂字符串映射。
- 监控 target 接受 `0/1/True/False/null`；null 表示未到表现期。
- 分布类指标使用全量样本；标签类指标只使用已表现样本。

## 缺失语义与缺失统计

- `Null / NaN / missing_values` 必须走共享底座。
- `profiler / evaluator / monitoring / scanner / detector` 不得各自维护一套缺失逻辑。
- `profile_stats` / `MarsProfileReport` 是高层画像入口，不是缺失率异常检测的底层计算接口。
- 新增缺失率相关能力时，优先复用共享 helper，例如：
  - `missing_condition_expr`
  - `missing_rate_expr`
  - `build_missing_by_period_stats`

## 分箱

- public `binning_type` 使用 `native`、`lite_opt`、`optimal`。
- 不恢复已删除的 `opt` 别名。
- `MarsNativeBinner` 默认无监督策略，CART 明确要求 y。
- `MarsOptimalBinner` 和 `MarsLiteOptBinner` 是监督式分箱器，fit 必须传 y。
- LiteOpt 支持 `ascending/descending/peak/valley/auto/auto_asc_desc`。
- `auto_asc_desc` 只在升序和降序中择优。
- 缺失箱和特殊值箱独立于正常箱，不占用正常箱数量。

## PSI

- 默认不包含缺失箱和特殊值箱，除非 API 明确覆盖。
- 缺失率通常单独监控。
- Analysis/feature 可暴露 `psi_include_missing` 和 `psi_include_special`。
- 建模 score/feature PSI 通常只需要 `psi_include_missing`，不要人为制造特殊值语义。
- 公式、平滑和 contribution 明细使用共享 stability 能力。
- 缺失率异常检测与 PSI 互补，不能互相替代。

## 缺失率异常检测

- 建模前：`scanner`。
  - 典型命名示例：`MarsMissingShiftScanner`
  - 职责是离线扫描异常特征和异常时间窗口。
- 上线后：`detector`。
  - 典型命名示例：`MarsMissingShiftDetector`
  - 职责是对最新分组或最新日期给出异常判别。
- 两者定位都是数据质量异常检测器，不自动删特征，不直接阻断流程。
- 如提供 `feature_data_source`，可用于数据源级异常聚合和同源多特征联动识别。

## Selector

- `MarsStatsSelector` 的阈值和指标口径放 init；数据、target、features、来源和分组上下文放 fit。
- `feature_data_source` 是本次候选特征全集的配置。
- 传给 evaluator 前必须按当前 active features 裁剪，但 selector 决策报告仍可保留已过滤特征来源。
- `transform(X)` 默认返回选中特征；高层 StatsSelector 可选择保留 target。
- 质量筛选器与异常检测器是互补关系，不要把缺失异常扫描直接等同于“自动筛掉特征”。

## Modeling

- `MarsModelEvaluator` 是建模评估器，不等同于完整模型监控平台。
- Modeling Pipeline 处于快速迭代阶段，接口变化要同步 docs。
- Session 的 tune/replay 参数应显式可见，避免 public `**kwargs`。
- backend 策略对象类型必须明确，模型实例本身可在边界使用 `Any`。
- 支持内置 metric 和统一签名的自定义 metric：`func(y_true, y_pred) -> float`。
- replay 应支持 Top-K 和显式 trial 编号。
- artifact 每次运行使用独立目录，保留 history、配置、元信息、模型和失败信息。
- 多 target 中只有主 target 参与训练，辅助 target 只用于独立表现期评估。

## Pipeline

- 树模型默认：Selection -> Modeling。
- LR/评分卡可选：Selection -> WOE Binning -> Selection -> Modeling。
- 如果已有 WOE step，LR backend 使用 numeric 模式，避免重复 WOE。
- 如果没有 WOE step 且模型为 LR，允许 LR backend 使用自身 WOE 模式。
- Pipeline 维护 active features、feature map、step results 和最终 modeling result。

## Monitoring

- 特征监控和模型监控共用通用指标计算逻辑。
- 用户负责窗口、基准样本、模型版本、调度、业务阈值、看板和处置流程。
- Report 缺少某类表时，报警器跳过对应检查并说明，不应直接失败。
- 趋势表排序必须记录在 metadata；报警器读取 report 的真实顺序识别基准期和最新期。
