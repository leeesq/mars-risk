---
description: MARS 0.0.26 的用户可见变化、兼容性说明和升级检查项。
---

# Release Notes

## 0.0.26

该版本将基础包的运行范围扩展到 Python 3.8–3.12，同时保持业务 public API、报告 schema
和指标定义不变。

### Python 与依赖兼容

- Python 3.8 固定使用 Polars 1.8.2，并将 scikit-learn 限制在 1.3.x；仓库通过
  `constraints/python38.txt` 固定验证栈。
- Python 3.9 使用现代 Polars 与 scikit-learn 1.6.x；Python 3.10–3.12 延续当前现代依赖。
- Python 3.8 已停止官方安全维护。MARS 的兼容承诺仅表示冻结栈可以运行，不延长解释器的
  安全支持周期。
- `ml`、`tuning`、`notebook`、`docs` 和 `dev` extras 要求 Python 3.10+；Python 3.6、3.7
  以及 3.13+ 不在本版本支持范围内。

### 实现与结果口径

- 新增内部 Polars 兼容层，统一 membership 与 streaming collect 的跨版本差异。
- KS/AUC 的前一累计分布改为“当前累计值减当前箱分布”，以兼容 Polars 1.8 的窗口表达式
  限制；指标结果与现代 Polars 保持一致。
- Python 3.8 语言兼容改造只涉及注解求值、dataclass、zip 和字符串后缀处理，不改变公开
  参数、返回类型或序列化字段。
- 特征筛选的监督指标和 WOE 相关性只使用 target 非空的已表现样本；质量、分布和 PSI 仍
  使用全量样本。
- Optimal Binner 的失败特征统一批量回退到 Native Binner，避免随失败特征数增长的重复拟合。

### 升级检查

- Python 3.8 环境按约束文件重建，不要在已有环境中强制覆盖整套依赖。
- 使用可选建模或文档依赖时升级到 Python 3.10+。
- 发布前同时验证 Python 3.8 冻结栈、Python 3.9 依赖边界和 Python 3.10–3.12 现代栈。

## 0.0.24

从当前公开版本 `0.0.21` 升级到 `0.0.24` 时，重点核对分析报告链路、基准样本语义、
趋势图时间范围、HTML 大报告和 Modeling/Pipeline 契约。

### 用户可见变化

- `profile_risk()` 返回 `MarsRiskProfile`，同时提供 `report`、`binner`、`targets` 和 `metadata`。
- `benchmark_df` 统一用于基准期分箱和 PSI expected distribution，不进入当前期 Total。
- `MarsStatsSelector.fit()` 支持 `benchmark_df`，筛选指标仍在当前 `df` 上计算。
- 风险趋势图的时间范围只来自有效 `time_col`；`group_col` 只负责面板分组。
- HTML 报告支持可检索视图、图表数量控制、图片资产模式和懒加载。
- Modeling/Pipeline 增加结果对象、replay、artifact 和多 target 评估能力，状态仍为 Experimental。

### 升级检查

- 将旧代码中直接假定 `profile_risk()` 返回 report 的访问改为 `risk_profile.report`。
- 使用固定分箱规则时改用 `MarsBinEvaluator.evaluate(..., binner=...)`。
- 生成趋势图或 Charts HTML 前显式提供有效 `time_col`。
- 使用 `MarsStatsSelector` 默认 PSI/RC 阈值时提供 `group_col` 或 `time_col`；静态筛选应显式关闭
  对应阈值。
- 升级 Modeling/Pipeline 后重新核对结果字段和 artifact 路径。

!!! note "发布状态"

    本页随 `0.0.24` 源码维护。版本发布前，`main` 站点内容仅作为预览；PyPI 发布和 release tag
    通过版本一致性检查后，安装命令才表示正式可用。
