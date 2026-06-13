# MARS 架构边界

## 目录职责

- `mars.core`
  - 基类、异常、内部数值常量、稳定协议。
  - 不依赖 analysis、feature、modeling、monitoring、pipeline、reporting。
- `mars.utils`
  - 轻依赖基础工具：可选依赖、日期、日志、轻量 HTML/表格 helper。
  - 不承载重 HTML 模板、重绘图、重导出和具体风控工作流。
- `mars.compute`
  - 表达式工厂、共享统计算子、共享缺失语义、缺失统计 helper、物化策略。
  - 为 analysis、feature、modeling、monitoring 提供底层计算复用。
- `mars.analysis`
  - 数据画像、分箱评估、PSI 稳定性、风险画像和分析工作流。
  - `MarsBinEvaluator` 负责通用分箱评估；高层画像与分析入口消费共享 `compute` 能力。
- `mars.feature`
  - 分箱器和特征筛选器。
  - `base.py` 放共享分箱能力；Native、Optimal、LiteOpt 分文件维护。
  - 不得反向依赖 `mars.modeling`。
- `mars.modeling`
  - 数据切分、后端策略、调参、replay、建模评估、结果和 artifact。
  - 建模评估可复用 analysis 的稳定性或分箱评估能力，但应通过清晰 adapter。
- `mars.pipeline`
  - 串联 Selection、可选 WOE Binning 和最终 Modeling。
  - 不在 pipeline 内重新定义 modeling / monitoring / reporting 规则。
- `mars.monitoring`
  - 特征/模型监控指标计算层、趋势表和报警摘要。
  - 不是调度、看板、告警发送或业务处置平台。
- `mars.reporting`
  - HTML / Excel / plot / 模板 / 导出。
  - 消费结构化结果对象，不反向承载核心计算。
- `mars.scoring`
  - 评分卡和部署转换能力。

## 推荐依赖方向

```text
core <- utils <- compute
core/utils/compute <- feature
core/utils/compute/feature <- analysis
core/utils/compute/feature <- monitoring
core/utils/compute/feature/analysis <- modeling
core/utils/compute/feature/modeling <- pipeline
core/utils <- reporting
reporting <- analysis/modeling/monitoring/scoring (results only)
```

补充约束：

- 允许 modeling 复用 analysis 的稳定性/分箱评估能力，但要通过清晰 adapter。
- 不要把 analysis 的实现细节散落进 modeling 评估主体。
- reporting 只消费结果对象、summary/detail/trend 表和 metadata，不重新承载底层计算。

## API 分层

- Public API
  - 对外稳定承诺入口；当前稳定入口只收口到顶层 `mars` 与一级领域包。
- Internal API
  - 允许随重构演进；默认不承诺稳定路径。
- Experimental API
  - 用于试验性能力，不直接伪装成稳定 public surface。

新增能力先判断属于哪一层，不要默认直接暴露到 public 入口。

## Pandas 白名单

MARS 不是“绝对纯 Polars”项目，但 Pandas 只能存在于明确白名单区域：

- logistic backend
- scorecard export
- report / plot rendering

新增核心链路默认优先保持在 Polars / `pl.Expr` 内完成。
如果必须转 Pandas，要说明原因并尽量通过统一物化 helper 收口。

## 结果对象

- 分析结果使用 `Report` 或 `Profile`。
- 调参、replay、特征增长等运行产物使用 `Result`。
- Report 应尽量保留 summary、detail、trend、metadata 等多粒度结构化表，方便二次加工、
  看板接入和 Agent 定制分析。
- 文件导出是 Report 的能力之一，不应成为唯一结果。
- scanner / detector 一类数据质量能力可以输出结构化表、summary 或轻量 result，不应被迫先变成 HTML/Excel。

## 拆分过胖模块

拆分原则：

1. public class/function 收口到顶层 `mars` 或一级领域包；深层模块不再默认承诺公开路径。
2. 先拆纯 helper、adapter、表构造、绘图、导出和工作流函数。
3. 不为了文件变短而制造循环依赖或把一个流程拆成难追踪的碎片。
4. 内部模块以 `_` 开头，例如 `_risk_profile.py`、`_report_utils.py`、`_spec_utils.py`。
5. 底部 re-export 如为规避循环导入，可使用局部 `# noqa: E402`，但必须说明结构原因。
