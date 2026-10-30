# 分客群分析

风险画像、数据画像和统计筛选支持一个独立的 `segment_col`。
它与已有 `group_col` 分别表示客群和趋势分组。提供 `group_col` 时，仍优先按该列组织趋势；
`time_col` 提供日期与附表，不额外增加时间轴。

## 人口与固定规则

All 表示完整输入人口。Total 表示当前 scope 跨原分组的汇总。
这些统计分别聚合原始人口，不平均各客群或月份的坏率、KS、IV、AUC 或 PSI。

| 参数 | 位置 | 含义 |
| --- | --- | --- |
| `segment_col` | 运行方法 | 本次运行的客群列，默认 None |
| `binning_scope` | 拟合入口的构造策略、便捷函数 | `shared` 共用整体规则；`per_segment` 各自学习规则 |
| `selection_scope` | StatsSelector 构造策略 | `global` 只筛整体；`per_segment` 分别筛整体与各客群 |

有 `benchmark_df` 时，分箱用它拟合。建议显式传入开发样本，不能仅凭 Train 等分组名称推断。
shared 模式下，整体和客群共享切点、类别及已学转换映射。客群内的统计 WOE 描述当前人口，
不会覆盖转换用的映射。per_segment 的 All 仍使用整体规则，子客群使用各自基准人口的规则。
同一 scope 在评估期和其他目标中复用固定规则，不重新拟合。
独立规则适合研究客群内区分度；不同 scope 的箱号不能直接当成相同区间比较。

PSI 的 expected 始终匹配当前 scope。共享规则不表示共享 expected 分布。
seg2 在 benchmark 中没有样本时，seg2 的 PSI 为 null，并在 `calculation_status` 记录原因，
不会借用整体分布。无 benchmark 时，各客群使用同一个整体首组；缺该组则记录缺参考。
RC 的 total、benchmark 和 first_group 同样使用当前人口。
PSI expected 与 RC 参考分别判断。缺少某个客群的 benchmark 会使该客群 PSI 不可计算，
但它在当前人口中的 first_group 或 total RC 仍可有效。反过来，缺整体首组也不会清空有效 benchmark PSI。

每个目标、特征和分组（含 Total）分别检查指标适用性。当前有效权重只有一种标签时，
IV、KS、AUC 及对应逐箱贡献为 null；零有效权重、无标签和单类别保留各自原因。
人数、分布、合法坏率、金额坏率及独立有效的 PSI 继续保留，Lift 按自己的分母判断。
规则中已学的 WOE 转换映射不随当前客群的统计 WOE 改变。
某期排除缺失／特殊箱后没有适用人口，该期 PSI 为 null；其他有效期和 Total 继续计算。
原始表、图、HTML、Excel 和恢复快照保留这些 null，不能将它解释为稳定的 0。

数据画像 PSI 的退化诊断检查实际规则拟合人口。显式 benchmark 整体为常量或全缺失时，
shared 模式也保留失败状态；整体拟合有效时，某个客群局部 benchmark 为常量仍允许使用共享规则。

数据画像在平滑之前，按当前 scope、feature 和 group 检查适用人口。
某期排除缺失／特殊箱后没有观测，只将该特征、该期设为 Float64 null，
并记录 `not_computed / no_effective_current_population`。原期间列保留，Total 使用跨期合并计数重算。
参考分布无适用人口时，该特征的各期和 Total 均为 null，原因为
`no_effective_reference_population`。无显式 benchmark 时，参考仍是整体首组对应的客群人口，
不会换成后续有数据的期间。批次大小、同批特征和特征顺序不改变这些结果与状态。

`psi_include_missing` 和 `psi_include_special` 是 `generate_profile()`／`profile_stats()` 的运行参数。
纳入对应箱后，只要分布获得适用人口，就按原 PSI 公式计算；规则拟合失败仍单独解释。
`get_table("calculation_status")` 可按 scope、特征和具体 group 查询原因，
`show_trend()` 只显示当前投影涉及的状态。旧快照按已保存统计读取，加载时不重算历史数值。

分布与缺失使用完整人口。风险表现使用当前目标已观察标签，保留权重和金额的现有定义。
多目标仍由首目标拟合；当前趋势只覆盖首目标。均值、缺失率等数据统计不受 binning_scope 影响，
数据画像的 binning_scope 只用于 PSI。没有请求 PSI 却传独立分箱模式会报错。
声明的目标缺列或全部未表现时，沿用无标签评估，保留目标身份及 null 事件率，
不注入伪造标签。非法标签或角色冲突仍按输入契约拒绝。

## 筛选与转换

| selection_scope | binning_scope | 结果 |
| --- | --- | --- |
| global | shared | 只筛整体，客群列只登记角色并排除候选 |
| per_segment | shared | 独立漏斗与名单，各阶段共享整体固定规则 |
| per_segment | per_segment | 独立漏斗、名单与规则 |
| global | per_segment | 拒绝，本期没有客群证据合成整体名单的规则 |

每个 scope 从原始声明候选集开始。shared 模式按阶段所需候选并集学习规则，
整体已淘汰的特征仍可进入客群漏斗。各客群的 WOE 更新与剪枝不影响其他 scope。
完整 WOE 相关矩阵覆盖当前 scope 的完整适用人口和全部剩余候选，列批不会拆散筛选池。

`selected_features_`、默认 `transform()` 和 `get_report()` 仍表示 All。
`selected_features_by_segment_` 以 `scopes_` 中的稳定 `scope_id` 为键。
`[]` 表示成功得到空名单，`None` 表示客群未成功完成。不能把失败名单当成 All。

```python
selector.fit(df, target="y", features=features, segment_col="segment",
             group_col="split", benchmark_df=development)
seg2_data = selector.transform(df, segments=["seg2"], include_all=False)
seg2_decisions = selector.get_report(segments=["seg2"], include_all=False)
seg2_correlation = selector.get_correlation_report(segments=["seg2"], include_all=False)
```

显式客群 transform 仅返回该客群的行，保持相对行序，并使用该名单。
`keep_target` 和外部 y 按同一行身份对齐；`fit_transform` 采用同样的选择。
不同客群列集合不会自动取并集、交集或逐行拼接。
重新生成分箱报告时，仍须显式重传原 benchmark，筛选器不保存原始宽表。
相关矩阵、邻居查询和漏斗每次只选择一个 scope。Linear 和 Importance Selector 暂不支持独立客群筛选。
同一 StatsSelector 可重复 fit。公开入口在验证新输入前重建运行状态，失败后也不暴露旧规则或名单。
内部子任务保留本次统一首组与 scope；切换策略、分组或 benchmark 后，结果与同参数的新实例一致。

## 查询、绘图和导出

开启客群的完整公共表新增 `scope_id`，`scopes` 目录保存原始 dtype、受限编码值、名称、顺序和状态。
真实字符串 All、Total、Missing、null、空字符串与缺失客群彼此独立。
null 和 NaN 归为缺失客群；字符串 NaN 保留原值。嵌套对象、混合不支持对象和无穷值会报错。
客群列自动排除于默认特征，显式声明成特征、目标、分组、日期、权重或金额会报角色冲突。

`get_table()` 默认保留完整表。人工摘要、Notebook 图和 AI context 默认选择 All。
`segments` 接收业务原值列表，`include_all=False` 表示只取子客群。各条件取交集。
分页引用记录有效 scope 条件，跨 scope 的投影仍保留身份。
未指定 `segments` 时，可用 `filters={"scope_id": identity}` 直接选择客群；
`include_all=False` 始终排除整体 ID `all`。`segments=[]` 明确不选任何子客群，filter 不能恢复它们。
显式 `scope_id=None` 是 null 条件。它与没有 scope filter 不同，也不会自动改成 All。
`show_trend()` 的状态只解释当前显示的客群、特征和期间。

```python
page = report.query_page("summary", segments=["seg2"], include_all=False,
                         features=["income"], limit=5)
replayed = report.get_table("summary", **page["reference"]["query"])
figures = report.build_risk_trend_figures("income", segments=["seg2"], include_all=False)
report.write_excel("seg2.xlsx", segments=["seg2"], include_all=False)
report.write_html("segments.html", segments=["seg1", "seg2", "seg3"])
```

图形沿用原有字体与布局，每个 feature/target/scope 一张图。
`include_total` 只控制当前 scope 的 Total，`max_plots` 限制实际图总数。
HTML 各人口复用原离线页面，通过客群选择器切换；来源、中文搜索、特征、目标与分组在当前人口内生效。
原生 Excel 一次只导出一个 scope，保留 22 列模板；请循环导出独立工作簿。
同时请求整体与子客群会报错。客群说明在语义附页，不插入模板列。

`.marsreport` 保存完整统计、目录、参数与持久身份，不恢复原始宽表或 estimator。
格式仍为 1，历史无客群文件继续按原合同恢复。恢复后的 Excel 是带 scope 的静态长表，
不能代替原生透视模板。分箱快照的 `build_risk_trend_figures` 使用已保存统计 adapter。
独立规则通过 `MarsRiskProfile.save_binning_rules/load_binning_rules` 保存和加载，用于固定转换，
报告快照没有这个职责。缺规则或未知客群不会隐式重新拟合。

## 可运行的完整示例

示例构造三个模拟客群，包含整体与两种分箱模式、独立名单、缺客群参考、规则转换，
以及保存后新进程分页、预算 Agent 查询、HTML、Excel、图片和证据重放。
另保存有效开发规则下的单类别客群和某期无适用 PSI 人口；新进程直接读取快照复验局部 null。
无需 API Key，不接入 LLM。运行时可加 `--output-dir output/segmented` 保留产物。
完整示例的 Agent 消费需要 Python 3.10+；核心画像、筛选与快照查询仍支持 Python 3.8。

```python
--8<-- "docs/snippets/segmented_analysis.py"
```

shared 风险画像按特征批复用当前和参考的分箱转换；数据画像 PSI 复用批内频数。
各 scope 的风险基础聚合和统计筛选阶段仍分别执行，新增客群会增加计算和报告成本。
资源与环境证据、未完成项见[本地实施记录](../project/segmented-analysis-implementation.md)。
模型分交叉、规则评估和缺失异常诊断保留在后续批次。

下面的短例子只构造 80 行模拟数据。seg2 的 x 在 02 期全缺失，01 和 Total 保持有效，
并比较 shared／per_segment 在 batch=1、2、8 下的统计与局部状态。

```python
--8<-- "docs/snippets/profile_psi_population.py"
```
