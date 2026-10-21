---
description: 核心分析与报告消费的合成规模验收、独立进程 RSS、原始轮次和使用边界。
---

# 核心规模与报告消费验收

以下记录来自 2026-10-01 的验收，使用合成数据，不能代表真实业务性能。
验收起点与参考源码均为 `26d0f2d1f38c1a0483fe7c9baa2a2b68124d9909`。开始时工作树干净，修复后的测量使用尚未提交的工作树代码。
我们用 `git archive` 将参考源码提取到临时目录，再让测量脚本（harness）通过 `--source` 切换源码。
记录包含实际导入的 `mars.__file__`、源码哈希，以及两个 harness 文件的哈希；参考提交之后的代码未被覆盖。

## 环境与测量方法

运行日期为 2026-10-01 起（Asia/Shanghai；每轮 JSON 单独记录 UTC 时间）。环境为 Windows 11
10.0.26200、Intel Core i7-14650HX、24 逻辑 CPU、63.7 GiB 物理内存；Python 3.10.19、
NumPy 2.2.6、Pandas 2.3.3、Polars 1.37.1、scikit-learn 1.7.2、psutil 7.2.2。
PyArrow 等依赖的实际版本、每轮可用内存和原生线程池见 JSON。环境中没有检测到 cgroup，也未暴露 Windows job 的限制，
因此物理内存容量不能保证容器实际可用的内存。

固定随机种子为 `20261001`，线程配置为 4。在导入 NumPy 和 Polars 前，先设置 Polars、OMP、OpenBLAS、MKL、
NumExpr 和 VECLIB 的环境变量，再记录 Polars 实际线程数，以及 threadpoolctl 检测到的原生线程池。
每个案例默认限时 300 秒，进程树 RSS 预算为 8 GiB。若保守估计已明显超过预算，脚本将状态记为 `not_run`。
实际运行超限或超时时，脚本终止 worker 及其子进程，保存阶段、退出码和有限日志，再继续其他案例，不自动降低规模或无限重试。

预算状态与操作系统退出状态分别记录。直属 worker 的退出码只由它的 `Popen` 对象回收；
psutil 负责发现和控制后代进程，并在限定时间内等待后代，不等待直属 worker。
未知退出状态保存为 null，并注明 unknown，不补成 0 或固定非零数。
worker 自然退出后，脚本仍会清理已经发现的后代。读取诊断或清理失败时，记录原因、已完成阶段和残留进程，避免无限等待。

每个案例、输入后端、预热和测量轮次都使用独立 Python 进程，大型案例串行运行。
预热结果单独记录，不参与中位数计算。正式 standard 与报告 large 的两侧各运行 3 轮；扩规模诊断只有 1 轮时，仅记录单次观察。
单轮结果不能支持稳定收益结论，三次样本也不能用于推算 P95。下文的复核阈值不替代原有规则发布门禁。

主要内存口径是观察到的 RSS，配置采样间隔为 10 ms，包含 Arrow、NumPy 和 Polars 的原生内存。
系统调用与调度可能延长实际间隔。主进程 RSS 与进程树 RSS 总和分别记录；共享页可能重复计数，瞬时峰值可能漏采，两者都不等于独占物理内存。

各阶段记录起始 RSS、结束 RSS、观察峰值和相对起始值的增量。全流程包括导入、原始宽表准备、正确性验证、保存，以及外部加载和查询。
嵌套阶段不能相加作为总耗时。公开计算入口中的报告装配计入 `public_compute`；规则和注入矩阵场景另测 `report_construct`，不计作规则挖掘或 `selector.fit` 的耗时。

## 统一 CLI 与实际 workload

入口是 `benchmarks/benchmark_core_capacity.py`，`benchmark_core_workloads.py` 只提供固定场景和数据。
用 `--help` 查看规模档位，用 `--case` 选择一个或多个案例；`--backend` 可选 `pandas`、`polars` 或 `both`。
默认输出路径带微秒时间戳。显式指定路径时，脚本也会拒绝覆盖已有文件。

```bash
python benchmarks/benchmark_core_capacity.py --suite all --scale smoke --backend both --repeats 1
python benchmarks/benchmark_core_capacity.py --suite all --scale standard --backend both --repeats 3
python benchmarks/benchmark_core_capacity.py --suite reports --scale large --backend both --repeats 3
python benchmarks/benchmark_core_capacity.py --suite analysis --scale large --backend polars --repeats 1
python benchmarks/benchmark_core_capacity.py --case rule_report --scale large --backend polars --repeats 3 --timeout 300 --memory-budget-mib 8192
python benchmarks/benchmark_core_capacity.py --case profile_columns binning_rows --scale large --backend polars --repeats 1
```

要在同一环境中对照两版源码，或用当前源码恢复旧快照，先将参考提交的 src 提取到独立临时目录，再运行：

```bash
python benchmarks/benchmark_core_capacity.py --suite reports --scale large --source /path/to/reference/src --backend both --repeats 3 --output /tmp/reference.json
python benchmarks/benchmark_core_capacity.py --suite reports --scale large --baseline-json /tmp/reference.json --backend both --repeats 3 --output /tmp/current.json
python benchmarks/benchmark_core_capacity.py --case rule_report correlation_long --scale smoke --source /path/to/reference/src --consumer-source ./src --backend polars --repeats 1
```

Windows 使用可写临时路径替换 `/tmp`；本轮实际解释器位于 Conda `mars` 环境。
对照时会核对规模、随机种子、线程、超时、预算、依赖和硬件。规则与相关性报告逐表检查 schema、行列顺序，以及完整数据的原生批次哈希。
随机生成的 report_id 和时间不参与数值差异比较，但同一快照中的身份必须精确保留。
早期 harness 把已安装 `mars-risk` distribution 也算入依赖，参考 src 没有 egg-info 时取到了本机旧包
`0.0.16`，拒绝产生比值。实际导入路径与参考源码哈希已核对，目标源码版本均为 `0.0.28`；修正后
目标源码版本与已安装 distribution 单列，环境对照仍严格检查所有数值依赖。该修正只影响元数据校验，
不改变采样、计时、数据或算法。Pandas 交叉基准的半样本包含端点错误另行修正后重新建立分析对照，
原调试记录不参与正式比较。

基线工具记录工作负载、独立进程测量方式、快照加载与查询的检查项，以及每轮生效的算法参数、诊断分支、资源策略和计算依赖。
线性诊断区分四种情况：包未安装、已安装但导入失败、成功执行，以及明确跳过。记录中保留 VIF 和系数表的实际行数；输入尺寸相同，不能证明工作量相同。

源码和 harness 哈希只用于追溯来源，优化前后的生产代码 commit 无需相同。仅修改注释或日志，也不会使对照失效。
更改测试夹具、测量口径或加载与查询任务时，必须更新对应的比较约定，不能只沿用旧标识。

`comparison` 分别记录可比较、不可比较、执行失败、语义不一致，以及有意改变资源策略的对照。
诊断或工作量不同时，记录具体差异路径，不生成常规提速比例或退化结论。若有意改变线程、batch_size、n_jobs 等资源策略，需用 `--comparison-purpose` 说明目的。
输出会保留这些差异，不将结果当作完全相同配置的对照。历史 JSON 缺少新的比较字段时，明确标注兼容信息不足，不猜测补齐，也不覆盖旧轮次。
2026-10-01 正式线性对照的两侧均记录 `optional_diagnostics=available`；这次检查加固不影响已经发布的性能记录。

| 案例 | smoke | standard | large | 实际计算路径 |
| --- | --- | --- | --- | --- |
| rule_report | 1,000 审计行 | 10,000 | 50,000 | 合成真实契约统计 → to_report；不挖掘全部候选 |
| rule_mining / rule_states | 480 行 | 2,000 | 4,000 | 独立 train/validation，真实 mine_rules，候选预算 50 |
| correlation_short / long | 40 特征 | 500 | 3,000 | 固定 1,500 行 NumPy 矩阵 → 完整关系报告 |
| correlation_1000_short / long | 40 | 1,000 | 默认不重复执行 | 同一矩阵注入诊断 |
| profile / binning | 800×8 | 50,000×200 | 200,000×1,000 | generate_profile / evaluate，基础确定性分箱 |
| *_batch100 | 同上 | 同上 | 同上 | batch_size=100；默认案例为 50 |
| selection | 800×8 | 50,000×200 | 200,000×1,000 | 完整 MarsStatsSelector.fit，默认筛选步骤与阈值 |
| linear_selection | 400×8 | 3,000×40 | 5,000×80 | 真实 MarsLinearSelector.fit，含默认可用诊断 |
| score_cross / wide50 / wide500 | 2,000 行 | 1,000,000 | 1,000,000 | cross_scores，7 个必要列，加 0/50/500 个真实无关列 |
| optimal_binning | 1,000×4 | 5,000×10 | 10,000×20 | 独立监督分箱 workload，不与基础算法混算 |
| profile_columns / binning_columns | 800×8 | 50,000×3,000 | 同 standard | 显式单独扩列 |
| profile_rows / binning_rows | 800×8 | 1,000,000×100 | 同 standard | 显式单独扩行 |

500、1,000、3,000 特征分别保存 124,750、499,500、4,498,500 对完整关系。短名约 5 字符，
长名约 69 字符；完整候选顺序、相关系数符号、真实对角和 unavailable 保留，未改成 top-k 存储。
注入矩阵时，缺失值按 NumPy 的 NaN 传播规则处理。真实 selector 的样本范围、数据表示和缺失处理由报告参数记录。
注入矩阵不执行完整筛选，测量结果需与真实 selector 分开解释。

分析宽表由 8 个潜变量和噪声生成，混合 float32/float64（每 5 列有一列 float64），包含少量常量列和低基数列。
NaN、-999 和合法的 0 注入到固定位置。主标签约有 1/19 未表现，辅助标签约有 1/7 未表现；数据还包含 4 个客群分组、均匀正权重、金额和重复客户。
交叉分数额外包含 ±inf。画像只测量现有支持的能力，权重和金额用于评估与交叉入口。
每列固定随机数生成器（RNG），扩宽时必要列、标签和权重保持不变，避免其他数据变化影响宽度对照。

规则统计覆盖同一 ID 的两轮审计、单特征与多特征规则、入选与淘汰、train/validation、两个标签和客群切片。
32 个长英文 ID 对应多个业务 data_source，其中一些中文显示名重复。`rule_report` 始终接收 Polars 合成统计；requested backend 的两侧用于诊断重复加载与查询，不能解释为 Pandas 与 Polars 的原始数据挖掘对比。
真实规则和宽表入口才对照实际输入后端。结果表的后端与体积逐表记录；即使输入是 Pandas，产生的 Polars 报告仍标记为 Polars 表。

## 已解决的问题与正确性证据

### 已完成的报告 large 对照

下表展示 Polars 输入的结果：两侧各测量三轮，耗时取中位数，RSS 取阶段内观察到的最高值。另一后端的全部轮次也保留在原始记录中。
合成规则统计不用于比较后端性能。两侧原始报告表完全一致，但参考规则案例在 5,000 字符预算下未保留证据，整体状态为 failed。
表中只引用参考案例已经完成的构造阶段，不将部分成功记为案例通过。

| 场景 | 参考构造 s | 当前构造 s | 参考阶段峰值 MiB | 当前阶段峰值 MiB | 当前表存储 MiB | 快照 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 行规则审计 | 1.554 | 1.452 | 1246.1 | 1223.6 | 198.8 | 14.59 |
| 3,000 特征短名，4,498,500 对 | 2.378 | 0.321 | 1072.0 | 597.1 | 108.4 | 65.73 |
| 3,000 特征长名，4,498,500 对 | 3.429 | 0.303 | 3973.3 | 606.2 | 108.6 | 66.90 |

长名关系构造阶段的 RSS 下降约 84.7%，中位耗时下降约 91.2%。这项收益来自原生 Enum 构造，不能代表完整 `selector.fit`，也不能代表报告加载与查询全程的收益。
规则构造变化较小，不能得出稳定提速的结论。投影修复的直接证据是：未入选统计不再全部展开为 Python 字典。

| 当前场景 | 保存 s | 新进程冷加载 s | 有限页 ms | 5k 上下文 ms | 16k 上下文 ms | 有限 HTML ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 审计 | 0.129 | 0.097 | 14.6 | 23.8 | 17.0 | 14.5 |
| 3,000 短名 | 0.128 | 0.167 | 12.1 | 35.4 | 31.0 | 170.5 |
| 3,000 长名 | 0.113 | 0.175 | 11.6 | 36.5 | 27.6 | 152.4 |

规则审计的去重 ID 为 33,333，桥接 44,444 行、evaluation 399,996 行、slices 799,992 行；
后部来源交集查询返回 20 / 2,088 行，next_offset=40，5k 保留 1 行、16k 保留 20 行。
相关性后部查询返回 20 / 2,999 行，next_offset=40；固定 1,500 行矩阵计算另记为约 0.14–0.16 秒。
全流程规则主进程最高观察约 2271 MiB，包含生成合成统计时的大量 Python 行；相关性全进程主进程
峰值应取阶段与旁路采样的最大值（约 606 MiB），含独立消费者的进程树约 1154 MiB。
旁路采样和阶段采样可能漏到不同瞬时峰值，不能只选较小的旁路数值。

首次查询耗时和父进程等待消费者时的 RSS，有少数项目触发了 20%/15% 的复核阈值。原始 comparison 保留触发项，复核结果单独记录。
构造阶段的收益不能抵消加载与查询的成本。新的 5k 输出保留了有效证据，参考输出为空，两者不用于比较速度收益。

### 真实公共分析入口与分维度扩展

standard 两侧各测量三轮。下表记录当前 `public_compute` 的中位耗时（秒）和阶段内主进程的最高 RSS，耗时已包含入口内部的报告装配。
画像和分箱的输入均为 50,000×200，交叉输入为百万行。表体积按报告的实际后端估计。

| standard 案例 | Pandas 计算 s / RSS MiB | Polars 计算 s / RSS MiB | Polars 表 MiB / 快照 MiB | Polars 冷 load ms / 页 ms |
| --- | ---: | ---: | ---: | ---: |
| 画像 batch50 | 0.888 / 371.1 | 0.757 / 370.2 | 0.115 / 0.121 | 41.3 / 0.70 |
| 画像 batch100 | 0.897 / 418.5 | 0.736 / 413.1 | 0.115 / 0.121 | 36.8 / 0.86 |
| 基础分箱 batch50 | 1.560 / 790.8 | 1.467 / 774.0 | 2.658 / 1.412 | 47.2 / 0.76 |
| 基础分箱 batch100 | 1.566 / 1134.6 | 1.511 / 1142.7 | 2.658 / 1.415 | 63.2 / 0.75 |
| 统计筛选（资源继承修复后） | 20.387 / 776.3 | 20.308 / 795.9 | 0.076 / 0.104 | 见原始轮次 |
| 线性筛选 3,000×40 | 0.861 / 250.3 | 0.837 / 253.1 | 0.036 / 0.050 | 33.7 / 0.47 |
| 交叉：必要列 | 0.266 / 647.9 | 0.165 / 501.0 | 0.148 / 0.116 | 32.9 / 2.10 |
| 交叉：加 50 列 | 0.277 / 889.4 | 0.202 / 722.6 | 0.148 / 0.116 | 37.7 / 2.32 |
| 交叉：加 500 列 | 0.290 / 2948.6 | 0.172 / 2821.8 | 0.148 / 0.116 | 38.9 / 1.93 |
| 监督分箱 5,000×10 | 6.573 / 260.4 | 6.621 / 257.0 | 0.088 / 0.103 | 39.2 / 0.69 |

统计筛选在两个后端均从 200 个输入特征选出 50 个相关性候选，最终保留相同的 50 个特征及顺序。
画像、评估和矩阵各计算一次，评估复用已经拟合的 binner。线性筛选从 40 个特征中保留全部 40 个，仍执行默认相关性分析，以及可用的 VIF/Logit 诊断。
各预筛阶段的特征数和最终英文 ID 顺序保存在每轮 JSON 中。

交叉计算所需输入的实际体积为 Pandas 100.1 MiB / Polars 46.0 MiB；增加 500 列后为 2389.0 / 2334.8 MiB，
差异包含 Pandas 表示客群字符串的成本。拦截转换后确认，主样本只传入 7 列，半样本参考只传入 2 个分数列。
每轮拟合两个轴，再通过一次联合聚合同时处理 bad/late。两侧报告与固定分段的政策回放结果一致。
原始宽表准备约 8 秒，逐轮值见 JSON，不能用约 0.2 秒计算入口隐去准备成本及约 2.8 GiB RSS。

large 分析以及扩行、扩列的每个案例，都只有一轮测量和一次独立预热。下表是单次观察，不能视为稳定分布。
本轮扩行、扩列使用 Polars；除与 standard 相同的百万行交叉外，large Pandas 分析仍未验证。

| 案例与实际输入 | 计算 s | 计算阶段主进程 RSS MiB | 表 MiB / 快照 MiB | 状态 |
| --- | ---: | ---: | ---: | --- |
| 画像 200,000×1,000，batch50 | 12.546 | 1495.1 | 0.575 / 0.446 | passed |
| 同上 batch100 | 12.451 | 1704.8 | 0.575 / 0.446 | passed |
| 基础分箱 200,000×1,000，batch50 | 24.958 | 3246.1 | 13.320 / 6.707 | passed |
| 同上 batch100 | 25.192 | 5508.7 | 13.320 / 6.725 | passed |
| 统计筛选 200,000×1,000（修复后） | 124.625 | 4234.7 | 0.934 / 0.760 | passed |
| 画像单独扩列 50,000×3,000 | 13.139 | 1149.2 | 1.725 / 1.257 | passed |
| 分箱单独扩列 50,000×3,000 | 50.427 | 1858.7 | 40.031 / 20.395 | passed |
| 画像单独扩行 1,000,000×100 | 6.749 | 1816.0 | 0.058 / 0.083 | passed |
| 分箱单独扩行 1,000,000×100 | 10.759 | 5208.0 | 1.321 / 0.734 | passed |
| 线性筛选 5,000×80 | 3.649 | 259.9 | 0.107 / 0.100 | passed |
| 监督分箱 10,000×20 | 10.415 | 284.9 | 0.193 / 0.155 | passed |

large 的百万行交叉三个宽度也各完成一轮（计算 0.162–0.183 秒），尺寸与 standard 相同，
不能将其算作新增行数容量。large 统计筛选为 1,000→250→250，保留完整相关性证据和最终顺序。

在本机上，基础分箱 batch100 没有耗时优势。standard 阶段的主进程峰值比 batch50 增加约 48%，large 则由 3246 MiB 增至 5509 MiB。
使用类似的分组、权重和基准配置时，建议从 50 开始测量；这个建议不能保证在其他机器、指标或算法上仍是最优参数。
我们还对 50,000×200 的两个后端逐表比较全部报告，按业务键对齐，浮点容差为 1e-8，结果全部通过。
原生表哈希因浮点归约产生的细小差异，没有被误判为统计结果变化。

### 筛选预算失败与修复

首次 large 筛选预热在 `public_compute` 约 45 秒被采样保护终止，退出码 15，主进程峰值
4010 MiB、进程树峰值 8193 MiB；状态为 `memory_budget_exceeded`，计划测量轮次为 `not_run`。
日志显示画像、粗筛及精筛预分箱已完成，没有将该退出猜作操作系统 OOM 或正常无规则。

源码显示，精筛没有把 selector 的 `n_jobs=4` 传给最优分箱器，后者因此使用默认值 `-1` 启动 loky 进程池。
修复后，分箱器默认继承 selector 的配置；显式传入的 `binning_params['n_jobs']` 仍优先。候选数、精筛步骤和阈值均保持原样。
复跑使用与首次失败相同的输入、harness 和预算，完整通过：计算耗时 124.625 秒，阶段进程树峰值为 5378.5 MiB，主进程为 4234.7 MiB。
这里只测量了一轮，不能据此得出稳定容量或提速比例的结论。原始失败与修复后的复跑记录分别保存。

standard 原参考 Polars 计算中位 17.999 秒、修复后 20.308 秒，约增加 12.8%；Pandas
18.568→20.387 秒，约增加 9.8%。这是执行既有资源限制的吞吐权衡，没有宣称速度收益。
精筛任务的并发配置控制进程数，4 线程环境配置控制各计算库在单进程中的线程数，两者分别生效。
独立数值核对记录精筛 binner 实际 n_jobs 从 23 变为 4。
独立最优分箱诊断保留原公共入口的默认进程配置；不能将所有场景解释为全进程树仅有四个线程。

### 实施范围

1. 完整相关性报告此前先将长名展开为平方级 NumPy Unicode 数组，再转为 Enum。现在从原生整数端点
   直接构造既有 Enum 类型，用 Polars 构造相关值和状态，保留全部关系、schema 和顺序；快照格式不变。
2. 规则解释曾将全部验证命中统计转 Python 字典，然后筛入选 ID。现在先在原生表中筛选，只展开
   需要的入选规则；候选、审计和切片完整保留。测试拦截 to_dicts，确认只转换 5 条解释所需统计。
3. 5,000 字符上下文此前会删光证据，却仍返回正常 JSON。现在根据保留行删除无关的业务元数据，重复的 null
   定义只在表级保存一次，并优先裁剪可以继续查询的说明与导航。每个非空查询至少保留一条完整证据；
   预算容不下必需口径和一条证据时，明确拒绝请求。完整 describe 和快照元数据保持原样。

上下文预算按最终 Unicode 字符数计算。验收测量 5,000 和 16,000 字符的输出，并单独记录 512 字符下明确拒绝的结果。
包含全部 45 列的规则评估表，在 7,000 字符预算下无法保留一条完整证据，因此会拒绝请求；只选择所需列后，可以返回有效证据。
原始 JSON 记录省略原因、行数和继续查询的位置，空证据不记为 passed。

小规模数值参考包含 NumPy 有效值均值，独立的已表现标签/权重/坏样本/金额合计（rtol/atol 1e-6），
交叉样本数和已表现分母精确整数检查，signed 子矩阵 1e-12。其他统计口径由定向与基础回归覆盖。
规则 sources 筛选验证特征与来源命中同一规则的不同成员，统计行不展开；不同轮次同 ID 保留审计粒度。
无候选、全部淘汰、有候选未入选的状态都保存后在新 Python 中查询，并保留高级关系 not_computed 状态。

每个成功案例保存后，都另启解释器，只提供快照、查询配置和源码安装位置。
新解释器不导入 workload 模块，也不调用数据生成、拟合或挖掘。它逐表验证数据、schema、顺序、report_id 和 describe，再按有限查询的引用重放证据。
相关性还恢复 30×30 矩阵和 20 个关联特征；交叉报告查询 cells、显示有限矩阵并回放固定分段政策。
query_page 使用 limit=20，检查后部 offset、投影、筛选和排序，只将筛选后的有限页展开为 Python 对象或 JSON。
完整 HTML/Excel 只在少数 smoke 小报告执行；平方级报告只展示有限表或 30×30 矩阵。

## 同进程重复消费和资源边界

每个独立消费者先预热，再至少执行 10 次 load → query_page → context → release，逐轮记录加载、查询和上下文的延迟，以及 RSS。
新进程冷加载与同进程重复操作分别记录。内存池可能保留内存，执行 del 后不要求 RSS 归零。
判断泄漏时，先比较后半段的变化范围和逐轮轨迹；若仍持续增长，再排查引用或缓存，不能只看某次末值。

3,000 长名额外重复 100 次：释放后 RSS 首轮 593.5 MiB、末轮 617.5 MiB、最高 621.0 MiB；
每十轮平均值为 603.2、610.2、606.0、611.1、612.4、605.4、613.8、612.4、611.9、613.0 MiB。
本次观察符合内存池逐渐进入波动平台，没有后半段持续单向增长；仍不能证明任意长会话不存在泄漏。

large 短名的首次页耗时触发 20% 复核阈值后，重新测量了三轮：参考为 10.57/17.98/20.32 ms，当前为
10.28/13.88/22.03 ms，没有复现中位耗时退化。等待消费者时，父进程 RSS 仍增加约 16%；这些内存是在不同构造路径后保留的分配，应计入总预算。
规则首次页复测的中位耗时为 14.64→18.04 ms（约 +23%），仍触发复核。独立消费者的中位耗时为 20.94→13.45 ms，不能用这个结果抵消首次页的变化。
分页的生产实现没有变化，三个样本仍有明显噪声，因此保留这项限制，不能宣称加载与查询都已提速。
standard 中，少数毫秒级的保存、准备、哈希和首次页操作也触发比例阈值。原始值与触发项完整保留，但没有逐项扩大复测。
仅凭这些比例，无法判断是否存在稳定退化，也不能认定退化已经消除。

超时诊断使用 0.05 秒预算，准确记为启动阶段 timeout；1 GiB 参考长名矩阵触发构造阶段
memory_budget_exceeded；512 MiB 配置对大画像/分箱保守预估为 not_run，未分配宽表。
这些是保护机制验收，与 5k/16k 的有效证据成功、512 字符的明确拒绝分别记录。
本轮先验证 standard 双后端，再用单后端分别扩行和扩列。large Pandas 的新增规模、超过上述规模的输入，以及真实业务数据都未运行。
未运行不等于资源超限，也不能根据较小规模的结果推断通过。

建议先运行 smoke，再运行 standard，并显式设置本机预算。有限查询和排序仍可能扫描完整原生关系表，耗时不保证恒定。
批量 50/100 的结论只适用于本页固定的 dtype、分组、基准和指标配置，没有继续无限搜索参数。
默认 30×30 展示与 20 行分页是消费范围，完整关系仍在报告中。监督算法单独列出，不以关闭原本
应运行步骤或改变阈值取得性能收益。

耗时中位数增长超过 20%，或计算阶段最高观察 RSS 增长超过 15%，只表示需要复核。
应先检查各轮结果、阶段和样本规模，再用相同工作负载复测。采样预算不能保证瞬时值从不超限，异常退出也不能直接判断为 OOM。
较小规模通过不能替代未运行、更大规模或其他机器的验证。

## CI 与复现限制

普通 CI 增加三个独立 report smoke，分别检查规则报告、相关性和交叉宽表投影。
基础测试套件检查结构、版本隔离、预算终止，以及新进程加载和查询。push 不执行 large，可通过统一 CLI 手动运行。
Python 3.8–3.12 核心边界不变，Agent 保持 >=3.10；外部快照消费不依赖 mars.agent 或模型服务。
大型宽表、Parquet、快照与导出都在拥有的临时目录，结束清理，不纳入仓库。

本轮修改了报告构造、限定长度的上下文、规则解释前的数据投影，以及精筛资源参数的继承方式。
暂停模块未增加功能，也未增加旧入口别名。

回归结果：Python 3.10 基础套件 708 passed、1 skipped、7 deselected；Python 3.8
冻结依赖基础套件 583 passed、11 skipped、7 deselected，Agent 测试按版本隔离；
Python 3.12（Pandas 3.0.3、Polars 1.42.0）报告、规则、交叉、批次和筛选定向套件 127 passed、
1 deselected。没有可用 Python 3.9 解释器，其语法按 3.9 AST 检查、CI 保留真实版本矩阵，
不能描述为本机完整运行过 3.9。Ruff、154 个生产文件的 Mypy、公共与私有 docstring 检查、
严格 MkDocs 构建通过；私有 docstring 检查仍有六条既有提示。

## 2026-10-02 审查修复验证

本节记录基于 `b73491c80617c24767f289921cc7924f71c9737f` 的修复工作树，与前文 2026-10-01 的性能记录分别保留。
开始时，本地 HEAD 与远端 main 一致，工作树干净。本轮没有覆盖后续提交，也没有修改历史结果 JSON。
原 Linux CI 的两个 worker 用例将 psutil 提前回收的直属进程记为退出码 0；修复后，只由 Popen 回收直属进程。

行为回归先确认缺陷，再验证最终行为：

| 场景 | 修复后的实际结果 |
| --- | --- |
| 受控正常／异常退出 | 分别保留退出码 0／7，异常退出与预算判定分别记录 |
| timeout／memory_budget_exceeded | Windows 实测退出码 1、`exit_status=known`；后代停止、无残留 PID，`deliberate_wait` 阶段与诊断保留；不固定 Linux 信号码 |
| 重复值右闭最小箱复现 | 切点由共享左闭 `[2.0]` 转为参考样本真实前值 `[1.0]`；最终正常箱人数 4、6，`actual_n_bins=2`，约束诊断为 satisfied |
| 自定义 `inf`／`-inf` 缺失 | 首算、提取直接复用、相同配置显式复用，以及保存后新进程加载复用均为正常箱 2、missing 1、invalid 0；统计与政策回放相同 |
| key／feature 同列快照 | Polars／Pandas 保存前后均能筛选、分页和生成证据；最小复现证据为 `feature=x, mean=2.0`，原统计行不展开 |

最小箱检查只针对拟合参考样本。显式切点或复用固定定义时，边界保持不变；OOT 不重新约束箱占比。
无法完成约束的拟合保留失败／未满足诊断。默认 native 分箱原本未开启
`merge_small_bins` 的路径不增加边界适配扫描，既有容量 workload 的计算路径与报告形状未改变。

本地环境为 Windows 11、Intel i7-14650HX（24 个逻辑处理器）、约 63.7 GiB 内存。
各解释器使用当前源码；Python 3.8 的 psutil 7.2.2 从现有环境复制至临时依赖目录，
使用其兼容 ABI，不卸载或替换系统依赖。Python 3.10 的 DLL 导入需在获准的沙箱外进程执行。

基础套件命令为 `python -m pytest -q -m "not docs_ml and not optional_ml"`，
同时指定独立临时目录、禁用 pytest 缓存。Python 3.10 指定 `tests` 收集目录，
避开既有忽略产物的目录 ACL；仓库所有测试文件均位于该目录。

| Python | Pandas／Polars | 最终基础套件 |
| --- | --- | --- |
| 3.8.20 | 2.0.3／1.8.2 | 706 passed、13 skipped、7 deselected |
| 3.10.19 | 2.3.3／1.37.1 | 833 passed、1 skipped、7 deselected |
| 3.11.15 | 3.0.3／1.42.0 | 793 passed、5 skipped、7 deselected |
| 3.12.13 | 3.0.3／1.42.0 | 795 passed、3 skipped、7 deselected |

各环境安装的可选依赖不同，因此适用的用例数也不同。基础套件运行后，worker 清理诊断又得到补强；
四个版本分别重跑 `tests/test_core_capacity_benchmark.py`，均为 35 passed。
Python 3.12 的五个 Score Cross 定向模块为 113 passed，四个报告／规则／快照定向模块为
91 passed。最后补强的非有限缺失配置复用回归为 2 passed；四个导出临时目录用例也已复验。
首次并行运行基础套件时，四个旧导出测试因共享 `tests/_artifacts` 发生竞态。修复后，各测试使用自己的 `tmp_path`，原断言保留。
源码搜索路径错误和目录权限也曾导致早期运行失败，纠正后均重新运行。早期失败没有计为通过，原有业务断言未削弱。

统一容量入口实际执行了以下 smoke，每个案例都包含独立预热、测量、保存，以及新进程冷加载和查询验证：

```bash
python benchmarks/benchmark_core_capacity.py --case rule_report correlation_short --backend polars --scale smoke --repeats 1 --output <tmp>/report-smoke.json
python benchmarks/benchmark_core_capacity.py --case score_cross linear_selection --backend pandas --scale smoke --repeats 1 --output <tmp>/analysis-smoke.json
```

四个案例全部 passed。线性筛选成功导入 statsmodels 0.14.6，实际执行的 VIF 和系数诊断各返回 8 行；关闭的 stepwise 分支明确记录为跳过。
另外，使用原提交源码和修复源码执行相同的 `correlation_short / polars / smoke` 对照，七个阶段均为 comparable。
两侧表内容、schema 和顺序摘要一致，源码哈希变化没有使有效对照被误拒绝。
这是一轮正确性与比较工具验收，不能据此宣称提速。

在同一解释器内受控模拟 statsmodels 导入失败后，400×8 输入的 VIF 和系数诊断从各返回 8 行变为跳过并返回空表。
比较结果为 incomparable，保留依赖和诊断差异的原因，不输出常规耗时比例。
回归测试还验证：工作负载、生效参数、测量约定或实际计算依赖发生变化时，拒绝常规对照；明确说明目的的资源策略变化则单独标记。
执行失败、语义不一致和不可比较分别报告。旧基线缺少新的兼容字段时，说明信息不足；已发布正式线性对照的两侧诊断均为 available，原数字保持原样且继续有效。

Ruff 全范围检查、Mypy（157 个生产文件）、公共 pydoclint、私有 docstring 检查及
严格 MkDocs 构建通过；私有检查保留六条既有提示。`git diff --check` 通过。
离线探索器的实际 Node JavaScript 行为测试随定向／基础套件执行；未做真实浏览器交互验收。
本轮没有无条件重跑 standard／large，未新增大型二进制产物或重做历史性能结论。

原始日志、smoke JSON、对照 JSON、参考源码和站点构建产物保存在本机独立临时目录
`mars-review-20261002-l9zhqkcb`；Python 3.10 的运行使用独立临时目录。
没有本地 Python 3.9 或普通 Linux 环境；随后通过下面的远端运行补齐，不能将其写成本机通过。
本地没有单独执行 `docs_ml`／`optional_ml` 标记套件。

### 远端 Linux CI 补验

修复提交 `a99e0fc53f0864936a5e8200034df4838ad75ec7` 已推送到独立分支
`codex/review-fixes-capacity-20261002`，见 [草稿 PR #2](https://github.com/leeesq/mars-risk/pull/2)。
现有 Ubuntu CI [run 36910816160](https://github.com/leeesq/mars-risk/actions/runs/36910816160)
的 11 个任务全部 success，没有取消或跳过版本矩阵：

| Python | CI 基础套件 |
| --- | --- |
| 3.8.18 | 710 passed、13 skipped、7 deselected |
| 3.9.25 | 710 passed、13 skipped、7 deselected |
| 3.10.21 | 799 passed、3 skipped、7 deselected |
| 3.11.16 | 799 passed、3 skipped、7 deselected |
| 3.12.14 | 799 passed、3 skipped、7 deselected |

五个版本均实际执行最终 worker 回归，原 timeout／memory_budget_exceeded 用例保留
真实 known 非零退出状态、后代不再运行和诊断保留断言。日志没有打印具体操作系统退出码，
因此不推断或写死 Linux 信号码。规则、相关性、模型分交叉的隔离容量 smoke 全部 passed；
quality、modeling（110 passed）、distribution、两版 installed-wheel smoke 也全部 success。
[Docs run 36910816417](https://github.com/leeesq/mars-risk/actions/runs/36910816417)
为 43 passed，严格构建通过；PR 条件下部署任务 skipped，未部署站点。
本地基础套件与远端依赖组合不同，以上各自记录实际通过数，不用其中一组替代另一组。

## 原始结果索引

结果保存在 `benchmarks/results/`，包含每轮数值、预热、环境、实际参数和拒绝结果。
合并文件的 `source_runs` 记录每次实际调用的参数和原文件哈希。这些文件归档了多次串行调用，不能将合并记录解释为一次运行的总耗时。
diagnostics 保留调试失败的简短索引，正式基线和实际预算失败记录均保留。

| JSON | 内容 |
| --- | --- |
| `core_capacity_baseline_initial.json` | 首次生产优化前的 standard 报告基线，三轮值 |
| `core_capacity_smoke_verified.json` | 全部 smoke、双输入后端、独立预热与测量，34 个组合通过 |
| `core_capacity_standard_reference.json` | 同环境参考源码：报告与真实分析全部 standard |
| `core_capacity_standard.json` | 当前全部 standard：34 个组合，三轮，筛选资源修复后结果 |
| `core_capacity_large_reports_reference.json` | 50k 审计、3k 完整关系及真实小规模挖掘流程的参考结果 |
| `core_capacity_large_reports.json` | 同场景当前结果、十个组合、三轮、完整报告语义对照 |
| `core_capacity_large_analysis.json` | large/扩行/扩列、原始筛选预算失败及修复复跑 |
| `core_capacity_diagnostics.json` | 桥接、跨快照版本、超时/预算/not_run、消费复测、100 次释放、完整批次与筛选数值核对 |
| `core_capacity_validation.json` | 实际验证命令、版本、通过数和检查结果 |

批次与筛选的独立数值核对代码随 diagnostics 的 `verification_script` 保存，仅生成标准规模
验证证据；外部快照消费者不导入这些代码。快照、原始宽表及导出产物均未保存到仓库。
