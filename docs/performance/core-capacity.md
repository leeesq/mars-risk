---
description: 核心分析与报告消费的合成规模验收、独立进程 RSS、原始轮次和使用边界。
---

# 核心规模与报告消费验收

以下原始验收记录来自 2026-10-01，合成数据不代表真实业务性能。起点与参考源码均为
`26d0f2d1f38c1a0483fe7c9baa2a2b68124d9909`，开始时仓库工作树干净；修复后的测量来自未提交工作树。
参考源码由 `git archive` 提取到临时目录，harness 通过 `--source` 切换源码，并记录实际导入的
`mars.__file__`、源码哈希与 harness 两个文件的哈希。没有覆盖参考提交之后的代码。

## 环境与测量方法

运行日期为 2026-10-01 起（Asia/Shanghai；每轮 JSON 单独记录 UTC 时间）。环境为 Windows 11
10.0.26200、Intel Core i7-14650HX、24 逻辑 CPU、63.7 GiB 物理内存；Python 3.10.19、
NumPy 2.2.6、Pandas 2.3.3、Polars 1.37.1、scikit-learn 1.7.2、psutil 7.2.2。
PyArrow 等实际版本、每轮可用内存和原生线程池见 JSON。没有检测到 cgroup，Windows job 的限制
未暴露，不能将物理内存视为容器可用内存保证。

固定种子 `20261001`、4 线程。在导入 NumPy/Polars 前设置 Polars、OMP、OpenBLAS、MKL、
NumExpr 和 VECLIB 环境变量，并记录 Polars 实际线程数及 threadpoolctl 的原生线程池。
默认每案例 300 秒、进程树 RSS 8 GiB 预算。明显超预算的保守估计会记 `not_run`；实际超限或超时
终止 worker 与其子进程，保存阶段、退出码和有限日志，继续其他案例，不降档或无限重试。

预算状态与操作系统退出状态是两项证据。直属 worker 的退出码只由其 `Popen` 对象回收；
psutil 发现、控制并有界等待后代，不等待该直属 worker。未知退出状态保存为 null 并注明
unknown，不补成 0 或固定非零数。自然退出后仍清理已观察到的后代；读取诊断或清理失败时
记录原因、已完成阶段及残留进程，不让一个案例无限等待。

每个案例、输入后端、预热和测量轮次都是独立 Python 进程；大型案例串行运行。预热单列，不参加
中位数。正式 standard 与报告 large 每侧各 3 轮；只有 1 轮的扩规模诊断仅报告单次观察，
不作稳定收益结论，也不从三次样本推算 P95。旧规则发布门禁不被新复核触发线替代。

主要内存口径为配置间隔 10ms 的观察 RSS，包含 Arrow/NumPy/Polars 原生内存；系统调用与调度
可能延长实际间隔。分别记录主进程与
进程树 RSS 和，不混作独占物理内存，共享页可能重复计数，瞬时峰值可能漏采。阶段包含起始、
结束、观察峰值与相对起始增量；全流程包含导入、原始宽表准备、正确性验证、保存以及外部消费。
嵌套阶段不能相加作总耗时。公开计算入口产生报告时，报告装配包含在 `public_compute` 中；
规则和注入矩阵场景另外单测 `report_construct`，不会将这些时间称为挖掘或 selector.fit。

## 统一 CLI 与实际 workload

入口为 `benchmarks/benchmark_core_capacity.py`；`benchmark_core_workloads.py` 只提供固定场景与
数据，没有独立平台或插件层。`--help` 列出档位；`--case` 可选一个或多个案例，`--backend` 可选
`pandas`、`polars`、`both`。输出默认使用带微秒的时间戳，显式输出路径也拒绝覆盖。

```bash
python benchmarks/benchmark_core_capacity.py --suite all --scale smoke --backend both --repeats 1
python benchmarks/benchmark_core_capacity.py --suite all --scale standard --backend both --repeats 3
python benchmarks/benchmark_core_capacity.py --suite reports --scale large --backend both --repeats 3
python benchmarks/benchmark_core_capacity.py --suite analysis --scale large --backend polars --repeats 1
python benchmarks/benchmark_core_capacity.py --case rule_report --scale large --backend polars --repeats 3 --timeout 300 --memory-budget-mib 8192
python benchmarks/benchmark_core_capacity.py --case profile_columns binning_rows --scale large --backend polars --repeats 1
```

同环境对照及旧快照用当前源码恢复（先将参考提交 src 提取到独立临时目录）：

```bash
python benchmarks/benchmark_core_capacity.py --suite reports --scale large --source /path/to/reference/src --backend both --repeats 3 --output /tmp/reference.json
python benchmarks/benchmark_core_capacity.py --suite reports --scale large --baseline-json /tmp/reference.json --backend both --repeats 3 --output /tmp/current.json
python benchmarks/benchmark_core_capacity.py --case rule_report correlation_long --scale smoke --source /path/to/reference/src --consumer-source ./src --backend polars --repeats 1
```

Windows 使用可写临时路径替换 `/tmp`；本轮实际解释器位于 Conda `mars` 环境。
对照会核对规模、种子、线程、超时、预算、依赖与硬件；报告规则/相关性逐表检查 schema、行列
顺序和完整数据的原生批次哈希，随机 report_id 与时间不参与数值差异比较，快照内部身份必须精确保留。
早期 harness 把已安装 `mars-risk` distribution 也算入依赖，参考 src 没有 egg-info 时取到了本机旧包
`0.0.16`，拒绝产生比值。实际导入路径与参考源码哈希已核对，目标源码版本均为 `0.0.28`；修正后
目标源码版本与已安装 distribution 单列，环境对照仍严格检查所有数值依赖。该修正只影响元数据校验，
不改变采样、计时、数据或算法。Pandas 交叉基准的半样本包含端点错误另行修正后重新建立分析对照，
原调试记录不参与正式比较。

基线工具现记录明确的 workload、独立进程测量和快照消费合同，以及每轮有效算法参数、
诊断分支、资源策略和实际参与计算的依赖。线性诊断区分包未安装、安装但导入失败、成功
执行和明确跳过，保留 VIF/系数表的实际行数。相同输入尺寸不足以证明工作量相同。
源码及 harness 哈希只用于溯源；生产优化前后的 commit 无需相同，注释或日志变化也不会
单独使对照失效。更改 fixture、测量口径或消费工作时需更新对应合同，不能只保持旧标识。

`comparison` 区分可比较、不可比较、执行失败、语义不一致和有目的的资源策略对照。
诊断或工作量不同时保留具体差异路径，不生成常规提速比例或退化结论。线程、batch_size、
n_jobs 等资源策略的刻意变化需用 `--comparison-purpose` 说明目的，输出中保留差异，
不会当作完全同配置对照。历史 JSON 缺少新合同字段时明确标注兼容信息不足，不猜测补齐，
也不覆盖旧轮次。2026-10-01 正式线性对照的双方均记录 `optional_diagnostics=available`；
本次加固不据此推翻已经发布的性能记录。

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
注入矩阵的缺失处理明确为 NumPy NaN 传播；真实 selector 的样本范围、表示和缺失处理由报告参数记录，
两条路径不能混称完整筛选性能。

分析宽表来自 8 个潜变量与噪声，混合 float32/float64（每 5 列一列 float64），少量常量和低基数；
按确定位置注入 NaN、-999 和合法 0。主/辅标签分别约 1/19 与 1/7 未表现，4 个真实客群分组，
均匀正权重、金额和重复客户。交叉分数额外包含 ±inf；画像只测其现有支持能力，权重与金额用于
评估和交叉入口。固定每列 RNG，使扩宽时必要列、标签和权重不变，避免把宽度变化混为数据变化。

规则统计包含两轮同 ID 审计、单/多特征、入选/淘汰、train/validation、两个标签及客群切片。
32 个长英文身份对应多个业务 data_source 和重复中文显示名。`rule_report` 输入始终为 Polars
合成统计，requested backend 两侧是重复消费诊断，不能解释为原始 Pandas/Polars 挖掘对比。
真实规则与宽表入口才有实际输入后端对照。结果表后端及体积逐表记录；Pandas 输入产生 Polars 报告时
仍标记为 Polars 表，避免混淆。

## 已解决的问题与正确性证据

### 已完成的报告 large 对照

下面为两侧各三轮的中位耗时和最高阶段观察 RSS，展示 Polars 输入列；另一后端的所有轮次也保留。
规则合成统计不作后端性能比较。原始报告表完全一致；参考规则案例因 5,000 字符空证据整体 failed，
表中仅引用其已完成的构造阶段，不将部分成功称为案例通过。

| 场景 | 参考构造 s | 当前构造 s | 参考阶段峰值 MiB | 当前阶段峰值 MiB | 当前表存储 MiB | 快照 MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 50,000 行规则审计 | 1.554 | 1.452 | 1246.1 | 1223.6 | 198.8 | 14.59 |
| 3,000 特征短名，4,498,500 对 | 2.378 | 0.321 | 1072.0 | 597.1 | 108.4 | 65.73 |
| 3,000 特征长名，4,498,500 对 | 3.429 | 0.303 | 3973.3 | 606.2 | 108.6 | 66.90 |

长名关系构造阶段 RSS 下降约 84.7%，中位耗时下降约 91.2%；这是原生 Enum 构造的诊断收益，
不等于完整 selector.fit 或整个报告消费链路的收益。规则构造变化小，不宣称稳定提速；投影修复的
直接证据为未入选统计不再完整展开为 Python 字典。

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

首次查询和父进程等待消费者时的 RSS 有少数 20%/15% 触发项，原始 comparison 保留；复核结果
单列，不用构造收益掩盖消费成本。新 5k 有效证据与参考空输出不作速度收益比较。

### 真实公共分析入口与分维度扩展

standard 两侧各三轮，以下为当前 public_compute 中位秒数、最高阶段主进程 RSS；时间已经包含
入口内部报告装配。画像和分箱均为 50,000×200；交叉为百万行。表体积按实际报告后端估计。

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

统计筛选两个后端均保持 200 个输入 → 50 个相关性候选 → 50 个入选身份及顺序；画像一次、
评估一次且复用已拟合 binner、矩阵一次。线性筛选 40→40，但确实执行默认相关性及可用的
VIF/Logit 诊断。具体预筛漏斗和最终英文顺序保存在每轮 JSON。

交叉的实际必要输入为 Pandas 100.1 MiB / Polars 46.0 MiB，加 500 列为 2389.0 / 2334.8 MiB；
差异包含 Pandas 客群字符串表示。拦截转换确认主样本只传 7 列、半样本参考只传 2 个分数列；
每轮两次轴拟合、一次联合聚合同时处理 bad/late。报告及固定分段政策回放一致。
原始宽表准备约 8 秒，逐轮值见 JSON，不能用约 0.2 秒计算入口隐去准备成本及约 2.8 GiB RSS。

large 分析和扩维每案例只有一轮测量加独立预热，下面是单次观察，不能视为稳定分布。
本轮扩维使用 Polars；除与 standard 相同的百万行交叉外，large Pandas 分析仍未验证。

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

基础分箱 batch100 在本机没有时间优势，standard 阶段主进程峰值比 batch50 增加约 48%，large
由 3246 MiB 增到 5509 MiB。建议同类分组、权重和基准配置从 50 开始；这不是跨机器、指标或
算法的最优参数保证。50,000×200 的两个后端另外比较所有报告表，按业务键对齐，浮点容差 1e-8，
全部通过；原生表哈希的细小浮点归约差异没有被误判为统计变化。

### 筛选预算失败与修复

首次 large 筛选预热在 `public_compute` 约 45 秒被采样保护终止，退出码 15，主进程峰值
4010 MiB、进程树峰值 8193 MiB；状态为 `memory_budget_exceeded`，计划测量轮次为 `not_run`。
日志显示画像、粗筛及精筛预分箱已完成，没有将该退出猜作操作系统 OOM 或正常无规则。

源码确认精筛没有把 selector 的 `n_jobs=4` 传给最优分箱器，后者使用默认 `-1` 启动 loky
进程池。修复为默认继承 selector 的配置，显式 `binning_params['n_jobs']` 仍优先；没有减少
候选、取消精筛或调整阈值。与首次失败同输入、同 harness、同预算的复跑完整通过，计算 124.625 秒，
阶段进程树峰值 5378.5 MiB，主进程 4234.7 MiB；只有一轮，不据此宣称稳定容量或提速比例。
原始失败和修复后复跑分别保存。

standard 原参考 Polars 计算中位 17.999 秒、修复后 20.308 秒，约增加 12.8%；Pandas
18.568→20.387 秒，约增加 9.8%。这是执行既有资源限制的吞吐权衡，没有宣称速度收益。
精筛任务并发是进程数，4 线程环境控制的是各计算库单进程线程，两者需要分别理解。
独立数值核对记录精筛 binner 实际 n_jobs 从 23 变为 4。
独立最优分箱诊断保留原公共入口的默认进程配置；不能将所有场景解释为全进程树仅有四个线程。

### 实施范围

1. 完整相关性报告曾将长名展开为平方级 NumPy Unicode 数组再转 Enum。现在由原生整数端点
   直接转既有 Enum 类型，相关值与状态由 Polars 构造，保留所有关系、schema 和顺序；没有快照格式变更。
2. 规则解释曾将全部验证命中统计转 Python 字典，然后筛入选 ID。现在先在原生表中筛选，只展开
   需要的入选规则；候选、审计和切片完整保留。测试拦截 to_dicts，确认只转换 5 条解释所需统计。
3. 5,000 字符上下文曾删光证据却返回正常 JSON。现在按保留行关联裁剪无关业务元数据，重复 null
   定义在表级存一次，优先裁剪可继续查询的说明导航，每个非空查询至少保留一条完整证据。
   必需口径和一条证据无法共存时明确拒绝。既有完整 describe 与快照元数据保持原样。

上下文预算为最终 Unicode 字符数，测 5,000、16,000，并独立记录 512 的明确拒绝。45 列完整规则
评估在 7,000 字符下无法保留一条完整证据，会明确拒绝；投影所需列后可产生有效证据。
省略原因、行数及继续查询定位都在原始 JSON 中，空证据不能算 passed。

小规模数值参考包含 NumPy 有效值均值，独立的已表现标签/权重/坏样本/金额合计（rtol/atol 1e-6），
交叉样本数和已表现分母精确整数检查，signed 子矩阵 1e-12。其他统计口径由定向与基础回归覆盖。
规则 sources 筛选验证特征与来源命中同一规则的不同成员，统计行不展开；不同轮次同 ID 保留审计粒度。
无候选、全部淘汰、有候选未入选的状态都保存后在新 Python 中查询，并保留高级关系 not_computed 状态。

每个成功案例 save 后另启解释器，仅提供快照、查询配置与源码安装位置。该解释器不导入 workload
模块，也不调用数据生成、拟合或挖掘，逐表验证数据/schema/顺序、report_id、describe 和有限引用回放。
相关性还恢复 30×30 矩阵和 20 个关联特征；交叉报告查询 cells、显示有限矩阵并回放固定分段政策。
query_page 为 limit=20，包含后部 offset、投影、筛选与排序；只对筛选后的有限页进行 Python/JSON 展开。
完整 HTML/Excel 只在少数 smoke 小报告执行；平方级报告只展示有限表或 30×30 矩阵。

## 同进程重复消费和资源边界

每个独立消费者先预热，再至少 10 次 load → query_page → context → release，逐轮记录三种延迟和 RSS。
独立冷加载延迟与同进程重复分开。内存池可能保留 RSS，不要求 del 后归零；比较后半段范围与逐轮
轨迹，持续增长才进一步定位引用或缓存，不据一次末值判定泄漏。

3,000 长名额外重复 100 次：释放后 RSS 首轮 593.5 MiB、末轮 617.5 MiB、最高 621.0 MiB；
每十轮平均值为 603.2、610.2、606.0、611.1、612.4、605.4、613.8、612.4、611.9、613.0 MiB。
本次观察符合内存池逐渐进入波动平台，没有后半段持续单向增长；仍不能证明任意长会话不存在泄漏。

large 短名首次页的 20% 触发项复测三轮，参考 10.57/17.98/20.32ms、当前
10.28/13.88/22.03ms，没有复现中位耗时退化；等待消费者时父进程 RSS 的约 16% 增量仍存在，
属于不同构造路径之后的保留分配，应计入总预算。规则首次页复测中位 14.64→18.04ms（约 +23%）
仍触发复核；独立消费者中位 20.94→13.45ms，不用这一结果抵消首次页变化。
分页生产实现未变，三个样本仍有明显噪声，因此保留该限制，未宣称消费全面提速。
standard 少数毫秒级保存、准备、哈希与首次页也触发比例线，原始值和触发项完整保留，
未逐项扩大复测，不能据比例单独认定稳定退化或已消除退化。

超时诊断使用 0.05 秒预算，准确记为启动阶段 timeout；1 GiB 参考长名矩阵触发构造阶段
memory_budget_exceeded；512 MiB 配置对大画像/分箱保守预估为 not_run，未分配宽表。
这些是保护机制验收，与 5k/16k 的有效证据成功、512 字符的明确拒绝分别记录。
large Pandas 新增规模、超出上述规模及真实业务数据没有运行，本轮选择标准双后端后用单后端逐维扩展，
不将未运行项描述为资源超限，也不由较小规模推断通过。

建议从 smoke、standard 推进，显式设置本机预算。有限查询与排序仍可能扫描完整原生关系表，
不承诺恒定时间。批量 50/100 的结论只适用于本页固定 dtype、分组、基准与指标配置，不进行无限搜索。
默认 30×30 展示与 20 行分页是消费范围，完整关系仍在报告中。监督算法单独列出，不以关闭原本
应运行步骤或改变阈值取得性能收益。

时间中位数增长超过 20% 或计算阶段最高观察 RSS 增长超过 15% 只触发复核；先看每轮、阶段与
样本规模，再在相同 workload 下复测。不用采样预算承诺永不瞬时超限，也不将异常退出猜作 OOM。
较小规模通过不能替代未运行、更大规模或其他机器的验证。

## CI 与复现限制

普通 CI 增加三个独立 report smoke（规则报告、相关性、交叉宽表投影）；结构、版本隔离、
预算终止与跨进程消费检查在基础测试套件运行。large 不在 push 上执行，统一 CLI 可手动运行。
Python 3.8–3.12 核心边界不变，Agent 保持 >=3.10；外部快照消费不依赖 mars.agent 或模型服务。
大型宽表、Parquet、快照与导出都在拥有的临时目录，结束清理，不纳入仓库。

本轮修改报告构造、有限上下文、解释投影和精筛资源参数继承；暂停模块没有新功能，
也没有为其新增旧入口别名。

回归结果：Python 3.10 基础套件 708 passed、1 skipped、7 deselected；Python 3.8
冻结依赖基础套件 583 passed、11 skipped、7 deselected，Agent 测试按版本隔离；
Python 3.12（Pandas 3.0.3、Polars 1.42.0）报告、规则、交叉、批次和筛选定向套件 127 passed、
1 deselected。没有可用 Python 3.9 解释器，其语法按 3.9 AST 检查、CI 保留真实版本矩阵，
不能描述为本机完整运行过 3.9。Ruff、154 个生产文件的 Mypy、公共与私有 docstring 检查、
严格 MkDocs 构建通过；私有 docstring 检查仍有六条既有提示。

## 2026-10-02 审查修复验证

本节记录 `b73491c80617c24767f289921cc7924f71c9737f` 上的本轮修复工作树，
与前文 2026-10-01 的性能记录分别保留。开始时本地 HEAD、远端 main 一致，工作树干净；
没有覆盖后续提交，也没有修改历史结果 JSON。原 Linux CI 的两个 worker 用例曾把被
psutil 提前回收的直属进程记为退出码 0；本轮由 Popen 独占直属进程的回收。

行为回归先确认缺陷，再验证最终行为：

| 场景 | 修复后的实际结果 |
| --- | --- |
| 受控正常／异常退出 | 分别保留退出码 0／7，异常退出与预算判定分别记录 |
| timeout／memory_budget_exceeded | Windows 实测退出码 1、`exit_status=known`；后代停止、无残留 PID，`deliberate_wait` 阶段与诊断保留；不固定 Linux 信号码 |
| 重复值右闭最小箱复现 | 切点由共享左闭 `[2.0]` 转为参考样本真实前值 `[1.0]`；最终正常箱人数 4、6，`actual_n_bins=2`，约束诊断为 satisfied |
| 自定义 `inf`／`-inf` 缺失 | 首算、提取直接复用、相同配置显式复用，以及保存后新进程加载复用均为正常箱 2、missing 1、invalid 0；统计与政策回放相同 |
| key／feature 同列快照 | Polars／Pandas 保存前后均能筛选、分页和生成证据；最小复现证据为 `feature=x, mean=2.0`，原统计行不展开 |

最小箱检查仅针对拟合参考样本。显式切点和固定定义复用保持固定，OOT 不重新约束占比。
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

不同环境安装的可选依赖不同，适用用例数因此不同。基础套件后补强的 worker 清理诊断
在四个版本各重跑 `tests/test_core_capacity_benchmark.py`，均为 35 passed。
Python 3.12 的五个 Score Cross 定向模块为 113 passed，四个报告／规则／快照定向模块为
91 passed。最后补强的非有限缺失配置复用回归为 2 passed；四个导出临时目录用例也已复验。
初次并行运行基础套件暴露了四个旧导出测试共享 `tests/_artifacts` 的竞态，现改用各自
`tmp_path` 并保留原断言；错误源码搜索路径和目录权限造成的早期运行失败分别纠正后重跑，
未把这些失败计为通过，也未削弱原有业务断言。

统一容量入口实际执行以下 smoke，均包含独立预热、测量、保存和冷消费者验证：

```bash
python benchmarks/benchmark_core_capacity.py --case rule_report correlation_short --backend polars --scale smoke --repeats 1 --output <tmp>/report-smoke.json
python benchmarks/benchmark_core_capacity.py --case score_cross linear_selection --backend pandas --scale smoke --repeats 1 --output <tmp>/analysis-smoke.json
```

四个案例全部 passed。线性筛选实际执行 VIF／系数各 8 行，statsmodels 0.14.6 成功导入；
关闭的 stepwise 分支明确记录跳过。另以原提交源码和修复源码执行相同
`correlation_short / polars / smoke` 对照，七个阶段均为 comparable，
表内容、schema 与顺序摘要一致，源码溯源哈希变化不会误拒绝。
这是一轮正确性与比较工具验收，不能据此宣称提速。

同一解释器内受控模拟 statsmodels 导入失败，400×8 输入的 VIF／系数由各 8 行变为
跳过且为空；比较结果为 incomparable，保留依赖与诊断差异原因，不输出常规耗时比例。
workload、有效参数、测量合同和实际参与依赖变化均有拒绝回归；明确比较目的的资源策略
变化单独标记。执行失败、语义不一致与不可比较分别报告。旧基线缺少新兼容字段时说明
信息不足；已发布正式线性对照两侧的诊断均为 available，原数字未重写或宣布失效。

Ruff 全范围检查、Mypy（157 个生产文件）、公共 pydoclint、私有 docstring 检查及
严格 MkDocs 构建通过；私有检查保留六条既有提示。`git diff --check` 通过。
离线探索器的实际 Node JavaScript 行为测试随定向／基础套件执行；未做真实浏览器交互验收。
本轮没有无条件重跑 standard／large，未新增大型二进制产物或重做历史性能结论。

原始日志、smoke JSON、对照 JSON、参考源码与站点构建保存在本机独立临时目录
`mars-review-20261002-l9zhqkcb`；Python 3.10 的运行使用独立临时目录。
没有本地 Python 3.9 或普通 Linux 环境，原远端失败已核对，但修复后的 Linux CI 尚待新提交实跑；
这些环境不能记为本轮通过。`docs_ml`／`optional_ml` 标记套件没有在本轮单独执行。

## 原始结果索引

结果在 `benchmarks/results/`，每轮值、预热、环境、实际参数和拒绝结果都保留。
合并文件的 `source_runs` 记录各真实调用的参数与原文件哈希，表示多次串行调用的归档，
不能解释为一次总耗时。调试失败的简短索引在 diagnostics，正式基线和实际预算失败不被删除。

| JSON | 内容 |
| --- | --- |
| `core_capacity_baseline_initial.json` | 首次生产优化前的 standard 报告基线，三轮值 |
| `core_capacity_smoke_verified.json` | 全部 smoke、双输入后端、独立预热与测量，34 个组合通过 |
| `core_capacity_standard_reference.json` | 同环境参考源码：报告与真实分析全部 standard |
| `core_capacity_standard.json` | 当前全部 standard：34 个组合，三轮，筛选资源修复后结果 |
| `core_capacity_large_reports_reference.json` | 50k 审计、3k 完整关系及真实小挖掘链路参考 |
| `core_capacity_large_reports.json` | 同场景当前结果、十个组合、三轮、完整报告语义对照 |
| `core_capacity_large_analysis.json` | large/扩行/扩列、原始筛选预算失败及修复复跑 |
| `core_capacity_diagnostics.json` | 桥接、跨快照版本、超时/预算/not_run、消费复测、100 次释放、完整批次与筛选数值核对 |
| `core_capacity_validation.json` | 实际验证命令、版本、通过数和检查结果 |

批次与筛选的独立数值核对代码随 diagnostics 的 `verification_script` 保存，仅生成标准规模
验证证据；外部快照消费者不导入这些代码。快照、原始宽表及导出产物均未保存到仓库。
