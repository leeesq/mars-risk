---
description: M12 宽表暴露的上游缺陷、旧环境回归、连续阶段容量与真实输入复现边界。
---

# M12 宽表上游修复验收

本页保留上一轮内存改造的原始测量和限制。
2026 年 10 月 23 日独立验收后的修补与复测另记在文末；历史计数和容量数字不代表本轮结果。

本轮在 Windows 当前工作区修改 MARS，未运行 M12 Agent、预测模型训练或 Runtime 恢复流程。
真实 M12 的 257,997 行风险画像和 297,440 行 selector 容量均未运行。
仓库中的 Home Credit 公开案例为另一份 119 候选输入，不能替代 1,111 候选验收。

## 源码与环境

工作区为 `D:/Desktop/credit-risk/mars-risk`，分支 `main`，初始工作树干净。
实现与验收时 HEAD 为 `625de258deb75d8d95a7e2bd36e9ae1774b8c6a2`。
完成验收时未提交工作树的 Python 源码内容 SHA256 为
`1fba73c237a2ef03ce60dd5152858dfdd7fdc9982cb909b8151b99597083c852`；版本仍为 0.0.28。
用户指定的基线 `c937fbfadbe083e39d3425ea1af206b8907fd0ff` 已通过只读获取提交对象取得；
它比验收时本地 HEAD 多一个文档提交，计算源码相同。实现与验收阶段没有切分支、reset、commit、push、发布或改版本号。
随后用户明确授权提交推送；推送前将 `main` 快进到该基线，仅合入规则生成器返回值的 docstring 说明，
未改变任何计算实现、版本号或本轮验收结论。
Git 确认的历史完整 SHA 为 `847a23053b1cbc2ba1be14fbc0c494eacecf8596`。
当前目录及祖先和相关源码目录未发现适用的 `AGENTS.md`。

所有最终回归将 `PYTHONPATH` 设置为当前仓库的绝对 `src` 路径，`mars.__file__` 指向该目录下的 `mars/__init__.py`。
这是必要的环境隔离：本机 mars38 默认会导入 site-packages 中的旧安装包。

| 环境 | Python | Polars | Pandas | 用途 |
| --- | --- | --- | --- | --- |
| mars38 | 3.8.20 | 1.8.2 | 2.0.3 | 冻结旧环境、公开 API 和多 chunk 回归 |
| mars | 3.10.19 | 1.37.1 | 2.3.3 | 合成容量和分箱回归 |
| mars312 | 3.12.13 | 1.42.0 | 3.0.3 | 现代公开 API、静态检查、画像回归及正式公开案例生成 |

单线程缺陷回归在导入 Polars 前设置 `POLARS_MAX_THREADS=1`。
容量 worker 在导入 NumPy/Polars 前设置全部既有线程环境，`n_jobs=1`，两侧配置相同。
未将 `n_jobs` 解释为 Polars 或 BLAS 线程数。

## 根因与修改

| 项目 | 修前行为 | 本轮实现与证据 |
| --- | --- | --- |
| P01 | 旧 Polars 的纯 literal select 可按 chunk 返回多行，transpose 只有一个 total 名称 | 非适用统计和常量用 typed 标量聚合；PSI 独立计算，保留不可算表和状态 |
| P02 | eager 样本先 with_columns/join，最后才转 lazy；fit 和 benchmark schema 校验提前物化 | 执行前选定 LazyFrame；schema 不执行样本转换，映射和非法规则校验仍保留 |
| P03 | Quality/Rough 未贯通列批宽；先过滤完整宽表；benchmark eager long 表驻留 | 贯通既有参数；每批先投影再过滤、转换、聚合；expected distribution 使用 lazy 聚合 |
| P04 | 完整监督人口的 WOE、填 null 副本和 NumPy 中心化工作数组同时存在 | 私有行块和列批生成 WOE，Float64 中心矩合并完整 Pearson；保留全 P² 报告 |
| P05 | 历史总量和临时外层分批无法定位实际阶段峰值 | 扩展现有 fresh-worker harness，连续公开调用记录阶段、人口、survivors 和报告保存 |
| P06 | 需同时防止旧环境回退、报告合同退化和案例指纹陈旧 | 旧版与现代回归、静态检查、文档及正式案例生成器验收，结果见下文 |

主要源码为 `analysis/_profiling/{metrics,pivot,psi}.py`、`analysis/profiler.py`、
`feature/binning/base.py`、`analysis/_evaluation/aggregation.py`、
`feature/selection/stats.py` 和 `compute/_correlation.py`。
新增回归进入既有 batching 套件及 `test_binner_execution.py`、`test_stats_blocked_correlation.py`。

P01 修前在真实 Python 3.8.20 / Polars 1.8.2 保存了四条 ShapeError：公开 mixed PSI、PSI-only、
单指标字符串 mean 和 zeros。新测试显式构造不等长 `[2, 5, 1]` chunks 并断言 `n_chunks()>1`。
280 组公开多 chunk/rechunk 对照覆盖单/多特征、Float32/64、String/Categorical、全 null、空表、
分组、时间、独立 benchmark 和缓存路径。还验证真实 null、文字 `null` 和 `（空分组）` 不混淆。
源值、dtype、行序和输入 chunk 结构均保持，PSI-only overview 保留身份。

P02 修前结构回归有八条失败。修后 guard 分开检查计划构建、schema 和 collect，允许小型映射表构造，
禁止 schema 阶段分配全行派生列及 eager 高基数 join。数值、replace、join、未知类别、Missing/Special/Other、
失败 refit、规则保存/新进程加载和 WOE 子集复用均覆盖。首次 WOE 的必要监督统计仍可能执行。

P03 的完整顺序为 Quality → Rough Fit → fit schema → benchmark schema → Rough Performance →
Fine（启用时）→ Stability PSI/RC → WOE Corr → 报告冻结/缓存清理 → 保存。
质量与分布使用完整当前人口；无监督拟合使用完整拟合人口；监督拟合仅使用原定义已表现标签。
独立 benchmark 继续提供切点和 expected 分布，当前监督统计和 WOE 来源没有改变。
真实 M12 关闭 Fine，本轮 Fine 回归不能算作真实失败热点的修复证据。

必要的 `_cache_X/_cache_y` 支持公共 `update_bins()` 和首次 WOE，不能为降低图中 RSS 清空。
投影通常共享原列 buffer；CART 的监督过滤先投影需要的特征和标签，再分配过滤结果。
表现统计和 expected long 表不跨批保留样本，跨批只累计小统计。
selector 保留已有清理 Stage 3 缓存的行为，未引入循环 GC 或外层补偿流程。

## 全局 Pearson 的空间与数值合同

设监督人数 N、候选数 P、内部行块 R、特征列批宽 B。内部 R 为 `min(4096, max(1, 8 MiB / (8P)))`。
生产代码保留常数份 Float64 R×P WOE/中心化块、当前列批和 O(P²) 的中心矩/临时乘积。
生成器交接时前后块可短暂共存，不能将空间解释为仅一份 8 MiB 缓冲区。
首次 WOE 还可需要拟合数据的监督统计。原表、训练缓存、规则、稳定性报告和完整相关性报告另计。
这不是总内存硬上限。极宽输入的最小一行也可超过 8 MiB。

时间为 O(NP²)，累计状态和全局输出为 O(P²)，块工作内存为 O(RP+RB)。没有临时落盘。
先减去首行原点，再按 Chan 公式合并块中心矩；不使用不稳定的平方和相减，也不以 Float32 累计。
仅 WOE null 填零，NaN/inf 传播；常量和零/单样本 helper 的不可用对角线不补 1。
公共 selector 仍拒绝不满足二分类合同的标签。

与旧 dense NumPy 参考存在浮点累计差。接近阈值舍入误差界的少量 pair，使用完整监督人口、
两列 WOE 的 `np.corrcoef` 复核，再按原 `>` 比较。误差界仅触发复核，不改变业务阈值。
复核为 O(N×2) 工作内存，即使列批宽为 1 也转换两列；阈值处于大量 pair 的舍入邻域时会增加运行时间。
实数结果沿用 NumPy 的 [-1, 1] 截限。跨列批的正负关系、IV/ID 优先级、白名单、完整低相关边、
signed 值和实际决策顺序均保留，最终名单仍按原候选顺序返回。
阈值大于等于 1 时，由于严格比较和截限不可能触发淘汰，直接跳过窄列复核。
这避免完全重复的 1,111 候选在阈值 1 时触发 616,605 次不必要的完整人口 pair 转换。

报告继续使用紧凑 Enum endpoint 构造完整上三角，不新增平方级长特征名 Python 字典列表。
保存、加载、局部矩阵、pairs、decisions、分页和 AI 查询仍使用原合同。

## 连续阶段容量测量

复用 `benchmark_core_capacity.py` 和 `benchmark_core_workloads.py`。每个预热/正式轮次独立 worker，
一次公开调用内连续记录所有阶段及对象共存。阶段隔离不是本轮容量通过的依据。
`selection_survivors` 要求至少 80% 候选进入全局 corr；`selection_attrition` 覆盖实际质量/PSI 淘汰。
两者含混合 Float32/64、null/sentinel、多 chunk、长特征名及完整分组，均使用独立 Train benchmark。

记录包括源码/依赖/参数、输入与 benchmark 人数、非空监督人数、dtype/chunks、候选进出、
实际转换批边界、wall time、RSS 起止/采样峰值、worker lifetime peak、父子进程及退出状态。
RSS 树可重复计共享页，10 ms 采样可漏瞬时峰值，lifetime 高水位不能当阶段增量。
Linux `ru_maxrss` 常见 KiB 转 bytes，Darwin 原生 bytes；Windows 使用进程 peak working set。
Darwin footprint/compressed/swap 与 RSS 分列，当前 Windows 没有对应 macOS 实测。
失败保持真实状态，不把包装器退出码、worker 退出码和内核终止证据混为一谈。

同合同中型 A/B 使用 Python 3.10.19 / Polars 1.37.1 / Pandas 2.3.3，Windows 11（build 26200），
24 逻辑 CPU、63.7 GiB 物理内存。两侧 50,000 行、200 候选、3 chunks、160 Float32/40 Float64，
benchmark 30,000 行，实际非空监督 28,965 行，列批宽 16、`n_jobs=1`、全部原生线程环境为 1。
每侧一轮排除预热和一轮正式测量，时间只作探索性观察，不估计 P95 或保证普遍加速。
本机另有回归进程运行，未声称独占 CPU 或内存；背景负载也是单次时间观察的限制。

| 场景 | 两版各阶段候选去向 | public 秒，旧→新 | public 采样峰 MiB，旧→新 | lifetime 峰 MiB，旧→新 |
| --- | --- | --- | --- | --- |
| survivors | Quality/Rough/Stability/Corr 输入均 200，最终 198 | 7.974→8.037 | 872.5→526.0 | 872.57→526.08 |
| attrition | 200→Quality 175→IV 150→PSI 125→最终 123 | 6.556→5.753 | 751.5→521.9 | 751.79→522.05 |

两场景最终名单顺序和 funnel 均一致。survivors 确实完成 200 候选的全局相关性，未用提前缩小到少数
特征替代容量。相同公开参数下，survivors 峰值下降约 39.7%，attrition 约 30.6%；
survivors 总耗时基本持平，全局相关性本身变慢，不能据此宣称所有阶段提速。

完整输出语义另做辅助复算，不计入性能样本。两场景所有表的身份、行序、schema、null 和非浮点列
精确相同；完整 pairs 最大绝对差为 `1.5543122344752192e-15`，对角和决策相关值最大差约 `2.22e-16`。
IV 优先级和阈值差为零，分箱切点、WOE、mappings、fit_report 等 10 项状态精确一致。
因此完整表 hash 不逐位相同，不能用 hash 不同宣称名单或业务口径改变，也不能把它们写成完全相同。
细节见 `.artifacts/m12_capacity_semantic_comparison.json` 及对应的小型合成快照/规则 JSON。

下表为 survivors 的连续阶段。峰值是该阶段所在完整 worker 的绝对 RSS，不能相加，也不是阶段独占增量。
fine 本场景关闭。报告 freeze 已包含完整 pairs 构造；后续 get report 只读取已冻结对象。

| 阶段 | 秒，旧→新 | 采样 RSS 峰 MiB，旧→新 |
| --- | --- | --- |
| 输入准备 | 0.169→0.192 | 313.4→313.3 |
| Quality | 0.449→0.475 | 442.2→354.9 |
| Rough Fit（包含末尾 schema） | 2.294→2.245 | 440.3→380.9 |
| fit schema 观察子阶段 | <0.001→0.003 | 375.3→380.5 |
| benchmark schema 校验 | 0.309→0.026 | 375.3→380.5 |
| Rough Performance | 0.655→0.590 | 683.6→420.7 |
| Rough 整体 | 3.303→2.908 | 683.6→421.5 |
| Stability PSI/RC | 3.408→2.957 | 872.5→518.1 |
| WOE + 全局 corr | 0.618→1.496 | 758.0→526.0 |
| 完整报告冻结 | 0.015→0.021 | 678.2→512.3 |
| 报告保存 | 0.015→0.021 | 679.0→511.4 |

schema 子阶段只观察 `collect_schema()` 的时间。旧版的 eager 派生分配发生在它之前，计入
Rough Fit，不能从 schema 子计时推断旧版没有隐藏物化。执行路径回归另有 guard 证明这一点。
新旧变换边界明确记录：旧 Rough Performance 一次计划 200 特征，旧 benchmark 校验 eager 200 特征；
新 Rough Performance 实际批宽不超过 16，benchmark 校验只建计划；全局 corr 只生成内部行块的列批。
fit 的 schema 计划可以引用全部规则，仍不分配样本派生数组。

完整 JSON 在忽略目录 `.artifacts/m12_capacity_{baseline,current}_standard_v2.json`，
简表为 `.artifacts/m12_capacity_acceptance_summary.json`。旧侧来自 `.artifacts/m12_reference_c937/src`，
修后本轮测量源码内容 SHA256 为 `42c0ac6ef11bc2c9c9a9a3270822132915d1572499b1a920e568e0ebe10786c6`。
随后画像跨 Float32/64 批次的 overview concat 修复另有实际回归；该变更不改变本轮 selector 的质量字段口径。

```bash
python benchmarks/benchmark_core_capacity.py --case selection_survivors selection_attrition --scale standard --source .artifacts/m12_reference_c937/src --backend polars --threads 1 --n-jobs 1 --batch-size 16 --repeats 1 --timeout 600 --output .artifacts/baseline.json
python benchmarks/benchmark_core_capacity.py --case selection_survivors selection_attrition --scale standard --baseline-json .artifacts/baseline.json --backend polars --threads 1 --n-jobs 1 --batch-size 16 --repeats 1 --timeout 600 --output .artifacts/current.json
```

在小型/中型验证后，另运行修后单个 fresh worker 的 200,000×1,000 survivors 场景，未再运行大规模旧版。
Train benchmark 120,000 行、实际监督 115,862 行，1,000 候选全部进入 Stability 和全局 corr，最终 998。
同为 Windows、Float32/64、3 chunks、列批 16、`n_jobs=1` 和线程 1。
公开调用 191.250 秒，完整 worker 203.502 秒；主/树采样峰约 2,634.31 MiB，lifetime 2,634.316 MiB。
完整 499,500 对关系保存为 7,466,447 bytes 的快照。

| 扩容阶段 | 秒 | 采样 RSS 峰 MiB |
| --- | --- | --- |
| Quality | 9.161 | 2,489.2 |
| Rough | 43.318 | 2,351.7 |
| Stability PSI/RC | 51.585 | 2,634.2 |
| WOE + 完整全局 corr | 86.542 | 2,305.4 |
| 已冻结报告读取 | 0.015 | 见原始 JSON |
| 完整快照保存 | 0.057 | 见原始 JSON |

Rough Performance 和 Stability 的实际列批均不超过 16。相关性共 12,033 次列批转换，每次不超过
609 个已表现行、16 特征；fit/benchmark 的 1,000 特征请求均为 lazy schema 计划。
该轮正常退出 0，无重试、取消或内核终止。单侧数据不能计算同条件降峰比例，也不能证明 Mac/16 GiB
真实数据容量。原始记录为 `.artifacts/m12_capacity_current_large.json`，测量源码内容 SHA256 为
`3e6e45ba5260cde3a68b3a58d4132e2b1855edc9e73ffd796540620661cd1130`。
随后的阈值 ≥1 复核早返不影响该轮默认 0.95 的行为。

## 真实输入复现

没有在当前工作区取得 M12 冻结 Prepared/manifest。未扫描用户主目录、上传原始数据或启动 Agent。
公开案例的 Home Credit 原文件可用于案例一致性检查，不能拼造 M12 的 1,111 个冻结特征。
请为现有冻结 Parquet/IPC 文件准备一个小 JSON 配置，路径可以是外部绝对路径，候选必须完整显式列出：

```json
{
  "input_path": "/local/frozen/selector.parquet",
  "benchmark_path": "/local/frozen/train.parquet",
  "features": ["original_feature_1", "original_feature_2"],
  "target": "frozen_target",
  "group_col": "dataset_flag",
  "supervision_group": "Train",
  "selector_parameters": {
    "skip_fine_scan": true,
    "psi_thr": 0.25,
    "corr_thr": 0.95
  }
}
```

这里的特征名只说明配置格式，实际必须使用原 1,111 候选、角色和参数。读取只保留冻结值与 dtype，
不做外层 rechunk、标签补齐或重试。`supervision_group` 校验非 Train 标签已经物理为 null，违规即拒绝。
读取成本独立计时，不重新执行 Agent bootstrap、人口准备或模型训练。
冻结 selector 入口只默认传入 CLI 的列批宽和 `n_jobs`，其余参数遵循配置及库默认，
不会继承合成场景的 sentinel、粗分箱箱数或关闭 RC 等配置。实际复现时应从原计划完整复制这些业务选项。

```bash
python benchmarks/benchmark_core_capacity.py --case frozen_selection --input-config /local/selector.json --backend polars --threads 1 --n-jobs 1 --batch-size 16 --repeats 1 --output .artifacts/m12-selector.json
```

风险画像应另用开发人口 257,997 行及 Train benchmark 180,893 行的配置，不能与 selector 全人口混算。
实际非空监督人数从冻结输入读取，不强行等于 Train 分区行数。

```bash
python benchmarks/benchmark_core_capacity.py --case frozen_risk --input-config /local/development-risk.json --backend polars --threads 1 --n-jobs 1 --batch-size 16 --repeats 1 --output .artifacts/m12-development-risk.json
```

`frozen_risk` 使用 `MarsBinEvaluator.evaluate()` 单次公开风险画像；`frozen_profile` 单独验证统计画像。
风险配置使用 `risk_parameters` 与 `risk_call_parameters` 保存原分箱/评估选项，文件路径及完整候选仍使用上面的结构。

## 回归与公开案例

定向缺陷、执行路径和数值回归在 Python 3.8/Polars 1.8.2 与 Python 3.12/Polars 1.42 均为
80 passed。分箱相关既有套件在旧环境及 Python 3.10 各 121 passed；最终新增分箱执行测试各 19 passed。
容量 harness 现代环境最终 43 passed，包含 frozen_profile/risk/selection 三个不同公开入口的 fresh worker 测试。
基础套件完成后，仅对 benchmark 冻结配置隔离新增两条回归；旧环境重跑这两条及三个 fresh 入口，5 passed。
下表基础套件计数不包含最后新增的这两条测试。
测试规模只证明对应合同，不能当作真实 M12 容量结论。

最终基础套件均正常退出 0，0 failed：

| 环境 | passed | skipped | deselected | 秒 |
| --- | --- | --- | --- | --- |
| Python 3.8.20 / Polars 1.8.2 / Pandas 2.0.3 | 1233 | 13 | 7 | 294.29 |
| Python 3.12.13 / Polars 1.42.0 / Pandas 3.0.3 | 1332 | 3 | 7 | 323.92 |

旧/现代定向套件各 80 passed。日志为 `.artifacts/m12-upstream/verified{38,312}.log` 和
`final-targeted{38,312}.log`，包含外部 protobuf、统计模型数值和系统中文字体等 warning，未屏蔽警告。
Ruff、Mypy（159 source files）、公共 pydoclint、私有 docstring checker 和 `git diff --check` 通过。
私有 checker 的简单 helper 长 docstring 提示为既有 warning，无阻断错误。
Python 3.9/3.11 未在本轮重跑完整套件，既有 Python 3.8–3.12 CI 矩阵和约束保持原样。

正式基础套件使用下列命令，环境变量在解释器启动前设置。`PYTHONPATH` 必须为绝对源码路径，
以保证下载 ZIP 在仓库外的新进程也读取本轮源码。Windows 的文本子进程统一设置 `PYTHONUTF8=1`。

```bash
python -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/m12-upstream/verified
python -m ruff check src tests scripts benchmarks
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
git diff --check
python -m mkdocs build --strict --site-dir .artifacts/m12-upstream/site
python scripts/check_case_assets.py --mode public --site-dir .artifacts/m12-upstream/site
```

`--ignore` 仅排除本机历史生成的 benchmark 测试产物，正式 `tests/` 没有跳过。
最初逐字运行不带 ignore 的命令，在收集旧产物时因 Windows ACL 失败；未改文件权限或删除旧数据来隐藏失败。
初次显式 tests 套件还发现相对 PYTHONPATH 的子进程导入旧包及 GBK 解码问题，已用环境配置纠正，未放宽断言。
新 nullable WOE 测试最初给浮点映射传入 None，触发既有合法性错误；随后改为合法映射和测试边界的
nullable 输出注入，产品仍拒绝非法映射。所有初始日志保留，不将中间失败写成正常退出。

独立附加 sandbox 的 frozen_profile 一次在 report_save 停留后被人工取消，worker exit 15，
pytest 记录失败，不能计作成功或 OOM；其前段 public_compute 完成并不代表整链成功。
正式解除已确认的沙箱临时目录/硬链接限制后的 harness 测试正常退出，不改 worker 回收或保存合同。
案例生成最初的 Python 3.10 wrapper 也曾人工取消，无产物；正式过程沿用原 Python 3.12 环境。

公开 Home Credit 案例通过正式 `compute all → consume → finalize`，最后 Stats 变更后重算 case 3。
共更新 59 个相互依赖的生成产物，约 42.4 MB，未手填指纹或身份。
准备摘要、元数据、预览与两张原生 PNG 逐字节不变；五份快照的 47 张查询表、9,325 行按正式容差对照通过。
全部既有 summary 业务字段保持，仅增加四个 WOE 算法描述参数。
源码指纹一致，public/site checker 通过 72 files、11 snapshots、seven cases；严格文档构建通过。
此处没有重跑 README/官网视觉，也没有将公开 119 候选案例误计为 M12 的 1,111 候选。

## 历史证据和 Agent 适配边界

历史 development risk 的 70 批成功是 257,997 行、1,111 特征、基准 180,893 行，日志 155.83 秒、
worker lifetime RSS 7,770,259,456 bytes；它没有证明 297,440 行原生 selector 成功。
v12 包装器返回 1，并报告 abnormal termination/Invalid signal。日志最大 RSS 为 9,058,844,672 bytes，
peak footprint 为 34,313,599,264 bytes，两个口径不同。最近失败位于 rough performance 后、stability 完成前；
没有瞬间分配栈，不能把尚未开始的全局 corr 当成唯一原因，也不能把所有退出码写成 137。

单线程旧 Polars 的多 chunk 上游缺陷及 schema eager 物化已有可撤除外层 workaround 的定向证据。
真实 1,111 特征完整风险画像和 selector 尚未跑通本轮独立验收，70 批外层拼报告及 Runtime 适配的
整体撤除仍需另行验证。Runtime Snapshot/Registry/审批/预算/身份/恢复策略、冷恢复 IPC hash 和
准备报告未持久化造成的重算均未修改，不能列为 MARS 已修复。

## 2026 年 10 月 23 日独立验收修补

本次起点为 `main` 的 `ff0a0fd2251215bfd8e788b58dcde052ed866a2a`，工作树干净。
未发现适用的 `AGENTS.md`。本节记录当前未提交修改，包版本仍为 0.0.28。
以上历史容量与回归记录保持原样，不能自动升级为本次新比较合同的验收证据。

已只读核对该提交的[远端 CI](https://github.com/leeesq/mars-risk/actions/runs/39807098259)：
3.9 和 3.11 的 WOE 用例失败，3.10 和 3.12 取消；
[Docs](https://github.com/leeesq/mars-risk/actions/runs/39807098257) 构建、部署成功。
用户提供的 1235 passed、1234 passed / 1 failed、1333 passed / 1 failed 只属于该次远端任务。
本次未推送，不能宣称修后远端 CI 全绿。

### Join 行序和监督标签

现代 Polars streaming Hash Join 可以重排左表。原表现统计在 Join 后按外部 Series 的位置挂标签，
同时改变了数值特征和类别特征的逐箱坏样本数。普通 collect 或单线程可能掩盖错误。
`mars._compat._left_join_preserving_order` 按实际 Join 签名使用 `maintain_order="left"`；
没有该参数的旧 Polars 使用内部行身份恢复左表行序。数值、Replace 和多个 Join 路径均保序。
无需唯一业务 ID，完全重复行仍保留原位置和行数。

`MarsBinnerBase.profile_bin_performance` 先投影本批特征，再绑定已规范化标签，
随后按该标签过滤已观测人口、转换并聚合。新增成本为本批一列标签，不复制全宽已观测表。
临时列名统一避让原字段、请求特征、派生列和已有内部列，并在输出前清理。
`MarsStatsSelector._woe_row_blocks` 和临界 pair 复核使用项目真实 streaming 入口；
各列批在同一已观测行序上写入 NumPy 行块。全局 Float64 Pearson、严格 `>` 阈值、IV/ID 优先级、
白名单、事件顺序及最终候选原序保持原合同。

Python 3.12.13 / Polars 1.42.0、导入前线程 4 的独立修前重放中，
Join 重排 68/70 个已观测位置，按位置挂载外部标签有 38/70 个错配；标签随样本携带时为 0。
修后重排和错配均为 0。下表按 bin_index 展示同一数值特征；修后 Join 与 Replace 完全一致。

| bin_index | count（两版相同） | 修前 Join bad | 修后 bad | 修前 Join WOE | 修后 WOE |
| --- | --- | --- | --- | --- | --- |
| -3 | 10 | 6 | 10 | 0.117782 | 12.429220 |
| -1 | 10 | 5 | 0 | -0.287680 | -12.716901 |
| 0 | 10 | 6 | 0 | 0.117782 | -12.716901 |
| 1 | 10 | 5 | 10 | -0.287680 | 12.429220 |
| 2 | 10 | 5 | 0 | -0.287680 | -12.716901 |
| 3 | 20 | 13 | 20 | 0.331356 | 13.122365 |

IV 从错误的约 0.070260 恢复为 Replace 参考的约 25.492693。
Hash Join 的错误排列不保证每次相同，这些数字只描述已保存的这一轮重放。
原远端失败用例保留；未改变期望 WOE、数值容差或最终统计表排序来消除失败。
详细修前/后记录为本机 `.artifacts/p01-streaming-evidence.json`。

### 空 PSI 的校验和状态

原画像入口按当前表 height 跳过整个 PSI 流程，也跳过了 benchmark 结构校验。
现在由 `get_psi_trend` 在空人口早返之前校验既有 active feature 范围；缺特征、类型不兼容和
空 benchmark 均保留原 `ValueError` 与说明，不要求基准携带当前 group/time 角色。
空当前人口不拟合分箱；结构合法的全 null 基准也不会触发无用拟合。

有 group、time 或合法 benchmark 的空人口返回 Float64 typed null，具体 PSI 状态行为
`not_computed / empty_current_population`。没有任何参考时继续使用
`not_computed / no_group_or_benchmark`。零候选不生成虚构特征。
身份、原 dtype、中文名、来源、单位和列序保持；其他指标状态和诊断继续保留。
状态可经目录、查询、AI 上下文、Notebook、HTML 以及保存后新进程加载读取。
修前/后小证据为 `.artifacts/p02-empty-psi-{before,after}.json`。

### 容量比较的自动门

`benchmark_core_workloads.py` 为 frozen current 和 benchmark 各做读前/读后 1 MiB 分块 SHA256，
并检查读入期间文件属性是否改变。记录算法、大小、实际路径和 reader/rechunk 配置；
路径只作本机溯源，相同字节复制到其他目录仍可比较。
合同绑定完整特征顺序、角色、业务参数、监督范围；合成场景绑定 fixture 版本、seed 与有效生成参数。
源码 SHA 仅作溯源，不作为同业务实现的拒比依据。

候选用完整特征表中的整数位置记录，覆盖 Quality、Rough、Fine、Stability、PSI、RC 和全局 Corr
的实际去向、corr 候选原序、最终名单原序及计算状态。至少 80% 进入 corr 仍只是规模 guard。
200→161、最终 198→159，同数量但字段或顺序不同，以及任一预热/正式轮漂移，均不得输出普通比例。

新增单个私有 `_capacity_evidence.py`，复用既有 ReportSnapshot 保存聚合语义证据。
每个预热和正式轮保留一个有大小边界的私有 `.marsreport`，先校验完整性、报告身份及本轮表签名，
再对照完整目录、逻辑 schema、行序、null、非浮点值、名单、动作、触发特征、保护状态与事件顺序。
有限浮点指标和决策系数采用 `atol=rtol=1e-12`；null、NaN、正负无穷分别核验。
阈值严格相等，容差不重做业务判定。约 `1e-15` 的相关系数差可以通过，WOE 12→0.1 必须拒绝。
仅排除已登记的报告身份和实现参数，以及画像报告自身的生成时间；未知元数据差异继续拒绝。

sidecar 默认位于已有忽略目录 `.artifacts`，可用 `--evidence-dir` 指定本机位置。
独占创建唯一文件名，失败清理本轮文件，重复执行不覆盖旧结果。后续比较必须保留 JSON 引用的快照；
缺身份、缺文件、损坏或语义证据不足的旧 JSON 明确为 incomparable，不补造历史身份或验收结论。
真实业务报告只留本机忽略目录，不提交或上传。

整个公开调用、生产必要的完整报告构造仍在一个 fresh worker 连续计量。
输入身份、报告签名/保存/完整性与 baseline 语义校验分别记录时间，不计入 public_compute。
资源策略有目的的比较保留既有规则，业务工作变化仍然拒比。
历史 39.7% / 30.6% 是上一轮人工复核的中型观察，没有被新门追认或推翻。
如需新的实现 A/B，必须用同一修正 harness 在两侧重新生成 baseline/current。

### 本次验证范围

`tests/test_binner_streaming.py` 每项启动新解释器，并在首次导入 Polars 前设置线程 1 或 4，
断言真实线程池大小。每组覆盖 eager、普通 collect、项目 collect_streaming、两个类别 Join、
19 特征和列批 1/16/100、非等长多 chunk、重复行、未表现标签、Missing/Special/Other、未知类别、
独立逐行计数和 Replace 数值参考、完整 dense WOE 矩阵、pairs、decisions、名单、cache/refit，
以及正式规则和报告保存后新进程加载查询。五组各 2 passed，均退出 0，无跳过：

| Python | Polars | Pandas | 新线程/引擎测试 |
| --- | --- | --- | --- |
| 3.8.20 | 1.8.2 | 2.0.3 | 2 passed |
| 3.9.20 | 1.36.1 | 2.3.3 | 2 passed |
| 3.10.19 | 1.37.1 | 2.3.3 | 2 passed |
| 3.11.15 | 1.42.0 | 3.0.3 | 2 passed |
| 3.12.13 | 1.42.0 | 3.0.3 | 2 passed |

这些均为本机 Windows 实测；未运行 Polars 1.44.2 或 Linux/macOS 矩阵。
既有 Python 3.8～3.12 CI 和 frozen constraints 未改变，未把 CI 全部任务改为单线程。
空 PSI 新增 83 项在 Python 3.8 / Polars 1.8.2 通过；现代画像关联组共 151 passed，均正常退出 0。
早期测试曾因旧 Categorical dtype 显示和子进程误导入旧安装包失败，最终测试显式使用当前 src。
沙箱内报告 hardlink 保存另曾触发 WinError 5；首次失败未计为通过，受影响检查随后在获准的限制外执行。

公开 Home Credit 三份原始 CSV 可用，已用正式生成器重算 all，随后 consume、finalize。
11 份快照的 80 张表、10,138 行通过正式语义对照，最大浮点绝对差为 `2.7755575615628914e-17`。
两张原生 PNG 逐字节相同；SVG 为正式重新导出，图形布局和字体代码未修改。
这份公开输入是 356,255 行、119 候选，不能替代真实 M12 的 1,111 候选容量验收。
本次没有取得 M12 冻结 Prepared/manifest，未运行真实 257,997 风险画像或 297,440 selector，
未启动 Runtime/Agent、训练预测模型、改业务输入、改包版本、提交、推送、发布或部署。

本次 fresh-worker smoke 包含 survivors 与 frozen_profile 各自的 baseline/current、预热/正式轮，
验证保存 JSON 后的下一次命令能读取保留 sidecar。另用同一个新 harness 在 `c937fbf` 旧 dense
源码和当前源码各生成一个独立 survivors smoke worker，二者均 passed、退出 0；
输入、执行、资源合同和阶段集合一致，完整 Report 内容核验通过。
这些 smoke 只证明比较门可用，没有重新测量或替代前文中型容量比例。

正式命令使用各环境的实际解释器，并在启动前设置当前源码路径及 UTF-8。
完整基础套件的 `--ignore` 只排除本机历史生成的 benchmark 产物，没有跳过正式 tests：

```powershell
$env:PYTHONPATH = (Join-Path $PWD 'src')
$env:PYTHONUTF8 = '1'
$env:POLARS_MAX_THREADS = '4'
python -m pytest -q -m "not docs_ml and not optional_ml" --ignore=benchmarks/_artifacts --basetemp=.artifacts/acceptance-full
python -m pytest -q tests/test_binner_streaming.py tests/test_binner_execution.py tests/test_analysis_batching.py tests/test_stats_blocked_correlation.py tests/test_profile_empty_psi.py tests/test_core_capacity_benchmark.py
python -m ruff check src tests scripts benchmarks docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python -m mkdocs build --strict --site-dir .artifacts/acceptance-site
python scripts/check_case_assets.py --mode public --site-dir .artifacts/acceptance-site
git diff --check
```

独立线程测试仍在其子进程内分别覆盖 1 和 4，不以父进程的线程值冒充两个环境。
案例正式流程为 `task_cases.py --phase compute --case all`、`--phase consume` 和 `--phase finalize`；
先在忽略目录完整生成并对照，再同步正式报告、manifest、ZIP 和下载资源，共更新 59 份生成文件。
公开 checker 通过 72 files、11 snapshots、seven cases；文档/案例回归为 100 passed、2 deselected，退出 0。
Ruff、Mypy（159 source files）、公共/私有 docstring 检查和 diff check 均通过。
私有 checker 的简单 helper 提示、外部依赖弃用及既有字体 warning 未屏蔽。

源码与测试冻结后，完整基础套件结果如下，failed 均为 0，进程退出码均为 0：

| 环境 | passed | skipped | deselected | 秒 |
| --- | --- | --- | --- | --- |
| Python 3.8.20 / Polars 1.8.2 / Pandas 2.0.3 | 1361 | 13 | 7 | 436.48 |
| Python 3.12.13 / Polars 1.42.0 / Pandas 3.0.3 | 1460 | 3 | 7 | 503.14 |

日志分别为 `.artifacts/acceptance-full38-frozen.txt` 和 `.artifacts/acceptance-full312-final.txt`。
7 deselected 来自既定 `not docs_ml and not optional_ml` 范围；skipped 保留实际报告，未计作通过。
最初旧环境整套为 1306 passed、12 failed、13 skipped、7 deselected，退出 1，
12 项均为新增测试对旧 Categorical 字符串的错误假设。该日志仍保留在
`.artifacts/acceptance-full38.txt`，不与最终结果合并计数。

最终容量文件在 Python 3.8 / Polars 1.8.2 和 Python 3.9 / Polars 1.36.1 各 84 passed，
无 skipped/deselected，退出 0；包含两个真实多 worker A/B 闭环及本轮自身表签名校验。
Python 3.11.15 / Polars 1.42.0 / Pandas 3.0.3 的分箱、列批、全局相关性、空 PSI 和容量
关联组为 241 passed，退出 0，日志 `.artifacts/acceptance-target311-final.txt`。
Python 3.9 和 3.10 的原分箱/列批/相关性/兼容定向组各 80 passed，退出 0。
本次未在 3.9、3.10、3.11 重跑完整基础套件，不以这些定向结果冒充整套或远端 CI。

交付时仍为 `main`，HEAD 未变化。工作区共有 75 个变更文件，
其中 59 个为正式生成的案例资源，16 个为实现、测试和文档；3 个新文件尚未跟踪。
CI 矩阵、constraints、包版本、原生图字体/布局、raw/binned KS 和 Excel 模板均未修改。

验收完成后，用户另行授权提交并推送本轮修复。上文未提交、未推送的说明以验收截止时点为准；
新提交的远端 CI 和 Docs 结果须另行核对，不沿用本机或旧提交的通过记录。
