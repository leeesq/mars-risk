---
description: MARS 0.0.28 分箱与规则性能基准的复现方法、结果和测量限制。
---

# 性能基准

性能结果必须同时记录代码版本、依赖、硬件、数据规模、参数和测量方法。此前缺少完整运行环境的
结果不再作为 `0.0.24` 正式性能结论展示。

## 复现命令

### 完整分析链路（本地测量）

新增 `benchmarks/benchmark_analysis.py` 复用既有固定种子宽表生成器与 RSS sampler。
每次 CLI 调用是独立进程；`--source` 可指向优化前 src 快照，两侧使用同一脚本、工作负载和线程。
结果输出放临时目录，例如 PowerShell：

```powershell
conda run -n mars python benchmarks/benchmark_analysis.py --rows 50000 --features 1001 --threads 4 --batch-size 50 --benchmark --output "$env:TEMP/analysis.json"
conda run -n mars python benchmarks/benchmark_analysis.py --stage profile --rows 50000 --features 1001 --threads 4 --batch-size 50 --output "$env:TEMP/profile.json"
```

记录环境：2026-09-30 至 2026-10-01（Asia/Shanghai），Windows 11 10.0.26200，Intel x64
Family 6 Model 183 Stepping 1，24 logical CPUs，Python 3.10.19、Polars 1.37.1、MARS 0.0.28，
固定 4 个 Polars 线程。优化前为 `d3201a2` 的 src 快照，优化后为本次未提交工作树。
随机种子 2026；50,000 行、1,001 个 Float32 特征，末批不足 50；含 NaN、特殊值 -999、
4 个分组、样本权重和金额。分箱场景用前 25,000 行作显式基准拟合 8 箱，RC 使用默认 total。
画像场景独立运行 missing/zeros/unique/mode/mean/std/min/max，不生成 sparkline。

两侧各 3 个独立进程，中位数如下（MiB = 1024² bytes）：

| 测量 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 完整分箱评估耗时 | 11.53 s | 11.34 s |
| 分箱评估计算阶段峰值 RSS | 1907.7 MiB | 959.9 MiB |
| 输入准备后分箱评估峰值 RSS 增量 | 1484.4 MiB | 536.9 MiB |
| 独立画像耗时 | 3.38 s | 2.61 s |
| 独立画像计算阶段峰值 RSS | 498.1 MiB | 498.4 MiB |
| 输入准备后画像峰值 RSS 增量 | 75.0 MiB | 75.5 MiB |

评估各阶段中位数：

| 阶段 | 优化前 | 优化后 |
| --- | ---: | ---: |
| 拟合与拟合诊断 | 6.163 s | 5.919 s |
| 当前及基准转换 | 1.263 s | 1.177 s |
| 当前样本聚合 | 1.812 s | 1.860 s |
| 基准分布 | 1.927 s | 1.953 s |
| 指标计算 | 0.074 s | 0.077 s |
| 报告表构造 | 0.209 s | 0.194 s |

阶段耗时不含所有输入预处理、对象装配和采样开销，不能直接相加等同总耗时。
分箱评估主要收益是减少全量分箱宽表和显式基准长表，计算阶段峰值 RSS 约减半；耗时变化小于
轮间波动，不宣称稳定提速。画像约减少 23% 耗时，计算阶段内存无明显收益。
全进程峰值也包含数据生成，分别约为分箱 1907.7→959.9 MiB、独立画像 753.1→742.5 MiB；
画像数据准备造成的峰值波动较大，不能据此宣称内存优化。

sampler 每 10 ms 采样 RSS，总峰值取全程与计算阶段 sampler 观测最大值；可能漏掉更短暂峰值。
输入准备完成后另起计算 sampler，增量相对该阶段起始 RSS。画像另起进程以避开评估后内存池。
没有把 Polars clone 当作底层全复制。不同硬件、线程、dtype、箱数及基准策略需要重新测量；
未验证百万行、3,000 特征或监督分箱器的大规模收益。

`--tables-output` 可将小结果表保存到临时目录供数值对照；本次固定数据的 19 张画像/风险表
已与优化前源码逐表比较（包括 schema，浮点容差 1e-6）。通常不需要在性能测量时保存表。

### 既有分箱与规则基准

```bash
python benchmarks/benchmark_binning_speed.py native \
  --rows 200000 --features 3000 --repeats 1

python benchmarks/benchmark_binning_speed.py optimal \
  --rows 50000 --features 1000 --repeats 3

python benchmarks/benchmark_rule_mining.py \
  --engine mars --rows 100000 --features 1000 --max-candidates 5000 \
  --output-json benchmarks/results/mars-rule.json

python benchmarks/benchmark_rule_mining.py \
  --engine deimos --rows 100000 --features 1000 --max-candidates 5000 \
  --output-json benchmarks/results/deimos-rule.json \
  --baseline-root ../deimos-rule

python benchmarks/benchmark_rule_mining.py \
  --engine gate \
  --mars-result benchmarks/results/mars-rule.json \
  --deimos-result benchmarks/results/deimos-rule.json

python benchmarks/benchmark_rule_stages.py \
  --stage all --rows 10000 --rules 100 --repeats 3 \
  --output-json benchmarks/results/rule-stages-current.json

python benchmarks/benchmark_rule_stages.py \
  --stage all --rows 10000 --rules 100 --repeats 3 \
  --baseline-json benchmarks/results/rule-stages-before.json
```

Native 对比需要额外安装 toad；Optimal 对比使用基础依赖中的 optbinning。
规则发布门禁要求相同机器、环境和数据上，对比 `deimos-rule`
`e6714c5e795054e44f0c58ad7097668b4117b4a2` 的组合生成与评估：MARS 总耗时不得退化超过 15%，
进程峰值 RSS 不得退化超过 20%。普通 CI 只运行 2k×20 smoke；100k×1000 对比在发布前手工运行。
子阶段 benchmark 分别覆盖 evaluator、压缩位图 IoU、命中矩阵 analysis 和 cascade；同机预热后
隔离运行 3 次取中位数，发布门禁要求 evaluator、IoU、analysis 相对改造前各至少提速 30%，
峰值 RSS 不得退化超过 10%。`rule-stages-before.json` 必须来自相同 commit 依赖环境和工作负载。
由于该来源提交调用的是旧版 `MarsStatsSelector(target=..., features=...)` 构造签名，benchmark
仅在 harness 中把这一调用适配为当前 `fit(..., target=..., features=...)`，预筛参数和排序口径不变。

## 0.0.28 规则门禁结果

2026-08-12 在同一 Windows 机器和 `mars` Conda 环境分别启动隔离进程测量。Mars 与 deimos
均使用已验证的 √预算单规则池和二阶 AND/OR 候选口径；完整精度切点与稳定排序差异使两者最终
候选数相差 30 条（0.7%）。

| 项目 | Mars 0.0.28 | deimos `e6714c5` | 比值 / 门槛 |
| --- | ---: | ---: | ---: |
| 组合生成＋评估耗时 | 12.094 s | 10.888 s | 1.1108 / ≤ 1.15 |
| 进程峰值 RSS | 2806.6 MB | 2756.8 MB | 1.0181 / ≤ 1.20 |
| 候选规则数 | 4240 | 4270 | 仅作工作量校验 |

同一环境的 10,000×100 分阶段 workload 预热后独立运行 3 次取中位数；原始结果保存在
`benchmarks/results/mars_rule_stages_0_0_28.json`：

| 子阶段 | 中位耗时 | 峰值 RSS | 校验值 |
| --- | ---: | ---: | ---: |
| evaluator | 0.0282 s | 266.8 MB | 1800 |
| IoU | 0.0068 s | 267.2 MB | 5 |
| analysis | 0.7921 s | 323.4 MB | 4950 |
| cascade | 0.0222 s | 302.6 MB | 1 |

客户指标由逐规则对 Python 集合改为一次性因子化后，在同一 workload 上 analysis 从
5.3525 s 降至 0.7921 s（85.2%），峰值 RSS 从 312.9 MB 增至 323.4 MB（3.4%）。

环境：Python 3.10.19、Polars 1.37.1、NumPy 2.2.6、scikit-learn 1.7.2、Windows
10.0.26200、Intel Core i7-14650HX（24 逻辑核）、63.7 GiB 内存。工作负载为 100,000 行、
1,000 个 `float32` 特征、`n_bins=10`、`max_candidates=5000`、`batch_size=100`、随机种子 42。

## 测量口径

- 分箱计时范围：数据生成、fit、WOE transform 和本轮清理。
- 规则计时范围：预先构造同一宽表，计入特征预筛、候选生成和长表评估，不计数据生成。
- 内存口径：主进程及子进程 RSS 的采样峰值和结束增量。
- MARS 与竞品分别构造对应 DataFrame，结果包含各自数据构造成本。
- Python、Polars、竞品版本和线程设置都会影响结果，不能跨环境直接比较绝对数值。

## 0.0.24 结果发布清单

正式填写结果表前必须记录：

| 项目 | 必填内容 |
| --- | --- |
| MARS | `0.0.24` 和 commit SHA |
| Runtime | Python、Polars、NumPy、竞品版本 |
| Hardware | CPU 型号、逻辑核心数、内存容量 |
| System | 操作系统和架构 |
| Workload | 行数、特征数、分箱数、重复次数、随机种子 |
| Result | 每轮耗时、平均耗时、峰值 RSS、校验值 |
| Date | 基准执行日期 |

在完整记录生成前，README 和首页不使用“快数倍”“更省内存”等无条件性能宣传。

## 公共报告消费的针对性测量

运行 `python benchmarks/benchmark_report_consumption.py`；该脚本直接构造统计表，不执行拟合。
2026-10-01 原始结果保存在 `benchmarks/results/report_consumption_20261001.json`。
环境为 Windows 10.0.26200、Intel Core i7-14650HX、24 逻辑核、63.7 GiB 内存，
Python 3.10.19、Pandas 2.3.3、Polars 1.37.1。运行时没有并行测试或文档构建。

工作负载：10,000 行 Pandas overview，加 1 个特征、365 个日期列的 stats.mean；分页
offset=9000、limit=10，查询/上下文运行 10 次取中位数，保存/恢复运行 3 次。

| 操作 | 中位耗时 | Python 分配峰值 | 校验结果 |
| --- | ---: | ---: | --- |
| 分页 | 0.211 ms | 9,549 B | 返回 10 行，总数 10,000 |
| 16,000 字符上下文 | 33.950 ms | 2,126,371 B | 13,487 字符；保留全部 365 天 |
| 5,000 字符上下文 | 198.513 ms | 2,120,399 B | 4,988 字符；保留 92 天，省略 273 天并给出位置 |
| 完整保存 | 94.228 ms | 4,511,040 B | 单文件 587,148 B |
| 完整恢复 | 60.756 ms | 5,539,585 B | 无原始宽表或分析器 |

字符计数包含整个最终 JSON。`tracemalloc` 只记录 Python 分配，不含 Arrow/Polars 等原生内存，
也不是进程 RSS。较小预算需要更多裁剪步骤，因此耗时更高。这里只报告本机绝对测量，
没有旧版对照，不证明提速或峰值内存改善；本轮没有重跑大规模拟合矩阵。
