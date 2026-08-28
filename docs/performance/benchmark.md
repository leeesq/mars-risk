---
description: MARS 0.0.28 分箱与规则性能基准的复现方法、结果和测量限制。
---

# 性能基准

性能结果必须同时记录代码版本、依赖、硬件、数据规模、参数和测量方法。此前缺少完整运行环境的
结果不再作为 `0.0.24` 正式性能结论展示。

## 复现命令

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
