---
description: MARS 0.0.23 分箱性能基准的复现方法、结果发布要求和测量限制。
---

# 性能基准

性能结果必须同时记录代码版本、依赖、硬件、数据规模、参数和测量方法。此前缺少完整运行环境的
结果不再作为 `0.0.23` 正式性能结论展示。

## 复现命令

```bash
python benchmarks/benchmark_binning_speed.py native \
  --rows 200000 --features 3000 --repeats 1

python benchmarks/benchmark_binning_speed.py optimal \
  --rows 50000 --features 1000 --repeats 3
```

Native 对比需要额外安装 toad；Optimal 对比使用基础依赖中的 optbinning。

## 测量口径

- 计时范围：数据生成、fit、WOE transform 和本轮清理。
- 内存口径：主进程及子进程 RSS 的采样峰值和结束增量。
- MARS 与竞品分别构造对应 DataFrame，结果包含各自数据构造成本。
- Python、Polars、竞品版本和线程设置都会影响结果，不能跨环境直接比较绝对数值。

## 0.0.23 结果发布清单

正式填写结果表前必须记录：

| 项目 | 必填内容 |
| --- | --- |
| MARS | `0.0.23` 和 commit SHA |
| Runtime | Python、Polars、NumPy、竞品版本 |
| Hardware | CPU 型号、逻辑核心数、内存容量 |
| System | 操作系统和架构 |
| Workload | 行数、特征数、分箱数、重复次数、随机种子 |
| Result | 每轮耗时、平均耗时、峰值 RSS、校验值 |
| Date | 基准执行日期 |

在完整记录生成前，README 和首页不使用“快数倍”“更省内存”等无条件性能宣传。
