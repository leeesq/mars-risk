# 性能对比

MARS 面向宽表分箱场景优化大样本、多特征下的分箱拟合、规则转换和报表链路衔接。下面结果来自 README 中记录的 benchmark。

## 复现命令

```bash
conda run -n mars python benchmarks/benchmark_binning_speed.py native --rows 200000 --features 3000 --repeats 1
conda run -n mars python benchmarks/benchmark_binning_speed.py optimal --rows 50000 --features 1000 --repeats 3
```

## 计时和内存口径

- 计时范围：数据生成 + fit + transform + 本轮清理。
- 内存口径：主进程及其子进程的 RSS。
- 结束增量：本轮结束 RSS - 起始 RSS。
- 峰值增量：采样峰值 RSS - 起始 RSS。
- 结束增量会受 Python、Polars、NumPy 内存分配器缓存影响；比较峰值压力时优先看峰值增量。

## 原生分箱：toad vs MarsNativeBinner

数据规模：`200,000` 行 × `3,000` 个数值特征，重复次数 `1`，随机种子 `2026`。

| 场景 | 方法 | 平均耗时(s) | 峰值增量(MB) | 相对基准 | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| 等频分箱 | toad Combiner + WOETransformer | 126.083 | 20516.1 | 1.60x | |
| 等频分箱 | MarsNativeBinner | 78.768 | 6768.8 | 1.00x | method=quantile |
| 等宽分箱 | toad Combiner + WOETransformer | 105.859 | 20468.4 | 1.31x | |
| 等宽分箱 | MarsNativeBinner | 81.058 | 6727.7 | 1.00x | method=uniform |

## 最优分箱：MarsOptimalBinner vs optbinning

数据规模：`50,000` 行 × `1,000` 个数值特征，重复次数 `3`，随机种子 `2026`。

| 场景 | 方法 | 平均耗时(s) | 峰值增量(MB) | 相对基准 | 备注 |
| --- | --- | ---: | ---: | ---: | --- |
| 最优分箱 | MarsOptimalBinner | 28.011 | 5531.5 | 1.00x | 单特征 time_limit=1s |
| 最优分箱 | optbinning.BinningProcess | 125.826 | 628.4 | 4.49x | 单特征 time_limit=1s |

## 如何理解结果

- MARS 的原生分箱在宽表场景下重点优化速度和峰值内存压力。
- 最优分箱比直接 `optbinning.BinningProcess` 更快，但峰值内存更高，适合需要大批量特征快速评估的场景。
- benchmark 是复现脚本结果，不代表所有业务数据。实际性能会受特征分布、缺失率、类别基数、CPU、内存和 Polars 版本影响。

