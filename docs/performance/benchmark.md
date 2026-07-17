# 性能对比

本页记录原生分箱与最优分箱脚本在固定宽表数据上的耗时和内存结果。结果只适用于列出的
数据规模、参数和运行环境。

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

- 表中的原生分箱结果对应 `200,000 × 3,000` 个数值特征和 `method=quantile/uniform`。
- 表中的最优分箱结果对应 `50,000 × 1,000` 个数值特征和单特征 `time_limit=1s`；该场景下
  `MarsOptimalBinner` 耗时更低、峰值内存更高。
- 使用自己的数据复现前，请同时记录特征分布、缺失率、类别基数、CPU、内存和 Polars 版本。
