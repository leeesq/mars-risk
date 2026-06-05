# MARS 快速开始

这份教程使用小型合成数据，演示 MARS 的三个典型工作流：

1. 数据画像
2. 特征分箱评估
3. Excel 报表导出

## 1. 准备数据

```python
import polars as pl

df = pl.DataFrame(
    {
        "month": [
            "2024-01", "2024-01", "2024-01", "2024-01",
            "2024-02", "2024-02", "2024-02", "2024-02",
            "2024-03", "2024-03", "2024-03", "2024-03",
        ],
        "income": [3200, 3600, -999, None, 3300, 4200, -999, 5800, 3400, 4300, None, 6100],
        "utilization": [0.12, 0.18, 0.52, 0.61, 0.14, 0.29, 0.54, 0.58, 0.16, 0.31, 0.56, 0.63],
        "segment": ["new", "repeat", "vip", "vip", "new", "repeat", "vip", "vip", "new", "repeat", "vip", "vip"],
        "target": [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
    }
)
```

## 2. 做一次数据画像

```python
from mars.analysis import MarsDataProfiler

profiler = MarsDataProfiler(
    missing_values=[-999],
)

profile_report = profiler.generate_profile(
    df,
    group_col="month",
    config_overrides={
        "enable_sparkline": False,
        "dq_metrics": ["missing", "zeros"],
        "stat_metrics": ["mean", "psi"],
    },
)

overview = profile_report.overview_table
```

你通常会先看：

- `overview`：全量特征概览。
- `profile_report.dq_tables["missing"]`：各月缺失率趋势。
- `profile_report.stats_tables["psi"]`：各月稳定性结果。

导出画像 Excel：

```python
profile_report.write_excel("tutorial_profile_report.xlsx")
```

## 3. 跑一次特征评估

```python
from mars.analysis import profile_risk

risk_profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization", "segment"],
    group_col="month",
    binning_type="native",
    binner_params={"method": "quantile", "n_bins": 4},
    plot=False,
)

eval_report = risk_profile.report
binner = risk_profile.binner
summary = eval_report.summary_table
detail = eval_report.detail_table
```

`profile_risk()` 返回 `MarsRiskProfile(report, binner, targets, metadata)`。这里最常用的是：

- `eval_report`：汇总表、趋势表、明细表和 Excel 导出入口。
- `binner`：保留了已经拟合好的分箱器，可以继续复用规则。

导出评估报表：

```python
eval_report.write_excel("tutorial_evaluation_report.xlsx", engine="openpyxl")
```

## 4. 单独使用分箱器

```python
from mars.feature import MarsNativeBinner

X = df.select(["income", "utilization", "segment"])
y = df.get_column("target")

binner = MarsNativeBinner(
    method="quantile",
    n_bins=4,
    special_values=[-999],
)

binner.fit(X, y, cat_features=["segment"])
X_binned = binner.transform(X, return_type="index")
X_woe = binner.transform(X, return_type="woe")
income_mapping = binner.get_bin_mapping("income")
```

## 5. 参数建议

- 快速评估宽表：优先从 `MarsNativeBinner(method="quantile")` 开始。
- 对评分卡单调性要求高：再切到 `MarsOptimalBinner`。
- 先要稳定的开源使用体验：`plot=False`，先把表和报表跑通。
- 团队共享结果：优先导出 `summary_table` 和 Excel。

## 6. 下一步

- 查看 [performance_audit.md](performance_audit.md) 了解当前性能整理方向。
- 运行 [../benchmarks/benchmark_binning_speed.py](../benchmarks/benchmark_binning_speed.py) 复现 5w×1000 特征速度对比。
