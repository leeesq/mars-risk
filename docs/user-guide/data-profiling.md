# 数据画像与特征分析

数据画像用于回答训练前最基础的问题：字段是否可用、缺失和特殊值是否异常、分布是否稳定、不同时间或业务分组之间是否出现漂移。

## 主要入口

```python
from mars.analysis import MarsDataProfiler, profile_stats
```

- `MarsDataProfiler`：配置好的画像器，适合反复分析不同数据集。
- `profile_stats`：轻量函数入口，适合快速计算少量指标。

## 基本用法

```python
profiler = MarsDataProfiler(missing_values=[-999])

report = profiler.generate_profile(
    df,
    features=["income", "utilization", "segment"],
    group_col="month",
    config_overrides={
        "dq_metrics": ["missing", "zeros"],
        "stat_metrics": ["mean", "psi"],
        "enable_sparkline": False,
    },
)
```

`generate_profile` 接收本次数据、特征范围、分组和配置覆盖项。`MarsDataProfiler` 构造函数只保留稳定策略，例如缺失值、特殊值和默认配置。

## 常见输出

| 字段 | 含义 |
| --- | --- |
| `overview_table` | 特征级画像总览，通常用于排序、筛选和看板展示 |
| `dq_tables` | 数据质量指标趋势表字典 |
| `stats_tables` | 统计指标趋势表字典 |
| `get_profile_data()` | 返回 `overview`、`dq_trends`、`stats_trends` 的结构化集合 |

```python
overview = report.overview_table
profile_data = report.get_profile_data()
dq_trends = profile_data.dq_trends
stats_trends = profile_data.stats_trends
```

## 分组与时间

已经有分组列时，使用 `group_col`：

```python
report = profiler.generate_profile(df, group_col="month")
```

只有原始日期列时，使用 `time_col` 和 `time_grain`：

```python
report = profiler.generate_profile(
    df,
    time_col="apply_dt",
    time_grain="month",
)
```

`time_grain` 可用于把日期聚合到天、周、月或自定义窗口。具体切分窗口由调用者根据业务需要选择。

## 数据源分组

数据源分组主要用于分箱评估和特征筛选链路，例如解释某类数据源的特征整体表现或筛选流失情况：

```python
from mars.analysis import profile_risk

risk_profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization", "segment"],
    group_col="month",
    feature_data_source={
        "user_profile": ["income"],
        "credit_usage": ["utilization"],
        "application": ["segment"],
    },
)
```

## 使用建议

- 画像阶段先关注缺失率、特殊值比例、稳定性和数据源分组。
- 对宽表建议先用 `features` 缩小范围，确认指标口径后再扩展到全量字段。
- 对上线监控或周期复盘，优先把 `dq_tables` 和 `stats_tables` 接入内部看板，而不是只看一次性 Excel。

