# MARS

<div align="center">

<img src="docs/assets/mars-logo.svg" alt="MARS" width="800">

<img src="docs/assets/mars-wordmark.svg" alt="MODELING ANALYSIS RISK SCORE" width="720">

<p align="center">
  <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
  <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
  <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square&label=Python&color=364f6b"></a>
  <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
  <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
</p>

<p align="center">
  <a href="https://leeesq.github.io/mars-risk/">文档站</a> ·
  <a href="https://leeesq.github.io/mars-risk/getting-started/quickstart/">Quickstart</a> ·
  <a href="https://leeesq.github.io/mars-risk/user-guide/binning-risk-evaluation/">分箱评估</a> ·
  <a href="https://leeesq.github.io/mars-risk/user-guide/monitoring/">监控</a> ·
  <a href="https://leeesq.github.io/mars-risk/user-guide/reports-and-exports/">报告导出</a>
</p>

</div>

MARS 为 Pandas 或 Polars 宽表提供数据画像、分箱评估、特征筛选、Modeling / Pipeline、监控和
Excel/HTML 报告入口。评估与监控工作流返回结构化 report，可继续读取汇总、明细、趋势和元数据。

## 安装

MARS 支持 Python 3.10+。

```bash
pip install mars-risk==0.0.23
```

| 场景 | 安装命令 |
| --- | --- |
| Notebook | `pip install "mars-risk[notebook]"` |
| 树模型与调参 | `pip install "mars-risk[ml,tuning]"` |
| 本地开发和文档 | `pip install -e ".[dev,ml,tuning,docs]"` |

## 最小风险评估

```python
import polars as pl

from mars.analysis import profile_risk

df = pl.DataFrame(
    {
        "apply_dt": [
            "2024-01-03", "2024-01-10", "2024-02-03", "2024-02-10",
            "2024-03-03", "2024-03-10", "2024-03-17", "2024-03-24",
        ],
        "month": [
            "2024-01", "2024-01", "2024-02", "2024-02",
            "2024-03", "2024-03", "2024-03", "2024-03",
        ],
        "income": [3200, 3600, 3300, 4200, 3400, 4300, 5800, 6100],
        "utilization": [0.12, 0.18, 0.52, 0.61, 0.14, 0.29, 0.54, 0.63],
        "target": [0, 0, 1, 1, 0, 1, 1, 1],
    }
)

risk_profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization"],
    group_col="month",
    time_col="apply_dt",
    method="quantile",
    n_bins=4,
)

report = risk_profile.report
report.write_html("risk_report.html", max_plots=500, chart_embed_mode="auto")
```

完整可运行说明见[10 分钟 Quickstart](https://leeesq.github.io/mars-risk/getting-started/quickstart/)。

## 关键约定

- 风险趋势图需要有效 `time_col`。`group_col` 决定面板分组，时间范围只来自 `time_col`，
  并显示为 `YYYY-MM-DD`；只有未传 `group_col` 时，`time_grain` 才生成时间分组。
- `benchmark_df` 是基准期样本。在 `MarsBinEvaluator.evaluate()` 和 `MarsMonitor.monitor()` 中，
  分箱规则优先级为显式 `binner` → `benchmark_df` → 当前 `df`；benchmark 同时提供 PSI 基准。
- `profile_risk()` 是高层自动建箱入口，不接收 `binner`。需要固定规则时使用
  `MarsBinEvaluator.evaluate(..., binner=...)`。
- `write_html()` 默认每个 target 最多展示 500 个特征；图表超过 50 张时，`auto` 自动生成旁路
  图片资产并懒加载，以保持大报告可搜索且快速打开。

## 从场景开始阅读

| 我想做什么 | 文档 |
| --- | --- |
| 数据质量、缺失、分布与 PSI | [数据画像](https://leeesq.github.io/mars-risk/user-guide/data-profiling/) |
| 建箱、五月建箱六月评估、趋势图 | [分箱与风险评估](https://leeesq.github.io/mars-risk/user-guide/binning-risk-evaluation/) |
| 特征质量与模型重要性筛选 | [特征筛选](https://leeesq.github.io/mars-risk/user-guide/feature-selection/) |
| 调参、replay、WOE 与 Pipeline | [Modeling / Pipeline](https://leeesq.github.io/mars-risk/user-guide/modeling-pipeline/) |
| 未表现期、PSI、报警摘要 | [特征/模型监控](https://leeesq.github.io/mars-risk/user-guide/monitoring/) |
| Excel、可检索 HTML、趋势图资产 | [报告导出与二次加工](https://leeesq.github.io/mars-risk/user-guide/reports-and-exports/) |
| 签名、默认值和异常 | [API Reference](https://leeesq.github.io/mars-risk/reference/) |

## 开发检查

```bash
ruff check src tests
mypy
pydoclint src/mars
mkdocs build --strict
pytest
```

## 许可证

见 [LICENSE](LICENSE)。
