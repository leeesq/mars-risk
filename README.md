# MARS

<div align="center">
  <img src="docs/assets/mars-logo.svg" alt="MARS" width="520">
  <p><strong>MODELING ANALYSIS RISK SCORE</strong></p>
  <p>面向信贷风控分析与建模的 Polars-first Python 工具库</p>
  <p>
    <a href="https://pypi.org/project/mars-risk/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mars-risk?style=flat-square&label=PyPI&color=2f6f8f"></a>
    <a href="https://leeesq.github.io/mars-risk/"><img alt="Docs" src="https://img.shields.io/badge/Docs-GitHub%20Pages-7c3aed?style=flat-square"></a>
    <a href="https://pypi.org/project/mars-risk/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mars-risk?style=flat-square&label=Python&color=364f6b"></a>
    <a href="https://pepy.tech/project/mars-risk"><img alt="Downloads" src="https://img.shields.io/pepy/dt/mars-risk?style=flat-square&label=Downloads&color=0f766e"></a>
    <a href="https://github.com/leeesq/mars-risk/actions/workflows/test.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/leeesq/mars-risk/test.yml?branch=main&style=flat-square&label=CI&color=1f7a5a"></a>
    <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/leeesq/mars-risk?style=flat-square&label=License&color=6c5ce7"></a>
  </p>
</div>

MARS 接受 Pandas 或 Polars 宽表，提供数据画像、分箱评估、特征筛选、建模、监控、结构化 report、
Excel/HTML 导出和评分卡能力。

## 安装

MARS `0.0.23` 支持 Python 3.10、3.11 和 3.12。

```bash
pip install mars-risk==0.0.23
```

建模与调参需要可选依赖：

```bash
pip install "mars-risk[ml,tuning]==0.0.23"
```

`0.0.23` 正式发布前，请从源码安装进行文档预览验收。

## 最小风险评估

```python
import polars as pl

from mars.analysis import profile_risk

df = pl.DataFrame(
    {
        "income": [3200, 3600, 5200, 6100, 3400, 4300, 5800, 6800],
        "utilization": [0.72, 0.61, 0.29, 0.18, 0.66, 0.48, 0.24, 0.12],
        "target": [1, 1, 0, 0, 1, 1, 0, 0],
    }
)

risk_profile = profile_risk(
    df,
    target="target",
    features=["income", "utilization"],
    method="quantile",
    n_bins=4,
)

summary = risk_profile.report.summary_table
binner = risk_profile.binner
```

完整的日期、分组、趋势和报告示例见
[10 分钟 Quickstart](https://leeesq.github.io/mars-risk/getting-started/quickstart/)。

## 从任务开始

| 目标 | 文档 |
| --- | --- |
| 检查缺失、分布和 PSI | [数据画像](https://leeesq.github.io/mars-risk/user-guide/data-profiling/) |
| 使用基准规则评估当前数据 | [分箱与风险评估](https://leeesq.github.io/mars-risk/user-guide/binning-risk-evaluation/) |
| 统计、线性或重要性筛选 | [特征筛选](https://leeesq.github.io/mars-risk/user-guide/feature-selection/) |
| 切分、调参、replay 与 Pipeline | [Modeling / Pipeline](https://leeesq.github.io/mars-risk/user-guide/modeling-pipeline/) |
| 分布、模型分和表现覆盖率监控 | [特征与模型监控](https://leeesq.github.io/mars-risk/user-guide/monitoring/) |
| Excel、HTML、评分卡与 SQL | [报告与评分卡](https://leeesq.github.io/mars-risk/user-guide/reports-and-exports/) |
| 精确签名、默认值和异常 | [API Reference](https://leeesq.github.io/mars-risk/reference/) |

## 稳定性

Analysis、Feature、Reporting 是当前 Stable 模块。Monitoring、Modeling、Pipeline、Scoring 为
Experimental；受控生产流程应固定精确版本，并为 report 字段、报警结果、评分映射、生成 SQL、
step 契约、replay 和 artifact 路径增加契约回归。

## 开发检查

```bash
python -m ruff check src tests scripts docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python -m pytest -q
python -m mkdocs build --strict
```

贡献要求见 [CONTRIBUTING.md](CONTRIBUTING.md)，许可证见 [LICENSE](LICENSE)。
