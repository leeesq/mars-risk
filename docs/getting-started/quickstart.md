---
description: 准备 Home Credit 数据，生成分箱趋势图和报告，再加载快照查询已保存的统计结果。
---

# 10 分钟 Quickstart

本页用两个 Home Credit 外部来源分演示分箱风险评估。运行后，你会得到原生趋势图、HTML 报告和 `.marsreport` 快照，还可以在新进程中加载快照，继续查询统计结果。

MARS 0.0.28 已发布到 PyPI，基础包支持 Python 3.8～3.12。

```bash
pip install mars-risk==0.0.28
```

需要当前开发分支时，从源码安装：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

## 准备输入

先按 [Home Credit 输入指南](../user-guide/home-credit-data.md)取得 `application_train.csv`、
`application_test.csv` 和 `HomeCredit_columns_description.csv`，放到仓库的
`examples/home-credit-2018/data/raw_data/`。原始特征与训练标签来自比赛，`APPLY_DATE` 是准备脚本生成的模拟日期。

准备函数会合并两张 application 表，划分五个分区，将比赛测试表的未知标签记为 null，
并提供中文字段说明和金额字段配置。单独复制示例时，将[准备脚本](../snippets/home_credit_data.py)放在脚本旁，并按输入指南设置数据路径。

## 分析、导图、保存与查询

在仓库根目录运行共享示例：

```bash
python docs/snippets/readme_quickstart.py
```

脚本执行以下代码。它先评估风险，再导出图片和 HTML，保存快照，最后查询一行汇总数据。

```python
--8<-- "docs/snippets/readme_quickstart.py:quickstart"
```

`profile_risk()` 返回 MarsRiskProfile，其中 `report` 是 MarsBinningReport，`binner` 保存本次拟合的分箱规则。
本例只用 `0train` 的 `benchmark_df` 拟合分箱，并以它作为 PSI 的参考分布。
`group_col` 指定按五个分区比较；`time_col` 保存各分区的模拟日期范围。
同时提供这两个参数时，报告按 `group_col` 分组；按模拟月份比较的视图由[完整分箱案例](../demos/binning-stability.md)另行生成。

无标签组参加分布统计，监督指标只使用已观测标签。无标签组的监督指标为空时，应查看报告记录的计算状态，不能按零风险解释。

运行后，当前目录会生成 `risk_charts/`、`risk_report.html` 和 `risk_report.marsreport`。
打开 HTML 查看趋势图，检查终端打印的汇总数据；详细表和指标状态见[分箱案例](../demos/binning-stability.md)。

## 在新进程继续查询

在生成文件的目录启动新进程，运行以下代码。`describe()` 列出可查询的表及口径，`query_page()` 返回数据和证据引用，`write_excel()` 导出静态表。

```python
from mars.reporting import load_report

saved = load_report("risk_report.marsreport")
print(saved.describe())
page = saved.query_page("summary", features="EXT_SOURCE_2", limit=1)
print(page["reference"], page["data"])
saved.write_excel("risk_report.xlsx")
```

加载后得到 ReportSnapshot。查询已有统计无需原 CSV、分析器或内置 Agent；需要新增分析维度、查看原始记录或重新拟合时，必须读取原数据重新计算。

[七类当前案例](../demos/index.md) · [外部 Agent 指南](../user-guide/external-agents.md) · [安装与依赖](installation.md) · [API](../reference/index.md)
