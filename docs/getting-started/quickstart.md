---
description: 使用 Home Credit 共享输入，完成固定分箱、原生图和保存查询。
---

# 10 分钟 Quickstart

MARS 0.0.28 已发布到 PyPI，基础包支持 Python 3.8-3.12。

```bash
pip install mars-risk==0.0.28
```

当前开发源码可使用以下命令。

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

## 准备输入

本页复用当前公开案例的真实原始特征与标签，APPLY_DATE 为模拟日期。
先按[Home Credit 输入指南](../user-guide/home-credit-data.md)取得三个 CSV。
从仓库根目录运行；单独复制示例时，将[共享 helper](../snippets/home_credit_data.py)放在脚本旁。
完整五分区、null 标签、中文字典与金额角色由 helper 统一准备。

## 分析、导图、保存与查询

```python
--8<-- "docs/snippets/readme_quickstart.py:quickstart"
```

profile_risk() 返回 MarsRiskProfile，report 为 MarsBinningReport，binner 为本次拟合规则。
benchmark_df 只来自 0train，用于拟合及 PSI 参考。group_col 比较五分区，time_col 保存模拟日期范围；模拟月视图由完整案例另行生成。
无标签组参加分布，监督指标只用已观测标签。详细表与状态见[分箱案例](../demos/binning-stability.md)。

## 在新进程继续查询

```python
from mars.reporting import load_report

saved = load_report("risk_report.marsreport")
print(saved.describe())
page = saved.query_page("summary", features="EXT_SOURCE_2", limit=1)
print(page["reference"], page["data"])
saved.write_excel("risk_report.xlsx")
```

恢复为 ReportSnapshot，查询已有统计无需原 CSV、分析器或内置 Agent。需要新的维度或重新拟合时，再读取原数据计算。
[七类当前案例](../demos/index.md) · [外部 Agent 指南](../user-guide/external-agents.md) · [安装与依赖](installation.md) · [API](../reference/index.md)
