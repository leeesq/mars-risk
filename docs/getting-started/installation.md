---
description: 区分 PyPI 已发布版本与当前源码，说明 Python 兼容策略和可选依赖。
---

# 安装

## 当前源码与本文新能力

本站及 README 对应源码 **0.0.28**，公共报告、外部 Agent 与新分析能力按当前源码验证。
请先从源码安装：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

或克隆仓库进行开发：

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
python -m pip install -e .
```

## 已发布版本

截至 2026-10-02，核验[PyPI](https://pypi.org/project/mars-risk/)和
[最新 GitHub release](https://github.com/leeesq/mars-risk/releases/tag/0.0.27)均为 **0.0.27**。
它不包含本文全部新能力；动态徽章展示发布状态，不代表 main 已发布。

```bash
pip install mars-risk==0.0.27
```

本任务不发布版本。正式版本变更以 PyPI、pyproject.toml 和 Release Notes 为准。

## Python 兼容策略

基础包的元数据要求 Python >=3.8,<3.13；当前核心 CI 验证 3.8–3.12。
Python 3.8 使用冻结依赖栈，Polars 1.8.2 与 scikit-learn 1.3.x；
3.9 使用依赖边界栈。可选模型、调参、Notebook、文档／开发工具及内置 Agent
以 Python 3.10+ 为支持环境；并非每个 extra 都通过包元数据自动禁止旧解释器安装。
Agent 的 SDK 带 Python >=3.10 条件，模块导入也会校验版本。

Python 3.8 已停止官方安全维护，运行兼容不等于解释器继续获得维护。
旧环境源码安装使用仓库约束：

```bash
python -m pip install -c constraints/python38.txt -e .
```

## 可选依赖

在源码目录内按任务安装，未发布版本不要使用 `mars-risk==0.0.28` 的 PyPI 命令：

| 场景 | 源码安装命令 |
| --- | --- |
| Notebook | `pip install -e ".[notebook]"` |
| 树模型 | `pip install -e ".[ml]"` |
| 调参 | `pip install -e ".[ml,tuning]"` |
| 文档构建 | `pip install -e ".[docs]"` |
| 开发检查 | `pip install -e ".[dev]"` |
| 可选内置 Agent SDK | `pip install -e ".[agent]"` |

基础包包含画像、分箱、筛选、报告与 Excel／HTML 导出；保留的监控、评分卡不需要模型 SDK。
`ml` 提供 XGBoost、LightGBM、CatBoost、SHAP 与 statsmodels，`tuning` 提供 Optuna。
监控、建模（含 Pipeline）、评分卡暂停功能迭代，现有功能保留，见[稳定性](../project/stability.md)。

## 验证安装

```bash
python -c "import mars; print(mars.__version__)"
```

当前源码输出 0.0.28；已发布安装输出其对应版本。随后运行[Quickstart](quickstart.md)；
公共报告交接见[外部 Agent 指南](../user-guide/external-agents.md)。

## 常见问题

### Modeling 导入失败

确认安装 ml,tuning extra，并检查模型库对当前 Python／系统的支持。

### Excel 或绘图依赖缺失

基础包包含 openpyxl、xlsxwriter、xlwings、matplotlib 和 seaborn；
按实际错误检查依赖，不在已有环境中盲目覆盖全部依赖。

### Pandas 与 Polars 怎么选

MARS 接收两种宽表，以 Polars 为计算基础。已有 Pandas 流程可直接传入，
在返回对象与展示边界核对表类型。
