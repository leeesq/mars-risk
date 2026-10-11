---
description: MARS 0.0.28 的版本安装、main 开发安装、Python 兼容策略和可选依赖。
---

# 安装

## 安装 0.0.28

**MARS 0.0.28 已发布到 PyPI**。使用以下命令安装固定版本。

```bash
pip install mars-risk==0.0.28
```

版本发布情况以 [PyPI](https://pypi.org/project/mars-risk/)和
[GitHub Releases](https://github.com/leeesq/mars-risk/releases)为准。
动态版本徽章展示发布来源；不能将仓库版本号或文档更新视为已经上传 PyPI。

## 源码安装与 main 开发

参与 main 开发或使用尚未发布的后续修复时，从当前源码安装：

```bash
pip install "git+https://github.com/leeesq/mars-risk.git"
```

需要修改代码或运行仓库内案例时，克隆仓库并在根目录安装：

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
python -m pip install -e .
```

版本安装固定发布产物；main 可能继续变化。复现分析时记录实际版本或源码提交，
升级变化见 [Release Notes](../project/release-notes.md)。

## Python 兼容策略

基础包的元数据要求 Python >=3.8,<3.13；当前核心 CI 验证 3.8–3.12。
Python 3.8 使用冻结依赖栈，Polars 1.8.2 与 scikit-learn 1.3.x；
Python 3.9–3.12 使用 `polars>=1.33.1,<2`，本版尚未支持 Polars 2.0 的类型规则。
3.9 使用依赖边界栈。可选模型、调参、Notebook、文档／开发工具及内置 Agent
以 Python 3.10+ 为支持环境；并非每个 extra 都通过包元数据自动禁止旧解释器安装。
Agent 的 SDK 带 Python >=3.10 条件，模块导入也会校验版本。

Python 3.8 已停止官方安全维护，运行兼容不等于解释器继续获得维护。
需要复现 Python 3.8 的冻结源码验证栈时，在克隆目录内使用仓库约束：

```bash
python -m pip install -c constraints/python38.txt -e .
```

## 可选依赖

以下版本命令安装已发布的 **0.0.28**；参与开发时使用右侧的源码命令。可选能力使用 Python 3.10+。

| 场景 | 0.0.28 版本安装 | main 源码安装 |
| --- | --- | --- |
| Notebook | `pip install "mars-risk[notebook]==0.0.28"` | `pip install -e ".[notebook]"` |
| 树模型 | `pip install "mars-risk[ml]==0.0.28"` | `pip install -e ".[ml]"` |
| 调参 | `pip install "mars-risk[ml,tuning]==0.0.28"` | `pip install -e ".[ml,tuning]"` |
| 文档构建 | `pip install "mars-risk[docs]==0.0.28"` | `pip install -e ".[docs]"` |
| 开发检查 | `pip install "mars-risk[dev]==0.0.28"` | `pip install -e ".[dev]"` |
| 可选内置 Agent SDK | `pip install "mars-risk[agent]==0.0.28"` | `pip install -e ".[agent]"` |

基础包包含画像、分箱、筛选、报告与 Excel／HTML 导出；保留的监控、评分卡不需要模型 SDK。
`ml` 提供 XGBoost、LightGBM、CatBoost、SHAP 与 statsmodels，`tuning` 提供 Optuna。
监控、建模（含 Pipeline）、评分卡暂停功能迭代，现有功能保留，见[稳定性](../project/stability.md)。

## 验证安装

```bash
python -c "import mars; print(mars.__version__)"
```

0.0.28 版本安装应输出 0.0.28；main 安装输出当前源码版本。随后运行[Quickstart](quickstart.md)；
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
