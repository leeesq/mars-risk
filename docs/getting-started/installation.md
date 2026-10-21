---
description: MARS 0.0.28 的版本安装、main 开发安装、Python 兼容策略和可选依赖。
---

# 安装

## 安装 0.0.28

MARS 0.0.28 已发布到 PyPI。使用以下命令安装固定版本：

```bash
pip install mars-risk==0.0.28
```

版本发布情况以 [PyPI](https://pypi.org/project/mars-risk/)和
[GitHub Releases](https://github.com/leeesq/mars-risk/releases)为准。
动态版本徽章来自发布服务。仓库中的版本号或文档更新不代表软件包已经上传到 PyPI。

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

指定版本号会安装对应的发布产物，main 分支则会继续变化。复现分析时，记录实际版本或源码提交。
升级前查看 [Release Notes](../project/release-notes.md)。

## Python 兼容策略

基础包要求 Python >=3.8,<3.13，当前核心 CI 验证 Python 3.8～3.12。
Python 3.8 使用冻结依赖栈，Polars 1.8.2 与 scikit-learn 1.3.x；
Python 3.9～3.12 使用 `polars>=1.33.1,<2`，本版尚未支持 Polars 2.0 的类型规则。
Python 3.9 使用兼容该版本的依赖边界配置。

可选模型、调参、Notebook、文档／开发工具和内置 Agent 支持 Python 3.10+。
部分 extra 的包元数据未阻止旧解释器安装，安装成功也不代表受支持。
Agent SDK 的依赖声明要求 Python >=3.10，模块导入时也会校验版本。

Python 3.8 已停止官方安全维护，运行兼容不等于解释器继续获得维护。
需要复现 Python 3.8 的冻结源码验证栈时，在克隆目录内使用仓库约束：

```bash
python -m pip install -c constraints/python38.txt -e .
```

## 可选依赖

左侧命令安装已发布的 0.0.28；参与开发时，先克隆仓库，再在根目录执行右侧命令。
可选能力使用 Python 3.10+。

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

安装 0.0.28 后应输出 `0.0.28`；从 main 安装则输出当前源码版本。随后运行 [Quickstart](quickstart.md)；
公共报告交接见[外部 Agent 指南](../user-guide/external-agents.md)。

## 常见问题

### Modeling 导入失败

确认已安装 `ml,tuning` extra，再检查模型库是否支持当前 Python 版本和操作系统。

### Excel 或绘图依赖缺失

基础包包含 openpyxl、xlsxwriter、xlwings、matplotlib 和 seaborn。
先按报错检查对应依赖，避免在已有环境中覆盖全部依赖。

### Pandas 与 Polars 怎么选

MARS 接受两种宽表，使用 Polars 计算。已有 Pandas 流程可以直接传入 DataFrame；
继续处理返回值或展示表格时，先确认结果是 Pandas 还是 Polars 表。
