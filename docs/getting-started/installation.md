---
description: 安装 MARS 0.0.28、Python 3.8 兼容栈、可选依赖和本地开发环境。
---

# 安装

MARS `0.0.28` 的基础包支持 Python 3.8–3.12，Python 3.13+ 暂不开放安装。

```bash
pip install mars-risk==0.0.28
```

!!! warning "发布前提"

    正式站点只应在 PyPI 已发布 `0.0.28` 后部署。发布前请从源码安装进行预览验收。

!!! warning "Python 3.8 生命周期"

    Python 3.8 已停止官方安全维护。MARS 只承诺冻结依赖栈下的运行兼容，不代表解释器仍有
    安全支持；可升级的生产环境应优先使用 Python 3.10–3.12。

Python 3.8 会安装 Polars 1.8.2 与 scikit-learn 1.3.x。源码开发或 CI 应使用仓库约束文件：

```bash
python -m pip install -c constraints/python38.txt -e .
```

## 可选依赖

| 场景 | 安装命令 |
| --- | --- |
| Notebook | `pip install "mars-risk[notebook]==0.0.28"` |
| 树模型 | `pip install "mars-risk[ml]==0.0.28"` |
| 调参 | `pip install "mars-risk[ml,tuning]==0.0.28"` |
| 文档构建 | `pip install "mars-risk[docs]==0.0.28"` |

基础安装已经包含画像、分箱、筛选、监控、Excel/HTML 导出和评分卡能力。`ml` 提供 XGBoost、
LightGBM、CatBoost、SHAP 与 statsmodels；`tuning` 提供 Optuna。
这些 extras 以及 `dev` 不纳入 Python 3.8/3.9 支持承诺，统一要求 Python 3.10+。

## 从源码安装

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[dev,ml,tuning,docs]"
```

## 验证安装

```bash
python -c "import mars; print(mars.__version__)"
```

输出应为 `0.0.28`。随后运行[10 分钟 Quickstart](quickstart.md)。

## 常见问题

### Modeling 导入失败

确认安装 `ml,tuning` extra，并检查模型库是否支持当前 Python 与操作系统。

### Excel 或绘图依赖缺失

基础安装包含 `openpyxl`、`xlsxwriter`、`xlwings`、`matplotlib` 和 `seaborn`。环境损坏时在新的
虚拟环境中重新安装，不建议在已有环境强制覆盖全部依赖。

### Pandas 与 Polars 怎么选

MARS 接受两类 DataFrame。宽表和大样本优先使用 Polars；已有 Pandas 工作流可以直接传入
Pandas DataFrame，并在边界处关注返回类型。
