---
description: 安装 MARS 0.0.25、可选依赖和本地开发环境。
---

# 安装

MARS `0.0.25` 支持 Python 3.10、3.11 和 3.12。

```bash
pip install mars-risk==0.0.25
```

!!! warning "发布前提"

    正式站点只应在 PyPI 已发布 `0.0.25` 后部署。发布前请从源码安装进行预览验收。

## 可选依赖

| 场景 | 安装命令 |
| --- | --- |
| Notebook | `pip install "mars-risk[notebook]==0.0.25"` |
| 树模型 | `pip install "mars-risk[ml]==0.0.25"` |
| 调参 | `pip install "mars-risk[ml,tuning]==0.0.25"` |
| 文档构建 | `pip install "mars-risk[docs]==0.0.25"` |

基础安装已经包含画像、分箱、筛选、监控、Excel/HTML 导出和评分卡能力。`ml` 提供 XGBoost、
LightGBM、CatBoost、SHAP 与 statsmodels；`tuning` 提供 Optuna。

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

输出应为 `0.0.25`。随后运行[10 分钟 Quickstart](quickstart.md)。

## 常见问题

### Modeling 导入失败

确认安装 `ml,tuning` extra，并检查模型库是否支持当前 Python 与操作系统。

### Excel 或绘图依赖缺失

基础安装包含 `openpyxl`、`xlsxwriter`、`xlwings`、`matplotlib` 和 `seaborn`。环境损坏时在新的
虚拟环境中重新安装，不建议在已有环境强制覆盖全部依赖。

### Pandas 与 Polars 怎么选

MARS 接受两类 DataFrame。宽表和大样本优先使用 Polars；已有 Pandas 工作流可以直接传入
Pandas DataFrame，并在边界处关注返回类型。
