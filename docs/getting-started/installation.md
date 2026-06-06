# 安装

MARS 支持 `Python >= 3.10`，包名为 `mars-risk`。

```bash
pip install mars-risk
```

基础安装已经包含数据画像、分箱、筛选、Excel/HTML 报表导出和图表报告能力。

## 可选依赖

| 场景 | 安装命令 |
| --- | --- |
| Notebook 交互 | `pip install "mars-risk[notebook]"` |
| 树模型与调参 | `pip install "mars-risk[ml,tuning]"` |
| 文档站构建 | `pip install "mars-risk[docs]"` |
| 本地开发 | `pip install -e ".[dev,ml,tuning,docs]"` |

`ml` 包含 XGBoost、LightGBM、CatBoost、SHAP 和 statsmodels。`tuning` 包含 Optuna 相关依赖。`docs` 包含 MkDocs Material 和 API Reference 生成工具。

## 从源码安装

```bash
git clone https://github.com/leeesq/mars-risk.git
cd mars-risk
pip install -e ".[dev,ml,tuning,docs]"
```

## 开发检查

提交代码前建议运行：

```bash
python -m ruff check src tests benchmarks scripts
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python -m mkdocs build --strict
MPLBACKEND=Agg python -m pytest -q --basetemp .pytest-tmp
```

Windows PowerShell 可使用：

```powershell
$env:MPLBACKEND = "Agg"
python -m pytest -q --basetemp .pytest-tmp
```

## 常见安装问题

### 建模模块导入失败

如果使用 `MarsModelingSession`、`MarsModelTuner` 或 `MarsModelReplayRunner`，请确认安装了建模可选依赖：

```bash
pip install "mars-risk[ml,tuning]"
```

### Excel 或绘图依赖缺失

基础安装应已包含 Excel/HTML 报表和绘图报告依赖。如果环境中缺少 `openpyxl`、`xlsxwriter`、`xlwings`、`matplotlib` 或 `seaborn`，建议重新安装：

```bash
pip install --upgrade --force-reinstall mars-risk
```

### Pandas 与 Polars 输入怎么选

MARS 同时支持 Pandas 和 Polars。宽表、大样本场景建议优先使用 Polars，以减少内存复制和跨框架转换。

