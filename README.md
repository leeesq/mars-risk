# 🚀 MARS: High-Performance Risk Modeling Framework

**MARS (Modern Analytical Risk System)** 是一个专为大规模风控建模场景设计的高性能 Python 框架。它深度集成了 **Polars** 的向量化计算引擎与 **Scikit-learn** 的设计模式，旨在解决亿级行、数千列宽表场景下的数据画像、特征工程与模型评估的性能瓶颈。

> **核心理念**：利用 Polars 实现极致的计算速度 (Vectorized Execution)，利用 Sklearn 保持优雅的 API 设计 (Fit/Transform)，实现 "零代码迁移" 的 Pandas/Polars 双向兼容。

## ✨ 核心特性 (Key Features)

### 1. 📊 高性能数据画像 (Data Profiling)
提供全链路的数据质量诊断与可视化报告，性能比传统 Pandas 方案快 10x-100x。
* **全量指标概览**: 一次性计算 Missing, Zero, Unique, Top1 等基础 DQ 指标。
* **Unicode Sparklines**: 在终端或 Notebook 中直接生成迷你分布图 (如 ` ▂▅▇█`)，快速洞察数据分布。
* **多维趋势分析**: 支持按时间 (Month/Vintage) 或客群进行分组分析，自动计算稳定性指标 (PSI, CV)。
* **Excel 自动化报告**: 导出带有热力图、数据条和条件格式的精美 Excel 报表。

### 2. 🧮 极速分箱引擎 (High-Performance Binning)
针对风控评分卡场景深度优化的分箱器。
* **MarsNativeBinner**: 完全基于 Polars 表达式实现的极速分箱。
    * 支持 **Quantile** (等频), **Uniform** (等宽), **CART** (决策树) 三种模式。
    * **并行加速**: 决策树分箱利用 `joblib` 实现多核并行，内存占用极低。
* **MarsOptimalBinner**: 混合动力最优分箱。
    * **Hybrid Engine**: 结合 Polars 的极速预分箱 (O(N)) 与 `optbinning` 的数学规划 (MIP/CP) 求解 (O(1))。
    * 支持**单调性约束** (Monotonic Trend) 和**特殊值/缺失值**的独立分层处理。

### 3. 📐 风控指标计算 (Risk Metrics)
* **连续值指标**: 精确计算 AUC, KS (基于 ROC 曲线)。
* **离散值指标**: 高速聚合计算 WOE, IV, Lift, Binned KS。
* **稳定性指标**: PSI (Population Stability Index), 形状一致性 (Shape Consistency)。

### 4. 🛠️ 工程化设计
* **Auto Polars**: 智能装饰器支持 Pandas DataFrame 无缝输入，内部自动转换为 Polars 计算，结果按需回退。
* **Pipeline Ready**: 所有组件均继承自 `MarsBaseEstimator` 和 `MarsTransformer`，完美兼容 Sklearn Pipeline。

---

## 📦 安装 (Installation)

```bash
# 推荐使用 pip 安装
pip install mars-risk

# 或者从源码安装
git clone [https://github.com/your-username/mars-risk.git](https://github.com/your-username/mars-risk.git)
cd mars-risk
pip install -e .
依赖项: polars, pandas, numpy, scikit-learn, scipy, xlsxwriter, colorlog. (可选: optbinning)

# ⚡️ 快速上手 (Quick Start)
## 场景 1：生成数据画像报告
```python
import polars as pl
from mars.analysis.profiler import MarsDataProfiler

# 1. 加载数据
df = pl.read_csv("your_data.csv")

# 2. 初始化分析器 (支持自定义缺失值，如 -999)
profiler = MarsDataProfiler(df, custom_missing_values=[-999, "unknown"])

# 3. 生成画像报告
report = profiler.generate_profile(
    profile_by="month",  # 可选：按月份分组分析趋势
    config_overrides={"enable_sparkline": True} # 开启迷你分布图
)

# 4. 展示与导出
report.show_overview()  # 在 Jupyter 中查看概览 (含热力图)
report.show_trend("mean") # 查看均值趋势
report.write_excel("data_profile_report.xlsx") # 导出为 Excel
```

## 场景 2：高性能特征分箱
```python
from mars.feature.binning import MarsNativeBinner, MarsOptimalBinner

# --- 方式 A: 极速原生分箱 (适合大规模预处理) ---
binner = MarsNativeBinner(
    features=["age", "income"],
    method="quantile",  # 等频分箱
    n_bins=10,
    special_values=[-1] # 特殊值独立成箱
)
binner.fit(X_train, y_train)
X_train_binned = binner.transform(X_train)

# --- 方式 B: 最优分箱 (适合评分卡精细建模) ---
opt_binner = MarsOptimalBinner(
    features=["credit_score"],
    n_bins=5,
    solver="cp", # 使用约束编程求解
    monotonic_trend="ascending" # 强制单调递增
)
opt_binner.fit(X_train, y_train)
print(opt_binner.bin_cuts_) # 查看最优切点
```

# 📂 项目结构 (Project Structure)
```Plaintext
mars/
├── analysis/           # 数据分析与画像模块
│   ├── profiler.py     # MarsDataProfiler 核心逻辑
│   ├── report.py       # MarsProfileReport 报告容器
│   └── config.py       # 分析配置类
├── feature/            # 特征工程模块
│   ├── binning.py      # NativeBinner & OptimalBinner
│   ├── encoding.py     # (开发中) 编码器
│   └── imputer.py      # (开发中) 缺失值填补
├── risk/               # 风控专用模块
│   └── validator.py    # MarsTrendValidator 趋势/稳定性校验
├── metrics/            # 数学指标计算
│   └── calculation.py  # KS, AUC, PSI, WOE 计算引擎
├── modeling/           # 建模策略模块
│   ├── base.py
│   └── tuner.py        # 自动调参器
├── core/               # 核心基类
│   ├── base.py         # MarsBaseEstimator (Sklearn 兼容)
│   └── exceptions.py   # 自定义异常
└── utils/              # 工具库
    ├── logger.py       # 全局日志配置
    └── decorators.py   # @time_it, @auto_polars 装饰器
```

## 🤝 贡献 (Contributing)
欢迎提交 Issue 和 Pull Request！ 在提交 PR 前，请确保通过了所有的单元测试，并遵循现有的代码风格 (Type Hinting + Numpy Docstrings)。

## 📄 许可证 (License)
MIT License