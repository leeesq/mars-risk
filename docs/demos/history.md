---
description: LightGBM 历史案例的维护边界、独立数据及可选依赖；保留原 Notebook URL。
---

# 历史／保留案例

[LightGBM 建模与监控](lgb-modeling-monitoring.ipynb)继续保留原 URL、完整代码和数据角色说明。
建模（包含 Pipeline）、监控、评分卡暂停功能迭代；现有功能与文档保留，并处理必要正确性、运行修复和上游适配。
Experimental 表示成熟度，暂停表示投入，两者不能互相代替。

该 Notebook 使用 seed `1206`、240 行独立合成数据完成时间切分、轻量调参、打分与监控。
它不属于七个核心实战案例的共享 18,000 行数据，也不作为正在扩展的训练路线。

## 在 0.0.28 环境中运行

使用 Python 3.10+。0.0.28 发布后安装可选依赖，再取得并打开上面的 Notebook：

```bash
python -m pip install "mars-risk[ml,tuning,notebook]==0.0.28"
```

若 PyPI 尚未提供该版本，先按[安装指南](../getting-started/installation.md)克隆 main，
在仓库根目录使用源码安装并打开 Notebook：

```bash
python -m pip install -e ".[ml,tuning,notebook]"
python -m jupyter lab docs/demos/lgb-modeling-monitoring.ipynb
```

本页对应 `0.0.28`；旧材料中的 `mars-risk[ml,tuning]==0.0.24` 不适用于当前 Notebook。
版本命令须在 PyPI 提供 0.0.28 后使用；发布前的 main 安装与固定版本安装分别记录。
LightGBM、其他模型后端与 Optuna 属于可选 `ml,tuning` extra；七个核心案例不要求这些训练依赖。

!!! warning "渲染与执行分别验收"

    MkDocs 的 `mkdocs-jupyter.execute: false` 只渲染 Notebook。
    保留页面、严格构建通过或旧输出存在，都不能证明本轮重新执行了模型训练。
    是否实际执行，以本轮[验收记录](../project/task-cases-validation.md)为准。

[实战案例索引](index.md) · [保留的 Score Cross Notebook](correlation_and_score_cross.ipynb) ·
[稳定性与兼容性](../project/stability.md)
