---
description: LightGBM 历史案例的维护边界、独立数据及可选依赖；保留原 Notebook URL。
---

# 历史／保留案例

[LightGBM 建模与监控](lgb-modeling-monitoring.ipynb)保留原 URL、完整代码和数据角色说明。
建模（包含 Pipeline）、监控和评分卡暂停功能迭代。我们保留现有功能和文档，修复必要的正确性与运行问题，并在上游变化时修改调用代码。
Experimental 说明接口成熟度，暂停说明开发投入，两个标记分别使用。

这本 Notebook 使用随机种子 `1206` 生成 240 行独立合成数据，再按时间切分、运行轻量调参、打分并监控。
七个核心实战案例使用 Home Credit 的 356,255 条记录；这本历史 Notebook 继续使用独立数据，保留原 URL。

## 在 0.0.28 环境中运行

使用 Python 3.10+。0.0.28 已发布，先安装可选依赖，再取得并打开上面的 Notebook：

```bash
python -m pip install "mars-risk[ml,tuning,notebook]==0.0.28"
```

需要开发版时，按[安装指南](../getting-started/installation.md)克隆 main，在仓库根目录使用源码安装并打开 Notebook：

```bash
python -m pip install -e ".[ml,tuning,notebook]"
python -m jupyter lab docs/demos/lgb-modeling-monitoring.ipynb
```

本页对应 `0.0.28`；旧材料中的 `mars-risk[ml,tuning]==0.0.24` 不适用于当前 Notebook。
固定版本的安装命令要求 PyPI 已提供 0.0.28，发布前则从 main 安装。两种安装方式分别记录。
LightGBM、其他模型后端和 Optuna 属于可选的 `ml,tuning` extra，七个核心案例无需这些训练依赖。

!!! warning "渲染与执行分别验收"

    MkDocs 配置 `mkdocs-jupyter.execute: false` 时，只渲染已保存的 Notebook。
    页面保留、严格构建通过或存在旧输出，都不能证明本轮重新执行过模型训练。
    是否实际执行，以本轮[验收记录](../project/task-cases-validation.md)为准。

[实战案例索引](index.md) · [保留的 Score Cross Notebook](correlation_and_score_cross.ipynb) ·
[稳定性与兼容性](../project/stability.md)
