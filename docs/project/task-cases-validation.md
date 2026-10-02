---
description: 原生分箱展示、报告交互及七个任务案例的当前来源、验证与限制。
---

# 原生分箱展示与案例验收

本轮从 main `97a1c36` 创建 `codex/native-binning-showcase-fixes` 分支，
修复双语 README、首页、分箱与交叉 HTML，以及共享案例。
原 Logo 字形、英文全称和六个动态徽章保留；主展示使用 MARS 原生分箱图。
没有合并 main、发布包或额外部署；PR 分支不会触发正式 Pages 部署。

## 同源产物

[七个任务](../demos/index.md)使用 18,000 行合成申请、seed `20261001`，
发现、验证、观察各 6,000 行。生成源码来源、实际依赖、快照身份与文件哈希记录在
[manifest](../assets/cases/manifest.json)和[环境记录](../assets/cases/generation-environment.json)。
报告身份随生成变化，以这批材料为准，不复用旧截图里的身份。

[主分原生图](../assets/cases/binning-native-main-score.svg)直接调用
`save_risk_trend_images()` 导出，PNG 为 300 dpi；没有重新绘制曲线或裁剪原布局。
图中是三个样本分区与 Total，日期取真实范围；不是九个月的风险分组。
九个模拟日期均为每月 15 日，日缺失表也只展示这些实际日期。
[图表证据](../assets/cases/binning-native-evidence.json)保留同报告的 summary/detail 查询。
单独下载的[分箱 HTML](../assets/cases/binning.html)内嵌原生图，无旁路图片依赖。

规则格子转 DSL 排除定义中的缺失、特殊值、非法概率和非有限分数；
逐个 `sample_id` 核对正常箱与规则命中成员相同。
[规则案例](../demos/rule-evidence.md)另外运行真实组合生成器，24 候选预算、独立验证、
单独报告与明确的输入域剔除人数，不与格子种子混为同一批规则。

筛选恢复质量、IV/Lift、PSI/RC 与相关性门槛，实际保留与淘汰理由来自 selector；
模拟漂移字段明确标记，raw 与 WOE 证据分别解释。
保存后另起进程消费快照；报告查询不调用 LLM，未宣称已执行自主 Agent 或真实业务收益。

## 复现与检查

```bash
python docs/snippets/task_cases.py --case all --rows 18000 --seed 20261001 --output-dir docs/assets/cases
python scripts/check_case_assets.py
python -m pytest -q tests/test_task_cases.py tests/test_documentation.py -m "not docs_ml"
python -m mkdocs build --strict
python scripts/check_case_assets.py --site-dir site --skip-recompute
```

900 行、自定义 seed `20261002` 用于接口、成员集合、保存与独立消费回归，
不能代替公开规模。ZIP 在没有 Git 的解压目录实际生成和消费成功；
未知源码提交标记 unknown/null，同时记录脚本和已安装模块指纹。
`previews.txt`、机器证据、Excel 与快照来自同批计算；校验会重算公开规模并重放查询。

## 当前验证与限制

源码检查包括 Ruff、157 文件 Mypy、pydoclint 与私有 docstring；六条既有私有 docstring 建议保留。
七页新增人工短代码全部实际执行通过；同目录重复保存的安全拒绝未放宽。
案例定向 17 项通过；交叉 HTML 与数值相关定向 128 项通过；分箱 HTML 定向 24 项通过。
最终文档、案例、外部规则消费者及两种 HTML 定向检查为 90 passed、2 deselected，
使用仓库内 `--basetemp .pytest-tmp-final` 复现 Docs CI 的临时目录配置；
公开 18,000 行重算、56 个文件与 8 个快照校验通过。sdist/wheel 构建与 Twine 检查通过。
严格 MkDocs 构建与构建产物校验通过；最终远端 CI 结果在 [PR #5](https://github.com/leeesq/mars-risk/pull/5) 中更新。

首次 PR CI 暴露两个真实案例问题：未加权报告的消费查询请求不存在的权重字段，
以及 ZIP 的“仓库外”测试沿用 CI 位于 checkout 内的临时目录。
消费者现在按报告的实际 `weights_col` 选择人数或权重，并在文字结论中保留分母；
加权报告缺少权重证据仍明确失败。ZIP 测试改用 checkout 的兄弟临时目录，
保留未知提交、指纹、独立生成与消费的全部断言。18,000 行加权快照消费复核通过，
查询行与证据引用保持一致；计算报告与原生图无需重算，下载源码、任务材料及 ZIP 同步刷新。

本地完整基础套件的首次运行在公开产物刷新前执行：1,040 passed、4 skipped、7 deselected，
12 failed。四项属于尚未生成的新图片/下载材料及已修正的链接断言；
另外八项为未改动的 worker/optimal 相关测试，在本执行容器受到 PID namespace 与 `/proc` 不对应影响。
受控 Python 的 `os.getpid()` 与 `/proc/self/status` 的 Pid 不同，
`psutil.Process(Popen.pid)` 读取另一进程或返回不存在，导致 RSS、后代发现及 Loky 采样不可用。
线程限制后单个 optimal 测试可返回 passed，但后台 RSS 采样仍失败，不能记作环境限制已消失。
没有为此改动 worker 代码、删断言或放宽预算，完整基础检查依靠正常 Linux CI。

本地未安装 LightGBM/Optuna，完整 optional_ml/docs_ml 不在本地验收范围。
云浏览器拒绝访问本地 HTTP 与 file 协议，因此本轮本地 HTML 没有完成真实浏览器交互验收；
导航行为由实际 Node JS 回归与生成后 HTML 集成检查覆盖。
真实 GitHub 托管的中英文 README 已实际打开并点击语言入口：两页主标题居中，
原生 PNG、动画 GIF、英文全称与六个徽章均加载成功，互链指向当前分支。
[中文桌面截图](../assets/case-validation/native-readme-zh-desktop.jpg)记录本轮原生主图与门面布局。
既有旧手机/暗色记录不代表新版已验收。
原生 Excel 外观未验收，静态工作表与关键数值由自动校验核对。

先前版本的截图与验收材料保留在 Git 历史及已有 case-validation 目录，
不作为这次原生主图、慢变色 Logo 或新版 HTML 的验证证据。
