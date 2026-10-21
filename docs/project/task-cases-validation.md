---
description: PR #5 原生分箱展示的历史验收记录，当前 Home Credit 迁移另见专门记录。
---

# 原生分箱展示与案例验收（历史）

本页保留 PR #5 当时使用的合成数据、运行命令和环境限制。当前公开案例已更换数据，不适用这里的统计结论。
当前案例已迁移到 Home Credit，见[本次迁移验收](home-credit-migration-validation.md)。
本页旧统计材料链接指向当时的 Git 版本，避免将历史数字对到当前报告。

当时从 main `97a1c36` 创建 `codex/native-binning-showcase-fixes` 分支，
修复双语 README、首页、分箱与交叉 HTML，以及共享案例。
原 Logo 字形、英文全称和六个动态徽章保留；主展示使用 MARS 原生分箱图。
当时没有合并 main、发布软件包或额外部署；PR 分支不会触发正式 Pages 部署。

## 同源产物

[七个任务](../demos/index.md)使用 18,000 行合成申请、seed `20261001`，
发现、验证、观察三个分区各 6,000 行。生成时使用的源码、实际依赖、快照报告标识和文件哈希记录在
[manifest](https://github.com/leeesq/mars-risk/blob/aa43995b2d670aeafdfb9536e24be4f171941032/docs/assets/cases/manifest.json)和[环境记录](https://github.com/leeesq/mars-risk/blob/aa43995b2d670aeafdfb9536e24be4f171941032/docs/assets/cases/generation-environment.json)。
报告标识随生成变化，本次验收使用这批材料中的标识，没有复用旧截图中的标识。

[主分原生图](https://github.com/leeesq/mars-risk/blob/aa43995b2d670aeafdfb9536e24be4f171941032/docs/assets/cases/binning-native-main-score.svg)直接调用
`save_risk_trend_images()` 导出，PNG 为 300 dpi；没有重新绘制曲线或裁剪原布局。
图中展示三个样本分区与 Total，日期取各分区实际范围。九个模拟月份没有被用作风险分组。
九个模拟日期均为每月 15 日，日缺失表也只展示这些实际日期。
[图表证据](https://github.com/leeesq/mars-risk/blob/aa43995b2d670aeafdfb9536e24be4f171941032/docs/assets/cases/binning-native-evidence.json)保留同报告的 summary/detail 查询。
单独下载的[分箱 HTML](https://github.com/leeesq/mars-risk/blob/aa43995b2d670aeafdfb9536e24be4f171941032/docs/assets/cases/binning.html)内嵌原生图，无旁路图片依赖。

格子规则转换为 DSL 时，排除分段定义中的缺失值、特殊值、非法概率和非有限分数。
验收逐个核对 `sample_id`，确认正常箱与规则命中的样本相同。
[规则案例](../demos/rule-evidence.md)另外运行真实组合生成器，24 候选预算、独立验证、
单独生成报告，并记录因输入值超出适用范围而剔除的人数。这批规则与格子种子规则分别验收。

筛选恢复了质量、IV/Lift、PSI/RC 和相关性门槛，保留与淘汰理由均来自 selector 的实际结果。
模拟漂移字段单独标记，raw 与 WOE 相关性证据分别解释。
保存报告后另起进程加载快照、查询统计。报告查询不调用 LLM，本次验收没有宣称执行了自主 Agent 或取得真实业务收益。

## 当时的复现与检查

以下命令属于当时版本；当前命令见上面的迁移记录。

```bash
python docs/snippets/task_cases.py --case all --rows 18000 --seed 20261001 --output-dir docs/assets/cases
python scripts/check_case_assets.py
python -m pytest -q tests/test_task_cases.py tests/test_documentation.py -m "not docs_ml"
python -m mkdocs build --strict
python scripts/check_case_assets.py --site-dir site --skip-recompute
```

900 行、自定义 seed `20261002` 用于接口、成员集合、保存与独立加载查询的回归，
不能代替公开规模验收。ZIP 在没有 Git 的解压目录中实际生成报告，并成功加载查询；
未知源码提交标记 unknown/null，同时记录脚本和已安装模块指纹。
`previews.txt`、机器证据、Excel 与快照来自同批计算；校验会重算公开规模并重放查询。

## 当前验证与限制

当时的源码检查包括 Ruff、覆盖 157 个文件的 Mypy、pydoclint 和私有 docstring 检查；六条已有私有 docstring 建议仍保留。
七个页面新增的短代码示例全部实际执行通过；同目录重复保存时仍按原规则拒绝覆盖。
案例定向 17 项通过；交叉 HTML 与数值相关定向 128 项通过；分箱 HTML 定向 24 项通过。
最终文档、案例、外部规则消费者及两种 HTML 定向检查为 90 passed、2 deselected，
使用仓库内 `--basetemp .pytest-tmp-final` 复现 Docs CI 的临时目录配置；
公开 18,000 行重算、56 个文件与 8 个快照校验通过。sdist/wheel 构建与 Twine 检查通过。
严格 MkDocs 构建与构建产物校验通过；最终远端 CI 结果在 [PR #5](https://github.com/leeesq/mars-risk/pull/5) 中更新。

首次 PR CI 发现两个案例问题：未加权报告的查询请求了不存在的权重字段，
ZIP 的“仓库外”测试却使用了 CI 中位于 checkout 内的临时目录。
修复后，查询代码按报告的实际 `weights_col` 选择人数或权重，并在文字结论中说明分母；
加权报告缺少权重证据仍明确失败。ZIP 测试改用 checkout 的兄弟临时目录，
保留未知提交、指纹、独立生成、加载查询的全部断言。18,000 行加权快照的加载查询复核通过，
查询行与证据引用保持一致；计算报告与原生图无需重算，下载源码、任务材料及 ZIP 同步刷新。

当时首次运行完整基础测试套件时，公开产物尚未刷新。结果为 1,040 passed、4 skipped、7 deselected，
12 failed。四项属于尚未生成的新图片/下载材料及已修正的链接断言；
另外八项为未改动的 worker/optimal 相关测试，当时的执行容器中 PID namespace 与 `/proc` 不对应，影响了这些测试。
受控 Python 的 `os.getpid()` 与 `/proc/self/status` 的 Pid 不同，
`psutil.Process(Popen.pid)` 读取另一进程或返回不存在，导致 RSS、后代发现及 Loky 采样不可用。
限制线程数后，单个 optimal 测试可返回 passed，但后台 RSS 采样仍失败，因此环境限制仍然存在。
没有为此改动 worker 代码、删断言或放宽预算，完整基础检查依靠正常 Linux CI。

当时本地未安装 LightGBM/Optuna，未执行完整 optional_ml/docs_ml 检查。
云浏览器拒绝访问本地 HTTP 与 file 协议，因此这次验收没有完成本地 HTML 的真实浏览器交互；
导航行为由实际 Node JS 回归与生成后 HTML 集成检查覆盖。
验收实际打开了 GitHub 托管的中英文 README，并点击语言切换入口。两页主标题居中，
原生 PNG、动画 GIF、英文全称与六个徽章均加载成功，互链指向当前分支。
[中文桌面截图](../assets/case-validation/native-readme-zh-desktop.jpg)记录当时的原生主图与 README 布局。
既有旧手机/暗色记录不代表新版已验收。
原生 Excel 外观未验收，静态工作表与关键数值由自动校验核对。

先前版本的截图与验收材料保留在 Git 历史及已有 case-validation 目录，
不作为这次原生主图、慢变色 Logo 或新版 HTML 的验证证据。
