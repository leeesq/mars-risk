---
description: 2026-10-02 七个真实案例与双语 README 的来源、执行、下载、浏览器验收和限制。
---

# 七案例与双语 README 验收

本轮从远端 `main` 的 `746b8fa76439e6841b466bd590c35249a975ab56` 开始，
工作树干净，专用分支为 `codex/task-cases-bilingual-readme`。没有修改核心分析引擎，
没有合并、包发布或额外部署。2026-10-02 实时 PyPI 为 `0.0.27`，源码为 `0.0.28`。
既有 Docs 工作流只对 main push 部署；任务分支的源码和本地预览与当前线上站点分别说明。

## 同源证据与产物

[七个任务](../demos/index.md)、[能力与格式矩阵](../demos/index.md#formats)、
[共享文件来源](../assets/cases/manifest.json)和[下载包](../assets/cases/cases.zip)来自同批公开计算。
生成 seed=`20261001`、18,000 行；发现、验证、观察各 6,000 行。
观察期 `late60` 全部未表现，`bad30` 仍有 5,515 个有效标签，不能把两个目标合成一个分母。

案例 4/6/7 的交叉快照身份为 `066dccc0-b803-4f2a-8f3c-2ace86cf0ada`。
真实预览对应 discovery / 202601 / bad30 的 `b1/b3`（X3/Y4）：122 人、117 已表现人，
已表现权重 119.31605708882033、坏权重 67.67070266830763，
加权坏率 56.71550361233672%，Δ 行基线 +37.51357433077731 pp，整体 Lift 3.549613382315548。
完整行、精确查询和身份见[截图证据](../assets/cases/preview-evidence.json)。

规则实际有 5 个候选、2 个保留，拒绝阶段和原因来自审计；Rule 仍为 Experimental。
人工回答是依据确定性证据整理的示例，没有调用真实 LLM 或付费服务。
保存后结束计算进程，再由另一个进程只加载 ReportSnapshot；不访问原宽表。
分页、空结果、无效列请求与聚合 policy 回放均有实际记录。
最终上下文为 8,929 Unicode 字符／9,000 字符上限，40 行请求被裁剪为 30 行，
后续应使用裁剪后的 offset=30。[案例 6](../demos/saved-reports.md)提供实际查询。

## 执行环境与命令

| 用途 | 实际环境 |
| --- | --- |
| 公开生成 | Windows；Python 3.11.15；源码 `mars.__version__=0.0.28` |
| 生成依赖 | NumPy 2.4.6、Pandas 3.0.3、Polars 1.42.0、scikit-learn 1.9.0、PyArrow 24.0.0、openpyxl 3.1.5、XlsxWriter 3.2.9 |
| 本地测试／构建／浏览器 | Python 3.12.13；MkDocs 1.6.1、Material 9.7.7、Playwright 1.63.0、Chrome/Chromium 154.0.8037.95 |
| 轻量旧解释器回归 | Python 3.8.20、NumPy 1.24.4、Pandas 2.0.3、Polars 1.8.2 |

公开生成环境通过 checkout 的 `src/mars` 执行，环境残留安装元数据为 `0.0.19`，
没有将其冒充实际导入源码版本。两种版本及真实生成依赖分别记录在
[generation-environment.json](../assets/cases/generation-environment.json)。
更新截图后的 finalize 保留原生成环境；不以运行 finalize 的解释器替代生成记录。

以下从已安装当前源码的仓库根目录运行。新进程消费与生成分开，构建只消费已验证静态资产：

```bash
python docs/snippets/task_cases.py --case all --rows 18000 --seed 20261001 --output-dir docs/assets/cases
python docs/snippets/task_cases.py --case 4 --rows 18000 --seed 20261001 --output-dir output/task-cases
python docs/snippets/task_cases.py --phase consume --output-dir output/task-cases
python scripts/check_case_assets.py
python -m pytest -q tests/test_task_cases.py tests/test_documentation.py --basetemp .pytest-tmp-cases
python -m mkdocs build --strict
python scripts/check_case_assets.py --site-dir site
```

case 6/7 需要前面已有的交叉与 policy 快照；不会重复拟合宽表。
`--phase finalize` 只从冻结结果生成预览表、manifest 与 ZIP。
900 行轻量夹具只检验 API／语义；公开 checker 另重算实际 18,000 行参数并比较统计和查询。
合理排除随机报告身份、UTC 时间与实测耗时；同批身份仍逐一重放验证。
`previews.txt` 是七页与七卡唯一首屏表来源，CI 比较真实表内容，不能只更新图片文件名。

## 已执行检查

| 检查 | 结果与范围 |
| --- | --- |
| 受影响基线 | 131 passed、2 deselected，退出 0；文档、外部规则、查询及快照 |
| 本轮定向回归 | 207 passed、2 skipped，退出 0；两语短代码、七例、公共证据与规则 |
| 最终文档与七例复核 | 56 passed、2 skipped，退出 0；含两语新进程查询与当前静态 Excel 实际值验证；严格构建与公开规模重算分别通过 |
| 900 行七例 | Python 3.12 与 3.8 均通过；标签人数、权重、金额、状态、raw/WOE、独立加载、预算 |
| 静态检查 | Ruff 通过；Mypy 157 源文件通过；pydoclint 通过；私有 docstring 检查通过 |
| 文件 | 7 个快照 load 与查询重放；Excel 实际工作表和关键值对齐；UTF-8 TXT/源码、有限 JSON、ZIP 与 manifest 校验 |
| 外部链接 | 仓库、PyPI、Pages 首页／Quickstart／外部 Agent／稳定性、GitHub markup／写作指南、PePy 均 HTTP 200 |

第一次基线因既有 Temp/.pytest_cache ACL 返回权限错误，改用明确可写的 basetemp 后通过，
未删除断言。第一次文档构建同样受既有 Notebook 缓存 ACL 影响；在授权环境重新构建。
既有 protobuf、Pandas 未来弃用、Excel 旧模板扩展及私有 docstring 建议单独记录，不属于新增计算失败。

## 本轮真实浏览器

浏览器使用真实 headless Chromium 引擎，语言 zh-CN。文档由严格构建后的本地 HTTP 提供；
离线报告通过 `file://` 打开，非本地网络请求被阻断。没有使用 jsdom 或静态扫描代替视觉验收。

首页、索引和七页在 1440／390px 检查亮暗主题、主体溢出、实际页签、代码复制、折叠、
键盘焦点和文件下载。交叉报告补 768px；200% 使用 Chrome 设置页真实缩放。
选格、target/group/period、正常箱规则应用／错误／恢复、已保存 policy 与证据复制逐项对公共 Python 查询。
自包含交叉 HTML 没有外部请求；它自身使用固定浅色配色，和文档站主题分别说明。

另运行现有七个边界夹具，包含 empty、low_sample、unobserved、invalid_denominator、
not_requested、valid 及有效零值；不为凑状态改公开数据。
7 个夹具完成生成进程结束后的独立导出，目录、schema、数值、身份保持。
真实 cells 中有 empty 1,118、low_sample 380、valid 115、unobserved 40、
invalid_denominator 3、not_requested 168；5 个夹具覆盖有效零值。
1440／1024／768／390px、200%、原生剪贴板及拒绝/fallback 都通过，
pageerror、console error、CSP、失败和外部请求均为零。

手机原来缩放桌面预览导致数字难读，已改为同源真实选格与 JSON 手机图。
[桌面产品预览](../assets/cases/readme-preview.png)与[手机产品预览](../assets/cases/readme-preview-mobile.png)
分别约 197／66 KiB，完整数字以 JSON 为准。

代表性本轮截图与[浏览器行为记录](../assets/case-validation/browser-evidence.json)已保存：
[桌面交叉](../assets/case-validation/cross-desktop.png)、
[暗色索引](../assets/case-validation/index-desktop-dark.png)、
[手机规则交互](../assets/case-validation/cross-rule-phone.png)。
手机预览的[修改前](../assets/case-validation/before-score-cross-phone.png)与
[修改后](../assets/case-validation/after-score-cross-phone.png)保留真实对照；
[原桌面索引](../assets/case-validation/before-index-desktop.png)记录原巨大预览问题。
公开记录将本机 file URL 改为静态相对路径，移除私有绝对路径，不改变检查结果。

本地文档浏览器环境中的六个远程动态徽章图片未全部加载，链接与原动态来源已完整核对；
徽章远程图片是否可显示另以真正 GitHub 渲染核验，不使用手工状态图片替代。

浏览器可复现命令（`<output>` 为独立可写验收目录；浏览器不属于核心安装依赖）：

```bash
python tests/browser/task_cases.py --phase preview --assets docs/assets/cases --output <output>
python tests/browser/task_cases.py --phase verify --assets docs/assets/cases --site-dir site --output <output>
python tests/browser/fixtures.py --phase generate --output <output>/boundaries
python tests/browser/fixtures.py --phase export --output <output>/boundaries
python tests/browser/score_cross.py --manifest <output>/boundaries/manifest.json --output <output>/boundaries/browser --channel chrome
python tests/browser/readme.py --ref codex/task-cases-bilingual-readme --output <output>/github
```

## 实际边界与线上状态

本地缺少 LightGBM/Optuna，因此两项完整建模／Notebook 测试跳过；历史 Notebook 的首个数据 cell
已实际执行得到 240 行，完整训练没有伪报通过。Docs CI 使用既有可选依赖环境执行这些测试。
本轮静态 Excel 当前值已读取核对；原生 Excel 外观未完成验收。
该交付不使用旧透视缓存冒充新计算，也不承诺 Excel 与 HTML 交互等价。

GitHub README 真正托管页面和相关 CI 在分支推送后另外验收。
任务分支普通 push 不触发 Pages 部署；新站点页面仅在源码／本地严格构建预览中，
线上既有文档继续可访问。没有自动 merge、Release、PyPI 发布或额外生产部署。
历史验收记录保留在[2026-10-02 原记录](validation-20261002.md)，不代替本轮检查。
