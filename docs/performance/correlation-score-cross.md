# 相关性报告与模型分交叉：验证和性能记录

记录日期：2026-10-01。测量对象为 mars-risk 0.0.28 工作树中的本次未提交实现。完整原始记录在仓库 `benchmarks/results/correlation_score_cross_20261001.json`，可运行脚本为 `benchmarks/benchmark_correlation_score_cross.py`。

## 环境与测量方法

Windows 11；Python 3.12.13；NumPy 2.4.6；Pandas 3.0.3；Polars 1.42.0；SciPy 1.18.0；PyArrow 24.0.0。随机种子 20261001，Polars 线程数 4，OPENBLAS_NUM_THREADS=4。

```powershell
$env:PYTHONPATH = 'src'
$env:POLARS_MAX_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
python benchmarks/benchmark_correlation_score_cross.py
```

每阶段使用 perf_counter 计时，每 10ms 采样一次当前进程 RSS，并记录阶段起始 RSS、观察到的峰值及二者差值。这不是精确分配追踪，短暂峰值可能未被采到，内存分配器复用会影响增量；原始记录保留字节数。本记录为单次运行的测量值，不宣称稳定提速比例。

## Linear 矩阵计算次数

在同一份 200 行、12 特征的合成输入上，执行参考提交 `4b3c88f5af3145619a35028202356d702e83cfba` 中的原始 Linear 类和本次实现，通过拦截 Pandas DataFrame.corr 计数：原实现调用 **2 次**，本次实现调用 **1 次**；两者最终入选特征及顺序一致。这里统计特征间矩阵，不包含特征与 target 的关系计算。

## 相关性结果规模

每项使用 1,500 行合成数据，先测 NumPy 有符号相关矩阵计算，再测已有矩阵转换为完整候选报告；没有为了此基准进行监督分箱拟合。报告构造结束释放 selector 的密集缓存，查询、展示和保存使用 pairs 与实际对角信息。

### 500 特征 / 124,750 对

| 阶段 | 时间（秒） | 观察峰值 RSS（MiB） | RSS 增量（MiB） |
| --- | ---: | ---: | ---: |
| 有符号矩阵计算 | 0.0050 | 286.46 | 3.05 |
| 完整报告构造 | 0.0628 | 298.89 | 18.18 |
| 保存 | 0.0124 | 304.64 | 14.81 |
| 加载 | 0.0310 | 311.38 | 6.76 |
| 关联查询 20 条 | 0.0050 | 313.48 | 2.13 |
| 恢复 30×30 子矩阵 | 0.0262 | 313.93 | 0.47 |
| 30×30 矩阵 HTML 展示 | 0.0756 | 315.71 | 1.80 |

保存文件大小：1.86 MiB。

### 1000 特征 / 499,500 对

| 阶段 | 时间（秒） | 观察峰值 RSS（MiB） | RSS 增量（MiB） |
| --- | ---: | ---: | ---: |
| 有符号矩阵计算 | 0.0155 | 347.24 | 20.11 |
| 完整报告构造 | 0.2254 | 402.50 | 78.18 |
| 保存 | 0.0181 | 355.37 | 15.88 |
| 加载 | 0.0295 | 362.70 | 7.35 |
| 关联查询 20 条 | 0.0056 | 362.22 | 0.14 |
| 恢复 30×30 子矩阵 | 0.0294 | 362.23 | 0.04 |
| 30×30 矩阵 HTML 展示 | 0.0651 | 362.52 | 0.31 |

保存文件大小：7.34 MiB。

## 模型分交叉

两个分数、两个标签、TEST/OOT 实际分组，默认 5 段；联合聚合覆盖正常箱、缺失箱和非法值箱。宽表的额外列为 uint8 合成字段。拦截跨引擎转换确认两项测量都只转换 `x, y, bad, late, split` 五列。

总入口计时包含投影、分段拟合、赋箱、联合聚合、边际和报告构造。两个轴拟合及报告构造是该入口中的嵌套计时，包含采样器设置开销，不能把它们再加到总入口时间上。保存后删除原始 DataFrame 与分数数组，再执行加载、查询和规则回放。

### 200,000 行 / 55 输入列

| 阶段 | 时间（秒） | 观察峰值 RSS（MiB） | RSS 增量（MiB） |
| --- | ---: | ---: | ---: |
| 完整分析入口 | 0.1897 | 406.47 | 28.77 |
| X 轴拟合（嵌套） | 0.0034 | 382.09 | 0.13 |
| Y 轴拟合（嵌套） | 0.0038 | 382.09 | 0.02 |
| 报告构造（嵌套） | 0.0010 | 406.50 | 0.02 |
| 保存 | 0.0085 | 390.26 | 0.21 |
| 加载 | 0.0173 | 390.29 | 0.05 |
| 格子查询 20 条 | 0.0006 | 390.29 | 0.03 |
| 加载后规则回放 | 0.0122 | 390.56 | 0.29 |
| Notebook 矩阵 HTML | 0.0221 | 391.35 | 0.81 |

### 1,000,000 行 / 5 输入列

| 阶段 | 时间（秒） | 观察峰值 RSS（MiB） | RSS 增量（MiB） |
| --- | ---: | ---: | ---: |
| 完整分析入口 | 0.2924 | 585.59 | 146.93 |
| X 轴拟合（嵌套） | 0.0152 | 486.29 | 7.66 |
| Y 轴拟合（嵌套） | 0.0163 | 478.66 | 0.02 |
| 报告构造（嵌套） | 0.0010 | 585.61 | 0.02 |
| 保存 | 0.0080 | 552.97 | 0.16 |
| 加载 | 0.0180 | 552.97 | 0.03 |
| 格子查询 20 条 | 0.0006 | 552.97 | 0.02 |
| 加载后规则回放 | 0.0124 | 552.98 | 0.04 |
| Notebook 矩阵 HTML | 0.0199 | 553.00 | 0.04 |

## 实际验证

- Python 3.12 完整默认测试：604 passed、3 skipped、7 deselected；71.37 秒。
- Python 3.8 最小依赖环境完整默认测试：542 passed、10 skipped、7 deselected；61.28 秒。
- 默认测试命令：`python -m pytest -q -m "not docs_ml and not optional_ml" -p no:cacheprovider`。实际运行另指定了工作区内可写的 basetemp。
- 原有 Linear 相关性/VIF/stepwise 定向可选测试另行运行：1 passed。
- 新功能定向测试：18 passed，包含手工计数、独立 Wilson 预期、矩阵一次计算、实际 WOE 口径、生命周期、保存后操作、宽表投影及规则布尔参考。
- Ruff、Mypy（src 下 152 个文件）、pydoclint、私有 docstring 检查通过；私有检查仅报告 6 项历史长度提示。
- 严格 MkDocs 构建通过；仅在构建进程中将 Jupyter 缓存重定向至工作区内可写目录，未修改项目缓存配置。
- 合成完整示例在 Python 3.8 最小环境实际运行；Notebook 的 6 个代码单元实际执行。HTML/Excel 结构及转义测试、Node 脚本语法检查通过。

## 验证边界

本地 HTML 的 file:// 页面被浏览器工具安全策略拒绝，未完成真实浏览器点击验证；未绕过该策略。HTML 包含本地脚本、聚合结果和必要元数据，代码与结构测试覆盖标签/分组切换、格子详情及预计算规则结果展示，但不将这些检查称为浏览器交互验证。

未验证超过 1,000 特征、超过 1,000,000 行、超过 55 列的性能，也未在 Python 3.9–3.11 或其他操作系统重复本轮完整验证。相关性报告的完整关系存储仍为 O(p²)。规则回放精度限于完整固定分段；加权二项置信区间不支持，报告明确使用 unweighted 字段表示整数 bad/n 的 Wilson 区间。

本记录生成时 README 未修改，没有 commit、push、发布或部署；后续提交与推送按用户授权单独执行。

## 2026-10-02 Score Cross 接入复核

本轮复用上述报告体系，接入共享画像分箱器、安全分箱表达式和离线交互。没有重新测量上述性能数字，也没有将附件的合成统计写入计算代码。

- Windows 现有 `mars312` 环境，Python 3.12.13 / Polars 1.42.0：默认测试 **761 passed、3 skipped、7 deselected**，92.25 秒。命令为 `python -m pytest -q -p no:cacheprovider -m "not docs_ml and not optional_ml"`，另外指定独立 basetemp。设置 `PYTHONUTF8=1` 和 `PYTHONIOENCODING=utf-8`，避免 Windows GBK 父进程解码 UTF-8 子进程输出的环境冲突。
- Windows 现有 `mars38` 环境，Python 3.8.20 / Polars 1.8.2：五个 Score Cross 测试模块 **102 passed、2 skipped**；两项独立 Agent 测试按既有 Python >=3.10 支持范围跳过。未在此环境重新运行完整仓库测试。
- 新测试覆盖实际 native / optimal / lite_opt 分箱路径、非 TRAIN 参考集、监督目标、每轴一次拟合、右闭切点、单箱退化、保存定义无拟合复用，以及方向、特殊箱、权重、金额、多个实际 scope、Wilson 和旧快照回放。
- 表达式 Python/JavaScript 结果对照与导出脚本的 Node 语法检查通过；五维独立 Agent 查询、跨进程加载后 HTML/Excel 导出及 policy 回放通过。这些是计算、脚本和产物检查，不是实际浏览器交互检查。
- Ruff、Mypy（157 个源文件）、pydoclint、私有 docstring 检查通过；私有检查仍有 6 项历史长度提示。严格 MkDocs 构建、wheel/sdist 构建、分发内容校验和 Twine 元数据检查通过；没有发布包或升级依赖。
- 未运行 `docs_ml` / `optional_ml` 标记测试及其他 Python 版本的本地完整测试。

本地验收产物位于忽略目录 `output/score-cross-acceptance/`：手工计数的加权、多目标、多组、月份及特殊箱测试报告 `weighted-scopes.marsreport`，加载后导出的 `score-cross.html` / Excel，以及独立 policy 和证据 JSON。它是明确标识的测试夹具，不代表真实业务收益。

另从仓库已有 `output/agent-rule-case/score-cross.marsreport` 加载并重导出至 `output/score-cross-review-existing/`，没有重新生成输入数据；已有快照保留原来的模拟数据来源说明。两类 HTML 都来自实际 ScoreCrossReport / ReportSnapshot，而非样板网页的固定计数。

附件 `MARS-score-cross-prototype.html` 在当前 Windows 工作区存在、可读，内容标识 Visual Prototype 05；Library 版本 4 是独立编号。受支持浏览器工具实际尝试打开其 `file://` 地址后，被安全策略拒绝（仅允许 HTTP/HTTPS，且禁止绕过）。因此附件和最终 HTML 的真实浏览器打开、离线点击、剪贴板、控制台、1440/1280/390px 与 200% 缩放检查均为 **未运行**，没有生成或声称存在验收截图。HTML 的自包含资源和禁止外网连接 CSP 已由代码/产物测试检查；不能据此宣称完成浏览器离线验收。

<span id="2026-10-02-真实浏览器验收"></span>

## 2026-10-02 真实浏览器验收

本节是后续实际浏览器结果，不改写上文历史通过数和浏览器缺口。
开始时工作树干净；重新通过 GitHub 与 HTTPS fetch 核对：main 为
`b73491c80617c24767f289921cc7924f71c9737f`，[PR #2](https://github.com/leeesq/mars-risk/pull/2)
仍为未合并草稿，head 为 `35d2b6eab67124e84220b6ade5693ee7ddfb3ad1`。
本轮独立本地分支 `codex/browser-acceptance-20261002` 从该 head 继续。
Score Cross 修复提交为 **`8ae6e46314af23bdf1cb82543c13b378dd604989`**，
共享风险 HTML 工具条修复为 **`aac62f7b67343ffc6993733f293c1bfef738e7fa`**；
元数据表滚动修复为 **`ebe16eddf1d63635a92abf4f3c57b0c35ec98455`**，这是最终生产代码提交。
本节及浏览器入口在后续验证提交追加，生产代码不再变化。
没有回退 main、重做上轮计算修复、推送、合并、发布或部署。

### 环境、夹具与可重复入口

Windows 11（10.0.26200）、Python 3.12.13、Polars 1.42.0、Pandas 3.0.3、NumPy 2.4.6；
Playwright 1.63.0 驱动本机 **Chrome 154.0.8037.92 / Chromium**，headless、`zh-CN`。
Score Cross 直接 `file://` 打开，新进程重导出 HTML；主流程 1440×1000，
另检查 1024/768/390×1000、系统浅色/深色偏好。报告保持已有固定配色，没有增加主题开关。
200% 使用独立临时 Chrome profile，在浏览器设置中实际选择页面缩放 200%；
窗口宽 1440，CSS innerWidth 707、devicePixelRatio 2、visualViewport.scale 1。
这不是 CSS zoom、缩小 viewport 或手工调用页面 render。

入口位于 `tests/browser/fixtures.py`、`score_cross.py`、`representatives.py`、`docs.py`；
完整命令和 Chrome/Chromium 前提见本分支仓库根 `CONTRIBUTING.md` 的“离线报告的真实浏览器验收”；
本轮尚未推送，该新节不宣称已经出现在 main 或线上站点。
`browser` 是 Python >=3.10 的独立开发 extra，不进入核心 dependencies；不安装其他浏览器引擎，
没有新增前端框架、验收平台或 CI 发布流。本机有 Firefox 143.0，但本轮未配置其自动化驱动、未做跨引擎验收；
WebKit 未配置，也未为了浏览器矩阵下载多套引擎。

`fixtures.py --phase generate` 与 `--phase export` 分别在结束后的独立进程运行，PID 为 2748/14136。
后续纯加载/导出补验 PID 为 37292，manifest 保留 export history；7 份 Score Cross 身份不变。
代表报告的修复后原生 HTML 另由公共 profile_risk 等入口重新计算/导出，记录在 `representatives-refresh.log`，
没有手工修改产物 HTML 或注入 CSS 冒充公共导出。
夹具复用原 Score Cross 测试与文档 snippet 的公共入口，生成 7 份紧凑报告：
standard、weighted、no_labels、one_bin、empty、zero_overall、medium。
覆盖相反方向、3×4、双目标、不同组不同可用周期、真实零值/低样本/未表现/空格/零权重分母，
非均匀权重与金额缺失、Null/NaN/有限缺失码/自定义 ±inf 缺失/非法值/特殊箱/概率域外，
中英文长身份、组名及 `< & " </script>` 字面文本。one_bin 是 1×2；medium 是 8×10、8 个实际 scope，
只检查展示与连续操作，不作为容量或性能结论。
公共 API 允许有显式固定分段的空宽表，产生零 scope，已实际导出并验收控件空态。
未给显式切点的空输入自动拟合仍以 `ValueError` 拒绝，原错误与明确的参数建议记录在 manifest。

### 真实复现与修复

修改既有 `_score_cross_html.py` 的展示/状态处理、共享 `html_assets.py` 的工具栏布局，
以及 `_binning_html.py` 的元数据表容器；
统计算法、固定定义、快照与公共 API 不变。

| 缺陷与触发 | 修复及回归 |
| --- | --- |
| 390px 打开原 3×2 加权报告，主体 scrollWidth 为 451，matrix-card 宽 440；矩阵撑开页面 | 允许 grid/card 收缩，矩阵、表及图在各自容器滚动；同场景最终主体为 390。所有夹具四宽度及长文本检查通过 |
| 零 scope 报告仍能应用默认规则，显示 `Cannot read properties of undefined (reading 'find')`，并伪称上一有效规则；详情/KPI 留白 | 明确无范围/无格子与不可用指标，禁用规则、复制、scope、policy、指标控件，清空覆盖进度并隐藏空图；点击/键盘不制造统计 |
| 3×2 或单箱页面语法提示仍写 Y4/前两行 | 提示按实际正常箱数生成合法标签；默认输入与镜像示例均实际提交检查 |
| 切样本集重建按钮后 document.activeElement 变为 BODY | 焦点回到新选中组按钮，保留滚动位置；连续切组及键盘回归检查真实 activeElement |
| 原风险报告 390px 的五列工具条被 hero 裁切：Go 在 x604–649、来源筛选在 x910–952、Export 在 x949–1129，主体宽度却仍为 390 | 共享工具条按可用宽度换行，允许列收缩；已有搜索、Regex、跳转、来源选择和导出实际操作回归 |
| 原风险 Business Metadata 的长表宽度最高 1933px，却没有滚动容器，被外层 section 裁掉右侧 | 用已有 mars-table-scroll 包裹，保留转义并加可聚焦表身份；1440/390 下真实点击/ArrowRight 滚到右侧数据 |

视觉前后证据为 `initial-390.png` / `initial-after.png`；空态为 `empty-before.png` / 浏览器入口的 `browser/empty-390.png`；
风险工具条为 `representative-browser/risk-toolbar-before-390.png` / `risk-toolbar-after-390.png`。
元数据为同目录 `risk-metadata-before-390.png` / `risk-metadata-after-390.png`，均有 DOM 尺寸记录。
固定色阶、risk_rank、cutpoints 和样本/表现分母未改变；选格只改变详情与双向梯度。

### 数字、交互和离线证据

浏览器使用真实点击、select、fill、Enter、Tab/方向键及滚动；只读 DOM 获取实际值。
期望来自加载报告的公共 `get_score_cell`、`query_page`、`evaluate_score_policy`，不调用前端聚合函数充当期望。
覆盖 30 个实际 scope、268 次正常格点击（medium 每 scope 抽查 3 格）、主流程 204 次规则数字/状态对照，
另补 56 次能改变命中组合的优先级/括号对照与 60 次比较符对照，共 **320 次**；snippet 另有 2 个 scope，
以及 3,780 行嵌入 policy 业务数据。标准/加权的全部正常格均逐个对照，
其余特殊格保存在特殊箱表并按公共表核对；不是将 884 条全部格子重放误写成 884 次浏览器点击。

人数、已观测人数、坏人数、全量占比、坏率、Lift、Δ、Wilson、真实行列边际及双向图一致。
加权坏率按保存权重，人数保持整数；90% 未加权 Wilson 与 weighted CI unsupported 明示。
六种状态分别保留，有效零在图中保留为 0；整体零坏率时 Lift 不可用。
周期回退保持目标、组和固定分段；规则应用后换范围重算，未提交输入/选格/示例复制不改规则。
合法同轴整数/标签、大小写、优先级/括号、OR 去重、无命中、空格、未观测和零权重均对照 Python。
错误输入、长度/词元/深度边界保留上一有效规则并标 `aria-invalid=true`；恢复正常视图仅清规则。
指标切换保留选格；全报告色阶和相同数值颜色跨范围一致，不重新标色。
键盘边界、activeElement、aria-pressed、live region 和详情同步；折叠控件和已有说明均实际操作。

证据使用真实 report_id 与五维 target/group/period/x_bin/y_bin 查询，逐格回交 Python 返回当前记录。
保存前后全部表/元数据/schema/定义/状态及每份 14 条表达式结果相同，共 884 条公开单元格证据重放一致。
7 份 Excel 的实际表头与行数一致。另一个新进程实际执行：

```bash
python docs/snippets/correlation_and_score_cross.py --report <tmp>/weighted/report.marsreport --rule "X <= X2 AND Y <= Y3" --output <tmp>/snippet-reexport
```

复制覆盖 file 原生 Clipboard、明确拒绝、execCommand fallback 及 localhost 安全上下文，
成功路径实际读取剪贴板并比对文本；拒绝显示手动复制提示，不声称已复制、不崩溃。
不可用/拒绝路径由浏览器初始化脚本注入 API 失败，随后用真实按钮触发；成功路径没有替换复制 API。
离线报告 pageerror、console error、CSP 违规、requestfailed 和外部请求均为 0；
每份报告只有自身 file 加载，没有 CDN、字体、遥测或 API 请求。非本地请求阻断辅助检查。
HTML 标题/字段中的 `</script>` 为文字，没有新增执行节点；输入继续为受限 AST。

### 代表 HTML、文档与检查状态

画像/风险、规则快照和相关性快照通过 file 打开，抽查已有表选择/过滤/分页/排序与原风险报告搜索、
目标切换、导航；规则按真实 evaluation 粒度与业务元数据核对，没有按特征重复计数。
相关性 Styler 的数值与保存的同一计算矩阵一致，只检查已有静态展示。
金额使用公共保存表，没有另建前端算法。详见 `representative-browser/results.json`。

本地 MkDocs 1.6.1 / Material 9.7.7 严格构建后以临时 HTTP 预览；首页、相关指南、analysis API 页
在 1440×1000 / 390×844、两种现有主题及真实 200% 下通过；搜索、导航抽屉、首页页签、交叉链接和宽表滚动实际操作。
首页原 Logo、英文全称、六枚完整动态徽章与任务入口保留，没有修改 README、首页样式或导航。
文档无 pageerror/CSP/本地加载失败；四条 console 403（普通/缩放上下文各两条）来自既有 GitHub repo/release 统计 API，
动态 shields 徽章加载正常。它们是文档既有线上内容，不混算为离线报告外部依赖。

| 检查 | 本轮实际结果 |
| --- | --- |
| 五个 Score Cross 模块（prompt 的第一组） | 113 passed |
| query/portable/rule/correlation 四模块（第二组） | 91 passed |
| `python -m pytest -q tests -p no:cacheprovider -m "not docs_ml and not optional_ml"` | Score Cross 修复 `8ae6e46` 的 799 passed、3 skipped、7 deselected，281.94s；3 个 modeling 模块因缺 xgboost 在收集时跳过 |
| 共享 HTML 布局收口后 reporting_contracts/evaluator/profiler 三模块 | 133 passed；没有把之前的完整套件冒充 CSS 收口后的重新全跑 |
| 元数据表容器收口后 reporting_contracts/portable_reports 两模块 | 50 passed，使用最终生产源码 |
| 默认文档测试 | 41 passed、2 docs_ml deselected |
| Ruff 全范围、Mypy、公共/私有 docstring | 通过；Mypy 157 文件，私有检查保留 6 条历史提示 |
| 严格 MkDocs、真实 Chromium 三个验收入口、git diff --check | 通过 |
| 其他 Python 本地矩阵、Firefox/WebKit、docs_ml/optional_ml、打包/发布 | 未运行；本轮为展示修复，保留实际配置和前轮来源 |
| standard/large 与 worker 全版本容量矩阵 | 未重复，不宣称提速 |

最初沙箱网络安装/fetch 被拒，改为获准的网络调用后成功；没有绕过浏览器安全策略。
初次验收脚本暴露折叠表可见性/搜索等待及定位问题，修正入口后完整重跑；不记为产品缺陷或最终通过。
初次定向 pytest 的不可写旧缓存提示由 `-p no:cacheprovider` 避免，业务断言未删改。

实际定向命令如下；运行时另指定产物根下独立 `--basetemp` 并设置 Windows UTF-8 环境。

```bash
python -m pytest -q tests/test_score_cross_html.py tests/test_score_cross_portable_ui.py tests/test_score_cross_policy_expression.py tests/test_score_cross.py tests/test_score_cross_binning.py
python -m pytest -q -p no:cacheprovider tests/test_report_query.py tests/test_portable_reports.py tests/test_rule_portable_report.py tests/test_correlation_report.py
python -m pytest -q -p no:cacheprovider tests/test_reporting_contracts.py tests/test_evaluator.py tests/test_profiler.py
python -m pytest -q -p no:cacheprovider tests/test_reporting_contracts.py tests/test_portable_reports.py
python -m pytest -q -p no:cacheprovider -m "not docs_ml" tests/test_documentation.py
python -m ruff check src tests scripts docs/snippets
python -m mypy src/mars
pydoclint src/mars
python scripts/check_private_docstrings.py src/mars
python -m mkdocs build --strict --site-dir <tmp>/docs/site
git diff --check
```

开始时核对的 [PR CI](https://github.com/leeesq/mars-risk/actions/runs/36911712699) 与
[PR Docs](https://github.com/leeesq/mars-risk/actions/runs/36911712732) 均 success，属于旧 head `35d2b6e`。
本轮未推送，远端 CI/Docs **未运行**；本地预览通过不代表 GitHub Pages 部署。
无合并/部署授权，不检查或修改线上页面，本轮提交尚未部署；线上实际 SHA 本轮未核查。
Monitoring、Modeling/Pipeline、Scoring 未新增功能或兼容层，核心 Python 3.8–3.12 与 mars.agent 隔离不变。

### 产物索引

本机独立目录为 `D:\Desktop\credit-risk\browser-acceptance-20261002`，不提交 HTML、Excel、快照、截图或大日志。
`manifest.json`、`environment.json` 和 `fixtures-*.log` 记录来源与分进程闭环；
`browser/score-cross-browser.json` 记录浏览器动作、范围计数、网络/异常、色阶、特殊箱、剪贴板、缩放和截图位置。
记录保留运行起点 `35d2b6e` 与 dirty 状态，并验证受测模板归一化 SHA256 与 `8ae6e46` git blob
一致（`311a495fdb03f3c34f3cd93f1e54ba30446780fa4bccdbd750ef8878109ed131`）；
不把追加来源信息说成在新 HEAD 无条件重跑。补充剪贴板/状态/snippet/缩放检查实际在 `8ae6e46` 后执行。
`representative-browser/results.json`、`docs/results.json` 分别记录代表报告与站点。
Score Cross 截图位于 `browser/`，包含 `standard-1440.png`、`weighted-390-applied.png`、`weighted-error.png`、各 `state-*.png`、
`empty-390.png`、`score-cross-zoom200.png`；文档为 `docs/home-desktop-default.png` / `docs/home-desktop-slate.png`。
静态/基础/文档命令日志另行保存。没有把截图或目测流畅度当作性能或统计证据。
