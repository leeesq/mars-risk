---
description: 0.0.28 发布收尾的依赖边界、安装产物、文档和报告浏览器验收。
---

# 0.0.28 发布收尾验收

2026-10-08 在 main `acc6c1f` 基础上整理发布内容，版本保持 0.0.28。
这次工作调整安装说明、PyPI 展示、更新说明和必要的依赖范围，没有创建 tag、GitHub Release 或上传 PyPI。
当时的后续发布要求是：选择包含本次收尾修改，且 CI 与 Docs 均成功的提交。

## 发布内容与依赖边界

- 中英文 README 从同一脚本生成；图片、语言切换、源码、许可证和贡献链接使用公开 HTTPS。
  保留原 Logo、英文全称、六个徽章、居中标题及原生高清分箱图。
- README 与网站优先给出 `mars-risk==0.0.28`；PyPI 尚未提供该版本时使用源码安装。
  不把仓库版本号写成已经完成发布；Release Notes 合并本版变化并补充升级清单。
- 新建 Python 3.12 环境首次安装解析到 Polars 2.0.0。文档/案例检查出现
  2 failed、58 passed、2 deselected、8 errors；案例错误均来自同一个生成夹具。
  实际原因是 Polars 2.0 不再隐式转换 `is_in` 的 Float64/List(Int64) 操作数，
  参见[官方迁移说明](https://docs.pola.rs/releases/upgrade/2/)。
- 本版将 Python 3.9～3.12 的依赖限定为 `polars>=1.33.1,<2`；Python 3.8 仍固定 1.8.2。
  不通过改变缺失、特殊值或筛选语义适配 2.0。最终实际验证版本为 Polars 1.44.2。
- 监控、建模（含 Pipeline）和评分卡继续暂停功能迭代；三个模块仍须直接修改调用代码来适配上游变化。

## 本地验证

独立 Python 3.12 环境，NumPy 2.4.6、Pandas 3.0.6、Polars 1.44.2。
计算检查限制 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`、`POLARS_MAX_THREADS=2`。

| 检查 | 实际结果 |
| --- | --- |
| 文档与七案例测试，排除 `docs_ml` | 68 passed、2 deselected |
| 公开案例按 18,000 行、seed 20261001 重算及站点下载检查 | 56 个文件、8 个快照、七个案例通过 |
| 严格 MkDocs | 通过 |
| Ruff、版本/tag 一致性、diff 检查 | 通过 |
| 单次构建 wheel/sdist、分发资源与 Twine | 通过 |
| wheel 内实际 METADATA | 版本和依赖正确，UTF-8 描述正文与 README 一致 |
| `readme_renderer.markdown.render` 清洗后长描述 | 9 张图含六徽章；Logo、英文全称、主图、标题和语言链接保留，外部资源全为 HTTPS |
| 同一 wheel 在另一 Python 3.12 环境安装，仓库外运行安装 smoke 与 `pip check` | 通过；MARS 从 site-packages 导入 |

本地未执行可选建模、Notebook 的两项 `docs_ml`。完整 Python 3.8～3.12 矩阵、可选建模和
最终源码文档构建结果，以当次提交的 [CI](https://github.com/leeesq/mars-risk/actions/workflows/test.yml)
与 [Docs](https://github.com/leeesq/mars-risk/actions/workflows/docs.yml) 为准。
依赖扫描记录了可选 HIGHS solver 的原生符号导入提示；上述检查及默认安装 smoke 正常通过，
不将其描述为已验收 HIGHS 后端。

## 当前报告的真实浏览器交互

验收时通过受支持的云端 Chrome 浏览器打开 GitHub Pages 当时托管的报告，桌面视口为 1363×936。
本次提交没有修改报告 HTML 和案例快照；下表记录的是 2026-10-08 当时公开产物的补充验收。

| 页面/操作 | 实际观察 |
| --- | --- |
| [分箱报告](../assets/cases/binning.html)：Business Metadata | 导航跳到独立页签，特征、来源、单位和业务上下文可见 |
| Missing By Day | 4 个特征、2026-01-15 至 2026-09-15 九个实际日期，非空统计 |
| Grouped Pivot | 默认 bad30 有统计；切换 late60 后仍有统计，页面均显示 17 rows shown |
| Charts 搜索 main_score | 从四张图筛到一张；内嵌原生图实际加载，尺寸 1726×614，已查看渲染截图 |
| [交叉报告](../assets/cases/score-cross.html)：选 X4/Y4 | 显示 `(−∞, 546.043]` / `(567.174, +∞)`，完整精度边界保留；Bad Rate 67.68% |
| 切换 Bad Rate、应用 `X >= 2 AND Y >= 3` | 命中 6/16 格；732 人、670 人已表现、238 人事件、加权 Bad Rate 35.93% |
| 复制证据 | 复制成功；实际 JSON 及报告身份可在 Python 新进程回放 |
| late60 / observation / 202607 | 规则命中 781 人，已表现为 0；状态 unobserved，Bad Rate 显示 `—` |

复制的证据来自报告 `cba02fb3-e2e3-44ae-8c72-32ab920fd80c` 的 `cells` 表，查询条件为
`bad30 / discovery / 202601 / x_bin=b0 / y_bin=b3`。Python 加载对应快照得到
124 人、115 人已表现、78 人事件，加权 Bad Rate 为 0.6767961979535294；
规则使用公共 `evaluate_score_policy()` 回放，正常与未表现结果均与浏览器一致。
人数和加权率分别核对，加权率不能用人数直接相除得到。

验收时，报告页面的控制台未出现 error/warn；浏览器扩展自身的 metadata 提示另行记录。
这次验收是 HTTPS 桌面交互，未重验 file:// 离线模式、手机视口、200% 缩放或全部浏览器夹具。
截图已在验收会话实际查看，但未能同步为仓库附件；不引用旧截图作为这次证据。
