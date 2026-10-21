---
description: 当前 Home Credit 2018 公开案例的输入、分区、复现命令与验收范围。
---

# Home Credit 当前案例复现

当前七个案例使用完整的 application train、application test 和官方字段说明。原始 CSV 仅保存在本地，公开报告和下载包保存统计结果。
日期生成方式、分区算法与字段角色见 [Home Credit 数据指南](../user-guide/home-credit-data.md)。

## 输入与分区

| 来源／分区 | 记录数 | 模拟日期 | 已观测 TARGET |
| --- | ---: | --- | ---: |
| application_train | 307,511 | 2026-01-01 至 2026-06-30 | 307,511 |
| application_test | 48,744 | 2026-07-01 至 2026-07-31 | 0 |
| 0train | 215,374 | 2026-01-01 至 2026-05-07 | 215,374 |
| 1val | 30,720 | 2026-05-08 至 2026-05-25 | 30,720 |
| 2test | 30,657 | 2026-05-26 至 2026-06-12 | 30,657 |
| 3oot | 30,760 | 2026-06-13 至 2026-06-30 | 30,760 |
| 4_oot_undue | 48,744 | 2026-07-01 至 2026-07-31 | 0 |

合并两张表，并新增 `APPLY_DATE` 和 `dataset_flag` 后，共有 356,255 行、124 列，其中候选特征 119 个。
`SK_ID_CURR`、`TARGET`、`APPLY_DATE`、`dataset_flag` 和金额字段 `AMT_CREDIT` 不进入候选列表。

准备脚本根据 ID 的稳定 SHA256 映射，为训练和测试数据生成模拟日期，seed 为 `20261001`；生成日期时不使用 TARGET。
脚本从最后一个日期向前，按完整自然日依次划出 OOT、Test 和 Val，每个分区的目标人数约为 30,000，其余样本归入 Train。

规则、分箱和参考分布均使用 `0train` 拟合。`1val` 可参与筛选，`2test` 和 `3oot` 只评估固定后的结果。
`4_oot_undue` 的标签全部为 null，可以计算人数、分布和 PSI；坏率、IV、KS、AUC 和 Lift 保留未观测状态。
这个演示分区来自标签未公开的竞赛测试集，不能证明真实 OOT 或真实贷款未到期样本的表现。

## 可复制命令

在仓库根目录执行，Python 使用已安装的 MARS 开发环境。Windows PowerShell 设置 `$env:PYTHONPATH='src'`；其他系统设置 `PYTHONPATH=src`。源码完整生成使用当前仓库，已发布 `mars-risk==0.0.28` 可消费下载中的 `.marsreport` 快照。

```bash
python docs/snippets/home_credit_data.py --data-dir examples/home-credit-2018/data/raw_data --summary-json output/home-credit-migration/preparation-summary.json
python docs/snippets/task_cases.py --case all --phase generate --data-dir examples/home-credit-2018/data/raw_data --output-dir .artifacts/home-credit-current
python scripts/check_case_assets.py --assets-dir .artifacts/home-credit-current --mode public
python scripts/check_case_assets.py --assets-dir docs/assets/cases --mode recompute --data-dir examples/home-credit-2018/data/raw_data
python -m zipfile -e docs/assets/cases/cases.zip .artifacts/home-credit-consume
python .artifacts/home-credit-consume/task_cases.py --case 6 --phase consume --output-dir .artifacts/home-credit-consume
python scripts/write_bilingual_readme.py
python -m mkdocs build --strict
python scripts/check_case_assets.py --mode public --site-dir site
```

请在 ZIP 的解压副本中加载快照、导出文件，避免改写网站资产，使文件哈希与清单不符。
公开检查不读取原始 CSV，只重放查询引用，并核对快照、JSON、HTML、Excel、原生 PNG/SVG、清单和 ZIP。

完整重算必须显式提供数据目录或三个 CSV 参数。检查脚本会核对输入指纹，在新进程中重新计算，并比较统计口径和结果。
如果缺少文件或重算失败，脚本会报错，不会改为只检查公开产物。

仓库外解压 `cases.zip` 后，同目录中的 `task_cases.py`、`home_credit_data.py` 和 `external_agent_rule_case.py` 可独立工作。先安装当前 MARS 源码环境，再执行：

```bash
python task_cases.py --case 6 --phase consume --output-dir .
python task_cases.py --case all --phase generate --train-csv /path/application_train.csv --test-csv /path/application_test.csv --columns-description-csv /path/HomeCredit_columns_description.csv --output-dir regenerated
```

## 报告结果与验收范围

原生主图使用字段 `EXT_SOURCE_2` 和原始标签 `TARGET`，完整保留五个分区及 Total。
300 dpi PNG 和 SVG 从同一批报告导出。Train、Val、Test、OOT 的 KS 分别为 21.1206、21.5852、21.8119、21.9017，均为百分制；未观测组 PSI 为 0.0067，KS 留空。

Stats 对 119 个候选特征粗筛后保留 26 个，本次显式跳过精筛。Linear 从指定的 11 个原始数值字段中保留 10 个；30 天和 60 天社交圈观察人数的 Spearman 相关系数为 0.9975。

4 个候选种子规则中保留 1 个；组合生成器生成 24 个候选，保留 1 个。
固定规则在 Test 中命中 1,829 条，其中 387 条的 TARGET=1，坏率为 21.1591%，Lift 为 2.7828。
在 OOT 中命中 1,757 条，其中 417 条的 TARGET=1，坏率为 23.7336%，Lift 为 2.9437。

本次迁移还修正了本地源码中的统计与导出行为。分箱 Total 中没有标签的行，不再因空聚合而生成 IV=0、AUC=1；分箱汇总表中属于同一指标的字段按字段名稳定排序。
Excel 中较长的业务语义 JSON 按 `key/part/parts/value` 分段保存，避免超过单元格长度上限而被截断。
规则评估保留未观测样本的分布计数，只使用已观测样本计算监督指标。
这些修复属于本次工作区修改，不计入原始 0.0.28 的发布变更。

下载 HTML 为宽规则表和相关矩阵提供局部滚动容器，并声明手机视口，防止表格撑宽整页。
显示调整不重新计算统计，也不改变快照的报告标识或文件字节。

下载文件 `generation-environment.json` 和 `generation-timings.json` 记录了完整生成环境与耗时；`preparation-summary.json` 记录输入指纹、实际分区和元数据。
耗时包含相应阶段的导出，本次没有测量峰值内存，也没有据此宣称性能提升倍数。
小型测试数据（fixture）只用于定向测试；生成正式公开产物时，如果原始数据缺失，流程会报错，不会回退到 fixture。

本次迁移的浏览器截图和文章写作检查保存在本地 `output/home-credit-migration/`，未放入网站、案例 ZIP 或软件发布包。
迁移任务保持软件版本 0.0.28，该次任务约定不自动提交或发布。
