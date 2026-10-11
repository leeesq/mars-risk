---
description: 当前 Home Credit 2018 公开案例的输入、分区、复现命令与验收范围。
---

# Home Credit 当前案例复现

当前七个案例使用完整 application train、test 和官方字段说明。原始 CSV 只放本地；公开报告和下载包保存统计结果。日期、分区算法与字段角色详见 [Home Credit 数据指南](../user-guide/home-credit-data.md)。

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

两表对齐后为 356,255 × 124，候选特征 119 个。`SK_ID_CURR`、`TARGET`、`APPLY_DATE`、`dataset_flag` 与金额角色 `AMT_CREDIT` 不进入候选列表。训练与测试来源通过 ID 的稳定 SHA256 映射生成日期，seed 为 `20261001`，不会用 TARGET 决定日期。按完整自然日向后依次划出 OOT、Test、Val，目标人数分别约 30,000，余下为 Train。

规则、分箱和参考分布拟合使用 `0train`。`1val` 可参与筛选，`2test`、`3oot` 仅评估固定结果。`4_oot_undue` 全部标签为 null；人数、分布与 PSI 可计算，坏率、IV、KS、AUC 和 Lift 保留未观测状态。这是竞赛标签未公开的演示分区，不能证明真实 OOT 或真实贷款未到期表现。

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

快照消费在 ZIP 的解压副本中执行，避免重导出文件改写网站资产的清单哈希。公开检查不读取原 CSV，重放查询引用，核对快照、JSON、HTML、Excel、原生 PNG/SVG、清单和 ZIP。完整重算必须显式提供数据目录或三个 CSV 参数，核对输入指纹，在新进程重新计算并比较统计语义；缺文件和重算失败不会降级成公开检查。

仓库外解压 `cases.zip` 后，同目录中的 `task_cases.py`、`home_credit_data.py` 和 `external_agent_rule_case.py` 可独立工作。先安装当前 MARS 源码环境，再执行：

```bash
python task_cases.py --case 6 --phase consume --output-dir .
python task_cases.py --case all --phase generate --train-csv /path/application_train.csv --test-csv /path/application_test.csv --columns-description-csv /path/HomeCredit_columns_description.csv --output-dir regenerated
```

## 报告结果与验收范围

原生主图字段为 `EXT_SOURCE_2`，目标为原始 `TARGET`，完整保留五分区与 Total，300 dpi PNG 和 SVG 来自同批报告。Train、Val、Test、OOT 的 KS 为 21.1206、21.5852、21.8119、21.9017，单位为百分制；未观测组 PSI 为 0.0067，KS 留空。

Stats 对 119 个候选粗筛后保留 26 个，显式跳过精筛。Linear 在指定的 11 个原始数值字段中保留 10 个；30/60 天社交圈观察人数 Spearman 相关系数为 0.9975。种子规则 4 个候选保留 1 个，组合生成器 24 个候选保留 1 个。固定规则在 Test 命中 1,829 条，其中 TARGET=1 为 387 条，坏率 21.1591%、Lift 2.7828；OOT 命中 1,757 条，其中 TARGET=1 为 417 条，坏率 23.7336%、Lift 2.9437。

本次迁移同时修正本地源码中的交付边界：分箱 Total 无标签行不再由空聚合生成 IV=0、AUC=1；分箱汇总表同指标字段按字段名稳定排序；Excel 长业务语义 JSON 按 `key/part/parts/value` 分段保存，避免 Excel 单格长度截断。规则评估保留未观测样本的分布计数，并将监督指标限制在已观测样本。这些是本次工作区修复，不反写为原始 0.0.28 发布变更。

下载 HTML 的宽规则表与相关矩阵使用局部滚动容器，并声明手机视口，避免整页被表格撑宽。显示调整不重算统计，也不改变快照身份或字节。

完整生成环境和耗时在下载文件 `generation-environment.json`、`generation-timings.json` 中；输入指纹、实际分区和元数据在 `preparation-summary.json` 中。耗时包含相应阶段导出，未测峰值内存，也没有据此宣称性能倍数提升。小 fixture 仅用于定向测试，正式公开生成拒绝 fixture 回退。

浏览器截图和文章写作检查保存在本地 `output/home-credit-migration/`，不会进入网站、案例 ZIP 或软件发布包。软件版本保持 0.0.28，本任务不自动提交或发布。
