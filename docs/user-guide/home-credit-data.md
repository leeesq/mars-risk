---
description: Home Credit 2018 两表输入、模拟日期、五分区、未公开标签和共享准备入口。
---

# Home Credit 输入与分区

当前案例使用 Home Credit Default Risk 2018 的 `application_train.csv`、
`application_test.csv` 和 `HomeCredit_columns_description.csv`。原始特征与训练表的
`TARGET` 是公开比赛数据；`APPLY_DATE` 是本例生成的模拟日期。
这次案例迁移发生在 0.0.28 发布之后，没有重发软件，也没有改变包版本。

## 获取本地输入

从 [Kaggle 官方数据页](https://www.kaggle.com/competitions/home-credit-default-risk/data)
获取三个文件。先查看[当前比赛规则](https://www.kaggle.com/competitions/home-credit-default-risk/rules)，
按官方页面提示登录并自行处理需要的规则确认；项目不会替你接受条款。
页面依赖 JavaScript，本次验证环境未能读取完整规则正文，因此不另行声明数据许可。

在仓库中可放到 `examples/home-credit-2018/data/raw_data/`，也可使用自己的目录。
仅需上述三个文件，不读取 bureau、previous_application、installments 等子表。
原始数据、清洗宽表、真实 ID、样本分区成员和逐行结果只留在本地。
项目 MIT 许可不替代原始数据的规则；公开案例包只含统计报告、字段解释和源码。

使用本次更新的仓库源码运行完整复算：

```bash
pip install -e .
python docs/snippets/home_credit_data.py --data-dir examples/home-credit-2018/data/raw_data
python docs/snippets/task_cases.py --case all --data-dir examples/home-credit-2018/data/raw_data --output-dir output/home-credit-cases
```

也可以分别指定三个路径：

```bash
python docs/snippets/home_credit_data.py --train-csv /my/data/application_train.csv --test-csv /my/data/application_test.csv --columns-description-csv /my/data/HomeCredit_columns_description.csv
```

完整生成器接受相同三个路径参数。缺文件、schema、规模或标签异常会明确失败，
不会改用合成数据。小型测试必须显式提供 fixture 输入，并关闭正式规模核对。
已发布的 `mars-risk==0.0.28` 可加载和查询公共快照；从 CSV 重新计算还需要本次源码中的
无标签规则评估修复。此修复没有通过本任务发布到 PyPI。

[共享准备脚本](../assets/cases/home_credit_data.py)随[案例下载包](../assets/cases/cases.zip)提供。
在仓库中从 `docs/snippets` 导入；下载包中可从脚本所在目录直接导入：

```python
from home_credit_data import prepare_home_credit_data

inputs = prepare_home_credit_data("my-local-home-credit")
df = inputs.df
fcols = inputs.fcols
benchmark_df = inputs.benchmark_df
```

## 原表、标签和字段角色

本地已核对训练表 **307,511 行、122 列**，比赛测试表 **48,744 行、121 列**。
测试表原本没有 `TARGET`。对齐原列后纵向合并是 **356,255 行、122 列**；
新增 `APPLY_DATE` 和 `dataset_flag` 后是 **356,255 行、124 列**。

准备过程先核对 ID 非空、表内唯一、两表互斥、列名不重复、除 `TARGET` 外字段集合相同。
训练标签必须为原始 0/1；测试表临时用 Int8 的 -1 区分来源。
日期与分区准备完成后将 -1 转为 null，进入 MARS 的标签只有 0、1、null。
训练标签不修改，不把比赛测试表补成好样本。

官方 `TARGET=1` 表示客户出现比赛定义的还款困难：贷款前 Y 期中至少一期逾期超过 X 天；
X/Y 没有在字段说明中公开。`TARGET=0` 表示其他情况，不能改写成 bad30、bad60
或自行补一个表现期限。无标签组的标签指标要读取报告的 `unobserved`、`not_computed`
等实际状态，不能把 null 当作零风险。

| 角色 | 字段或输入 | 本例口径 |
| --- | --- | --- |
| 特征 | `features=fcols` | 119 个原始候选字段，按实际模块支持范围分析 |
| 目标 | `target="TARGET"` | 训练表原始 0/1；比赛测试表 null |
| 分组 | `group_col="dataset_flag"` | 五个演示分区 |
| 时间 | `time_col="APPLY_DATE"` | 明确的 Date 类型、模拟日期 |
| 基准 | `benchmark_df=inputs.benchmark_df` | 仅 0train；用于参考分布与适用拟合 |
| 金额 | `amount_col="AMT_CREDIT"` | 贷款金额，原始金额单位，币种未公开 |
| 元数据 | `feature_metadata=inputs.feature_metadata` | 中文名、官方定义、来源、单位 |

这是角色映射；每个入口只传其实际支持的参数。默认不设置权重。
`fcols` 排除 ID、目标、日期、分组和单独承担金额角色的 `AMT_CREDIT`，不增加衍生特征。
20 个 `display_features` 用于有限图表；候选全集、模块支持子集和实际参与计算的字段在产物中分别记录。

## 模拟日期如何生成

准备版本为 `home-credit-2018-v1`，默认 seed 为 **20261001**。
把 `"{seed}:{SK_ID_CURR}"` 按 UTF-8 编码；ID 为没有前导零的十进制整数。
使用 SHA256 全部 32 字节按 big-endian 转整数，再对区间天数取模。
训练来源映射到 2026-01-01 至 2026-06-30（181 天），测试来源映射到
2026-07-01 至 2026-07-31（31 天），包含首尾。

0/1 的标签值、外部分、收入和分析结果均不参与区间内日期分配。
同一配置下打乱 CSV 行序、重新运行和新进程的结果相同。
这些日期不是真实申请、放款或表现日期，不能证明业务季节性、线上漂移或真实跨期泛化。
July 与前六个月同时存在原始数据来源差异，不能把它当作生产时间变化。

## 五组按完整自然日划分

对有标签样本按日期升序、ID 作为稳定次序排列，汇总每天人数。
从末日向前分别选择最接近 30,000 人的完整日期段，依次记为 `3oot`、`2test`、`1val`；
剩余归 `0train`。误差相同选择更早起始日，并保留前组至少一个非空自然日。
不拆同一天、不按标签分层凑人数。不足四个非空有标签自然日时明确失败。
原始比赛测试表全部归 `4_oot_undue`。

以下是本次全量执行的实际结果，人数不是预设整数目标。

| 分区 | 人数 | 日期范围 | 已观测标签 | TARGET=1 | 已观测 Bad Rate |
| --- | ---: | --- | ---: | ---: | ---: |
| 0train | 215,374 | 01-01～05-07 | 215,374 | 17,465 | 8.1091% |
| 1val | 30,720 | 05-08～05-25 | 30,720 | 2,549 | 8.2975% |
| 2test | 30,657 | 05-26～06-12 | 30,657 | 2,331 | 7.6035% |
| 3oot | 30,760 | 06-13～06-30 | 30,760 | 2,480 | 8.0624% |
| 4_oot_undue | 48,744 | 07-01～07-31 | 0 | 未观测 | 未观测 |

0train 用于拟合分箱、发现规则、开发筛选和 PSI 参考。1val 若参与筛选，不能再称为完全独立的最终评估。
2test、3oot 用于固定结果的独立复核，不用于反复调整候选门槛；3oot 是模拟时间外分区。
`2test` 是有标签训练来源的一个分区，与原始 `application_test.csv` 不同。
`4_oot_undue` 是指定的演示名称，只表示比赛标签未公开，不表示真实贷款未到期或承诺以后补标签。

缺失率、分布和适用 PSI 使用全量样本；是否纳入缺失/特殊箱按报告参数核对。
IV、KS、AUC、Bad Rate 等只用观测标签，实际分母和状态看报告表。
不要把所有表中的 `count` 都解释成全样本人数：规则的监督统计与分箱分布可能采用不同口径。

## 字段说明与复现边界

实际字段说明是 cp1252 编码，表名使用 `application_{train|test}.csv`。
共享入口只读取 application 定义，122 个原字段均有说明；相同重复定义合并，
冲突或缺少定义记录为未知。业务元数据保持 MARS 的四个公共字段；完整原定义与检查状态单独保存在准备摘要中。
外部来源分 1/2/3 只登记为公开说明中的归一化外部来源分，不猜测提供机构，也不称为本项目训练的模型。

`DAYS_EMPLOYED=365243` 在合并表中实际出现 **64,648 次**。原值保留，
在需要的分析中仅对该字段登记特殊值；官方字段说明未解释此值，不当作实际就业时长。
金额的单位使用原始公开金额单位，不推断人民币、损失或利润。

准备摘要保存三文件 SHA256、实际行列数、散列配置、切分边界、各组标签覆盖与字段范围，
不含真实 ID 或成员清单。原始 CSV 的指纹在准备前后复核，文件不修改。
有原始输入的本地复算验证完整计算链路；公开 CI 加载已提交快照，核对哈希、查询、状态与展示证据，
不下载 Kaggle 数据，也不把快照检查称为从 CSV 完整重算。
