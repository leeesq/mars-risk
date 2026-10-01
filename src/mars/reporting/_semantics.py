"""明确的指标定义；由表目录和实际配置补充口径。"""

from __future__ import annotations

# 定义来自 compute/binning、profiling/metrics；业务单位未登记时不能推断。
_DEFINITIONS: dict[str, tuple[str, str]] = {
    "missing": ("ratio", "缺失箱样本量/全样本量；画像包括 Null、NaN 和配置缺失值"),
    "zeros": ("ratio", "原始零值数/全样本数，非数值字段为 0"),
    "unique": ("ratio", "原始不同值数/全样本数，含缺失；overview 超百万行使用近似去重"),
    "mode": ("ratio", "原始最高频值数/全样本数，含缺失"),
    "pct": ("ratio", "箱样本量/分组全样本量，配置权重时为权重和之比"),
    "bad_rate": ("ratio", "bad/observed_count；未表现标签不计入分母"),
    "base_br": ("ratio", "RC 参考箱的 bad/observed_count"),
    "cum_bad_rate": ("ratio", "累计 bad/累计 observed_count"),
    "amt_bad_rate": ("ratio", "bad_amt/observed_amt；未表现金额不计入分母，分母非正时为空"),
    "ks": ("points_0_100", "按实际配置排序的分箱累计坏样本与好样本分布最大绝对差 ×100"),
    "ks_bin": ("points_0_100", "当前箱累计坏样本与好样本分布绝对差 ×100"),
    "auc": ("ratio", "按配置排序的分箱 ROC 梯形面积；报告归一到不小于 0.5"),
    "auc_bin": ("ratio", "分箱 ROC 梯形面积贡献；汇总行进行 AUC 归一"),
    "iv": ("dimensionless", "各箱 (bad_dist-good_dist)×WOE 之和，复用稳定化规则"),
    "iv_bin": ("dimensionless", "(bad_dist-good_dist)×WOE"),
    "woe": ("dimensionless", "稳定化的 ln(bad_dist/good_dist)"),
    "psi": ("dimensionless", "各箱 (actual-expected)×ln(actual/expected) 之和；参考及箱范围见参数"),
    "psi_bin": ("dimensionless", "当前箱 PSI 贡献；缺失箱和特殊箱由参数控制"),
    "lift": ("dimensionless", "箱坏率/总体坏率，复用稳定化规则"),
    "lift_min": ("dimensionless", "Total 正常箱 Lift 的最小值"),
    "lift_max": ("dimensionless", "Total 正常箱 Lift 的最大值"),
    "lift_amt": ("dimensionless", "箱金额坏率/总体金额坏率，复用稳定化规则"),
    "risk_corr": ("correlation", "正常箱坏率与 RC 参考坏率的 Spearman 相关系数，参考来源见参数"),
    "mono": ("correlation", "Total 正常箱索引与坏率的 Spearman 相关系数；单箱或坏率恒定时无法定义"),
    "count": ("count_or_weight", "全样本数；配置 weights_col 时为样本权重和"),
    "observed_count": ("count_or_weight", "已表现标签样本数或权重和；无标签时未计算"),
    "bad": ("count_or_weight", "已表现坏样本数或权重和"),
    "good": ("count_or_weight", "observed_count-bad"),
    "total_count": ("count_or_weight", "分组全样本数或权重和"),
    "cum_count": ("count_or_weight", "按分箱索引展示顺序的累计全样本量或权重和"),
    "cum_observed_count": ("count_or_weight", "按分箱索引展示顺序的累计已表现样本量或权重和"),
    "cum_bad": ("count_or_weight", "按分箱索引展示顺序的累计坏样本量或权重和"),
    "cum_good": ("count_or_weight", "按分箱索引展示顺序的累计好样本量或权重和"),
    "bad_dist": ("ratio", "箱坏样本量/(分组坏样本总量+epsilon)"),
    "good_dist": ("ratio", "箱好样本量/(分组好样本总量+epsilon)"),
    "cum_bad_dist": ("ratio", "按实际有序指标配置累计的坏样本分布"),
    "cum_good_dist": ("ratio", "按实际有序指标配置累计的好样本分布"),
    "expected_count": ("count_or_weight", "PSI 参考箱的原始样本数或权重和"),
    "expected_pct": ("ratio", "PSI 参考箱样本分布；箱范围与稳定化规则见计算参数"),
    "actual_pct": ("ratio", "PSI 当前箱样本分布；箱范围与稳定化规则见计算参数"),
    "avg_amt": (
        "unknown_business_unit",
        "tot_amt/count；配置样本权重时 count 为权重和；业务币种未知",
    ),
    "observed_amt": ("unknown_business_unit", "已表现样本金金额和；业务币种未知"),
    "tot_amt": ("unknown_business_unit", "全样本金金额和；业务币种未知"),
    "good_amt": ("unknown_business_unit", "好样本金金额和；业务币种未知"),
    "bad_amt": ("unknown_business_unit", "坏样本金金额和；业务币种未知"),
    "unseen": ("ratio", "当前有效类别中不在基准有效类别的样本比例"),
    "unseen_rate": ("ratio", "当前有效类别中不在基准有效类别的样本比例"),
    "feature": ("identifier", "原输入特征名称"),
    "dtype": ("type", "原输入字段的数据类型"),
    "data_source": ("category", "调用方登记的特征来源；未映射字段可能标为 UNMAPPED"),
    "source": ("category", "已保存的参考来源，例如 total、first_group 或 benchmark_df"),
    "bin_index": ("index", "分箱规则索引；缺失和特殊箱为负，汇总行由 bin_type 标识"),
    "bin_index_max": ("index", "分组中的最大分箱索引，用于首尾箱标识"),
    "bin_label": ("category", "来自拟合分箱规则的箱标签"),
    "bin_type": ("category", "正常、缺失、特殊或汇总箱类型"),
    "trend": ("category", "Total 正常箱 WOE 形态；无标签时未计算"),
    "y": ("identifier", "目标列标识；真实目标及无标签状态见参数"),
    "target": ("identifier", "真实目标列标识"),
    "mode_value": ("original_value", "原始最高频取值的字符串表示，含缺失"),
    "distribution": ("display", "已有 sparkline 展示；抽样配置见参数，不用于精确统计"),
    "metric": ("identifier", "计算状态适用的指标或指标族"),
    "status": (
        "state",
        "computed/not_computed/unobserved/undefined/insufficient_samples/failed/skipped；结合 reason",
    ),
    "reason": ("state_reason", "计算失败、跳过、无法定义或尚未表现的具体原因"),
    "group": ("dimension", "状态所属分组；Total 为全量"),
}
for _metric, _meaning in {
    "mean": "非加权有效值均值",
    "median": "有效值中位数",
    "sum": "有效值和，空有效集沿用 0",
    "std": "有效值样本标准差（ddof=1）",
    "min": "有效值最小值",
    "max": "有效值最大值",
    "p25": "有效值 25% 分位数（nearest 插值）",
    "p75": "有效值 75% 分位数（nearest 插值）",
    "skew": "有效值偏度（bias=True）",
    "kurtosis": "有效值超额峰度（fisher=True, bias=True）",
}.items():
    _DEFINITIONS[_metric] = (
        "dimensionless" if _metric in {"skew", "kurtosis"} else "unknown_business_unit",
        f"{_meaning}；排除 Null、NaN、配置缺失值和特殊值；非数值字段未计算",
    )


def _definition(column: str) -> dict[str, str]:
    """映射显式登记的指标变体；其余字段明确标为未知。"""
    aliases = {
        "missing_rate": "missing",
        "zeros_rate": "zeros",
        "unique_rate": "unique",
        "mode_rate": "mode",
        "rc": "risk_corr",
        "rc_min": "risk_corr",
        "psi_max": "psi",
        "missing_min": "missing",
        "missing_max": "missing",
    }
    name = aliases.get(column, column)
    unit, meaning = _DEFINITIONS.get(name, ("unknown", "描述字段或未登记指标；含义未知"))
    if name != column and column.endswith("_max"):
        meaning += "；跨有效分组最大值"
    elif name != column and column.endswith("_min"):
        meaning += "；跨有效分组最小值"
    return {"unit": unit, "meaning": meaning}
