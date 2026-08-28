"""规则定义、筛选配置、结果对象与 RuleSet artifact。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence, Tuple, Union

import polars as pl

from mars import __version__ as mars_version
from mars.compute import FrameLike, to_polars_frame
from mars.rule._dsl import (
    expression_complexity,
    expression_features,
    expression_has_missing,
    expression_to_polars,
    expression_to_sql,
    parse_expression,
)
from mars.rule.exceptions import MarsRuleArtifactError, MarsRuleDeploymentError

RuleDirection = Literal["high_risk", "low_risk"]
RuleComparison = Literal["<", "<=", "==", "!=", ">=", ">"]
RuleProfile = Literal["explore", "production"]
RuleQualification = Literal["exploratory", "validated", "temporally_validated"]


@dataclass(frozen=True)
class MarsRule:
    """表示与样本评估结果无关的可部署规则。

    Parameters
    ----------
    expression : str
        Mars Rule DSL 表达式。
    source : str
        规则来源，例如 ``manual``、``combination`` 或 ``tree``。
    labels : tuple[str, ...]
        调用方附加的稳定标签。

    Attributes
    ----------
    rule_id : str
        由规范化表达式确定性生成的规则标识。
    complexity : int
        原子条件数量。
    required_features : tuple[str, ...]
        规则引用的输入列。

    Examples
    --------
    >>> rule = MarsRule("age <= 25 AND debt > 0.5")
    >>> rule.rule_id.startswith("mr_")
    True
    """

    expression: str
    source: str = "manual"
    labels: Tuple[str, ...] = ()
    rule_id: str = field(init=False)
    complexity: int = field(init=False)
    required_features: Tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        """规范化表达式并计算稳定派生字段。"""
        ast = parse_expression(self.expression)
        normalized: str = expression_to_sql(ast, missing_policy="dsl")
        digest: str = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        object.__setattr__(self, "expression", normalized)
        object.__setattr__(self, "rule_id", f"mr_{digest}")
        object.__setattr__(self, "complexity", expression_complexity(ast))
        object.__setattr__(self, "required_features", tuple(sorted(expression_features(ast))))
        object.__setattr__(self, "labels", tuple(str(label) for label in self.labels))


@dataclass(frozen=True)
class MarsRuleMetricCondition:
    """定义单个规则评估指标条件。

    Parameters
    ----------
    metric : str
        固定长表中的指标列名。
    operator : {"<", "<=", "==", "!=", ">=", ">"}
        比较运算符。
    value : float
        比较阈值。
    """

    metric: str
    operator: RuleComparison
    value: float

    def __post_init__(self) -> None:
        """校验指标条件。"""
        if not isinstance(self.metric, str):
            raise TypeError("MarsRuleMetricCondition.metric 必须是字符串。")
        if not isinstance(self.value, (int, float)) or isinstance(self.value, bool):
            raise TypeError("MarsRuleMetricCondition.value 必须是数值。")
        allowed_metrics = {
            "sample_count",
            "event_count",
            "coverage",
            "event_rate",
            "lift",
            "amount_total",
            "event_amount",
            "amount_coverage",
            "amount_event_rate",
            "amount_lift",
            "customer_count",
            "event_customer_count",
            "customer_coverage",
            "customer_event_rate",
            "customer_lift",
            "event_rate_ci_lower",
            "event_rate_ci_upper",
            "lift_ci_lower",
            "lift_ci_upper",
            "p_value",
            "q_value",
        }
        if self.metric not in allowed_metrics:
            raise ValueError(f"不支持的规则指标：{self.metric!r}。")
        if self.operator not in {"<", "<=", "==", "!=", ">=", ">"}:
            raise ValueError(f"不支持的比较运算符：{self.operator!r}。")


@dataclass(frozen=True)
class MarsRuleFilter:
    """定义规则筛选阈值与跨目标、切片通过策略。

    Parameters
    ----------
    conditions : tuple[MarsRuleMetricCondition, ...]
        单行评估结果必须同时满足的指标条件。
    targets : "primary"、"all" 或目标列元组
        参与筛选的目标范围。
    target_scope : {"all", "any"}
        多目标条件的聚合方式。
    slice_pass_rate : float | None
        提供切片结果时要求通过的最小切片比例；``None`` 表示只使用 overall。
    """

    conditions: Tuple[MarsRuleMetricCondition, ...]
    targets: Union[Literal["primary", "all"], Tuple[str, ...]] = "primary"
    target_scope: Literal["all", "any"] = "all"
    slice_pass_rate: float | None = None

    def __post_init__(self) -> None:
        """校验聚合范围与通过率。"""
        if not self.conditions:
            raise ValueError("MarsRuleFilter.conditions 不能为空。")
        if any(not isinstance(condition, MarsRuleMetricCondition) for condition in self.conditions):
            raise TypeError("conditions 只接受 MarsRuleMetricCondition。")
        if self.targets not in ("primary", "all") and not (
            isinstance(self.targets, tuple)
            and bool(self.targets)
            and all(isinstance(target, str) and target for target in self.targets)
        ):
            raise TypeError("targets 必须是 'primary'、'all' 或非空目标名 tuple。")
        if self.target_scope not in {"all", "any"}:
            raise ValueError("target_scope 必须是 'all' 或 'any'。")
        if self.slice_pass_rate is not None and not 0 <= self.slice_pass_rate <= 1:
            raise ValueError("slice_pass_rate 必须位于 [0, 1]。")


def _high_candidate_filter() -> MarsRuleFilter:
    """构造默认高风险候选筛选器。"""
    return MarsRuleFilter(
        conditions=(
            MarsRuleMetricCondition("lift", ">=", 1.2),
            MarsRuleMetricCondition("event_count", ">=", 3.0),
            MarsRuleMetricCondition("coverage", ">=", 0.01),
            MarsRuleMetricCondition("coverage", "<=", 0.50),
        ),
    )


def _high_validation_filter() -> MarsRuleFilter:
    """构造默认高风险验证筛选器。"""
    return MarsRuleFilter(
        conditions=(
            MarsRuleMetricCondition("lift", ">=", 2.0),
            MarsRuleMetricCondition("event_count", ">=", 3.0),
            MarsRuleMetricCondition("coverage", ">=", 0.01),
            MarsRuleMetricCondition("coverage", "<=", 0.50),
        ),
        slice_pass_rate=0.8,
    )


@dataclass(frozen=True)
class MarsRuleMiningSpec:
    """规则挖掘的可复用、可序列化策略。

    Parameters
    ----------
    profile : {"explore", "production"}
        探索模式保留点估计筛选；生产模式强制独立验证与统计门禁。
    direction : {"high_risk", "low_risk"}
        规则风险方向。
    candidate_filter : MarsRuleFilter
        训练集候选筛选规则。
    validation_filter : MarsRuleFilter
        验证集最终筛选规则。
    grade_filters : Mapping[str, MarsRuleFilter]
        规则等级到筛选条件的映射。
    selection_strategy : {"ranked", "cascade"}
        最终规则选择策略。
    top_k : int
        最终最多保留的规则数。
    max_candidates : int
        合并生成器后允许评估的候选上限。
    iou_threshold : float
        命中人群 IoU 去重阈值。
    batch_size : int
        规则批量评估大小。
    iou_batch_size : int
        IoU 掩码处理批次大小。
    max_rounds : int
        cascade 最多轮数。
    random_state : int
        默认生成器随机种子。
    on_generator_error : {"raise", "record"}
        单生成器失败时的处理方式。
    confidence_level : float
        production Wilson 单侧置信水平。
    max_fdr : float
        production Benjamini-Hochberg 最大 q 值。
    min_time_slices : int
        获得时间验证资格所需的最少有效切片数。
    """

    profile: RuleProfile = "explore"
    direction: RuleDirection = "high_risk"
    candidate_filter: MarsRuleFilter = field(default_factory=_high_candidate_filter)
    validation_filter: MarsRuleFilter = field(default_factory=_high_validation_filter)
    grade_filters: Mapping[str, MarsRuleFilter] = field(default_factory=dict)
    selection_strategy: Literal["ranked", "cascade"] = "ranked"
    top_k: int = 10
    max_candidates: int = 100_000
    iou_threshold: float = 0.3
    batch_size: int = 100
    iou_batch_size: int = 512
    max_rounds: int = 10
    random_state: int = 42
    on_generator_error: Literal["raise", "record"] = "raise"
    confidence_level: float = 0.95
    max_fdr: float = 0.05
    min_time_slices: int = 3

    def __post_init__(self) -> None:
        """校验挖掘预算和策略。"""
        if not isinstance(self.candidate_filter, MarsRuleFilter):
            raise TypeError("candidate_filter 必须是 MarsRuleFilter。")
        if not isinstance(self.validation_filter, MarsRuleFilter):
            raise TypeError("validation_filter 必须是 MarsRuleFilter。")
        if any(not isinstance(value, MarsRuleFilter) for value in self.grade_filters.values()):
            raise TypeError("grade_filters 的值只接受 MarsRuleFilter。")
        if self.profile not in {"explore", "production"}:
            raise ValueError("profile 必须是 'explore' 或 'production'。")
        if self.direction not in {"high_risk", "low_risk"}:
            raise ValueError("direction 必须是 'high_risk' 或 'low_risk'。")
        if self.selection_strategy not in {"ranked", "cascade"}:
            raise ValueError("selection_strategy 必须是 'ranked' 或 'cascade'。")
        if self.top_k < 1 or self.max_candidates < 1 or self.batch_size < 1:
            raise ValueError("top_k、max_candidates 和 batch_size 必须至少为 1。")
        if self.iou_batch_size < 1 or self.max_rounds < 1:
            raise ValueError("iou_batch_size 和 max_rounds 必须至少为 1。")
        if not 0 <= self.iou_threshold <= 1:
            raise ValueError("iou_threshold 必须位于 [0, 1]。")
        if not 0.5 < self.confidence_level < 1:
            raise ValueError("confidence_level 必须位于 (0.5, 1)。")
        if not 0 < self.max_fdr < 1:
            raise ValueError("max_fdr 必须位于 (0, 1)。")
        if self.min_time_slices < 1:
            raise ValueError("min_time_slices 必须至少为 1。")

    @classmethod
    def explore(cls, **overrides: Any) -> MarsRuleMiningSpec:
        """构造允许 in-sample 结果的探索策略。

        Parameters
        ----------
        **overrides : Any
            覆盖默认 dataclass 字段的显式配置。

        Returns
        -------
        MarsRuleMiningSpec
            ``profile='explore'`` 的挖掘策略。
        """
        return cls(profile="explore", **overrides)

    @classmethod
    def production(cls, **overrides: Any) -> MarsRuleMiningSpec:
        """构造强制独立验证和统计门禁的生产策略。

        Parameters
        ----------
        **overrides : Any
            覆盖默认 dataclass 字段的显式配置。

        Returns
        -------
        MarsRuleMiningSpec
            ``profile='production'`` 的挖掘策略。
        """
        return cls(profile="production", **overrides)

    @classmethod
    def low_risk(cls, **overrides: Any) -> MarsRuleMiningSpec:
        """构造默认低风险挖掘策略。

        Parameters
        ----------
        **overrides : Any
            覆盖默认 dataclass 字段的显式配置。

        Returns
        -------
        MarsRuleMiningSpec
            Lift 上限分别为 0.9 和 0.8 的低风险策略。
        """
        candidate_filter = MarsRuleFilter(
            conditions=(
                MarsRuleMetricCondition("lift", "<=", 0.9),
                MarsRuleMetricCondition("coverage", ">=", 0.01),
                MarsRuleMetricCondition("coverage", "<=", 0.50),
            ),
        )
        validation_filter = MarsRuleFilter(
            conditions=(
                MarsRuleMetricCondition("lift", "<=", 0.8),
                MarsRuleMetricCondition("coverage", ">=", 0.01),
                MarsRuleMetricCondition("coverage", "<=", 0.50),
            ),
            slice_pass_rate=0.8,
        )
        values: Dict[str, Any] = {
            "direction": "low_risk",
            "candidate_filter": candidate_filter,
            "validation_filter": validation_filter,
        }
        values.update(overrides)
        return cls(**values)

    def to_dict(self) -> Dict[str, Any]:
        """返回可 JSON 序列化的完整策略。"""
        return asdict(self)


@dataclass
class MarsRuleSet:
    """有序、可部署的规则集合。

    Parameters
    ----------
    rules : Sequence[MarsRule]
        有序规则定义。
    grades : Mapping[str, Sequence[str]]
        等级到规则 ID 的映射。
    metadata : Mapping[str, Any]
        不包含样本指标的 artifact 元数据。
    qualification : {"exploratory", "validated", "temporally_validated"}
        规则集部署资格；手工构造和 explore 结果默认为 ``exploratory``。
    validation_summary : Mapping[str, Any]
        生成资格状态所依据的验证摘要。
    """

    rules: Sequence[MarsRule] = field(default_factory=tuple)
    grades: Mapping[str, Sequence[str]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    qualification: RuleQualification = "exploratory"
    validation_summary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """冻结集合顺序并校验 ID、表达式和等级引用。"""
        frozen_rules: Tuple[MarsRule, ...] = tuple(self.rules)
        if any(not isinstance(rule, MarsRule) for rule in frozen_rules):
            raise TypeError("MarsRuleSet.rules 只接受 MarsRule。")
        if self.qualification not in {
            "exploratory",
            "validated",
            "temporally_validated",
        }:
            raise MarsRuleArtifactError(f"未知规则部署资格：{self.qualification!r}。")
        if not isinstance(self.validation_summary, Mapping):
            raise MarsRuleArtifactError("validation_summary 必须是 mapping。")
        seen: Dict[str, str] = {}
        for rule in frozen_rules:
            existing: str | None = seen.get(rule.rule_id)
            if existing is not None and existing != rule.expression:
                raise MarsRuleArtifactError(f"规则 ID 哈希冲突：{rule.rule_id}。")
            if existing is not None:
                raise MarsRuleArtifactError(f"规则集包含重复 rule_id：{rule.rule_id}。")
            seen[rule.rule_id] = rule.expression
        frozen_grades: Dict[str, Tuple[str, ...]] = {
            str(grade): tuple(rule_ids) for grade, rule_ids in self.grades.items()
        }
        grade_columns: List[str] = [re_safe_column_fragment(grade) for grade in frozen_grades]
        if len(grade_columns) != len(set(grade_columns)):
            raise MarsRuleArtifactError("规则等级名称规范化后产生重复输出列。")
        unknown_ids: List[str] = sorted(
            {
                rule_id
                for rule_ids in frozen_grades.values()
                for rule_id in rule_ids
                if rule_id not in seen
            }
        )
        if unknown_ids:
            raise MarsRuleArtifactError(f"规则等级引用未知 rule_id：{unknown_ids}。")
        self.rules = frozen_rules
        self.grades = frozen_grades
        self.metadata = dict(self.metadata)
        self.validation_summary = dict(self.validation_summary)
        _validate_validation_summary(
            self.validation_summary,
            qualification=self.qualification,
            rule_count=len(frozen_rules),
        )

    @property
    def required_features(self) -> Tuple[str, ...]:
        """返回规则集引用的全部输入列。"""
        return tuple(sorted({feature for rule in self.rules for feature in rule.required_features}))

    def transform(self, df: FrameLike) -> FrameLike:
        """追加逐规则、总命中数和等级命中数。

        Parameters
        ----------
        df : FrameLike
            待应用规则的数据集。

        Returns
        -------
        pandas.DataFrame | polars.DataFrame
            与输入类型一致的命中结果表。

        Raises
        ------
        ValueError
            输入缺少任一规则引用列时抛出。
        """
        input_is_polars: bool = isinstance(df, pl.DataFrame)
        frame: pl.DataFrame = to_polars_frame(df)
        missing: List[str] = [feature for feature in self.required_features if feature not in frame.columns]
        if missing:
            raise ValueError(f"规则应用缺少必需列：{missing}。")
        hit_columns: List[str] = []
        expressions: List[pl.Expr] = []
        for rule in self.rules:
            column_name: str = f"rule__{rule.rule_id}"
            hit_columns.append(column_name)
            ast = parse_expression(rule.expression)
            hit_expr: pl.Expr = (
                expression_to_polars(ast, frame.schema)
                .fill_null(False)
                .cast(pl.Int8)
                .alias(column_name)
            )
            expressions.append(hit_expr)
        result: pl.DataFrame = frame.with_columns(expressions)
        hit_count: pl.Expr = (
            pl.sum_horizontal(hit_columns).cast(pl.Int32)
            if hit_columns
            else pl.lit(0, dtype=pl.Int32)
        )
        result = result.with_columns(hit_count.alias("rule_hit_count"))

        grade_expressions: List[pl.Expr] = []
        for grade, rule_ids in self.grades.items():
            grade_columns: List[str] = [f"rule__{rule_id}" for rule_id in rule_ids]
            safe_grade: str = re_safe_column_fragment(grade)
            if grade_columns:
                grade_expressions.append(
                    pl.sum_horizontal(grade_columns)
                    .cast(pl.Int32)
                    .alias(f"grade__{safe_grade}__hit_count")
                )
            else:
                grade_expressions.append(
                    pl.lit(0, dtype=pl.Int32).alias(f"grade__{safe_grade}__hit_count")
                )
        if grade_expressions:
            result = result.with_columns(grade_expressions)
        return result if input_is_polars else result.to_pandas()

    def to_dict(self) -> Dict[str, Any]:
        """返回严格、带版本的规则集 artifact。"""
        return {
            "artifact_type": "mars_rule_set",
            "schema_version": 1,
            "expression_version": 2,
            "mars_version": mars_version,
            "rules": [
                {
                    "rule_id": rule.rule_id,
                    "expression": rule.expression,
                    "source": rule.source,
                    "labels": list(rule.labels),
                }
                for rule in self.rules
            ],
            "grades": {grade: list(rule_ids) for grade, rule_ids in self.grades.items()},
            "metadata": dict(self.metadata),
            "qualification": self.qualification,
            "validation_summary": dict(self.validation_summary),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MarsRuleSet:
        """从严格 artifact 恢复规则集。

        Parameters
        ----------
        payload : Mapping[str, Any]
            由 :meth:`to_dict` 生成的规则集对象。

        Returns
        -------
        MarsRuleSet
            完成 ID 和引用校验的规则集。

        Raises
        ------
        MarsRuleArtifactError
            artifact 类型、版本、规则条目、ID 或等级引用非法时抛出。
        """
        required_keys = {
            "artifact_type",
            "schema_version",
            "expression_version",
            "mars_version",
            "rules",
            "grades",
            "metadata",
            "qualification",
            "validation_summary",
        }
        if set(payload) != required_keys:
            raise MarsRuleArtifactError(
                f"规则集 artifact 字段必须严格等于：{sorted(required_keys)}。"
            )
        if payload.get("artifact_type") != "mars_rule_set":
            raise MarsRuleArtifactError("artifact_type 必须是 'mars_rule_set'。")
        if payload.get("schema_version") != 1 or payload.get("expression_version") != 2:
            raise MarsRuleArtifactError("不支持的规则集 schema 或 expression 版本。")
        raw_rules: Any = payload.get("rules")
        raw_grades: Any = payload.get("grades", {})
        raw_metadata: Any = payload.get("metadata", {})
        raw_qualification: Any = payload.get("qualification")
        raw_validation_summary: Any = payload.get("validation_summary")
        if not isinstance(raw_rules, list) or not isinstance(raw_grades, dict):
            raise MarsRuleArtifactError("规则集 rules 必须是列表，grades 必须是对象。")
        if not isinstance(raw_metadata, dict):
            raise MarsRuleArtifactError("规则集 metadata 必须是对象。")
        if raw_qualification not in {
            "exploratory",
            "validated",
            "temporally_validated",
        }:
            raise MarsRuleArtifactError("规则集 qualification 非法。")
        if not isinstance(raw_validation_summary, dict):
            raise MarsRuleArtifactError("规则集 validation_summary 必须是对象。")
        if not isinstance(payload.get("mars_version"), str) or not payload["mars_version"]:
            raise MarsRuleArtifactError("mars_version 必须是非空字符串。")

        rules: List[MarsRule] = []
        for row in raw_rules:
            if not isinstance(row, dict):
                raise MarsRuleArtifactError("规则条目必须是对象。")
            if set(row) != {"rule_id", "expression", "source", "labels"}:
                raise MarsRuleArtifactError("规则条目字段必须严格匹配 schema_version=1。")
            if not isinstance(row.get("labels"), list):
                raise MarsRuleArtifactError("规则 labels 必须是字符串列表。")
            try:
                rule = MarsRule(
                    expression=str(row["expression"]),
                    source=str(row.get("source", "manual")),
                    labels=tuple(str(label) for label in row.get("labels", [])),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise MarsRuleArtifactError(f"非法规则条目：{row!r}。") from exc
            if row.get("rule_id") != rule.rule_id:
                raise MarsRuleArtifactError(
                    f"规则 ID 与表达式不匹配：{row.get('rule_id')!r}。"
                )
            rules.append(rule)
        grades: Dict[str, Tuple[str, ...]] = {
            str(grade): tuple(str(rule_id) for rule_id in rule_ids)
            for grade, rule_ids in raw_grades.items()
            if isinstance(rule_ids, list)
        }
        if len(grades) != len(raw_grades):
            raise MarsRuleArtifactError("每个等级值都必须是 rule_id 列表。")
        return cls(
            rules=rules,
            grades=grades,
            metadata=raw_metadata,
            qualification=raw_qualification,
            validation_summary=raw_validation_summary,
        )

    def save_json(self, path: Union[str, Path]) -> None:
        """原子写出规则集 JSON。

        Parameters
        ----------
        path : Union[str, Path]
            输出文件；父目录必须已经存在。

        Raises
        ------
        FileNotFoundError
            输出目录不存在时抛出。
        Exception
            临时文件写入、同步或原子替换失败时原样抛出。
        """
        output_path = Path(path)
        if not output_path.parent.exists():
            raise FileNotFoundError(f"规则集输出目录不存在：{output_path.parent}。")
        text: str = json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(output_path.parent),
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write(text)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, output_path)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> MarsRuleSet:
        """读取并严格校验规则集 JSON。

        Parameters
        ----------
        path : Union[str, Path]
            Mars RuleSet artifact 文件。

        Returns
        -------
        MarsRuleSet
            恢复后的规则集。

        Raises
        ------
        MarsRuleArtifactError
            JSON 语法、顶层结构或 artifact 契约非法时抛出。
        """
        input_path = Path(path)
        try:
            payload: Any = json.loads(input_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise MarsRuleArtifactError(f"规则集 JSON 无法解析：{input_path}。") from exc
        if not isinstance(payload, dict):
            raise MarsRuleArtifactError("规则集 JSON 顶层必须是对象。")
        return cls.from_dict(payload)

    def generate_sql(
        self,
        *,
        table_alias: str = "",
        include_grade_counts: bool = True,
        minimum_qualification: Literal["validated", "temporally_validated"] | None = "validated",
        missing_policy: Literal["reject", "normalized_to_null"] = "reject",
    ) -> str:
        """生成 ANSI SQL 命中列和汇总列片段。

        Parameters
        ----------
        table_alias : str
            输入表别名；空字符串表示不添加前缀。
        include_grade_counts : bool
            是否输出等级命中计数列。
        minimum_qualification : Literal["validated", "temporally_validated"] | None
            SQL 导出要求的最低部署资格；``None`` 仅用于显式开发导出。
        missing_policy : Literal["reject", "normalized_to_null"]
            ``IS MISSING`` 的 ANSI SQL 策略；默认拒绝，声明上游已完成 NaN 到
            NULL 规范化后才允许映射。

        Returns
        -------
        str
            可嵌入 ``SELECT`` 的逗号分隔 SQL 片段。

        Raises
        ------
        MarsRuleDeploymentError
            资格不足、策略非法或 MISSING 无法安全导出时抛出。
        """
        qualification_rank: Dict[str, int] = {
            "exploratory": 0,
            "validated": 1,
            "temporally_validated": 2,
        }
        if minimum_qualification not in {None, "validated", "temporally_validated"}:
            raise MarsRuleDeploymentError("minimum_qualification 非法。")
        if missing_policy not in {"reject", "normalized_to_null"}:
            raise MarsRuleDeploymentError("missing_policy 非法。")
        if (
            minimum_qualification is not None
            and qualification_rank[self.qualification]
            < qualification_rank[minimum_qualification]
        ):
            raise MarsRuleDeploymentError(
                f"规则集资格 {self.qualification!r} 低于 SQL 导出要求 "
                f"{minimum_qualification!r}。"
            )
        blocks: List[str] = []
        hit_expressions: Dict[str, str] = {}
        for rule in self.rules:
            alias: str = f"rule__{rule.rule_id}"
            ast = parse_expression(rule.expression)
            if expression_has_missing(ast) and missing_policy == "reject":
                raise MarsRuleDeploymentError(
                    f"规则 {rule.rule_id} 包含 IS MISSING，不能直接导出 ANSI SQL。"
                )
            condition: str = expression_to_sql(
                ast,
                table_alias,
                missing_policy=missing_policy,
            )
            hit_expression: str = f"CASE WHEN {condition} THEN 1 ELSE 0 END"
            hit_expressions[rule.rule_id] = hit_expression
            blocks.append(f'{hit_expression} AS "{alias}"')
        if not blocks:
            return ""
        hit_terms: str = " + ".join(hit_expressions.values())
        blocks.append(f"({hit_terms}) AS \"rule_hit_count\"")
        if include_grade_counts:
            for grade, rule_ids in self.grades.items():
                safe_grade: str = re_safe_column_fragment(grade)
                terms: List[str] = [hit_expressions[rule_id] for rule_id in rule_ids]
                expression: str = " + ".join(terms) if terms else "0"
                blocks.append(f'({expression}) AS "grade__{safe_grade}__hit_count"')
        return ",\n".join(blocks)


def _validate_validation_summary(
    summary: Mapping[str, Any],
    *,
    qualification: RuleQualification,
    rule_count: int,
) -> None:
    """严格校验资格状态所依赖的验证摘要。"""
    if not summary:
        if qualification != "exploratory":
            raise MarsRuleArtifactError("非 exploratory 规则集必须包含验证摘要。")
        return
    required_keys = {
        "profile",
        "qualification",
        "validation_status",
        "confidence_level",
        "max_fdr",
        "min_time_slices",
        "selected_count",
        "temporal_assessed",
        "minimum_time_slice_count",
        "minimum_time_slice_pass_rate",
    }
    if set(summary) != required_keys:
        raise MarsRuleArtifactError(
            f"validation_summary 字段必须严格等于：{sorted(required_keys)}。"
        )
    if summary["qualification"] != qualification:
        raise MarsRuleArtifactError("validation_summary 与规则集 qualification 不一致。")
    if type(summary["selected_count"]) is not int or summary["selected_count"] != rule_count:
        raise MarsRuleArtifactError("validation_summary.selected_count 与规则数量不一致。")
    if type(summary["min_time_slices"]) is not int or summary["min_time_slices"] < 1:
        raise MarsRuleArtifactError("validation_summary.min_time_slices 必须是正整数。")
    if (
        type(summary["minimum_time_slice_count"]) is not int
        or summary["minimum_time_slice_count"] < 0
    ):
        raise MarsRuleArtifactError(
            "validation_summary.minimum_time_slice_count 必须是非负整数。"
        )
    for key in ("confidence_level", "max_fdr"):
        value: Any = summary[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise MarsRuleArtifactError(f"validation_summary.{key} 必须是数值。")
    confidence_level: float = float(summary["confidence_level"])
    max_fdr: float = float(summary["max_fdr"])
    if not 0.5 < confidence_level < 1.0 or not 0.0 <= max_fdr <= 1.0:
        raise MarsRuleArtifactError("validation_summary 的置信水平或 FDR 阈值非法。")
    pass_rate: Any = summary["minimum_time_slice_pass_rate"]
    if pass_rate is not None and (
        isinstance(pass_rate, bool)
        or not isinstance(pass_rate, (int, float))
        or not 0.0 <= float(pass_rate) <= 1.0
    ):
        raise MarsRuleArtifactError("validation_summary 时间切片通过率非法。")
    expected_temporal: bool = qualification == "temporally_validated"
    if type(summary["temporal_assessed"]) is not bool or summary["temporal_assessed"] != expected_temporal:
        raise MarsRuleArtifactError("validation_summary.temporal_assessed 与资格不一致。")
    if qualification == "exploratory":
        if summary["profile"] != "explore":
            raise MarsRuleArtifactError("exploratory 规则集必须来自 explore profile。")
        if summary["validation_status"] not in {"in_sample", "independent"}:
            raise MarsRuleArtifactError("explore validation_status 非法。")
        return
    if summary["profile"] != "production" or summary["validation_status"] != "independent":
        raise MarsRuleArtifactError("可部署资格必须来自独立 validation 的 production profile。")
    if expected_temporal and (
        summary["minimum_time_slice_count"] < summary["min_time_slices"]
        or pass_rate is None
    ):
        raise MarsRuleArtifactError("temporally_validated 摘要不满足时间切片资格。")


def re_safe_column_fragment(value: str) -> str:
    """把用户等级名规范为稳定输出列片段。"""
    safe: str = "".join(char if char.isalnum() or char == "_" else "_" for char in value)
    return safe or "unnamed"
