"""受限规则 DSL 的词法分析、AST 与执行编译器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Mapping, Sequence, Set, Tuple, Union

import polars as pl

from mars.rule.exceptions import MarsRuleDeploymentError, MarsRuleSchemaError, MarsRuleSyntaxError


@dataclass(frozen=True)
class _Token:
    """保存词法单元类型、值与原始位置。"""

    kind: str
    value: Any
    position: int


@dataclass(frozen=True)
class _Comparison:
    """表示列与字面量之间的比较。"""

    identifier: str
    operator: str
    value: Any


@dataclass(frozen=True)
class _NullCheck:
    """表示 ``IS NULL`` 或 ``IS NOT NULL``。"""

    identifier: str
    negated: bool


@dataclass(frozen=True)
class _MissingCheck:
    """表示 ``IS MISSING`` 或 ``IS NOT MISSING``。"""

    identifier: str
    negated: bool


@dataclass(frozen=True)
class _Not:
    """表示逻辑取反。"""

    operand: _Expression


@dataclass(frozen=True)
class _Logical:
    """表示二元 AND/OR 逻辑。"""

    operator: str
    left: _Expression
    right: _Expression


_Expression = Union[_Comparison, _NullCheck, _MissingCheck, _Not, _Logical]

_NUMBER_RE = re.compile(r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*")
_OPERATORS: Tuple[str, ...] = ("<=", ">=", "!=", "==", "=", "<", ">")
_KEYWORDS: Set[str] = {
    "AND",
    "OR",
    "NOT",
    "IS",
    "NULL",
    "MISSING",
    "TRUE",
    "FALSE",
}
_MAX_EXPRESSION_LENGTH = 16_384
_MAX_TOKENS = 512
_MAX_AST_NODES = 256
_MAX_AST_DEPTH = 32
_FLOAT_TYPES: Set[pl.DataType] = {pl.Float32, pl.Float64}
_INTEGER_TYPES: Set[pl.DataType] = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}


def parse_expression(expression: str) -> _Expression:
    """解析并校验规则表达式。

    Parameters
    ----------
    expression : str
        Mars Rule DSL 表达式。

    Returns
    -------
    _Expression
        完成校验的内部 AST。

    """
    tokens: List[_Token] = _tokenize(expression)
    parsed: _Expression = _Parser(tokens).parse()
    _validate_ast_limits(parsed)
    return _simplify_expression(parsed)


def normalize_expression(expression: str) -> str:
    """返回表达式的确定性规范文本。

    Parameters
    ----------
    expression : str
        Mars Rule DSL 表达式。

    Returns
    -------
    str
        使用稳定引号、运算符和括号格式的表达式。
    """
    return expression_to_sql(parse_expression(expression), missing_policy="dsl")


def expression_to_polars(
    expression: _Expression,
    schema: Mapping[str, pl.DataType],
) -> pl.Expr:
    """把规则 AST 编译为 Polars 表达式。

    Parameters
    ----------
    expression : _Expression
        已解析规则 AST。
    schema : Mapping[str, pl.DataType]
        用于缺列、字面量类型和浮点 NaN 语义的输入 schema。

    Returns
    -------
    polars.Expr
        可用于 ``filter`` 或 ``with_columns`` 的布尔表达式。
    """
    validate_expression_schema(expression, schema)
    if isinstance(expression, _Comparison):
        column: pl.Expr = pl.col(expression.identifier)
        literal: pl.Expr = pl.lit(expression.value)
        operators = {
            "<": lambda: column < literal,
            "<=": lambda: column <= literal,
            "==": lambda: column == literal,
            "!=": lambda: column != literal,
            ">=": lambda: column >= literal,
            ">": lambda: column > literal,
        }
        comparison: pl.Expr = operators[expression.operator]()
        if schema[expression.identifier] in _FLOAT_TYPES:
            comparison = ~column.is_nan() & comparison
        return comparison
    if isinstance(expression, _NullCheck):
        null_expr: pl.Expr = pl.col(expression.identifier).is_null()
        return ~null_expr if expression.negated else null_expr
    if isinstance(expression, _MissingCheck):
        column = pl.col(expression.identifier)
        missing_expr: pl.Expr = column.is_null()
        if schema[expression.identifier] in _FLOAT_TYPES:
            missing_expr = missing_expr | column.is_nan()
        return ~missing_expr if expression.negated else missing_expr
    if isinstance(expression, _Not):
        return ~expression_to_polars(expression.operand, schema)
    if expression.operator == "AND":
        return expression_to_polars(expression.left, schema) & expression_to_polars(
            expression.right,
            schema,
        )
    return expression_to_polars(expression.left, schema) | expression_to_polars(
        expression.right,
        schema,
    )


def expression_to_sql(
    expression: _Expression,
    table_alias: str = "",
    *,
    missing_policy: Literal["reject", "normalized_to_null", "dsl"] = "reject",
) -> str:
    """把规则 AST 编译为 ANSI SQL 条件。

    Parameters
    ----------
    expression : _Expression
        已解析规则 AST。
    table_alias : str
        可选表别名前缀；空字符串表示直接引用列。
    missing_policy : Literal["reject", "normalized_to_null", "dsl"]
        ``IS MISSING`` 的输出策略；``dsl`` 仅用于内部规范化文本。

    Returns
    -------
    str
        经过标识符和字符串安全转义的 ANSI SQL 条件。

    Raises
    ------
    MarsRuleDeploymentError
        包含 MISSING 且策略为 ``reject`` 时抛出。
    """
    if isinstance(expression, _Comparison):
        identifier: str = _quote_identifier(expression.identifier, table_alias)
        comparison_operator: str = "=" if expression.operator == "==" else expression.operator
        return f"{identifier} {comparison_operator} {_format_literal(expression.value)}"
    if isinstance(expression, _NullCheck):
        identifier = _quote_identifier(expression.identifier, table_alias)
        null_operator: str = "IS NOT NULL" if expression.negated else "IS NULL"
        return f"{identifier} {null_operator}"
    if isinstance(expression, _MissingCheck):
        identifier = _quote_identifier(expression.identifier, table_alias)
        if missing_policy == "reject":
            raise MarsRuleDeploymentError(
                "ANSI SQL 无法可移植地表示 NaN；请先将 NaN 规范化为 NULL，"
                "并设置 missing_policy='normalized_to_null'。"
            )
        keyword: str = "MISSING" if missing_policy == "dsl" else "NULL"
        missing_operator: str = f"IS {'NOT ' if expression.negated else ''}{keyword}"
        return f"{identifier} {missing_operator}"
    if isinstance(expression, _Not):
        return (
            f"NOT ({expression_to_sql(expression.operand, table_alias, missing_policy=missing_policy)})"
        )
    left: str = expression_to_sql(
        expression.left,
        table_alias,
        missing_policy=missing_policy,
    )
    right: str = expression_to_sql(
        expression.right,
        table_alias,
        missing_policy=missing_policy,
    )
    return f"({left} {expression.operator} {right})"


def expression_features(expression: _Expression) -> Set[str]:
    """返回 AST 引用的全部列名。"""
    if isinstance(expression, (_Comparison, _NullCheck, _MissingCheck)):
        return {expression.identifier}
    if isinstance(expression, _Not):
        return expression_features(expression.operand)
    return expression_features(expression.left) | expression_features(expression.right)


def expression_complexity(expression: _Expression) -> int:
    """返回表达式包含的原子条件数量。"""
    if isinstance(expression, (_Comparison, _NullCheck, _MissingCheck)):
        return 1
    if isinstance(expression, _Not):
        return expression_complexity(expression.operand)
    return expression_complexity(expression.left) + expression_complexity(expression.right)


def expression_has_missing(expression: _Expression) -> bool:
    """返回 AST 是否包含 ``IS MISSING`` 语义。"""
    if isinstance(expression, _MissingCheck):
        return True
    if isinstance(expression, (_Comparison, _NullCheck)):
        return False
    if isinstance(expression, _Not):
        return expression_has_missing(expression.operand)
    return expression_has_missing(expression.left) or expression_has_missing(expression.right)


def _simplify_expression(expression: _Expression) -> _Expression:
    """规范逻辑顺序、移除重复条件并拒绝明显矛盾。"""
    if isinstance(expression, (_Comparison, _NullCheck, _MissingCheck)):
        return expression
    if isinstance(expression, _Not):
        operand: _Expression = _simplify_expression(expression.operand)
        return operand.operand if isinstance(operand, _Not) else _Not(operand)

    operator: str = expression.operator
    terms: List[_Expression] = []
    for child in (
        _simplify_expression(expression.left),
        _simplify_expression(expression.right),
    ):
        terms.extend(_flatten_logical(child, operator))
    unique_terms: Dict[str, _Expression] = {
        _expression_sort_key(term): term for term in terms
    }
    ordered: List[_Expression] = [unique_terms[key] for key in sorted(unique_terms)]
    if operator == "AND":
        _raise_on_contradiction(ordered)
        ordered = sorted(_simplify_and_terms(ordered), key=_expression_sort_key)
    result: _Expression = ordered[0]
    for term in ordered[1:]:
        result = _Logical(operator, result, term)
    return result


def _flatten_logical(expression: _Expression, operator: str) -> List[_Expression]:
    """展平同类逻辑节点以获得稳定排序。"""
    if isinstance(expression, _Logical) and expression.operator == operator:
        return [
            *_flatten_logical(expression.left, operator),
            *_flatten_logical(expression.right, operator),
        ]
    return [expression]


def _expression_sort_key(expression: _Expression) -> str:
    """生成不依赖输入条件顺序的内部排序键。"""
    if isinstance(expression, _Comparison):
        return f"C:{expression.identifier}:{expression.operator}:{expression.value!r}"
    if isinstance(expression, _NullCheck):
        return f"N:{expression.identifier}:{expression.negated!r}"
    if isinstance(expression, _MissingCheck):
        return f"M:{expression.identifier}:{expression.negated!r}"
    if isinstance(expression, _Not):
        return f"T:{_expression_sort_key(expression.operand)}"
    return (
        f"L:{expression.operator}:{_expression_sort_key(expression.left)}:"
        f"{_expression_sort_key(expression.right)}"
    )


def _raise_on_contradiction(terms: Sequence[_Expression]) -> None:
    """拒绝同一列上可静态证明不可能同时成立的原子条件。"""
    by_identifier: Dict[str, List[Union[_Comparison, _NullCheck, _MissingCheck]]] = {}
    for term in terms:
        if isinstance(term, (_Comparison, _NullCheck, _MissingCheck)):
            by_identifier.setdefault(term.identifier, []).append(term)
    for identifier, conditions in by_identifier.items():
        null_checks: Set[bool] = {
            condition.negated
            for condition in conditions
            if isinstance(condition, _NullCheck)
        }
        missing_checks: Set[bool] = {
            condition.negated
            for condition in conditions
            if isinstance(condition, _MissingCheck)
        }
        comparisons: List[_Comparison] = [
            condition for condition in conditions if isinstance(condition, _Comparison)
        ]
        has_missing_required: bool = False in missing_checks
        has_present_required: bool = True in missing_checks
        if (
            len(null_checks) > 1
            or len(missing_checks) > 1
            or (False in null_checks and comparisons)
            or (has_missing_required and comparisons)
            or (False in null_checks and has_present_required)
        ):
            raise MarsRuleSyntaxError(f"规则在列 {identifier!r} 上包含矛盾条件。")
        _raise_on_comparison_contradiction(identifier, comparisons)


def _raise_on_comparison_contradiction(
    identifier: str,
    comparisons: Sequence[_Comparison],
) -> None:
    """检查相等、上下界和排除值之间的直接矛盾。"""
    value_families: Set[str] = {_literal_family(item.value) for item in comparisons}
    if len(value_families) > 1:
        raise MarsRuleSyntaxError(f"规则在列 {identifier!r} 上混用了不可比较的字面量类型。")
    equals: List[Any] = [item.value for item in comparisons if item.operator == "=="]
    excluded: List[Any] = [item.value for item in comparisons if item.operator == "!="]
    if equals and (any(value != equals[0] for value in equals[1:]) or equals[0] in excluded):
        raise MarsRuleSyntaxError(f"规则在列 {identifier!r} 上包含矛盾条件。")

    lower: Tuple[Any, bool] | None = None
    upper: Tuple[Any, bool] | None = None
    for condition in comparisons:
        if condition.operator in {">", ">="}:
            candidate: Tuple[Any, bool] = (condition.value, condition.operator == ">=")
            if lower is None or candidate[0] > lower[0] or (
                candidate[0] == lower[0] and not candidate[1]
            ):
                lower = candidate
        elif condition.operator in {"<", "<="}:
            candidate = (condition.value, condition.operator == "<=")
            if upper is None or candidate[0] < upper[0] or (
                candidate[0] == upper[0] and not candidate[1]
            ):
                upper = candidate
    try:
        impossible_bounds: bool = bool(
            lower is not None
            and upper is not None
            and (
                lower[0] > upper[0]
                or (lower[0] == upper[0] and not (lower[1] and upper[1]))
            )
        )
        equality_outside: bool = bool(
            equals
            and (
                lower is not None
                and (equals[0] < lower[0] or (equals[0] == lower[0] and not lower[1]))
                or upper is not None
                and (equals[0] > upper[0] or (equals[0] == upper[0] and not upper[1]))
            )
        )
    except TypeError:
        impossible_bounds = False
        equality_outside = False
    if impossible_bounds or equality_outside:
        raise MarsRuleSyntaxError(f"规则在列 {identifier!r} 上包含矛盾条件。")


def _literal_family(value: Any) -> str:
    """返回 DSL 字面量的比较类型族。"""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, str):
        return "str"
    return "number"


def _simplify_and_terms(terms: Sequence[_Expression]) -> List[_Expression]:
    """删除同一列上被更严格相等或边界条件支配的比较。"""
    passthrough: List[_Expression] = [
        term
        for term in terms
        if not isinstance(term, (_Comparison, _NullCheck, _MissingCheck))
    ]
    identifiers: List[str] = sorted(
        {
            term.identifier
            for term in terms
            if isinstance(term, (_Comparison, _NullCheck, _MissingCheck))
        }
    )
    for identifier in identifiers:
        atomic = [
            term
            for term in terms
            if isinstance(term, (_Comparison, _NullCheck, _MissingCheck))
            and term.identifier == identifier
        ]
        null_checks: List[_NullCheck] = [
            term for term in atomic if isinstance(term, _NullCheck)
        ]
        missing_checks: List[_MissingCheck] = [
            term for term in atomic if isinstance(term, _MissingCheck)
        ]
        comparisons: List[_Comparison] = [
            term for term in atomic if isinstance(term, _Comparison)
        ]
        passthrough.extend(null_checks)
        passthrough.extend(missing_checks)
        equals: List[_Comparison] = [
            item for item in comparisons if item.operator == "=="
        ]
        if equals:
            passthrough.append(equals[0])
            continue
        lower = _tightest_bound(comparisons, lower=True)
        upper = _tightest_bound(comparisons, lower=False)
        if lower is not None:
            passthrough.append(lower)
        if upper is not None:
            passthrough.append(upper)
        passthrough.extend(item for item in comparisons if item.operator == "!=")
    return passthrough


def _tightest_bound(
    comparisons: Sequence[_Comparison],
    *,
    lower: bool,
) -> _Comparison | None:
    """返回同方向比较中约束最严格的边界。"""
    operators: Set[str] = {">", ">="} if lower else {"<", "<="}
    candidates: List[_Comparison] = [
        item for item in comparisons if item.operator in operators
    ]
    if not candidates:
        return None

    def key(item: _Comparison) -> Tuple[Any, int]:
        """让更大下界或更小上界优先，并在同值时选择开区间。"""
        exclusive: int = 1 if item.operator in {">", "<"} else 0
        return item.value, exclusive if lower else -exclusive

    return (max if lower else min)(candidates, key=key)


def _tokenize(expression: str) -> List[_Token]:
    """把 DSL 文本拆成带位置的词法单元。"""
    if not isinstance(expression, str) or not expression.strip():
        raise MarsRuleSyntaxError("规则表达式不能为空。")
    if len(expression) > _MAX_EXPRESSION_LENGTH:
        raise MarsRuleSyntaxError(
            f"规则表达式长度不能超过 {_MAX_EXPRESSION_LENGTH} 个字符。"
        )

    tokens: List[_Token] = []
    index = 0
    while index < len(expression):
        char: str = expression[index]
        if char.isspace():
            index += 1
            continue
        if char in "()":
            tokens.append(_Token("PAREN", char, index))
            index += 1
            continue
        if char == '"':
            value, index = _read_quoted(expression, index, '"', doubled_escape=True)
            tokens.append(_Token("IDENT", value, index - len(value) - 2))
            continue
        if char == "'":
            value, index = _read_quoted(expression, index, "'", doubled_escape=True)
            tokens.append(_Token("LITERAL", value, index - len(value) - 2))
            continue

        matched_operator = next(
            (operator for operator in _OPERATORS if expression.startswith(operator, index)),
            None,
        )
        if matched_operator is not None:
            normalized_operator: str = "==" if matched_operator == "=" else matched_operator
            tokens.append(_Token("OP", normalized_operator, index))
            index += len(matched_operator)
            continue

        number_match = _NUMBER_RE.match(expression, index)
        if number_match is not None:
            raw_number: str = number_match.group(0)
            number: Union[int, float]
            number = float(raw_number) if any(mark in raw_number.lower() for mark in (".", "e")) else int(raw_number)
            tokens.append(_Token("LITERAL", number, index))
            index = number_match.end()
            continue

        identifier_match = _IDENT_RE.match(expression, index)
        if identifier_match is not None:
            value = identifier_match.group(0)
            keyword: str = value.upper()
            if keyword in _KEYWORDS:
                if keyword == "TRUE":
                    tokens.append(_Token("LITERAL", True, index))
                elif keyword == "FALSE":
                    tokens.append(_Token("LITERAL", False, index))
                else:
                    tokens.append(_Token("KEYWORD", keyword, index))
            else:
                tokens.append(_Token("IDENT", value, index))
            index = identifier_match.end()
            continue
        raise MarsRuleSyntaxError(
            f"规则第 {index} 个字符无法解析：{char!r}。",
            context={"position": index},
        )
    if len(tokens) > _MAX_TOKENS:
        raise MarsRuleSyntaxError(f"规则 token 数不能超过 {_MAX_TOKENS}。")
    return tokens


def _read_quoted(
    text: str,
    start: int,
    quote: str,
    *,
    doubled_escape: bool,
) -> Tuple[str, int]:
    """读取 SQL 风格双写转义的引号内容。"""
    index = start + 1
    parts: List[str] = []
    while index < len(text):
        if text[index] != quote:
            parts.append(text[index])
            index += 1
            continue
        if doubled_escape and index + 1 < len(text) and text[index + 1] == quote:
            parts.append(quote)
            index += 2
            continue
        return "".join(parts), index + 1
    raise MarsRuleSyntaxError(
        f"规则第 {start} 个字符开始的引号未闭合。",
        context={"position": start},
    )


class _Parser:
    """使用递归下降算法解析受限布尔表达式。"""

    def __init__(self, tokens: Sequence[_Token]) -> None:
        self._tokens = tokens
        self._position = 0

    def parse(self) -> _Expression:
        """解析完整表达式并拒绝尾随 token。"""
        if not self._tokens:
            raise MarsRuleSyntaxError("规则表达式不能为空。")
        expression: _Expression = self._parse_or()
        if self._position != len(self._tokens):
            token: _Token = self._tokens[self._position]
            raise MarsRuleSyntaxError(
                f"规则存在无法归位的 token：{token.value!r}。",
                context={"position": token.position},
            )
        return expression

    def _parse_or(self) -> _Expression:
        """解析 OR 层级。"""
        expression: _Expression = self._parse_and()
        while self._accept("KEYWORD", "OR") is not None:
            expression = _Logical("OR", expression, self._parse_and())
        return expression

    def _parse_and(self) -> _Expression:
        """解析 AND 层级。"""
        expression: _Expression = self._parse_not()
        while self._accept("KEYWORD", "AND") is not None:
            expression = _Logical("AND", expression, self._parse_not())
        return expression

    def _parse_not(self) -> _Expression:
        """解析一元 NOT。"""
        if self._accept("KEYWORD", "NOT") is not None:
            return _Not(self._parse_not())
        return self._parse_factor()

    def _parse_factor(self) -> _Expression:
        """解析括号或原子条件。"""
        if self._accept("PAREN", "(") is not None:
            expression: _Expression = self._parse_or()
            self._expect("PAREN", ")")
            return expression
        return self._parse_predicate()

    def _parse_predicate(self) -> _Expression:
        """解析比较或空值判断。"""
        identifier_token: _Token | None = self._accept("IDENT")
        if identifier_token is None:
            identifier_token = self._accept("KEYWORD", "MISSING")
        if identifier_token is None:
            identifier_token = self._expect("IDENT")
        identifier: str = str(identifier_token.value)
        if self._accept("KEYWORD", "IS") is not None:
            negated: bool = self._accept("KEYWORD", "NOT") is not None
            if self._accept("KEYWORD", "NULL") is not None:
                return _NullCheck(identifier, negated)
            self._expect("KEYWORD", "MISSING")
            return _MissingCheck(identifier, negated)
        operator: str = str(self._expect("OP").value)
        literal: _Token = self._expect("LITERAL")
        return _Comparison(identifier, operator, literal.value)

    def _accept(self, kind: str, value: Any = None) -> Union[_Token, None]:
        """匹配当前 token，失败时不移动游标。"""
        if self._position >= len(self._tokens):
            return None
        token: _Token = self._tokens[self._position]
        if token.kind != kind or (value is not None and token.value != value):
            return None
        self._position += 1
        return token

    def _expect(self, kind: str, value: Any = None) -> _Token:
        """强制匹配 token 并产生带位置的业务错误。"""
        token: Union[_Token, None] = self._accept(kind, value)
        if token is not None:
            return token
        expected: str = str(value if value is not None else kind)
        if self._position >= len(self._tokens):
            raise MarsRuleSyntaxError(f"期望 {expected}，但规则已结束。")
        actual: _Token = self._tokens[self._position]
        raise MarsRuleSyntaxError(
            f"期望 {expected}，实际为 {actual.value!r}。",
            context={"position": actual.position},
        )


def _quote_identifier(identifier: str, table_alias: str) -> str:
    """安全引用 ANSI SQL 标识符和可选表别名。"""
    quoted: str = '"' + identifier.replace('"', '""') + '"'
    if not table_alias:
        return quoted
    alias: str = '"' + table_alias.replace('"', '""') + '"'
    return f"{alias}.{quoted}"


def _format_literal(value: Any) -> str:
    """把受支持字面量格式化为 ANSI SQL。"""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, (int, float)):
        return repr(value)
    raise MarsRuleSyntaxError(f"不支持的规则字面量类型：{type(value).__name__}。")


def validate_expression_schema(
    expression: _Expression,
    schema: Mapping[str, pl.DataType],
) -> None:
    """校验规则引用列和字面量类型与输入 schema 一致。"""
    missing: List[str] = sorted(expression_features(expression) - set(schema))
    if missing:
        raise MarsRuleSchemaError(f"规则执行缺少必需列：{missing}。")
    _validate_node_schema(expression, schema)


def _validate_node_schema(
    expression: _Expression,
    schema: Mapping[str, pl.DataType],
) -> None:
    """递归校验比较节点的数据类型族。"""
    if isinstance(expression, _Comparison):
        dtype: pl.DataType = schema[expression.identifier]
        family: str = _literal_family(expression.value)
        compatible: bool = (
            family == "number" and dtype in (_FLOAT_TYPES | _INTEGER_TYPES)
            or family == "bool" and dtype == pl.Boolean
            or family == "str" and dtype in {pl.Utf8, pl.Categorical, pl.Enum}
        )
        if not compatible:
            raise MarsRuleSchemaError(
                f"规则列 {expression.identifier!r} 的类型 {dtype} "
                f"与 {family} 字面量不兼容。"
            )
        return
    if isinstance(expression, (_NullCheck, _MissingCheck)):
        return
    if isinstance(expression, _Not):
        _validate_node_schema(expression.operand, schema)
        return
    _validate_node_schema(expression.left, schema)
    _validate_node_schema(expression.right, schema)


def _validate_ast_limits(expression: _Expression) -> None:
    """限制 AST 规模，避免病态表达式耗尽递归和编译资源。"""
    def visit(node: _Expression, depth: int) -> int:
        """返回子树节点数并检查最大嵌套深度。"""
        if depth > _MAX_AST_DEPTH:
            raise MarsRuleSyntaxError(f"规则 AST 嵌套深度不能超过 {_MAX_AST_DEPTH}。")
        if isinstance(node, (_Comparison, _NullCheck, _MissingCheck)):
            return 1
        if isinstance(node, _Not):
            return 1 + visit(node.operand, depth + 1)
        return 1 + visit(node.left, depth + 1) + visit(node.right, depth + 1)

    node_count: int = visit(expression, 1)
    if node_count > _MAX_AST_NODES:
        raise MarsRuleSyntaxError(f"规则 AST 节点数不能超过 {_MAX_AST_NODES}。")
