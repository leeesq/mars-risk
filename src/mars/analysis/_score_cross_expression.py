"""仅表达完整正常分箱集合的受限规则；Python 和离线页面共享语法与镜像示例。"""

from __future__ import annotations

import re
from typing import Any

_EXPRESSION_LIMITS = {"max_length": 240, "max_tokens": 96, "max_depth": 12}
_OPERATORS = {"<", "<=", "=", "==", "!=", ">=", ">"}
_TOKEN = re.compile(r"\s+|<=|>=|==|!=|[<>=()]|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|\S")
_LABEL = re.compile(r"([XY])([0-9]+)\Z")


class _ExpressionParser:
    """严格词法分析后按 OR/AND/括号解析，不接受连续值、函数或任意源码。"""

    def __init__(self, expression: str, x_count: int, y_count: int) -> None:
        if not isinstance(expression, str):
            raise ValueError("规则必须是文本。")
        if len(expression) > _EXPRESSION_LIMITS["max_length"]:
            raise ValueError("规则长度超限：最多 240 个字符。")
        self.tokens: list[str] = []
        for match in _TOKEN.finditer(expression):
            token = match.group().upper()
            if token.isspace():
                continue
            if not (
                token in {"X", "Y", "AND", "OR", "(", ")"}
                or token in _OPERATORS
                or _LABEL.fullmatch(token)
                or re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token)
            ):
                raise ValueError(f"非法词元：{token!r}。只支持分箱比较、AND、OR 和括号。")
            self.tokens.append(token)
            if len(self.tokens) > _EXPRESSION_LIMITS["max_tokens"]:
                raise ValueError("规则复杂度超限：最多 96 个词元。")
        if not self.tokens:
            raise ValueError("规则不能为空。")
        self.index = 0
        self.counts = {"X": x_count, "Y": y_count}

    def _peek(self) -> str | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _logical(self, operator: str, depth: int) -> dict[str, Any]:
        """AND 优先于 OR，连续同类组合保存为扁平 AST，避免不必要深链。"""
        operands = [self._logical("AND", depth) if operator == "OR" else self._atom(depth)]
        while self._peek() == operator:
            self.index += 1
            operands.append(self._logical("AND", depth) if operator == "OR" else self._atom(depth))
        return operands[0] if len(operands) == 1 else {"kind": operator.lower(), "operands": operands}

    def _atom(self, depth: int) -> dict[str, Any]:
        """检查成对括号和同轴整数标签，编号对应保存的风险序位。"""
        token = self._peek()
        if token == "(":
            if depth >= _EXPRESSION_LIMITS["max_depth"]:
                raise ValueError("括号嵌套超限：最多 12 层。")
            self.index += 1
            node = self._logical("OR", depth + 1)
            if self._peek() != ")":
                raise ValueError("括号不成对：缺少右括号。")
            self.index += 1
            return node
        if token == ")":
            raise ValueError("括号不成对或括号内缺少比较条件。")
        if token not in {"X", "Y"}:
            raise ValueError("需要 X 或 Y 轴的比较条件。")
        axis = str(token)
        self.index += 1
        operator = self._peek()
        if operator not in _OPERATORS:
            raise ValueError("缺少比较符：允许 <、<=、=、==、!=、>=、>。")
        self.index += 1
        value = self._peek()
        label = _LABEL.fullmatch(value or "")
        if label:
            if label.group(1) != axis:
                raise ValueError(f"轴不匹配：{axis} 只能与 {axis} 分箱标签比较。")
            rank = int(label.group(2))
        elif value is not None and re.fullmatch(r"[0-9]+", value):
            rank = int(value)
        else:
            raise ValueError("箱号必须是整数或同轴分箱标签，不支持原始连续阈值。")
        if not 1 <= rank <= self.counts[axis]:
            raise ValueError(f"{axis} 箱号越界：允许 1..{self.counts[axis]}。")
        self.index += 1
        return {"kind": "compare", "axis": axis, "operator": operator, "rank": rank}


def _parse_score_expression(expression: str, x_count: int, y_count: int) -> dict[str, Any]:
    """返回有界、可序列化 AST；剩余词元一律报错，不静默截断规则。"""
    parser = _ExpressionParser(expression, x_count, y_count)
    node = parser._logical("OR", 0)
    if parser._peek() is not None:
        if parser._peek() == ")":
            raise ValueError("括号不成对：多余右括号。")
        raise ValueError("条件之间缺少 AND 或 OR，或存在多余词元。")
    return node


def _evaluate_score_expression(node: dict[str, Any], x_rank: int | None, y_rank: int | None) -> bool:
    """显式解释已校验 AST；任何特殊轴均排除，不隐含接受缺失或 invalid 箱。"""
    if x_rank is None or y_rank is None:
        return False
    kind = node["kind"]
    if kind == "and":
        return all(_evaluate_score_expression(child, x_rank, y_rank) for child in node["operands"])
    if kind == "or":
        return any(_evaluate_score_expression(child, x_rank, y_rank) for child in node["operands"])
    actual = x_rank if node["axis"] == "X" else y_rank
    expected = node["rank"]
    operator = node["operator"]
    if operator == "<":
        return bool(actual < expected)
    if operator == "<=":
        return bool(actual <= expected)
    if operator in {"=", "=="}:
        return bool(actual == expected)
    if operator == "!=":
        return bool(actual != expected)
    if operator == ">=":
        return bool(actual >= expected)
    return bool(actual > expected)


def _score_rule_examples(x_count: int, y_count: int) -> list[dict[str, Any]]:
    """生成低风险阶梯及其真实风险序位镜像；小矩阵缩减且描述绑定去重命中集合。"""
    if x_count < 1 or y_count < 1:
        return []
    rows: list[tuple[int, int]] = [(1, y_count)]
    if x_count >= 2:
        rows.append((2, min(2, y_count)))
    if x_count >= 3:
        rows.append((3, 1))
    examples: list[dict[str, Any]] = []
    for high in (False, True):
        clauses: list[str] = []
        phrases: list[str] = []
        cells: list[list[int]] = []
        for position, (low_x, width) in enumerate(rows):
            x = x_count + 1 - low_x if high else low_x
            y = y_count + 1 - width if high else width
            columns = range(y, y_count + 1) if high else range(1, y + 1)
            cells.extend([[x, column] for column in columns])
            if position == 0:
                clauses.append(f"(X = X{x})")
                phrases.append(f"X{x} 全行")
            elif width == 1:
                clauses.append(f"(X = X{x} AND Y = Y{y})")
                phrases.append(f"X{x} 第 {y} 列")
            else:
                comparator = ">=" if high else "<="
                clauses.append(f"(X = X{x} AND Y {comparator} Y{y})")
                phrases.append(f"X{x} {'后' if high else '前'} {width} 列")
        count = len(cells)
        examples.append(
            {
                "name": "高风险侧诊断示例" if high else "低风险侧示例",
                "expression": " OR ".join(clauses),
                "description": "、".join(phrases) + f"，共 {count} 格。" + ("仅用于诊断，不是审批规则。" if high else "仅表达完整分箱集合。"),
                "cells": cells,
            }
        )
    return examples


# 页面解释器使用同一受限 AST，任何输入均不拼接成源码；错误文本与 Python 一致。
_EXPRESSION_JAVASCRIPT = r"""
function parseScoreExpression(expression, xCount, yCount) {
  if (typeof expression !== 'string') throw new Error('规则必须是文本。');
  if ([...expression].length > 240) throw new Error('规则长度超限：最多 240 个字符。');
  const tokens = [], pattern = /\s+|<=|>=|==|!=|[<>=()]|[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?|\S/gu;
  const operators = ['<', '<=', '=', '==', '!=', '>=', '>'];
  for (const match of expression.matchAll(pattern)) {
    const token = match[0].toUpperCase();
    if (/^\s+$/u.test(token)) continue;
    if (!(['X','Y','AND','OR','(',')'].includes(token) || operators.includes(token) || /^[XY][0-9]+$/.test(token) || /^[0-9]+(?:\.[0-9]+)?$/.test(token))) {
      throw new Error("非法词元：'" + token + "'。只支持分箱比较、AND、OR 和括号。");
    }
    tokens.push(token);
    if (tokens.length > 96) throw new Error('规则复杂度超限：最多 96 个词元。');
  }
  if (!tokens.length) throw new Error('规则不能为空。');
  let index = 0;
  const counts = {X:xCount, Y:yCount}, peek = () => tokens[index];
  function logical(operator, depth) {
    const operands = [operator === 'OR' ? logical('AND', depth) : atom(depth)];
    while (peek() === operator) {
      index++;
      operands.push(operator === 'OR' ? logical('AND', depth) : atom(depth));
    }
    return operands.length === 1 ? operands[0] : {kind:operator.toLowerCase(), operands};
  }
  function atom(depth) {
    const token = peek();
    if (token === '(') {
      if (depth >= 12) throw new Error('括号嵌套超限：最多 12 层。');
      index++;
      const node = logical('OR', depth + 1);
      if (peek() !== ')') throw new Error('括号不成对：缺少右括号。');
      index++;
      return node;
    }
    if (token === ')') throw new Error('括号不成对或括号内缺少比较条件。');
    if (!['X','Y'].includes(token)) throw new Error('需要 X 或 Y 轴的比较条件。');
    const axis = token;
    index++;
    const operator = peek();
    if (!operators.includes(operator)) throw new Error('缺少比较符：允许 <、<=、=、==、!=、>=、>。');
    index++;
    const value = peek(), label = /^([XY])([0-9]+)$/.exec(value || '');
    let rank;
    if (label) {
      if (label[1] !== axis) throw new Error(`${axis} 只能与 ${axis} 分箱标签比较。`.replace(/^/, '轴不匹配：'));
      rank = Number(label[2]);
    } else if (value !== undefined && /^[0-9]+$/.test(value)) rank = Number(value);
    else throw new Error('箱号必须是整数或同轴分箱标签，不支持原始连续阈值。');
    if (!(rank >= 1 && rank <= counts[axis])) throw new Error(`${axis} 箱号越界：允许 1..${counts[axis]}。`);
    index++;
    return {kind:'compare', axis, operator, rank};
  }
  const node = logical('OR', 0);
  if (peek() !== undefined) {
    if (peek() === ')') throw new Error('括号不成对：多余右括号。');
    throw new Error('条件之间缺少 AND 或 OR，或存在多余词元。');
  }
  return node;
}
function evaluateScoreExpression(node, xRank, yRank) {
  if (xRank == null || yRank == null) return false;
  if (node.kind === 'and') return node.operands.every(child => evaluateScoreExpression(child,xRank,yRank));
  if (node.kind === 'or') return node.operands.some(child => evaluateScoreExpression(child,xRank,yRank));
  const actual = node.axis === 'X' ? xRank : yRank, expected = node.rank;
  switch (node.operator) {
    case '<': return actual < expected;
    case '<=': return actual <= expected;
    case '=': case '==': return actual === expected;
    case '!=': return actual !== expected;
    case '>=': return actual >= expected;
    case '>': return actual > expected;
  }
  return false;
}
"""
