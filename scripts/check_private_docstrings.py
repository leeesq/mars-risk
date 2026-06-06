"""检查复杂私有函数和私有方法是否具备必要说明。"""

from __future__ import annotations

import argparse
import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

CONTROL_NODES: tuple[type[ast.AST], ...] = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.Match,
)
NESTED_SCOPE_NODES: tuple[type[ast.AST], ...] = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.ClassDef,
)
COMPLEX_LINE_THRESHOLD = 15
CONTROL_COUNT_THRESHOLD = 4
CONTROL_DEPTH_THRESHOLD = 2
SKIP_DIR_NAMES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "dist",
}
KEYWORDS = {
    "woe",
    "iv",
    "psi",
    "ks",
    "lift",
    "bin",
    "bins",
    "binner",
    "binning",
    "score",
    "oot",
    "bad_rate",
    "lazy",
    "join",
    "batch",
    "sample",
    "parallel",
}
FULL_DOCSTRING_SECTIONS = {"Parameters", "Returns", "Raises", "Examples"}


@dataclass(frozen=True)
class _FunctionAnalysis:
    """保存单个私有函数的复杂度判断结果。"""

    path: Path
    line: int
    qualname: str
    reasons: tuple[str, ...]
    has_docstring: bool
    has_full_docstring: bool


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="检查复杂私有函数和私有方法是否缺少中文短 docstring。",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["src/mars"],
        help="需要检查的文件或目录，默认检查 src/mars。",
    )
    return parser.parse_args(argv)


def _is_private_name(name: str) -> bool:
    return name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def _iter_python_files(paths: Sequence[Path]) -> Iterator[Path]:
    for path in paths:
        if path.is_file() and path.suffix == ".py":
            yield path
            continue
        if not path.is_dir():
            continue
        for candidate in path.rglob("*.py"):
            if any(part in SKIP_DIR_NAMES for part in candidate.parts):
                continue
            yield candidate


def _walk_without_nested_scopes(node: ast.AST) -> Iterator[ast.AST]:
    stack: list[ast.AST] = list(reversed(list(ast.iter_child_nodes(node))))
    while stack:
        current = stack.pop()
        yield current
        if isinstance(current, NESTED_SCOPE_NODES):
            continue
        stack.extend(reversed(list(ast.iter_child_nodes(current))))


def _count_effective_lines(node: ast.FunctionDef | ast.AsyncFunctionDef, lines: Sequence[str]) -> int:
    body_start = node.body[0].lineno if node.body else node.lineno
    if node.body and _is_docstring_expr(node.body[0]):
        body_start = (node.body[0].end_lineno or node.body[0].lineno) + 1

    body_end = node.end_lineno or node.lineno
    effective_lines = 0
    for line in lines[body_start - 1 : body_end]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        effective_lines += 1
    return effective_lines


def _is_docstring_expr(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _count_control_nodes(node: ast.AST) -> int:
    return sum(isinstance(child, CONTROL_NODES) for child in _walk_without_nested_scopes(node))


def _max_control_depth(node: ast.AST) -> int:
    def visit(current: ast.AST, depth: int) -> int:
        if isinstance(current, NESTED_SCOPE_NODES) and current is not node:
            return depth
        next_depth = depth + 1 if isinstance(current, CONTROL_NODES) else depth
        child_depths = [visit(child, next_depth) for child in ast.iter_child_nodes(current)]
        return max([next_depth, *child_depths])

    return visit(node, 0)


def _has_raise(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Raise) for child in _walk_without_nested_scopes(node))


def _has_complex_return(node: ast.AST) -> bool:
    complex_nodes = (ast.Tuple, ast.Dict, ast.List, ast.Set)
    for child in _walk_without_nested_scopes(node):
        if isinstance(child, ast.Return) and isinstance(child.value, complex_nodes):
            return True
    return False


def _has_keyword_hit(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    tokens: set[str] = set()
    raw_names: list[str] = [node.name, *(arg.arg for arg in node.args.args)]
    for child in _walk_without_nested_scopes(node):
        if isinstance(child, ast.Name):
            raw_names.append(child.id)
        elif isinstance(child, ast.Attribute):
            raw_names.append(child.attr)

    for name in raw_names:
        tokens.update(_split_identifier(name))
    return bool(tokens & KEYWORDS)


def _split_identifier(name: str) -> set[str]:
    snake_parts = re.split(r"[^0-9A-Za-z]+", name)
    tokens: set[str] = set()
    for part in snake_parts:
        if not part:
            continue
        camel_parts = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", part).split()
        tokens.update(token.lower() for token in camel_parts if token)
    return tokens


def _is_template_hook(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    return node.name.endswith("_impl") or any(_is_abstractmethod(decorator) for decorator in node.decorator_list)


def _is_abstractmethod(node: ast.AST) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "abstractmethod"
    if isinstance(node, ast.Attribute):
        return node.attr == "abstractmethod"
    return False


def _has_full_docstring(docstring: str | None) -> bool:
    if docstring is None:
        return False
    section_pattern = re.compile(r"^\s*(Parameters|Returns|Raises|Examples)\s*$", re.MULTILINE)
    return bool(FULL_DOCSTRING_SECTIONS & set(section_pattern.findall(docstring)))


class _PrivateFunctionVisitor(ast.NodeVisitor):
    """遍历模块中的顶层私有函数和类私有方法。"""

    def __init__(self, path: Path, lines: Sequence[str]) -> None:
        self.path = path
        self.lines = lines
        self.class_stack: list[str] = []
        self.function_depth = 0
        self.analyses: list[_FunctionAnalysis] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self.function_depth == 0 and _is_private_name(node.name):
            self.analyses.append(self._analyze_function(node))

        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def _analyze_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> _FunctionAnalysis:
        reasons = _collect_complex_reasons(node, self.lines)
        docstring = ast.get_docstring(node, clean=False)
        qualname = ".".join([*self.class_stack, node.name])
        return _FunctionAnalysis(
            path=self.path,
            line=node.lineno,
            qualname=qualname,
            reasons=tuple(reasons),
            has_docstring=docstring is not None,
            has_full_docstring=_has_full_docstring(docstring),
        )


def _collect_complex_reasons(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    lines: Sequence[str],
) -> list[str]:
    effective_lines = _count_effective_lines(node, lines)
    control_count = _count_control_nodes(node)
    control_depth = _max_control_depth(node)
    reasons: list[str] = []

    if effective_lines > COMPLEX_LINE_THRESHOLD:
        reasons.append(f"有效代码行数 {effective_lines} > {COMPLEX_LINE_THRESHOLD}")
    if control_count >= CONTROL_COUNT_THRESHOLD:
        reasons.append(f"控制流节点数 {control_count} >= {CONTROL_COUNT_THRESHOLD}")
    if control_depth >= CONTROL_DEPTH_THRESHOLD:
        reasons.append(f"最大控制流嵌套层级 {control_depth} >= {CONTROL_DEPTH_THRESHOLD}")
    if _has_raise(node):
        reasons.append("包含主动 raise")
    if _has_complex_return(node):
        reasons.append("返回复杂结构")
    if _has_keyword_hit(node):
        reasons.append("命中风控或性能关键词")
    if _is_template_hook(node):
        reasons.append("疑似子类 hook/template method")

    return reasons


def _analyze_file(path: Path) -> list[_FunctionAnalysis]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    visitor = _PrivateFunctionVisitor(path=path, lines=source.splitlines())
    visitor.visit(tree)
    return visitor.analyses


def _analyze_paths(paths: Sequence[Path]) -> tuple[list[_FunctionAnalysis], list[_FunctionAnalysis]]:
    missing_docstrings: list[_FunctionAnalysis] = []
    simple_full_docstrings: list[_FunctionAnalysis] = []

    for path in _iter_python_files(paths):
        for analysis in _analyze_file(path):
            is_complex = bool(analysis.reasons)
            if is_complex and not analysis.has_docstring:
                missing_docstrings.append(analysis)
            elif not is_complex and analysis.has_full_docstring:
                simple_full_docstrings.append(analysis)

    return missing_docstrings, simple_full_docstrings


def _format_issue(analysis: _FunctionAnalysis) -> str:
    reasons = "；".join(analysis.reasons)
    return f"{analysis.path}:{analysis.line}: {analysis.qualname} 缺少中文短 docstring（{reasons}）"


def _format_warning(analysis: _FunctionAnalysis) -> str:
    return (
        f"{analysis.path}:{analysis.line}: {analysis.qualname} 是简单私有 helper，"
        "但包含完整 NumPy docstring，可考虑精简。"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """运行复杂私有函数 docstring 检查。

    Parameters
    ----------
    argv : Sequence[str] | None, default None
        命令行参数；为 `None` 时使用当前进程参数。

    Returns
    -------
    int
        进程退出码；存在缺失 docstring 的复杂私有方法时返回 `1`，否则返回 `0`。
    """
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    paths = [Path(path) for path in args.paths]
    missing_docstrings, simple_full_docstrings = _analyze_paths(paths)

    for warning in simple_full_docstrings:
        print(f"WARNING: {_format_warning(warning)}")

    if missing_docstrings:
        print("复杂私有函数/方法 docstring 检查失败：", file=sys.stderr)
        for issue in missing_docstrings:
            print(f"ERROR: {_format_issue(issue)}", file=sys.stderr)
        return 1

    print("复杂私有函数/方法 docstring 检查通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
