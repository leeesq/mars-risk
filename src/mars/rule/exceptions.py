"""MARS 规则模块异常类型。"""

from __future__ import annotations

from mars.core.exceptions import MarsError


class MarsRuleError(MarsError):
    """规则模块基础异常。"""


class MarsRuleSyntaxError(MarsRuleError):
    """规则 DSL 无法解析或包含禁用语法时抛出的异常。"""


class MarsRuleArtifactError(MarsRuleError):
    """规则集 artifact 结构、版本或内容非法时抛出的异常。"""


class MarsRuleSchemaError(MarsRuleError):
    """规则表达式与输入数据 schema 不兼容时抛出的异常。"""


class MarsRuleDeploymentError(MarsRuleError):
    """规则集未满足部署资格或导出前提时抛出的异常。"""
