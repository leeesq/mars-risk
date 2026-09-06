"""Python 3.10+ 的 Experimental 风控分析 Agent，仅通过此入口公开。"""

import sys

if sys.version_info < (3, 10):
    raise ImportError(
        "mars.agent requires Python >=3.10; core MARS remains available on Python 3.8+."
    )

from ._agent import MarsRiskAgent
from ._contracts import (
    MarsAgentMessage,
    MarsAgentProvider,
    MarsAgentReport,
    MarsAgentResponse,
    MarsAgentResult,
    MarsAgentTool,
    MarsAgentToolCall,
    MarsAgentToolResult,
)
from ._provider import MarsOpenAIProvider
from ._session import MarsAgentSession

__all__ = [
    "MarsRiskAgent",
    "MarsAgentSession",
    "MarsOpenAIProvider",
    "MarsAgentProvider",
    "MarsAgentMessage",
    "MarsAgentResponse",
    "MarsAgentTool",
    "MarsAgentToolCall",
    "MarsAgentToolResult",
    "MarsAgentReport",
    "MarsAgentResult",
]
