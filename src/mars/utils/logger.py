"""MARS 统一日志工具。"""

from __future__ import annotations

import logging
import sys
from typing import Dict, Union

# 尝试导入 colorlog 以支持彩色日志输出
# 如果环境中未安装 colorlog，将自动回退到标准无色日志
HAS_COLORLOG: bool = False
try:
    import colorlog
except ImportError:
    pass
else:
    HAS_COLORLOG = True

# 定义单例 Logger 的名称，确保整个项目使用的是同一个日志实例
LOGGER_NAME: str = "mars"
CONSOLE_ENCODING: str = getattr(sys.stdout, "encoding", None) or "utf-8"
SAFE_REPLACEMENTS: Dict[str, str] = {
    "⚠️": "[WARN]",
    "⚠": "[WARN]",
    "❌": "[ERROR]",
    "✅": "[OK]",
    "ℹ️": "[INFO]",
    "ℹ": "[INFO]",
    "⚙️": "[SETUP]",
    "⚙": "[SETUP]",
    "📈": "[PLOT]",
    "📊": "[REPORT]",
    "📉": "[CHECK]",
    "🧹": "[CLEAN]",
    "🔄": "[TRANSFORM]",
    "🧮": "[METRIC]",
    "🎲": "[SAMPLE]",
    "⏱️": "[TIMER]",
    "⏱": "[TIMER]",
    "∑": "[ROLLUP]",
}
def _sanitize_console_text(text: str) -> str:
    """
    替换控制台图形符号，并处理当前编码无法表示的字符。

    Parameters
    ----------
    text : str
        待写入控制台的原始文本。

    Returns
    -------
    str
        经过编码安全处理后的文本。
    """
    sanitized = text
    for src, dst in SAFE_REPLACEMENTS.items():
        sanitized = sanitized.replace(src, dst)
    return sanitized.encode(CONSOLE_ENCODING, errors="replace").decode(CONSOLE_ENCODING)


class SafePlainFormatter(logging.Formatter):
    """
    面向普通文本控制台的安全格式化器。

    Attributes
    ----------
    _style : logging.PercentStyle
        ``logging.Formatter`` 内部保存的格式化样式对象。

    Examples
    --------
    >>> formatter = SafePlainFormatter("%(levelname)s:%(message)s")
    >>> isinstance(formatter, logging.Formatter)
    True
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录并执行编码安全转换。

        Parameters
        ----------
        record : logging.LogRecord
            待格式化的日志记录对象。

        Returns
        -------
        str
            已经过控制台编码安全处理的日志文本。

        Examples
        --------
        >>> record = logging.LogRecord("mars", logging.INFO, __file__, 1, "ok", (), None)
        >>> SafePlainFormatter("%(message)s").format(record)
        'ok'
        """
        return _sanitize_console_text(super().format(record))


if HAS_COLORLOG:
    class SafeColoredFormatter(colorlog.ColoredFormatter):
        """
        面向彩色终端的安全格式化器。

        Attributes
        ----------
        _style : logging.PercentStyle
            底层 formatter 内部保存的格式化样式对象。

        Examples
        --------
        >>> formatter = SafeColoredFormatter("%(message)s")
        >>> isinstance(formatter, logging.Formatter)
        True
        """

        def format(self, record: logging.LogRecord) -> str:
            """
            格式化彩色日志记录并执行编码安全转换。

            Parameters
            ----------
            record : logging.LogRecord
                待格式化的日志记录对象。

            Returns
            -------
            str
                已经过控制台编码安全处理的彩色日志文本。

            Examples
            --------
            >>> record = logging.LogRecord("mars", logging.INFO, __file__, 1, "ok", (), None)
            >>> SafeColoredFormatter("%(message)s").format(record)
            'ok'
            """
            return _sanitize_console_text(super().format(record))

def get_mars_logger(level: int = logging.INFO) -> logging.Logger:
    """
    获取名为 ``mars`` 的全局日志记录器。

    首次配置时添加标准输出处理器；已存在可用处理器时直接返回，避免重复输出。

    Parameters
    ----------
    level : int
        首次配置时的日志过滤级别，默认 ``logging.INFO``。
        已有可用处理器时不修改级别；运行中修改级别请使用 ``set_log_level``。

    Returns
    -------
    logging.Logger
        配置完成的 Python Logger 对象。

    Notes
    -----
    使用 ``hasHandlers()`` 检查可用处理器，也会检查允许传播到的父级处理器。
    新建配置使用 ``propagate=False``，防止根日志记录器重复输出。
    日志格式为 ``[MARS] 时间 - 级别 - 消息``，安装 colorlog 时按级别着色。

    Examples
    --------
    >>> logger = get_mars_logger()
    >>> logger.name
    'mars'
    """
    # 获取指定名称的 logger 实例
    logger = logging.getLogger(LOGGER_NAME)

    # ---------------------------------------------------------
    # 1. 幂等性检查 (Idempotency Check)
    # ---------------------------------------------------------
    # hasHandlers() 也检查可传播到的父级处理器；已有可用配置时不重复添加。
    if logger.hasHandlers():
        return logger

    # 设置日志级别
    logger.setLevel(level)

    # ---------------------------------------------------------
    # 2. 处理器配置 (Handler Configuration)
    # ---------------------------------------------------------
    # 创建控制台处理器，输出到标准输出流 (stdout)
    console_handler = logging.StreamHandler(sys.stdout)

    # ---------------------------------------------------------
    # 3. 格式化器配置 (Formatter Configuration)
    # ---------------------------------------------------------
    formatter: logging.Formatter
    if HAS_COLORLOG:
        # 定义颜色映射方案：符合直觉的警示色
        log_colors: Dict[str, str] = {
            'DEBUG':    'cyan',      # 调试：青色
            'INFO':     'green',     # 信息：绿色
            'WARNING':  'yellow',    # 警告：黄色
            'ERROR':    'red',       # 错误：红色
            'CRITICAL': 'bold_red',  # 严重：加粗红
        }

        # colorlog 使用级别颜色包装日志文本。
        # %(log_color)s ... 切换颜色
        # %(reset)s ... 重置颜色
        formatter = SafeColoredFormatter(
            '%(log_color)s[MARS] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors=log_colors,
            reset=True,
            style='%'
        )
    else:
        # 回退方案：标准的无色格式化
        formatter = SafePlainFormatter(
            '[MARS] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    # 将格式化器应用到处理器
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(console_handler)

    # ---------------------------------------------------------
    # 4. 传播控制 (Propagation Control)
    # ---------------------------------------------------------
    # 禁止日志向上传播到 root logger。
    # 防止根日志记录器再次输出同一条消息。
    logger.propagate = False

    return logger

def set_log_level(level: Union[str, int]) -> None:
    """
    动态修改全局 Logger 的日志级别。

    允许在程序运行时切换日志详细程度，例如从 ``INFO`` 切换到 ``DEBUG``
    以便定位问题。

    Parameters
    ----------
    level : Union[str, int]
        目标日志级别。可以是 ``logging`` 常量，也可以是字符串形式的
        级别名称，例如 ``"DEBUG"``、``"info"`` 或 ``"WARNING"``。
        字符串忽略大小写，未知名称按 ``INFO`` 处理；同时更新现有处理器的级别。

    Returns
    -------
    None
        函数仅修改全局 logger 与 handler 的级别。

    Examples
    --------
    >>> from mars.utils.logger import set_log_level
    >>> set_log_level('DEBUG')  # 开启调试模式
    >>> set_log_level('INFO')   # 恢复正常模式
    """
    logger = logging.getLogger(LOGGER_NAME)

    # 如果传入的是字符串，尝试将其转换为 logging 的整数常量
    if isinstance(level, str):
        level_str = level.upper()
        # 映射表：字符串 -> logging 常量
        levels: Dict[str, int] = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        # 如果找不到匹配的字符串，默认回退到 INFO
        final_level = levels.get(level_str, logging.INFO)
    else:
        final_level = level

    # 修改 logger 自身的级别
    logger.setLevel(final_level)

    # 同时修改所有 Handlers 的级别，确保过滤规则立即生效
    # 有些情况下 handler 的级别可能比 logger 高，导致修改 logger 级别无效
    for handler in logger.handlers:
        handler.setLevel(final_level)

# ---------------------------------------------------------
# 模块级实例 (Module Level Instance)
# ---------------------------------------------------------
# 初始化一个默认 logger 实例。
# 其他模块可以直接 `from mars.utils.logger import logger` 使用，
# 无需每次都调用 get_mars_logger()。
logger: logging.Logger = get_mars_logger()
