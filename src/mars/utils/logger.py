import logging
import sys
from typing import Union, Optional, Dict

# 尝试导入 colorlog 以支持彩色日志输出
# 如果环境中未安装 colorlog，将自动回退到标准无色日志
try:
    import colorlog
    HAS_COLORLOG: bool = True
except ImportError:
    HAS_COLORLOG: bool = False

# 定义单例 Logger 的名称，确保整个项目使用的是同一个日志实例
LOGGER_NAME: str = "mars"

def get_mars_logger(level: int = logging.INFO) -> logging.Logger:
    """
    获取 MARS 框架专属的全局 Logger 实例（单例模式）。

    该函数负责创建和配置 logger，包括设置输出格式、颜色（如果可用）以及
    防止在交互式环境（如 Jupyter Notebook）中出现重复日志的问题。

    Parameters
    ----------
    level : int, optional
        日志的初始过滤级别 (例如 logging.INFO, logging.DEBUG)。
        默认值为 logging.INFO。

    Returns
    -------
    logging.Logger
        配置完成的 Python Logger 对象。

    Notes
    -----
    1. **单例模式**: 如果该名称的 Logger 已经存在且包含处理器，则直接返回，不重复添加 Handler。
    2. **独立性**: 设置 `propagate=False` 防止日志向上传播到根记录器，避免被其他库的配置污染。
    3. **样式**: 
       - 彩色模式: [MARS] 时间 - 级别 - 消息 (不同级别不同颜色)
       - 普通模式: [MARS] 时间 - 级别 - 消息
    """
    # 获取指定名称的 logger 实例
    logger = logging.getLogger(LOGGER_NAME)
    
    # ---------------------------------------------------------
    # 1. 幂等性检查 (Idempotency Check)
    # ---------------------------------------------------------
    # 如果 logger 已经有了处理器 (handlers)，说明已经被初始化过。
    # 直接返回以防止重复添加 Handler，这在 Jupyter Notebook 重复运行单元格时尤为重要。
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
    if HAS_COLORLOG:
        # 定义颜色映射方案：符合直觉的警示色
        log_colors: Dict[str, str] = {
            'DEBUG':    'cyan',      # 调试：青色
            'INFO':     'green',     # 信息：绿色
            'WARNING':  'yellow',    # 警告：黄色
            'ERROR':    'red',       # 错误：红色
            'CRITICAL': 'bold_red',  # 严重：加粗红
        }
        
        # 使用 colorlog 进行带颜色的格式化
        # %(log_color)s ... 切换颜色
        # %(reset)s ... 重置颜色
        formatter = colorlog.ColoredFormatter(
            '%(log_color)s[MARS] %(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors=log_colors,
            reset=True,
            style='%'
        )
    else:
        # 回退方案：标准的无色格式化
        formatter = logging.Formatter(
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
    # 这样可以防止 MARS 的日志被其他库（如 TensorFlow, PyTorch）的默认配置再次打印，
    # 确保日志输出的纯净性。
    logger.propagate = False

    return logger

def set_log_level(level: Union[str, int]) -> None:
    """
    动态修改全局 Logger 的日志级别。
    
    允许在程序运行时切换日志详细程度（例如从 INFO 切换到 DEBUG 以排查问题）。

    Parameters
    ----------
    level : Union[str, int]
        目标日志级别。
        可以是 logging 常量 (如 logging.DEBUG) 
        也可以是字符串 (如 'DEBUG', 'info', 'WARNING')。

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