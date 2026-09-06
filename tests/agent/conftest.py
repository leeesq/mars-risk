"""Agent 的最低运行版本为 Python 3.10，旧版核心矩阵不收集本目录。"""

import sys

collect_ignore_glob = ["test_*.py"] if sys.version_info < (3, 10) else []
