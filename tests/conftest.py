"""pytest 全局前置 — 将项目根目录加入 sys.path。

这样无论从哪个目录运行 pytest，`src` 下的模块都能正常导入。
"""

import sys
from pathlib import Path

# 项目根 = conftest.py 所在目录的父目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
