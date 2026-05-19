"""
统一日志配置 (Logger)

提供项目级的 logging 配置，支持：
  • 控制台彩色输出（按级别着色）
  • 文件日志轮转（按天切割，保留 30 天）
  • 模块级 logger 获取
  • 第三方库日志抑制
"""

from __future__ import annotations

import logging
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


# ANSI 颜色代码（控制台）
_COLORS = {
    logging.DEBUG:    "\033[36m",      # 青色
    logging.INFO:     "\033[32m",      # 绿色
    logging.WARNING:  "\033[33m",      # 黄色
    logging.ERROR:    "\033[31m",      # 红色
    logging.CRITICAL: "\033[1;31m",    # 亮红
}
_RESET = "\033[0m"


class _ColoredFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{_RESET}"
        record.name = f"\033[1m{record.name}\033[0m"
        return super().format(record)


# ── 默认格式 ──────────────────────────────────────────────────────
CONSOLE_FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
CONSOLE_DATE_FMT = "%H:%M:%S"

FILE_FMT = "%(asctime)s [%(levelname)s] %(name)s (%(filename)s:%(lineno)d): %(message)s"
FILE_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# ── 已初始化标记 ──────────────────────────────────────────────────
_initialized = False


def setup_logging(
    *,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    file_level: Optional[int] = None,
    suppress_libs: bool = True,
) -> None:
    """配置项目级日志。

    多次调用时只生效第一次（幂等）。

    Parameters
    ----------
    level : int
        控制台日志级别，默认 INFO。
    log_file : str or None
        日志文件路径。设置后自动启用按天轮转。
    console : bool
        是否输出到控制台，默认 True。
    file_level : int or None
        文件日志级别。None 时与 level 相同。
    suppress_libs : bool
        是否抑制第三方库的 DEBUG/INFO 日志，默认 True。
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # root 最宽，由 handler 控制细度

    # ── 清除已有 handler ──
    root.handlers.clear()

    # ── 控制台 handler ──
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(level)
        console_handler.setFormatter(_ColoredFormatter(CONSOLE_FMT, CONSOLE_DATE_FMT))
        root.addHandler(console_handler)

    # ── 文件 handler ──
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = TimedRotatingFileHandler(
            filename=str(log_path),
            when="midnight",
            interval=1,
            backupCount=30,
            encoding="utf-8",
        )
        file_handler.setLevel(file_level if file_level is not None else level)
        file_handler.setFormatter(logging.Formatter(FILE_FMT, FILE_DATE_FMT))
        # 文件名后缀格式：log.2024-01-15
        file_handler.suffix = "%Y-%m-%d"
        root.addHandler(file_handler)

    # ── 抑制第三方库日志 ──
    if suppress_libs:
        for lib in (
            "chromadb", "urllib3", "httpx", "openai",
            "sentence_transformers", "faiss", "langchain",
            "langchain_core", "langchain_community", "langchain_chroma",
        ):
            logging.getLogger(lib).setLevel(logging.WARNING)

    # 项目自身设为 DEBUG
    logging.getLogger("src").setLevel(logging.DEBUG)

    root.info("Logging initialized (level=%s)", logging.getLevelName(level))


def get_logger(name: str) -> logging.Logger:
    """获取命名 logger（等价 logging.getLogger，但确保根配置已存在）。

    Parameters
    ----------
    name : str
        Logger 名称，通常传 ``__name__``。

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)
