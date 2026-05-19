"""
配置加载器 (Config Loader)

从 YAML 配置文件读取应用参数，支持：
  • 环境变量占位符 ``${VAR_NAME}`` 自动替换
  • 嵌套键的点号路径访问（如 ``llm.openai.model``）
  • 单例缓存，避免重复解析
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"

# 单例缓存
_cache: Dict[str, dict] = {}

# 环境变量占位符正则：${VAR_NAME} 或 ${VAR_NAME:default}
_RE_ENV_VAR = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


def _resolve_env(value: Any) -> Any:
    """递归解析值中的环境变量占位符。

    ``${VAR_NAME}`` → ``os.environ["VAR_NAME"]``
    ``${VAR_NAME:default}`` → ``os.environ.get("VAR_NAME", "default")``
    """
    if isinstance(value, str):
        def _replace(m: re.Match) -> str:
            var = m.group(1)
            default = m.group(2)
            if default is not None:
                return os.environ.get(var, default)
            return os.environ.get(var, "")

        # 全值匹配 "${FOO}" → 返回原始类型（None 等）
        m = re.fullmatch(_RE_ENV_VAR, value)
        if m:
            var = m.group(1)
            default = m.group(2)
            if default is not None:
                raw = os.environ.get(var)
                return raw if raw is not None else default
            return os.environ.get(var)

        return _RE_ENV_VAR.sub(_replace, value)

    elif isinstance(value, dict):
        return {k: _resolve_env(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_resolve_env(v) for v in value]
    return value


# ═════════════════════════════════════════════════════════════════
# 加载函数
# ═════════════════════════════════════════════════════════════════

def load_config(name: str = "app_config") -> dict:
    """加载并缓存 YAML 配置文件。

    Parameters
    ----------
    name : str
        配置文件名（不含 .yaml 扩展名），如 ``"app_config"`` 或 ``"prompt_templates"``。

    Returns
    -------
    dict
        解析后的配置字典，环境变量已替换。
    """
    if name in _cache:
        return _cache[name]

    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    config = _resolve_env(raw)
    _cache[name] = config
    return config


def get_config_value(key_path: str, default: Any = None, config_name: str = "app_config") -> Any:
    """按点号路径读取配置值。

    Parameters
    ----------
    key_path : str
        点号分隔的配置路径，如 ``"llm.openai.model"``。
    default : Any
        路径不存在时返回的默认值。
    config_name : str
        配置文件名。

    Returns
    -------
    Any
    """
    config = load_config(config_name)
    keys = key_path.split(".")
    for key in keys:
        if isinstance(config, dict) and key in config:
            config = config[key]
        else:
            return default
    return config


def reload_config(name: str = "app_config") -> dict:
    """强制重新加载配置文件（清除缓存）。

    Parameters
    ----------
    name : str
        配置文件名。

    Returns
    -------
    dict
    """
    _cache.pop(name, None)
    return load_config(name)
