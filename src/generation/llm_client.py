"""
LLM 客户端 (LLM Client)

统一的大模型调用接口，支持多种后端：
  • OpenAI / 兼容 API（GPT-4, GPT-4o-mini, 及任何 OpenAI-compatible 服务）
  • 百度文心 (Qianfan) — 通过 qianfan SDK
  • 阿里通义千问 (DashScope) — 通过 dashscope SDK
  • 本地 vLLM — 通过 OpenAI-compatible API

所有后端统一 ``generate(prompt) -> str`` 调用签名，
内置指数退避重试 + 超时控制。
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from src.utils.config_loader import get_config_value

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# 基类
# ═════════════════════════════════════════════════════════════════

class BaseLLMClient(ABC):
    """LLM 客户端抽象基类。"""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """生成回复。

        Parameters
        ----------
        prompt : str
            用户提示词（或完整的纯文本 prompt）。
        system : str or None
            系统消息。为 None 时使用默认值。
        temperature : float or None
            采样温度。None 使用配置默认值。
        max_tokens : int or None
            最大输出 token 数。None 使用配置默认值。

        Returns
        -------
        str
            模型回复文本。
        """
        ...

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """给定 Chat 消息列表生成回复。

        Parameters
        ----------
        messages : List[Dict[str, str]]
            ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``
        temperature : float or None
        max_tokens : int or None

        Returns
        -------
        str
        """
        # 默认实现：拼接为纯文本后调用 generate
        parts = []
        for msg in messages:
            prefix = msg["role"].capitalize() if msg["role"] != "assistant" else "Assistant"
            parts.append(f"{prefix}: {msg['content']}")
        return self.generate("\n\n".join(parts), temperature=temperature, max_tokens=max_tokens, **kwargs)


# ═════════════════════════════════════════════════════════════════
# 重试 + 超时装饰器
# ═════════════════════════════════════════════════════════════════

def _retry_with_backoff(
    func,
    max_retries: int = 3,
    initial_wait: float = 1.0,
    backoff_factor: float = 2.0,
):
    """同步重试装饰器（指数退避）。"""
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return func()
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = initial_wait * (backoff_factor ** (attempt - 1))
                logger.warning(
                    "LLM 调用失败 (attempt %d/%d)，%0.1fs 后重试: %s",
                    attempt, max_retries, wait, exc,
                )
                time.sleep(wait)
    raise RuntimeError(f"LLM 调用失败，已重试 {max_retries} 次") from last_exc


# ═════════════════════════════════════════════════════════════════
# OpenAI / 兼容客户端
# ═════════════════════════════════════════════════════════════════

class OpenAIClient(BaseLLMClient):
    """OpenAI Chat Completions API 客户端。

    同时兼容任何 OpenAI-compatible API（vLLM、Ollama、local.ai 等）。

    Parameters
    ----------
    model : str
        模型名称。
    api_key : str or None
    base_url : str or None
    temperature : float
    max_tokens : int
    timeout : int
        请求超时秒数。
    max_retries : int
        最大重试次数。
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai 未安装，请运行: pip install openai")

        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=timeout,
            max_retries=0,  # 我们自己管理重试
        )

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """调用 OpenAI Chat Completions API。"""
        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        return self.generate_with_messages(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

    def generate_with_messages(
        self,
        messages: List[Dict[str, str]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """直接发送 Chat 消息列表。"""
        _temp = temperature if temperature is not None else self.temperature
        _max_tok = max_tokens if max_tokens is not None else self.max_tokens

        def _call() -> str:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=_temp,
                max_tokens=_max_tok,
                **kwargs,
            )
            content = resp.choices[0].message.content
            if content is None:
                raise RuntimeError("模型返回了空内容")
            return content.strip()

        return _retry_with_backoff(_call, max_retries=self.max_retries)


# ═════════════════════════════════════════════════════════════════
# 百度文心 (Qianfan)
# ═════════════════════════════════════════════════════════════════

class QianfanClient(BaseLLMClient):
    """百度文心大模型客户端。

    通过 qianfan SDK 调用 ERNIE 系列模型。

    Parameters
    ----------
    model : str
        模型名称，默认 ``ernie-speed-128k``。
    api_key : str or None
    secret_key : str or None
    temperature : float
    max_output_tokens : int
    timeout : int
    max_retries : int
    """

    def __init__(
        self,
        model: str = "ernie-speed-128k",
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        try:
            import qianfan
        except ImportError:
            raise ImportError("qianfan 未安装，请运行: pip install qianfan")

        self._client = qianfan.ChatCompletion(
            model=model,
            ak=api_key or os.getenv("QIANFAN_ACCESS_KEY"),
            sk=secret_key or os.getenv("QIANFAN_SECRET_KEY"),
        )

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        _temp = temperature if temperature is not None else self.temperature
        _max_tok = max_tokens if max_tokens is not None else self.max_output_tokens

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _call() -> str:
            resp = self._client.do(
                messages=messages,
                temperature=_temp,
                max_output_tokens=_max_tok,
                timeout=self.timeout,
                **kwargs,
            )
            # qianfan 返回的 body 结构
            content = resp.get("result", "")
            if not content:
                raise RuntimeError(f"文心返回空结果: {resp}")
            return content.strip()

        return _retry_with_backoff(_call, max_retries=self.max_retries)


# ═════════════════════════════════════════════════════════════════
# 阿里通义千问 (DashScope)
# ═════════════════════════════════════════════════════════════════

class DashScopeClient(BaseLLMClient):
    """阿里通义千问客户端。

    通过 dashscope SDK 调用 Qwen 系列模型。

    Parameters
    ----------
    model : str
        模型名称，默认 ``qwen-plus``。
    api_key : str or None
    temperature : float
    max_tokens : int
    timeout : int
    max_retries : int
    """

    def __init__(
        self,
        model: str = "qwen-plus",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries

        try:
            import dashscope
        except ImportError:
            raise ImportError("dashscope 未安装，请运行: pip install dashscope")

        # 设置 API Key
        key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if key:
            dashscope.api_key = key

        self._dashscope = dashscope

    def generate(
        self,
        prompt: str,
        *,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        _temp = temperature if temperature is not None else self.temperature
        _max_tok = max_tokens if max_tokens is not None else self.max_tokens

        messages: List[Dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        def _call() -> str:
            from dashscope import Generation

            resp = Generation.call(
                model=self.model,
                messages=messages,
                temperature=_temp,
                max_tokens=_max_tok,
                result_format="message",
                **kwargs,
            )

            if resp.status_code != 200:
                raise RuntimeError(
                    f"DashScope 返回错误 (code={resp.status_code}): {resp.message}"
                )

            content = resp.output.choices[0].message.content
            if not content:
                raise RuntimeError("通义千问返回空内容")
            return content.strip()

        return _retry_with_backoff(_call, max_retries=self.max_retries)


# ═════════════════════════════════════════════════════════════════
# vLLM 客户端（通过 OpenAI-compatible API）
# ═════════════════════════════════════════════════════════════════

class VLLMClient(OpenAIClient):
    """本地 vLLM 客户端。

    vLLM 提供 OpenAI-compatible API，因此直接继承 OpenAIClient。

    Parameters
    ----------
    base_url : str
        vLLM 服务地址，默认 ``http://localhost:8000/v1``。
    model : str or None
        模型名；None 时使用 vLLM 服务端默认。
    temperature : float
    max_tokens : int
    timeout : int
    max_retries : int
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        timeout: int = 120,
        max_retries: int = 3,
    ) -> None:
        super().__init__(
            model=model or "vllm",
            api_key="not-needed",  # vLLM 本地服务不需要
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )


# ═════════════════════════════════════════════════════════════════
# 工厂函数
# ═════════════════════════════════════════════════════════════════

def create_llm_client(backend: Optional[str] = None, **overrides) -> BaseLLMClient:
    """根据配置创建 LLM 客户端实例。

    Parameters
    ----------
    backend : str or None
        ``"openai"`` / ``"qianfan"`` / ``"dashscope"`` / ``"vllm"``。
        None 时从配置文件 ``llm.backend`` 读取。
    **overrides
        覆盖配置文件中的参数。

    Returns
    -------
    BaseLLMClient
    """
    if backend is None:
        backend = get_config_value("llm.backend", "openai")

    backend = backend.lower()

    # 从配置文件读取该后端的默认参数
    cfg = get_config_value(f"llm.{backend}", {})
    if cfg is None:
        cfg = {}

    # overrides 优先
    params = {**cfg, **overrides}

    if backend == "openai":
        return OpenAIClient(
            model=params.get("model", "gpt-4o-mini"),
            api_key=params.get("api_key"),
            base_url=params.get("base_url"),
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
            timeout=params.get("timeout", 60),
            max_retries=params.get("max_retries", 3),
        )
    elif backend == "qianfan":
        return QianfanClient(
            model=params.get("model", "ernie-speed-128k"),
            api_key=params.get("api_key"),
            secret_key=params.get("secret_key"),
            temperature=params.get("temperature", 0.0),
            max_output_tokens=params.get("max_output_tokens", 2048),
            timeout=params.get("timeout", 60),
            max_retries=params.get("max_retries", 3),
        )
    elif backend == "dashscope":
        return DashScopeClient(
            model=params.get("model", "qwen-plus"),
            api_key=params.get("api_key"),
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
            timeout=params.get("timeout", 60),
            max_retries=params.get("max_retries", 3),
        )
    elif backend == "vllm":
        return VLLMClient(
            base_url=params.get("base_url", "http://localhost:8000/v1"),
            model=params.get("model"),
            temperature=params.get("temperature", 0.0),
            max_tokens=params.get("max_tokens", 2048),
            timeout=params.get("timeout", 120),
            max_retries=params.get("max_retries", 3),
        )
    else:
        raise ValueError(
            f"不支持的 LLM 后端: '{backend}'，可选: openai / qianfan / dashscope / vllm"
        )
