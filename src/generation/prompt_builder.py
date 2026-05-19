"""
提示词构建器 (Prompt Builder)

将用户问题与检索到的 Top-K 文本块拼接为 LLM 提示词。

功能：
  • 基于 YAML 模板（LangChain PromptTemplate）
  • 多场景模板：精确查找 / 步骤指导 / 模糊匹配
  • Token 超限时自动截断上下文（保留最相关的 chunk）
  • 格式化为 OpenAI Chat Completions 消息格式
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.utils.config_loader import get_config_value

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
# Token 估算
# ═════════════════════════════════════════════════════════════════

# 全局 tiktoken 编码器（惰性初始化）
_encoding = None


def _get_encoding():
    """获取 tiktoken 编码器（惰性加载）。"""
    global _encoding
    if _encoding is not None:
        return _encoding
    try:
        import tiktoken
        encoding_name = get_config_value("token_limits.encoding_model", "cl100k_base")
        _encoding = tiktoken.get_encoding(encoding_name)
        logger.debug("tiktoken encoding loaded: %s", encoding_name)
    except Exception:
        logger.debug("tiktoken 不可用，将使用字符估算")
        _encoding = False  # sentinel — 不可用
    return _encoding


def estimate_tokens(text: str) -> int:
    """估算文本的 token 数。

    优先使用 tiktoken，不可用时回退到字符数 / chars_per_token 估算。
    """
    enc = _get_encoding()
    if enc and enc is not False:
        return len(enc.encode(text))
    # 回退：字符数估算
    ratio = get_config_value("token_limits.chars_per_token_estimate", 2.5)
    return max(1, int(len(text) / ratio))


# ═════════════════════════════════════════════════════════════════
# 上下文截断
# ═════════════════════════════════════════════════════════════════

def truncate_context(
    chunks: Sequence[Document],
    max_tokens: int,
    *,
    reserve_tokens: int = 500,
) -> List[Document]:
    """当上下文 chunk 超出 token 预算时，从尾部开始丢弃。

    默认认为 chunks 已按相似度降序排列（最相关的在前），
    因此从尾部丢弃能保留最相关的检索结果。

    Parameters
    ----------
    chunks : Sequence[Document]
        检索到的 Document 列表（按相关性降序）。
    max_tokens : int
        上下文的最大 token 预算。
    reserve_tokens : int
        预留给问题 + 系统提示词的 token 数。上下文实际可用
        ``max_tokens - reserve_tokens``。

    Returns
    -------
    List[Document]
        截断后的 chunk 列表（可能少于输入）。
    """
    budget = max_tokens - reserve_tokens
    if budget <= 0:
        logger.warning("Token 预算不足以容纳任何上下文 (budget=%d)", budget)
        return []

    kept: List[Document] = []
    used = 0

    for doc in chunks:
        doc_tokens = estimate_tokens(doc.page_content)
        if used + doc_tokens <= budget:
            kept.append(doc)
            used += doc_tokens
        else:
            # 尝试截断单篇过长的 chunk
            remaining = budget - used
            if remaining > 50:  # 至少保留 50 token
                truncated_text = _truncate_text_to_tokens(doc.page_content, remaining)
                kept.append(
                    Document(
                        page_content=truncated_text,
                        metadata=dict(doc.metadata),
                    )
                )
                used += remaining
            break  # 后续 chunk 不再添加

    dropped = len(chunks) - len(kept)
    if dropped:
        logger.info(
            "Token 预算限制：保留 %d/%d chunks，丢弃 %d (budget=%d, used=%d)",
            len(kept), len(chunks), dropped, budget, used,
        )
    return kept


def _truncate_text_to_tokens(text: str, max_tokens: int) -> str:
    """将文本截断到指定 token 数以内。

    优先用 tiktoken 精确截断；不可用时用字符比例估算。
    """
    enc = _get_encoding()
    if enc and enc is not False:
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])

    # 字符估算
    ratio = get_config_value("token_limits.chars_per_token_estimate", 2.5)
    max_chars = int(max_tokens * ratio)
    return text[:max_chars]


# ═════════════════════════════════════════════════════════════════
# 上下文格式化
# ═════════════════════════════════════════════════════════════════

def format_context(
    chunks: Sequence[Document],
    *,
    include_metadata: bool = True,
    max_chunks: Optional[int] = None,
) -> str:
    """将检索到的 chunk 列表格式化为 LLM 可读的上下文字符串。

    格式::

        [来源 1] 文件: xxx | 标题: xxx
        chunk 内容...

        [来源 2] 文件: xxx
        chunk 内容...

    Parameters
    ----------
    chunks : Sequence[Document]
        检索到的 Document 列表。
    include_metadata : bool
        是否在每条来源前附加文件/标题等元数据。
    max_chunks : int or None
        最多使用的 chunk 数量。None 表示全部使用。

    Returns
    -------
    str
        格式化后的上下文字符串。
    """
    if max_chunks is not None:
        chunks = chunks[:max_chunks]

    parts: List[str] = []
    for i, doc in enumerate(chunks, 1):
        meta = doc.metadata
        header_parts = [f"[来源 {i}]"]

        if include_metadata:
            source = meta.get("source", "")
            if source:
                # 只取文件名，路径太长
                import os
                fname = os.path.basename(source)
                header_parts.append(f"文件: {fname}")

            title = meta.get("title", "")
            if title:
                header_parts.append(f"标题: {title}")

            chunk_idx = meta.get("chunk_index")
            if chunk_idx is not None:
                header_parts.append(f"段落: {chunk_idx}")

        header = " | ".join(header_parts)
        content = doc.page_content.strip()
        parts.append(f"{header}\n{content}")

    return "\n\n".join(parts)


# ═════════════════════════════════════════════════════════════════
# 提示词构建
# ═════════════════════════════════════════════════════════════════

def build_prompt(
    question: str,
    chunks: Sequence[Document],
    *,
    scenario: str = "default",
    max_context_tokens: Optional[int] = None,
    include_metadata: bool = True,
    extra_vars: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """构建完整的 Chat 提示词。

    这是主入口，串联了 格式化 → 截断 → 模板填充 的完整流程。

    Parameters
    ----------
    question : str
        用户问题。
    chunks : Sequence[Document]
        检索到的 Document 列表（按相关性降序）。
    scenario : str
        场景模板名：``"default"`` / ``"exact_lookup"`` / ``"howto"`` / ``"fuzzy"``。
    max_context_tokens : int or None
        上下文 token 上限。None 时从配置文件读取。
    include_metadata : bool
        是否在上下文中包含来源元数据。
    extra_vars : dict or None
        额外的模板变量（如对话历史）。

    Returns
    -------
    List[Dict[str, str]]
        OpenAI Chat Completions 格式的消息列表：
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``
    """
    from src.utils.config_loader import load_config

    # ── 加载模板 ──
    templates = load_config("prompt_templates")
    scenario_templates = templates.get("scenarios", {}).get(scenario)
    if scenario_templates:
        system_tmpl = scenario_templates["system"]
        user_tmpl = scenario_templates["user"]
        logger.debug("Using scenario template: %s", scenario)
    else:
        default_tmpl = templates.get("default", {})
        system_tmpl = default_tmpl.get("system", "You are a helpful assistant.")
        user_tmpl = default_tmpl.get("user", "Context:\n{context}\n\nQuestion: {question}")
        logger.debug("Using default template (scenario '%s' not found)", scenario)

    # ── 格式化上下文 ──
    context_text = format_context(chunks, include_metadata=include_metadata)

    # ── Token 截断 ──
    if max_context_tokens is None:
        max_context_tokens = get_config_value("token_limits.max_context_tokens", 6000)

    # 计算系统 + 用户模板（不含上下文）的 token 占用
    template_overhead = estimate_tokens(system_tmpl + user_tmpl)
    # 额外留 300 token 给 question 本身
    reserve = template_overhead + 300

    context_tokens = estimate_tokens(context_text)
    if context_tokens + reserve > max_context_tokens:
        logger.info(
            "上下文超限 (%d + %d > %d)，执行截断...",
            context_tokens, reserve, max_context_tokens,
        )
        truncated_chunks = truncate_context(chunks, max_context_tokens, reserve_tokens=reserve)
        context_text = format_context(truncated_chunks, include_metadata=include_metadata)
    else:
        truncated_chunks = list(chunks)

    # ── 构建模板变量 ──
    template_vars: Dict[str, Any] = {
        "context": context_text,
        "question": question,
    }
    if extra_vars:
        template_vars.update(extra_vars)

    # ── 填充模板 ──
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_tmpl),
        ("user", user_tmpl),
    ])

    messages = chat_prompt.format_messages(**template_vars)

    # 转为 dict 列表
    result = [
        {"role": msg.type if msg.type != "human" else "user", "content": str(msg.content)}
        for msg in messages
    ]

    logger.info(
        "Prompt built: scenario=%s chunks_used=%d/%d tokens_est=%d",
        scenario,
        len(truncated_chunks),
        len(chunks),
        sum(estimate_tokens(m["content"]) for m in result),
    )

    return result


def build_prompt_text(
    question: str,
    chunks: Sequence[Document],
    **kwargs,
) -> str:
    """构建纯文本提示词（用于非 Chat 模型）。

    将 system + user 消息拼接为一个文本块。

    Parameters
    ----------
    question : str
        用户问题。
    chunks : Sequence[Document]
        检索到的 Document 列表。
    **kwargs
        传递给 ``build_prompt`` 的其他参数。

    Returns
    -------
    str
        纯文本提示词。
    """
    messages = build_prompt(question, chunks, **kwargs)
    parts: List[str] = []
    for msg in messages:
        if msg["role"] == "system":
            parts.append(f"System: {msg['content']}")
        elif msg["role"] == "user":
            parts.append(f"User: {msg['content']}")
    return "\n\n".join(parts)


def detect_scenario(question: str) -> str:
    """根据问题特征自动判断场景模板。

    Heuristics
    ---------
    - 包含 "多少" / "功率" / "参数" / "规格" → ``exact_lookup``
    - 包含 "怎么" / "如何" / "步骤" / "流程" → ``howto``
    - 包含 "类似" / "相关" / "大致" / "大概" → ``fuzzy``
    - 否则 → ``default``

    Parameters
    ----------
    question : str
        用户问题。

    Returns
    -------
    str
        场景名。
    """
    q = question.lower()

    exact_keywords = ["多少", "功率", "参数", "规格", "尺寸", "重量", "电压", "型号"]
    howto_keywords = ["怎么", "如何", "步骤", "流程", "操作", "配置", "设置", "安装"]
    fuzzy_keywords = ["类似", "相关", "大致", "大概", "差不多", "相近", "相似"]

    for kw in exact_keywords:
        if kw in q:
            return "exact_lookup"
    for kw in howto_keywords:
        if kw in q:
            return "howto"
    for kw in fuzzy_keywords:
        if kw in q:
            return "fuzzy"

    return "default"
