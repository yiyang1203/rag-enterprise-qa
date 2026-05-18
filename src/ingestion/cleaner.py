"""
文本清洗器 (Text Cleaner)

去除文档加载后常见的噪声：
  • 纯分隔线（====, ----, ****, #### 等）
  • 页眉/页脚横线
  • 过度空白行（3+ 连续换行 → 2 个）
  • 行首/行尾空白
  • 空文档 / 噪声页（PDF 的纯标题页等）

同时提供 LangChain Document 粒度的清洗接口。
"""

import re
import logging
from typing import List, Optional

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── 噪声行正则（行首到行尾仅包含这些字符） ──────────────────────
# 匹配整行都是重复分隔符的情况
_RE_PURE_SEPARATOR = re.compile(
    r"^\s*"           # 行首可选空白
    r"("
    r"={5,}"          # ===== 及以上
    r"|-{5,}"          # ----- 及以上
    r"|\*{5,}"         # ***** 及以上
    r"|#{5,}"          # ##### 及以上
    r"|~{5,}"          # ~~~~~ 及以上
    r"|_{5,}"           # _____ 及以上
    r")\s*$"
)

# 匹配整行只有空白
_RE_BLANK_LINE = re.compile(r"^\s*$")

# 纯页码残留（单独一行的数字，1-4 位）
_RE_PAGE_NUMBER = re.compile(r"^\s*\d{1,4}\s*$")

# 常见页眉/页脚短语（整行匹配）
_HEADER_FOOTER_LINES = {
    "Confidential", "Internal Use Only", "Draft", "机密", "内部资料",
    "版权所有", "All Rights Reserved", "Copyright", "©",
}


# ── 行级清洗 ───────────────────────────────────────────────────

def _is_noise_line(line: str) -> bool:
    """判断单行是否为噪声（应被移除）。

    注意：空白行不在此处处理，交由后续 collapse_blanks 逻辑统一管理。
    """
    stripped = line.strip()

    if not stripped:
        return False  # 空白行由 collapse 阶段处理

    if _RE_PURE_SEPARATOR.match(line):
        return True

    if _RE_PAGE_NUMBER.match(line):
        return True

    if stripped in _HEADER_FOOTER_LINES:
        return True

    return False


def clean_text(text: str, *, collapse_blanks: bool = True) -> str:
    """清洗纯文本，返回干净的字符串。

    Parameters
    ----------
    text : str
        待清洗的原始文本。
    collapse_blanks : bool
        是否将 3+ 连续空白行压缩为 2 个（默认开启）。

    Returns
    -------
    str
        清洗后的文本。
    """
    lines = text.split("\n")

    # 1. 过滤噪声行
    cleaned_lines = [line for line in lines if not _is_noise_line(line)]

    # 2. 去除各行首尾空白
    cleaned_lines = [line.strip() for line in cleaned_lines]

    # 3. 压缩连续空白行
    if collapse_blanks:
        result: List[str] = []
        blank_count = 0
        for line in cleaned_lines:
            if not line:
                blank_count += 1
                if blank_count <= 2:
                    result.append(line)
            else:
                blank_count = 0
                result.append(line)
        cleaned_lines = result

    # 4. 去除首尾空白行
    while cleaned_lines and not cleaned_lines[0]:
        cleaned_lines.pop(0)
    while cleaned_lines and not cleaned_lines[-1]:
        cleaned_lines.pop()

    return "\n".join(cleaned_lines)


# ── 文档级清洗 ──────────────────────────────────────────────────

def is_empty_or_noise(text: str, *, min_content_chars: int = 10) -> bool:
    """判断清洗后的文本是否为空或纯噪声。

    Parameters
    ----------
    text : str
        清洗前的文本（内部会先清洗再判断）。
    min_content_chars : int
        有效内容的最小字符数阈值。

    Returns
    -------
    bool
        True 表示该文本可丢弃。
    """
    cleaned = clean_text(text)
    # 去除空格后计数
    meaningful = re.sub(r"\s+", "", cleaned)
    return len(meaningful) < min_content_chars


def clean_document(doc: Document, *, min_content_chars: int = 10) -> Optional[Document]:
    """清洗单个 LangChain Document。

    Parameters
    ----------
    doc : Document
        待清洗的 Document。
    min_content_chars : int
        清洗后有效内容低于此阈值则返回 None。

    Returns
    -------
    Optional[Document]
        清洗后的 Document；若内容为空则返回 None。
    """
    original = doc.page_content
    cleaned = clean_text(original)
    meaningful = re.sub(r"\s+", "", cleaned)

    if len(meaningful) < min_content_chars:
        logger.debug(
            "Discarding empty/noise doc: source=%s chars_before=%d chars_after=%d",
            doc.metadata.get("source", "?"), len(original), len(meaningful),
        )
        return None

    return Document(page_content=cleaned, metadata=dict(doc.metadata))


def clean_documents(
    docs: List[Document],
    *,
    min_content_chars: int = 10,
) -> List[Document]:
    """批量清洗 LangChain Document 列表。

    行为：
      1. 对每个 doc 做行级清洗
      2. 丢弃清洗后有效字符 < ``min_content_chars`` 的文档
      3. 保留原 metadata

    Parameters
    ----------
    docs : List[Document]
        待清洗的 Document 列表。
    min_content_chars : int
        最小有效字符数阈值。

    Returns
    -------
    List[Document]
        清洗后的 Document 列表（顺序保持）。
    """
    cleaned: List[Document] = []
    dropped = 0
    for doc in docs:
        result = clean_document(doc, min_content_chars=min_content_chars)
        if result is not None:
            cleaned.append(result)
        else:
            dropped += 1

    if dropped:
        logger.info("Dropped %d empty/noise documents (threshold=%d chars)",
                     dropped, min_content_chars)
    logger.info("Cleaned: %d → %d documents", len(docs), len(cleaned))
    return cleaned
