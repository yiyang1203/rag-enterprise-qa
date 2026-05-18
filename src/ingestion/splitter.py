"""
文本分块器 (Text Splitter)

基于 LangChain RecursiveCharacterTextSplitter，将清洗后的文档
按语义边界切分为固定大小的块（chunk），用于后续向量化与检索。

特性：
  • 递归分割：优先按段落 → 句子 → 字符的优先级切分
  • chunk_size / chunk_overlap 均可配置
  • 保留原始 metadata 并为每个 chunk 附加序号
"""

import logging
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── 默认参数 ────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 500        # 每个 chunk 的目标最大字符数
DEFAULT_CHUNK_OVERLAP = 50      # 相邻 chunk 之间的重叠字符数


def split_documents(
    docs: List[Document],
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
    add_chunk_index: bool = True,
) -> List[Document]:
    """将文档列表切分为固定大小的块。

    Parameters
    ----------
    docs : List[Document]
        待切分的 Document 列表（通常已经过清洗）。
    chunk_size : int
        每个 chunk 的目标最大字符数，默认 500。
    chunk_overlap : int
        相邻 chunk 之间的重叠字符数，默认 50。
    separators : List[str] or None
        自定义分割符优先级列表。默认使用 LangChain 内置的
        ``["\\n\\n", "\\n", "。", ".", " ", ""]``。
    add_chunk_index : bool
        是否在每个 chunk 的 metadata 中附加 ``chunk_index``
        字段（从 0 开始计数），默认开启。

    Returns
    -------
    List[Document]
        切分后的 Document 列表。
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.split_documents(docs)

    if add_chunk_index:
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i

    logger.info(
        "Split %d documents → %d chunks (chunk_size=%d, overlap=%d)",
        len(docs), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def split_text(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    separators: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> List[Document]:
    """将单段文本切分为 Document 块。

    适用于只需切分纯文本、无需预先构建 Document 的场景。

    Parameters
    ----------
    text : str
        待切分的纯文本。
    chunk_size : int
        每个 chunk 的目标最大字符数。
    chunk_overlap : int
        相邻 chunk 之间的重叠字符数。
    separators : List[str] or None
        自定义分割符优先级列表。
    metadata : dict or None
        附加到每个 chunk 的元数据。

    Returns
    -------
    List[Document]
        切分后的 Document 列表。
    """
    if separators is None:
        separators = ["\n\n", "\n", "。", ".", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = splitter.create_documents([text], metadatas=[metadata or {}])

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info("Split text (%d chars) → %d chunks", len(text), len(chunks))
    return chunks
