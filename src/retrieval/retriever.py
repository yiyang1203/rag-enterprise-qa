"""
检索器 (Retriever)

基于向量存储的语义检索，支持：
  • 纯向量检索（相似度搜索）
  • 元数据过滤（按 source / file_type 等字段筛选）
  • 分数阈值过滤
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from src.ingestion.vectorizer import BaseVectorStore, load_vector_store
from src.utils.config_loader import get_config_value

logger = logging.getLogger(__name__)


class Retriever:
    """向量检索器。

    Parameters
    ----------
    vector_store : BaseVectorStore
        已初始化的向量存储实例。
    top_k : int
        默认返回的结果数。
    score_threshold : float or None
        最低相似度阈值。None 表示不过滤。
        注意：不同后端的分数度量不同（L2 距离 vs 余弦相似度）。
    """

    def __init__(
        self,
        vector_store: BaseVectorStore,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> None:
        self._store = vector_store
        self.top_k = top_k
        self.score_threshold = score_threshold

    # ── 检索接口 ──────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """检索与查询最相关的文档块。

        Parameters
        ----------
        query : str
            查询文本。
        top_k : int or None
            返回结果数。None 使用初始化默认值。
        filter : dict or None
            元数据过滤条件。

        Returns
        -------
        List[Document]
            按相关性降序排列的 Document 列表。
        """
        k = top_k if top_k is not None else self.top_k
        return self._store.similarity_search(query, k=k, filter=filter)

    def retrieve_with_scores(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """检索并返回相似度分数。

        Parameters
        ----------
        query : str
            查询文本。
        top_k : int or None
        filter : dict or None

        Returns
        -------
        List[Tuple[Document, float]]
            (文档, 分数) 列表。分数含义取决于后端。
        """
        k = top_k if top_k is not None else self.top_k
        results = self._store.similarity_search_with_score(query, k=k, filter=filter)

        # 分数阈值过滤
        if self.score_threshold is not None:
            results = [
                (doc, score)
                for doc, score in results
                if score >= self.score_threshold
            ]

        return results

    # ── 信息 ──────────────────────────────────────────────────

    @property
    def store_count(self) -> int:
        """向量库中的总文档数。"""
        return self._store.count()

    @property
    def vector_store(self) -> BaseVectorStore:
        """返回底层向量存储。"""
        return self._store


# ═════════════════════════════════════════════════════════════════
# 工厂函数
# ═════════════════════════════════════════════════════════════════

def create_retriever(
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> Retriever:
    """根据配置文件创建检索器。

    Parameters
    ----------
    top_k : int or None
        默认 Top-K。None 时从配置读取。
    score_threshold : float or None

    Returns
    -------
    Retriever
    """
    if top_k is None:
        top_k = get_config_value("retrieval.top_k", 5)
    if score_threshold is None:
        score_threshold = get_config_value("retrieval.score_threshold")

    backend = get_config_value("vector_store.backend", "chroma")
    persist_dir = get_config_value("vector_store.persist_dir", "data/vector_store")
    collection_name = get_config_value("vector_store.collection_name", "rag_enterprise_qa")
    index_path = get_config_value("vector_store.index_path", "data/vector_store/faiss_index")

    logger.info("Loading vector store: backend=%s persist_dir=%s", backend, persist_dir)
    store = load_vector_store(
        backend=backend,
        persist_dir=persist_dir,
        collection_name=collection_name,
        index_path=index_path,
    )

    return Retriever(
        vector_store=store,
        top_k=top_k,
        score_threshold=score_threshold,
    )
