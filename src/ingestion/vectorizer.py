"""
向量存储与持久化 (Vectorizer)

将嵌入后的 chunk 写入向量数据库，支持两种后端：
  • Chroma  — 开源向量数据库，自带持久化，适合中小规模场景
  • FAISS   — Meta 高效相似性搜索库，适合大规模 / 内存映射

功能：
  • 从空库创建索引（build）
  • 增量追加（add）
  • 相似性检索（search）
  • 元数据过滤（filter）
  • 持久化到磁盘
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore as LangChainVectorStore

if TYPE_CHECKING:
    from src.ingestion.embedder import BaseEmbedder

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
# 基类
# ═════════════════════════════════════════════════════════════════


class BaseVectorStore:
    """向量存储抽象基类。

    封装 LangChain VectorStore，提供统一的 build / add / search / persist 接口。
    """

    def __init__(self, store: LangChainVectorStore) -> None:
        self._store = store

    @property
    def native_store(self) -> LangChainVectorStore:
        """返回底层 LangChain VectorStore 实例。"""
        return self._store

    # ── 写入 ──────────────────────────────────────────────────

    def add_documents(
        self,
        docs: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """追加文档到向量库。

        Parameters
        ----------
        docs : List[Document]
            待入库的 Document 列表（page_content 会被向量化）。
        ids : List[str] or None
            文档 ID 列表；为 None 时自动生成 UUID。

        Returns
        -------
        List[str]
            入库文档的 ID 列表。
        """
        if not docs:
            return []
        logger.info("Adding %d documents to vector store ...", len(docs))
        result = self._store.add_documents(docs, ids=ids)
        logger.info("Added %d documents.", len(docs))
        return result

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """追加纯文本到向量库（绕过 Document 对象）。

        Parameters
        ----------
        texts : List[str]
            待入库文本。
        metadatas : List[dict] or None
            每条文本对应的元数据。
        ids : List[str] or None
            文档 ID。

        Returns
        -------
        List[str]
            入库文档的 ID 列表。
        """
        if not texts:
            return []
        logger.info("Adding %d texts to vector store ...", len(texts))
        result = self._store.add_texts(texts, metadatas=metadatas, ids=ids)
        logger.info("Added %d texts.", len(texts))
        return result

    # ── 检索 ──────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """相似性检索。

        Parameters
        ----------
        query : str
            查询文本。
        k : int
            返回结果数。
        filter : dict or None
            元数据过滤条件（Chroma 的 where 子句）。

        Returns
        -------
        List[Document]
            最相似的 k 个 Document。
        """
        return self._store.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """带相似度分数的检索。

        Returns
        -------
        List[Tuple[Document, float]]
            (Document, score) 列表。score 越小越相似（L2 距离）
            或越大越相似（余弦相似度），取决于后端。
        """
        return self._store.similarity_search_with_score(
            query, k=k, filter=filter
        )

    def similarity_search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """给定向量直接检索（绕过查询编码）。

        Parameters
        ----------
        embedding : List[float]
            查询向量。
        k : int
            返回结果数。
        filter : dict or None
            元数据过滤条件。

        Returns
        -------
        List[Document]
        """
        return self._store.similarity_search_by_vector(
            embedding, k=k, filter=filter
        )

    # ── 管理 ──────────────────────────────────────────────────

    def persist(self) -> None:
        """持久化到磁盘（FAISS 需要显式调用；Chroma 自动持久化）。"""
        if hasattr(self._store, "save_local"):
            self._store.save_local()
            logger.info("Vector store persisted to disk.")
        else:
            logger.debug("Backend does not require explicit persist.")

    def count(self) -> int:
        """返回当前库中的文档数。"""
        # 不同后端的计数方式不同
        try:
            if hasattr(self._store, "_collection"):
                return self._store._collection.count()  # Chroma
        except Exception:
            pass
        try:
            if hasattr(self._store, "index"):
                return self._store.index.ntotal  # FAISS
        except Exception:
            pass
        logger.debug("Unable to count documents for this backend.")
        return -1

    def delete(self, ids: List[str]) -> None:
        """按 ID 删除文档。"""
        if hasattr(self._store, "delete"):
            self._store.delete(ids=ids)
            logger.info("Deleted %d documents.", len(ids))
        else:
            raise NotImplementedError("Backend does not support deletion.")


# ═════════════════════════════════════════════════════════════════
# Chroma
# ═════════════════════════════════════════════════════════════════

class ChromaVectorStore(BaseVectorStore):
    """Chroma 向量存储。

    基于 LangChain Chroma 集成，数据自动持久化到磁盘目录。

    Parameters
    ----------
    persist_dir : str
        持久化目录路径。
    collection_name : str
        Collection 名称，默认 ``"rag_enterprise_qa"``。
    embedding_function : Callable
        嵌入函数，签名为 ``(List[str]) -> List[List[float]]``。
        通常来自 ``Embedder.embed_documents``。
    """

    def __init__(
        self,
        persist_dir: str,
        embedding_function: Callable[[List[str]], List[List[float]]],
        collection_name: str = "rag_enterprise_qa",
    ) -> None:
        try:
            import chromadb
            from langchain_chroma import Chroma
        except ImportError:
            raise ImportError(
                "langchain-chroma 未安装，请运行: pip install langchain-chroma chromadb"
            )

        self.persist_dir = str(Path(persist_dir).resolve())
        self.collection_name = collection_name
        self._embedding_function = embedding_function

        # 确保目录存在
        os.makedirs(self.persist_dir, exist_ok=True)

        # Chroma 持久化客户端
        self._client = chromadb.PersistentClient(path=self.persist_dir)

        # 嵌入函数适配器：将 callable 包装为 Chroma 兼容的 EmbeddingFunction
        self._ef = _ChromaEmbeddingFunction(embedding_function)

        # 获取或创建 collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._ef,
        )

        # 创建 LangChain Chroma wrapper
        self._langchain_store = Chroma(
            client=self._client,
            collection_name=collection_name,
            embedding_function=self._ef,
        )

        logger.info(
            "Chroma initialized: persist_dir=%s collection=%s count=%d",
            self.persist_dir, collection_name, self._collection.count(),
        )

        super().__init__(self._langchain_store)

    def count(self) -> int:
        return self._collection.count()

    def get_collection_stats(self) -> dict:
        """返回 collection 统计信息。"""
        return {
            "name": self.collection_name,
            "count": self._collection.count(),
            "persist_dir": self.persist_dir,
        }


class _ChromaEmbeddingFunction:
    """将我们的 ``embed_documents`` callable 适配为 Chroma 的 EmbeddingFunction 协议。"""

    def __init__(self, func: Callable[[List[str]], List[List[float]]]) -> None:
        self._func = func

    def __call__(self, input: List[str]) -> List[List[float]]:
        return self._func(input)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Chroma >= 0.5.x 要求的接口。"""
        return self._func(texts)

    def embed_query(self, text: str) -> List[float]:
        """Chroma >= 0.5.x 要求的接口。"""
        return self._func([text])[0]

    def name(self) -> str:
        """Chroma >= 0.6.x 要求的接口 — 返回嵌入函数标识名。"""
        return "rag_custom_embedder"


# ═════════════════════════════════════════════════════════════════
# FAISS
# ═════════════════════════════════════════════════════════════════

class FAISSVectorStore(BaseVectorStore):
    """FAISS 向量存储。

    基于 LangChain FAISS 集成，支持内存映射与磁盘持久化。

    Parameters
    ----------
    index_path : str
        FAISS 索引文件保存路径（不含扩展名，实际会生成 .faiss + .pkl）。
    embedding_function : Callable
        嵌入函数，签名为 ``(List[str]) -> List[List[float]]``。
    index_type : str or None
        FAISS 索引类型。None 时自动选择（数据量 < 10k 用 ``"IDMap,Flat"``
        即精确搜索，数据量大时使用 ``"IVF,SQ8"`` 近似搜索）。
    """

    def __init__(
        self,
        index_path: str,
        embedding_function: Callable[[List[str]], List[List[float]]],
        index_type: Optional[str] = None,
    ) -> None:
        try:
            from langchain_community.vectorstores import FAISS
        except ImportError:
            raise ImportError(
                "langchain-community 未安装，请运行: pip install langchain-community faiss-cpu"
            )

        self.index_path = str(Path(index_path).resolve())
        self._embedding_function = embedding_function
        # 确保父目录存在
        os.makedirs(str(Path(self.index_path).parent), exist_ok=True)

        # 尝试从磁盘加载已有索引
        faiss_file = self.index_path + ".faiss"
        pkl_file = self.index_path + ".pkl"

        if os.path.exists(faiss_file) and os.path.exists(pkl_file):
            logger.info("Loading existing FAISS index from %s ...", self.index_path)
            self._langchain_store = FAISS.load_local(
                folder_path=str(Path(self.index_path).parent),
                index_name=Path(self.index_path).name,
                embeddings=self._embedding_function,
                allow_dangerous_deserialization=True,
            )
            logger.info("FAISS index loaded. total=%d", self._langchain_store.index.ntotal)
        else:
            logger.info("Creating new FAISS index at %s ...", self.index_path)
            import faiss

            # 通过一条探测文本确定向量维度
            probe = embedding_function(["dimension probe"])[0]
            dim = len(probe)

            # 创建空 FAISS 索引
            if index_type is None:
                index = faiss.IndexFlatL2(dim)         # 精确检索
            elif index_type.upper() == "IVF,FLAT":
                # IVF 需要先训练，用占位数据；生产建议 Flat 起步
                quantizer = faiss.IndexFlatL2(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, 100)
                index.train(np.array([probe], dtype=np.float32))
            else:
                raise ValueError(f"Unsupported FAISS index_type: {index_type}")

            # 构建 LangChain FAISS wrapper（空 docstore）
            from langchain_community.docstore.in_memory import InMemoryDocstore

            self._langchain_store = FAISS(
                embedding_function=self._embedding_function,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={},
            )
            logger.info("FAISS index created. dim=%d", dim)

        super().__init__(self._langchain_store)

    # ── 检索 ──────────────────────────────────────────────────

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """FAISS 返回 L2 距离（越小越相似）。"""
        if filter:
            logger.warning("FAISS does not support metadata filtering; filter ignored.")
        return self._store.similarity_search_with_score(query, k=k)

    # ── 管理 ──────────────────────────────────────────────────

    def persist(self) -> None:
        """持久化 FAISS 索引到磁盘。"""
        self._langchain_store.save_local(
            folder_path=str(Path(self.index_path).parent),
            index_name=Path(self.index_path).name,
        )
        logger.info("FAISS index saved to %s", self.index_path)

    def count(self) -> int:
        return self._langchain_store.index.ntotal


# ═════════════════════════════════════════════════════════════════
# 工厂函数
# ═════════════════════════════════════════════════════════════════

def create_vector_store(
    backend: str = "chroma",
    embedding_function: Optional[Callable[[List[str]], List[List[float]]]] = None,
    persist_dir: str = "data/vector_store",
    collection_name: str = "rag_enterprise_qa",
    index_path: str = "data/vector_store/faiss_index",
) -> BaseVectorStore:
    """根据配置创建向量存储实例。

    Parameters
    ----------
    backend : str
        ``"chroma"`` 或 ``"faiss"``。
    embedding_function : Callable or None
        嵌入函数。为 None 时尝试从环境变量自动配置
        （OpenAI key 存在则用 OpenAI，否则抛异常）。
    persist_dir : str
        Chroma 持久化目录。
    collection_name : str
        Chroma collection 名称。
    index_path : str
        FAISS 索引文件路径（不含 .faiss/.pkl 扩展名）。

    Returns
    -------
    BaseVectorStore
    """
    if embedding_function is None:
        embedding_function = _auto_embedding_function()

    backend = backend.lower()
    if backend == "chroma":
        return ChromaVectorStore(
            persist_dir=persist_dir,
            embedding_function=embedding_function,
            collection_name=collection_name,
        )
    elif backend == "faiss":
        return FAISSVectorStore(
            index_path=index_path,
            embedding_function=embedding_function,
        )
    else:
        raise ValueError(
            f"不支持的向量存储后端: '{backend}'，可选: chroma / faiss"
        )


def _auto_embedding_function() -> Callable[[List[str]], List[List[float]]]:
    """尝试自动创建嵌入函数。"""
    from src.ingestion.embedder import create_embedder

    # 优先检测 OpenAI key
    if os.getenv("OPENAI_API_KEY"):
        backend = "openai"
    else:
        backend = "huggingface"

    logger.info("Auto-creating embedder with backend=%s", backend)
    embedder = create_embedder(backend=backend)
    return embedder.embed_documents


# ═════════════════════════════════════════════════════════════════
# 便捷函数：完整构建流程
# ═════════════════════════════════════════════════════════════════

def build_index_from_chunks(
    chunks: List[Document],
    *,
    embedder: "BaseEmbedder | None" = None,  # type: ignore[name-defined]
    vector_store: Optional[BaseVectorStore] = None,
    backend: str = "chroma",
    persist_dir: str = "data/vector_store",
    validate: bool = True,
    batch_size: int = 50,
) -> BaseVectorStore:
    """从 chunk 列表一键构建向量索引。

    这是最常用的顶层接口，串联了嵌入 → 验证 → 入库的完整流程。

    Parameters
    ----------
    chunks : List[Document]
        待入库的 chunk 列表（来自 splitter 的输出）。
    embedder : BaseEmbedder or None
        嵌入模型实例。为 None 时自动创建。
    vector_store : BaseVectorStore or None
        已有的向量存储实例。为 None 时自动创建。
    backend : str
        向量存储后端，``"chroma"`` 或 ``"faiss"``。
    persist_dir : str
        持久化目录（仅当 vector_store 为 None 时生效）。
    validate : bool
        是否在入库前逐条验证 chunk 向量化质量。
    batch_size : int
        每批入库的 chunk 数（避免一次性提交过多文本）。

    Returns
    -------
    BaseVectorStore
        已持久化的向量存储实例。
    """
    from src.ingestion.embedder import create_embedder, validate_chunks

    if embedder is None:
        embedder = create_embedder()

    # ── 验证 ──
    if validate:
        chunks = validate_chunks(embedder, chunks, strict=False, max_empty_ratio=0.05)
    else:
        # 至少过滤空文本
        chunks = [d for d in chunks if d.page_content and d.page_content.strip()]

    logger.info("Building index with %d chunks (backend=%s) ...", len(chunks), backend)

    # ── 创建 / 使用 vector_store ──
    if vector_store is None:
        vector_store = create_vector_store(
            backend=backend,
            embedding_function=embedder.embed_documents,
            persist_dir=persist_dir,
        )

    # ── 分批入库 ──
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        vector_store.add_documents(batch)
        logger.debug(
            "Batch %d/%d: %d chunks added.",
            i // batch_size + 1,
            (len(chunks) + batch_size - 1) // batch_size,
            len(batch),
        )

    # ── 持久化 ──
    vector_store.persist()

    logger.info(
        "Index build complete. store=%s count=%d",
        backend, vector_store.count(),
    )
    return vector_store


def load_vector_store(
    backend: str = "chroma",
    persist_dir: str = "data/vector_store",
    collection_name: str = "rag_enterprise_qa",
    index_path: str = "data/vector_store/faiss_index",
) -> BaseVectorStore:
    """加载已有的向量存储（仅读取，不重建）。

    Parameters
    ----------
    backend : str
        ``"chroma"`` 或 ``"faiss"``。
    persist_dir : str
        Chroma 持久化目录。
    collection_name : str
        Chroma collection 名称。
    index_path : str
        FAISS 索引路径。

    Returns
    -------
    BaseVectorStore
    """
    ef = _auto_embedding_function()
    return create_vector_store(
        backend=backend,
        embedding_function=ef,
        persist_dir=persist_dir,
        collection_name=collection_name,
        index_path=index_path,
    )
