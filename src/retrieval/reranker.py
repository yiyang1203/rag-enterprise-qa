"""
重排序器 (Reranker)

在检索阶段返回的 Top-K 候选块上，使用交叉编码器（Cross-Encoder）
进行精确重排序，提升答案相关块的排名。

技术选型：
  • 默认使用 BAAI/bge-reranker-base（中英双语）
  • 也支持通过 FlagEmbedding / HuggingFace 加载其他 reranker 模型

流程：
  query + candidate chunks → Cross-Encoder 逐对打分 → 按分数降序重排
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# 基类
# ═════════════════════════════════════════════════════════════════

class BaseReranker:
    """重排序器抽象基类。"""

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """对候选文档重排序。

        Parameters
        ----------
        query : str
            用户查询。
        documents : Sequence[Document]
            候选文档列表（通常来自检索器的 Top-K 结果）。
        top_k : int or None
            重排序后保留的文档数。None = 全部保留。

        Returns
        -------
        List[Document]
            按相关性降序排列的文档列表。
        """
        raise NotImplementedError


# ═════════════════════════════════════════════════════════════════
# BGE Reranker（基于 FlagEmbedding）
# ═════════════════════════════════════════════════════════════════

class BGEReranker(BaseReranker):
    """BGE Cross-Encoder 重排序器。

    使用 BAAI/bge-reranker 系列模型对 query-document 对进行联合编码打分。

    Parameters
    ----------
    model_name : str
        模型名称或本地路径，默认 ``BAAI/bge-reranker-base``。
    device : str or None
        推理设备。None 时自动选择。
    batch_size : int
        推理 batch size，默认 16。
    normalize : bool
        是否归一化分数，默认 True。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        device: Optional[str] = None,
        batch_size: int = 16,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            raise ImportError(
                "FlagEmbedding 未安装，请运行: pip install FlagEmbedding"
            )

        logger.info("Loading reranker model: %s ...", model_name)
        self._model = FlagReranker(
            model_name,
            use_fp16=True,
            device=device,
        )
        logger.info("Reranker loaded: %s", model_name)

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """对文档重排序。

        为每个 (query, doc) 对计算相关性分数，按分数降序排列。
        """
        if not documents:
            return []

        # 构建 (query, doc) 对
        pairs = [[query, doc.page_content] for doc in documents]

        # 批量打分
        scores = self._model.compute_score(
            pairs,
            batch_size=self.batch_size,
            normalize=self.normalize,
        )

        # 单结果时 scores 是标量
        if not isinstance(scores, list):
            scores = [scores]

        # 按分数降序排列
        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        # 将分数写入 metadata
        result: List[Document] = []
        for score, doc in scored_docs:
            new_meta = dict(doc.metadata)
            new_meta["rerank_score"] = float(score)
            result.append(Document(page_content=doc.page_content, metadata=new_meta))

        if len(documents) != len(result):
            logger.info(
                "Reranked: %d → %d documents (top_k=%s)",
                len(documents), len(result), top_k or "all",
            )

        return result


# ═════════════════════════════════════════════════════════════════
# HuggingFace Cross-Encoder（通用方案）
# ═════════════════════════════════════════════════════════════════

class CrossEncoderReranker(BaseReranker):
    """基于 HuggingFace CrossEncoder 的重排序器。

    适用于任意 sentence-transformers 兼容的 CrossEncoder 模型。

    Parameters
    ----------
    model_name : str
        HuggingFace 模型名，默认 ``cross-encoder/ms-marco-MiniLM-L-6-v2``。
    device : str or None
    batch_size : int
    max_length : int
        输入最大 token 数，默认 512。
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装，请运行: pip install sentence-transformers"
            )

        logger.info("Loading CrossEncoder: %s ...", model_name)
        self._model = CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )
        logger.info("CrossEncoder loaded: %s", model_name)

    def rerank(
        self,
        query: str,
        documents: Sequence[Document],
        top_k: Optional[int] = None,
    ) -> List[Document]:
        """对文档重排序。"""
        if not documents:
            return []

        pairs = [[query, doc.page_content] for doc in documents]
        scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

        if not isinstance(scores, list):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]

        scored_docs = list(zip(scores, documents))
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        if top_k is not None:
            scored_docs = scored_docs[:top_k]

        result: List[Document] = []
        for score, doc in scored_docs:
            new_meta = dict(doc.metadata)
            new_meta["rerank_score"] = score
            result.append(Document(page_content=doc.page_content, metadata=new_meta))

        return result


# ═════════════════════════════════════════════════════════════════
# 工厂函数
# ═════════════════════════════════════════════════════════════════

def create_reranker(
    backend: str = "bge",
    **kwargs,
) -> BaseReranker:
    """根据配置创建重排序器。

    Parameters
    ----------
    backend : str
        ``"bge"`` — BAAI/bge-reranker（推荐中文场景）
        ``"cross-encoder"`` — 通用 CrossEncoder
    **kwargs
        传递给具体实现。

    Returns
    -------
    BaseReranker
    """
    backend = backend.lower()
    if backend == "bge":
        return BGEReranker(**kwargs)
    elif backend == "cross-encoder":
        return CrossEncoderReranker(**kwargs)
    else:
        raise ValueError(
            f"不支持的重排序后端: '{backend}'，可选: bge / cross-encoder"
        )
