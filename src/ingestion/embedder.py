"""
嵌入模型封装 (Embedder)

提供统一的嵌入向量化接口，支持两种后端：
  • OpenAI Embedding API  — text-embedding-3-small / text-embedding-3-large / text-embedding-ada-002
  • 本地 HuggingFace 模型  — 任意 sentence-transformers 兼容模型

每条 chunk 入库前通过 ``validate_chunks()`` 进行逐条验证，
确保向量非空、维度一致且无 NaN/Inf。
"""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

import numpy as np
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════
# 基类
# ═════════════════════════════════════════════════════════════════


class BaseEmbedder(ABC):
    """嵌入模型抽象基类。

    所有嵌入后端必须实现 ``embed_documents`` 和 ``embed_query``。
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """将一批文本向量化。

        Parameters
        ----------
        texts : List[str]
            待嵌入的文本列表，每条一个字符串。

        Returns
        -------
        List[List[float]]
            与输入一一对应的向量列表，每个向量为 float 列表。
        """
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """将单条查询文本向量化。

        Parameters
        ----------
        text : str
            查询文本。

        Returns
        -------
        List[float]
            查询向量。
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度。"""
        ...


# ═════════════════════════════════════════════════════════════════
# OpenAI Embedder
# ═════════════════════════════════════════════════════════════════

# 已知模型的维度（避免每次调用 API 查询）
_OPENAI_DIMENSIONS: dict[str, int] = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002":  1536,
}

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI Embedding API 封装。

    Parameters
    ----------
    model : str
        模型名称，默认 ``text-embedding-3-small``。
    api_key : str or None
        API Key；未提供时从环境变量 ``OPENAI_API_KEY`` 读取。
    base_url : str or None
        自定义 API 端点（用于代理或兼容接口）。
    dimensions : int or None
        仅对 text-embedding-3 系列生效，缩减输出维度。
    batch_size : int
        单次 API 调用的最大文本条数，默认 100。
    max_retries : int
        失败重试次数，默认 3。
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        batch_size: int = 100,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._dimensions_override = dimensions

        # 延迟导入，避免 openai 未安装时阻塞整个模块
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai 未安装，请运行: pip install openai"
            )

        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "未设置 OpenAI API Key。请设置环境变量 OPENAI_API_KEY，"
                "或传入 api_key 参数，或切换到本地模型 (--embedding-backend huggingface)"
            )

        self._client = OpenAI(
            api_key=resolved_key,
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
            timeout=10,  # 连接超时 10 秒，快速失败
            max_retries=0,
        )

        # 确定维度（优先从已知表查找，避免不必要的 API 调用）
        if dimensions and model.startswith("text-embedding-3"):
            self._dim = dimensions
        elif model in _OPENAI_DIMENSIONS:
            self._dim = _OPENAI_DIMENSIONS[model]
        else:
            self._dim = self._probe_dimension()

    # ── 公共接口 ──────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档文本。

        自动分批 + 重试；返回的向量顺序与输入严格一致。
        """
        if not texts:
            return []

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            vectors = self._embed_batch_with_retry(batch)
            all_vectors.extend(vectors)

        return all_vectors

    def embed_query(self, text: str) -> List[float]:
        """向量化单条查询。"""
        vectors = self.embed_documents([text])
        return vectors[0]

    @property
    def dimension(self) -> int:
        return self._dim

    # ── 内部 ──────────────────────────────────────────────────

    def _embed_batch_with_retry(self, batch: List[str]) -> List[List[float]]:
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self._client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=(
                        self._dimensions_override
                        if self.model.startswith("text-embedding-3")
                        else None
                    ),
                )
                # 按 index 排序后提取向量
                sorted_data = sorted(resp.data, key=lambda d: d.index)
                return [d.embedding for d in sorted_data]
            except Exception as exc:
                last_exc = exc
                # 连接超时/网络错误不重试，直接抛出
                msg = str(exc).lower()
                if any(kw in msg for kw in ("timed out", "connect", "name resolution", "refused")):
                    raise ConnectionError(
                        f"无法连接 OpenAI API。请检查网络，或切换到本地模型："
                        f"python scripts/build_index.py --embedding-backend huggingface"
                    ) from exc
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Embedding API 调用失败 (attempt %d/%d)，%ds 后重试: %s",
                        attempt, self.max_retries, wait, exc,
                    )
                    time.sleep(wait)
        raise RuntimeError(
            f"Embedding API 调用失败，已重试 {self.max_retries} 次"
        ) from last_exc

    def _probe_dimension(self) -> int:
        """对未知模型发一次请求以获取向量维度。"""
        try:
            resp = self._client.embeddings.create(
                model=self.model,
                input=["dimension probe"],
            )
            dim = len(resp.data[0].embedding)
            logger.info("Probed embedding dimension for %s: %d", self.model, dim)
            return dim
        except Exception as exc:
            # 连接错误直接抛出，不要静默回退
            if "timed out" in str(exc).lower() or "connect" in str(exc).lower():
                raise ConnectionError(
                    f"无法连接 OpenAI API ({self.model}): {exc}"
                ) from exc
            logger.warning(
                "无法探测模型 '%s' 的维度，回退到默认 1536: %s", self.model, exc
            )
            return 1536


# ═════════════════════════════════════════════════════════════════
# HuggingFace / 本地 Embedder
# ═════════════════════════════════════════════════════════════════

class HuggingFaceEmbedder(BaseEmbedder):
    """本地 HuggingFace / sentence-transformers 嵌入模型。

    Parameters
    ----------
    model_name : str
        HuggingFace 模型名或本地路径，默认
        ``BAAI/bge-small-zh-v1.5``（中英双语小模型）。
    device : str or None
        推理设备（``cpu``, ``cuda``, ``cuda:0``...）。
        默认自动选择。
    normalize : bool
        是否对输出向量做 L2 归一化，默认 True。
    batch_size : int
        推理时的 batch size，默认 32。
    show_progress : bool
        是否显示 tqdm 进度条，默认 False。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        device: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> None:
        self.model_name = model_name
        self.normalize = normalize
        self.batch_size = batch_size
        self.show_progress = show_progress

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers 未安装，请运行: "
                "pip install sentence-transformers"
            )

        logger.info("Loading embedding model: %s ...", model_name)
        self._model = SentenceTransformer(
            model_name,
            device=device,
        )
        self._dim = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Model loaded. dimension=%d device=%s",
            self._dim,
            str(self._model.device),
        )

    # ── 公共接口 ──────────────────────────────────────────────

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化文档文本。"""
        if not texts:
            return []

        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        # numpy → list[float]
        return [vec.tolist() for vec in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """向量化单条查询。

        BGE 系列模型对查询建议添加 instruction prefix；
        其他模型直接编码。
        """
        if "bge" in self.model_name.lower():
            # BGE 模型推荐：为查询添加 "Represent this sentence for searching relevant passages: "
            if not text.startswith("Represent this sentence"):
                text = f"为这个句子生成表示以用于检索相关文章：{text}"
        vec = self._model.encode(
            text,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return vec.tolist()

    @property
    def dimension(self) -> int:
        return self._dim


# ═════════════════════════════════════════════════════════════════
# 工厂函数
# ═════════════════════════════════════════════════════════════════

def create_embedder(
    backend: str = "openai",
    **kwargs,
) -> BaseEmbedder:
    """根据配置创建嵌入模型实例。

    Parameters
    ----------
    backend : str
        ``"openai"`` 或 ``"huggingface"`` / ``"local"``。
    **kwargs
        传递给具体实现类的参数。

    Returns
    -------
    BaseEmbedder
    """
    backend = backend.lower()
    if backend == "openai":
        return OpenAIEmbedder(**kwargs)
    elif backend in ("huggingface", "local"):
        return HuggingFaceEmbedder(**kwargs)
    else:
        raise ValueError(
            f"不支持的嵌入后端: '{backend}'，可选: openai / huggingface"
        )


# ═════════════════════════════════════════════════════════════════
# 验证工具
# ═════════════════════════════════════════════════════════════════

class EmbeddingValidationError(Exception):
    """单条 chunk 向量化失败。"""

    def __init__(self, chunk_index: int, reason: str, preview: str = "") -> None:
        self.chunk_index = chunk_index
        self.reason = reason
        self.preview = preview
        super().__init__(
            f"Chunk {chunk_index} 向量化失败: {reason}"
            + (f" | preview: {preview!r}" if preview else "")
        )


def validate_chunks(
    embedder: BaseEmbedder,
    chunks: Sequence[Document],
    *,
    strict: bool = True,
    max_empty_ratio: float = 0.0,
) -> List[Document]:
    """逐条验证 chunk 能否成功向量化。

    将每条 chunk 单独送入嵌入模型，检查：
      1. 返回向量非空
      2. 维度与模型一致
      3. 不含 NaN / Inf
      4. 不全为零向量

    Parameters
    ----------
    embedder : BaseEmbedder
        已初始化的嵌入模型实例。
    chunks : Sequence[Document]
        待验证的 Document 列表。
    strict : bool
        True 时遇到任一失败即抛出 ``EmbeddingValidationError``；
        False 时跳过失败 chunk 并记录日志。
    max_empty_ratio : float
        允许的空白 chunk 最大比例 (0~1)。超过时即使 strict=False 也报错。
        默认 0，即不允许任何空白 chunk。

    Returns
    -------
    List[Document]
        通过验证的 chunk 列表（strict=False 时可能比输入少）。

    Raises
    ------
    EmbeddingValidationError
        当 ``strict=True`` 且某条 chunk 验证失败时抛出。
    """
    if not chunks:
        return []

    total = len(chunks)
    dim = embedder.dimension
    failed: list[tuple[int, str, str]] = []
    empty_count = 0

    # 逐条编码 — 用 batch_size=1 确保每条独立
    for i, doc in enumerate(chunks):
        text = doc.page_content

        # 空白文本检查
        if not text or not text.strip():
            empty_count += 1
            failed.append((i, "空白文本", text[:40]))
            continue

        try:
            vec = embedder.embed_query(text)  # 单条嵌入
        except Exception as exc:
            failed.append((i, f"嵌入异常: {exc}", text[:80]))
            continue

        # 向量检查
        vec_np = np.array(vec, dtype=np.float32)

        if len(vec) == 0:
            failed.append((i, "零长度向量", text[:80]))
        elif len(vec) != dim:
            failed.append(
                (i, f"维度不匹配 (期望 {dim}, 实际 {len(vec)})", text[:80])
            )
        elif np.any(np.isnan(vec_np)):
            failed.append((i, "向量含 NaN", text[:80]))
        elif np.any(np.isinf(vec_np)):
            failed.append((i, "向量含 Inf", text[:80]))
        elif np.all(vec_np == 0.0):
            failed.append((i, "全零向量", text[:80]))
        elif not np.any(vec_np):  # 捕捉 -0.0 等情况
            failed.append((i, "向量全为零", text[:80]))

    # ── 空白比例检查 ──
    empty_ratio = empty_count / total if total else 0
    if empty_ratio > max_empty_ratio:
        raise EmbeddingValidationError(
            -1,
            f"空白 chunk 比例 {empty_ratio:.1%} 超过阈值 {max_empty_ratio:.1%} "
            f"({empty_count}/{total})",
        )

    # ── 结果处理 ──
    if failed and strict:
        idx, reason, preview = failed[0]
        raise EmbeddingValidationError(idx, reason, preview)

    if failed:
        logger.warning(
            "validate_chunks: %d/%d 条失败，已跳过。失败详情:",
            len(failed), total,
        )
        for idx, reason, preview in failed[:5]:
            logger.warning("  chunk %d: %s | %s", idx, reason, preview)
        if len(failed) > 5:
            logger.warning("  ... 还有 %d 条", len(failed) - 5)

        # 返回通过验证的 chunk
        failed_indices = {f[0] for f in failed}
        return [doc for i, doc in enumerate(chunks) if i not in failed_indices]

    logger.info(
        "validate_chunks: %d/%d 条全部通过 (dim=%d)",
        total, total, dim,
    )
    return list(chunks)


def compute_embedding_stats(vectors: List[List[float]]) -> dict:
    """计算一批向量的统计信息（用于质量检查）。

    Parameters
    ----------
    vectors : List[List[float]]
        向量列表。

    Returns
    -------
    dict
        包含 count, dim, mean_norm, min_norm, max_norm 的字典。
    """
    if not vectors:
        return {"count": 0}

    arr = np.array(vectors, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1)

    return {
        "count": len(vectors),
        "dim": arr.shape[1],
        "mean_norm": float(np.mean(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "zero_vectors": int(np.sum(norms == 0.0)),
    }
