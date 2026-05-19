"""
离线索引构建脚本 (Build Index)
=================================

一键完成全量文档入库：加载 → 清洗 → 分块 → 嵌入 → 向量存储。

用法：
    # 使用默认配置构建索引
    python scripts/build_index.py

    # 指定数据源和输出
    python scripts/build_index.py --raw-dir data/raw --persist-dir data/vector_store

    # 强制重建（清空已有索引）
    python scripts/build_index.py --rebuild

    # 使用 FAISS 后端
    python scripts/build_index.py --backend faiss

    # 只处理特定类别
    python scripts/build_index.py --category markdown

    # 调整分块参数
    python scripts/build_index.py --chunk-size 800 --chunk-overlap 100

    # 跳过验证（更快）
    python scripts/build_index.py --no-validate
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.documents import Document

from src.ingestion.loader import MultiFormatLoader
from src.ingestion.cleaner import clean_documents
from src.ingestion.splitter import split_documents
from src.ingestion.embedder import create_embedder, validate_chunks, BaseEmbedder
from src.ingestion.vectorizer import (
    create_vector_store,
    build_index_from_chunks,
    BaseVectorStore,
)
from src.utils.config_loader import get_config_value
from src.utils.logger import setup_logging

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# 核心构建流程
# ═════════════════════════════════════════════════════════════════

class IndexBuilder:
    """离线索引构建器。

    将原始文档转换为可检索的向量索引，支持全量重建与增量更新。

    Parameters
    ----------
    raw_dir : str
        原始文档目录。
    persist_dir : str
        向量存储持久化目录。
    backend : str
        向量存储后端：``"chroma"`` 或 ``"faiss"``。
    chunk_size : int
        分块大小（字符数）。
    chunk_overlap : int
        分块重叠（字符数）。
    min_content_chars : int
        清洗后最小有效字符数。
    embedding_backend : str
        嵌入模型后端。
    validate : bool
        是否逐条验证 chunk 向量化。
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        persist_dir: str = "data/vector_store",
        backend: str = "chroma",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        min_content_chars: int = 10,
        embedding_backend: str = "openai",
        validate: bool = True,
    ) -> None:
        self.raw_dir = raw_dir
        self.persist_dir = persist_dir
        self.backend = backend
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_content_chars = min_content_chars
        self.embedding_backend = embedding_backend
        self.validate = validate

    def run(
        self,
        *,
        category: Optional[str] = None,
        rebuild: bool = False,
    ) -> BaseVectorStore:
        """执行完整的构建流程。

        Parameters
        ----------
        category : str or None
            只处理指定子目录。None = 全部。
        rebuild : bool
            True 时清空已有索引后重建。

        Returns
        -------
        BaseVectorStore
            构建完成的向量存储实例。
        """
        start_time = time.time()

        # ══════════════════════════════════════════════════════
        # Step 1: 加载
        # ══════════════════════════════════════════════════════
        print("\n📂 Step 1/5: 加载文档...", flush=True)
        loader = MultiFormatLoader(raw_dir=self.raw_dir)
        if category:
            docs = loader.load_by_category(category)
        else:
            docs = loader.load_all()

        print(f"   加载完成: {len(docs)} 个文档", flush=True)
        if not docs:
            print("❌ 未加载到任何文档，请检查 data/raw 目录。")
            sys.exit(1)

        # ══════════════════════════════════════════════════════
        # Step 2: 清洗
        # ══════════════════════════════════════════════════════
        print("\n🧹 Step 2/5: 清洗文档...", flush=True)
        cleaned_docs = clean_documents(docs, min_content_chars=self.min_content_chars)
        print(f"   清洗完成: {len(docs)} → {len(cleaned_docs)} 个文档"
              f" (丢弃 {len(docs) - len(cleaned_docs)} 个空/噪声文档)", flush=True)

        if not cleaned_docs:
            print("❌ 清洗后无有效文档，请检查数据质量。")
            sys.exit(1)

        # ══════════════════════════════════════════════════════
        # Step 3: 分块
        # ══════════════════════════════════════════════════════
        print(f"\n✂  Step 3/5: 分块 (size={self.chunk_size}, overlap={self.chunk_overlap})...", flush=True)
        chunks = split_documents(
            cleaned_docs,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        print(f"   分块完成: {len(cleaned_docs)} 文档 → {len(chunks)} chunks", flush=True)

        # 分块大小统计
        chunk_lens = [len(c.page_content) for c in chunks]
        avg_len = sum(chunk_lens) / len(chunk_lens) if chunk_lens else 0
        print(f"   Chunk 平均长度: {avg_len:.0f} chars "
              f"(min={min(chunk_lens)}, max={max(chunk_lens)})", flush=True)

        # ══════════════════════════════════════════════════════
        # Step 4: 嵌入 + 验证
        # ══════════════════════════════════════════════════════
        print(f"\n🧮 Step 4/5: 初始化嵌入模型 ({self.embedding_backend})...", flush=True)
        try:
            embedder = create_embedder(backend=self.embedding_backend)
        except ImportError as e:
            print(f"❌ 缺少依赖: {e}")
            print("   提示: 使用本地模型请先安装: pip install sentence-transformers")
            print("   或设置 OPENAI_API_KEY 环境变量后重试")
            sys.exit(1)
        except Exception as e:
            print(f"❌ 嵌入模型初始化失败: {e}")
            if "api_key" in str(e).lower() or "api key" in str(e).lower():
                print("   请设置环境变量: set OPENAI_API_KEY=sk-your-key")
            elif "timed out" in str(e).lower() or "connect" in str(e).lower():
                print("   无法连接 OpenAI API，请检查网络或切换到本地模型：")
                print("   python scripts/build_index.py --embedding-backend huggingface")
            else:
                print("   可尝试切换到本地模型：")
                print("   python scripts/build_index.py --embedding-backend huggingface")
            sys.exit(1)
        print(f"   模型维度: {embedder.dimension}", flush=True)

        if self.validate:
            print("   验证 chunk 向量化...", flush=True)
            chunks = validate_chunks(embedder, chunks, strict=False, max_empty_ratio=0.05)
            print(f"   验证完成: {len(chunks)} chunks 通过", flush=True)

        # ══════════════════════════════════════════════════════
        # Step 5: 构建向量索引
        # ══════════════════════════════════════════════════════
        print(f"\n📦 Step 5/5: 构建向量索引 (backend={self.backend})...", flush=True)

        if rebuild:
            # 清空已有索引目录
            import shutil
            store_path = Path(self.persist_dir)
            if store_path.exists():
                print(f"   🗑 清空已有索引: {store_path}", flush=True)
                shutil.rmtree(store_path, ignore_errors=True)

        vector_store = build_index_from_chunks(
            chunks=chunks,
            embedder=embedder,
            backend=self.backend,
            persist_dir=self.persist_dir,
            validate=False,  # 已在上一步验证过
        )

        elapsed = time.time() - start_time
        print(f"\n✅ 索引构建完成！", flush=True)
        print(f"   Chunks:  {len(chunks)}", flush=True)
        print(f"   存储:    {self.backend} ({self.persist_dir})", flush=True)
        print(f"   耗时:    {elapsed:.1f}s", flush=True)

        return vector_store


# ═════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG 离线索引构建 — 一键完成 加载→清洗→分块→嵌入→存储",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/build_index.py
  python scripts/build_index.py --rebuild
  python scripts/build_index.py --chunk-size 800 --chunk-overlap 100
  python scripts/build_index.py --backend faiss
  python scripts/build_index.py --category pdf
  python scripts/build_index.py --no-validate
        """,
    )

    # 路径
    parser.add_argument("--raw-dir", type=str, default="data/raw",
                        help="原始文档目录（默认 data/raw）")
    parser.add_argument("--persist-dir", type=str, default="data/vector_store",
                        help="向量库持久化目录（默认 data/vector_store）")

    # 向量存储
    parser.add_argument("--backend", "-b", type=str,
                        choices=["chroma", "faiss"],
                        default=None,
                        help="向量存储后端（默认从配置读取）")

    # 分块参数
    parser.add_argument("--chunk-size", type=int, default=None,
                        help="分块大小（默认从配置读取）")
    parser.add_argument("--chunk-overlap", type=int, default=None,
                        help="分块重叠（默认从配置读取）")

    # 嵌入模型
    parser.add_argument("--embedding-backend", type=str,
                        choices=["openai", "huggingface"],
                        default=None,
                        help="嵌入模型后端（默认从配置读取）")

    # 控制
    parser.add_argument("--category", "-c", type=str, default=None,
                        help="只处理指定类别子目录（如 pdf, markdown）")
    parser.add_argument("--rebuild", action="store_true",
                        help="强制重建（清空已有索引）")
    parser.add_argument("--no-validate", action="store_true",
                        help="跳过 chunk 向量化验证（更快）")
    parser.add_argument("--min-content-chars", type=int, default=10,
                        help="清洗后最小有效字符数（默认 10）")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细日志输出")

    args = parser.parse_args()

    # ── 日志 ──
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level, console=True)

    # ── 参数：CLI 参数优先，否则从配置读取 ──
    chunk_size = args.chunk_size or get_config_value("chunking.chunk_size", 500)
    chunk_overlap = args.chunk_overlap or get_config_value("chunking.chunk_overlap", 50)
    backend = args.backend or get_config_value("vector_store.backend", "chroma")
    embedding_backend = args.embedding_backend or get_config_value("embedding.backend", "openai")

    # ── 打印配置 ──
    print("=" * 60)
    print("  RAG 离线索引构建")
    print("=" * 60)
    print(f"  原始数据:    {args.raw_dir}")
    print(f"  向量存储:    {args.persist_dir} ({backend})")
    print(f"  嵌入模型:    {embedding_backend}")
    print(f"  分块参数:    size={chunk_size}, overlap={chunk_overlap}")
    print(f"  验证:        {'否' if args.no_validate else '是'}")
    print(f"  重建:        {'是' if args.rebuild else '否（增量）'}")
    if args.category:
        print(f"  类别过滤:    {args.category}")
    print("=" * 60)

    # ── 构建 ──
    builder = IndexBuilder(
        raw_dir=args.raw_dir,
        persist_dir=args.persist_dir,
        backend=backend,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_content_chars=args.min_content_chars,
        embedding_backend=embedding_backend,
        validate=not args.no_validate,
    )

    builder.run(
        category=args.category,
        rebuild=args.rebuild,
    )


if __name__ == "__main__":
    main()
