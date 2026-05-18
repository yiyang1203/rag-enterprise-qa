"""
统一多格式文档加载器 (Unified Multi-format Document Loader)

基于 LangChain 框架，支持从 data/raw 目录加载以下格式:
  • CSV (.csv)   → 按 content 列拆分, 其余列作为元数据
  • Markdown (.md) → TextLoader (保留原始 Markdown 结构)
  • PDF (.pdf)     → PyPDFLoader (按页加载)
  • TXT (.txt)     → TextLoader
  • Word (.docx)   → Docx2txtLoader

返回统一的 List[langchain_core.documents.Document]。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

from langchain_community.document_loaders import (
    CSVLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# ── 扩展名 → 内部类型映射 ──────────────────────────────────────────
_EXT_TO_TYPE: Dict[str, str] = {
    ".csv": "csv",
    ".md": "markdown",
    ".pdf": "pdf",
    ".txt": "text",
    ".docx": "word",
}

SUPPORTED_EXTENSIONS = frozenset(_EXT_TO_TYPE.keys())


class MultiFormatLoader:
    """多格式文档加载器。

    自动扫描 ``raw_dir`` 下的所有子目录，根据文件扩展名
    选择合适的 LangChain Loader，返回统一的 Document 列表。

    Parameters
    ----------
    raw_dir : str
        原始数据根目录，默认为 ``data/raw``。
    encoding : str
        文本文件编码，默认 ``utf-8``。
    """

    def __init__(self, raw_dir: str = "data/raw", encoding: str = "utf-8") -> None:
        self.raw_dir = Path(raw_dir)
        self.encoding = encoding
        if not self.raw_dir.is_dir():
            raise FileNotFoundError(f"raw_dir does not exist: {self.raw_dir}")

    # ── 公共接口 ──────────────────────────────────────────────────

    def load_all(self) -> List[Document]:
        """递归扫描 ``raw_dir``，加载所有支持格式的文档。

        Returns
        -------
        List[Document]
            所有成功加载的 LangChain Document 对象。
        """
        all_docs: List[Document] = []
        for ext in sorted(SUPPORTED_EXTENSIONS):
            for file_path in sorted(self.raw_dir.glob(f"**/*{ext}")):
                try:
                    docs = self._load_file(file_path)
                    all_docs.extend(docs)
                    logger.info("Loaded %d doc(s) from %s", len(docs), file_path)
                except Exception:
                    logger.exception("Failed to load %s", file_path)
        logger.info("Total documents loaded: %d", len(all_docs))
        return all_docs

    def load_by_category(self, category: str) -> List[Document]:
        """加载指定子目录（类别）下的所有文档。

        Parameters
        ----------
        category : str
            子目录名称，例如 ``"excel"``, ``"markdown"``, ``"pdf"`` 等。

        Returns
        -------
        List[Document]
        """
        category_dir = self.raw_dir / category
        if not category_dir.is_dir():
            raise FileNotFoundError(f"Category directory not found: {category_dir}")

        docs: List[Document] = []
        for ext in sorted(SUPPORTED_EXTENSIONS):
            for file_path in sorted(category_dir.glob(f"*{ext}")):
                try:
                    docs.extend(self._load_file(file_path))
                    logger.info("Loaded %s", file_path)
                except Exception:
                    logger.exception("Failed to load %s", file_path)
        return docs

    def load_single(self, file_path: str) -> List[Document]:
        """加载单个文件。

        Parameters
        ----------
        file_path : str
            文件路径（相对于项目根目录或绝对路径）。

        Returns
        -------
        List[Document]
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        return self._load_file(path)

    # ── 内部分派 ──────────────────────────────────────────────────

    def _load_file(self, file_path: Path) -> List[Document]:
        ext = file_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file extension: {ext}")

        loader_type = _EXT_TO_TYPE[ext]
        dispatch = {
            "csv": self._load_csv,
            "markdown": self._load_text,   # Markdown 使用 TextLoader, 保留原始结构
            "pdf": self._load_pdf,
            "text": self._load_text,
            "word": self._load_word,
        }
        return dispatch[loader_type](file_path)

    # ── 各格式加载器 ──────────────────────────────────────────────

    @staticmethod
    def _parse_csv_page_content(raw: str) -> Dict[str, str]:
        """将 CSVLoader 生成的 ``key: value\\n`` 字符串解析为字典。

        CSVLoader 将整行数据序列化为::

            chunk_id: CS210-001
            title: CS210产品概述
            content: CS210 是一款...

        本方法将其还原为 ``{col: val, ...}``。
        """
        result: Dict[str, str] = {}
        current_key: Optional[str] = None
        current_val: List[str] = []

        for line in raw.split("\n"):
            # 检查是否为新 key: value 行
            if ": " in line:
                # 尝试在第一个 ": " 处分割
                idx = line.index(": ")
                maybe_key = line[:idx]
                maybe_val = line[idx + 2:]

                # 如果 key 看起来像列名（不含空格、不含前导空白），则视为新字段
                if " " not in maybe_key and maybe_key.strip() == maybe_key:
                    # 保存上一个字段
                    if current_key is not None:
                        result[current_key] = "\n".join(current_val)
                    current_key = maybe_key
                    current_val = [maybe_val] if maybe_val else []
                    continue

            # 续行（多行 content 值）
            if current_key is not None:
                current_val.append(line)

        # 保存最后一个字段
        if current_key is not None:
            result[current_key] = "\n".join(current_val)

        return result

    def _load_csv(self, file_path: Path) -> List[Document]:
        """加载 CSV 文件。

        CSV 要求包含 ``content`` 列作为文档正文；
        ``chunk_id``, ``title``, ``category``, ``source_doc``, ``keywords``
        等列自动归入元数据。
        """
        loader = CSVLoader(
            file_path=str(file_path),
            encoding=self.encoding,
            csv_args={"delimiter": ",", "quotechar": '"'},
        )
        raw_docs = loader.load()

        documents: List[Document] = []
        for doc in raw_docs:
            # CSVLoader 将整行放在 page_content 中 (key: value\n 格式)
            parsed = self._parse_csv_page_content(doc.page_content)

            content = parsed.pop("content", "")
            if not content.strip():
                continue  # 跳过空行

            clean_meta: Dict[str, str] = {
                "source": str(file_path),
                "file_type": "csv",
                "title": parsed.pop("title", file_path.stem),
            }
            # 保留其它有效列
            for key in ("chunk_id", "category", "source_doc", "keywords"):
                val = parsed.pop(key, None)
                if val:
                    clean_meta[key] = val
            # 附加剩余字段（防止未来新增列被漏掉）
            clean_meta.update({k: v for k, v in parsed.items() if v})

            documents.append(
                Document(page_content=content.strip(), metadata=clean_meta)
            )

        return documents

    def _load_pdf(self, file_path: Path) -> List[Document]:
        """按页加载 PDF 文件。"""
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "pdf"
            doc.metadata["source"] = str(file_path)
        return docs

    def _load_text(self, file_path: Path) -> List[Document]:
        """加载纯文本或 Markdown 文件。"""
        loader = TextLoader(str(file_path), encoding=self.encoding)
        docs = loader.load()
        ext_type = _EXT_TO_TYPE.get(file_path.suffix.lower(), "text")
        for doc in docs:
            doc.metadata["file_type"] = ext_type
            doc.metadata["source"] = str(file_path)
        return docs

    def _load_word(self, file_path: Path) -> List[Document]:
        """加载 Word (.docx) 文件。

        优先使用 Docx2txtLoader；如果文件不是有效的 .docx 格式
        （例如实际为纯文本但扩展名被误设为 .docx），则回退到 TextLoader。
        """
        try:
            loader = Docx2txtLoader(str(file_path))
            docs = loader.load()
        except Exception:
            logger.warning(
                "%s is not a valid .docx file, falling back to TextLoader",
                file_path,
            )
            loader = TextLoader(str(file_path), encoding=self.encoding)
            docs = loader.load()

        for doc in docs:
            doc.metadata["file_type"] = "word"
            doc.metadata["source"] = str(file_path)
        return docs


# ── 便捷函数 ──────────────────────────────────────────────────────

def load_documents(
    raw_dir: str = "data/raw",
    encoding: str = "utf-8",
) -> List[Document]:
    """加载 ``data/raw`` 下所有支持格式的文档。

    便捷函数，等价于::

        loader = MultiFormatLoader(raw_dir, encoding)
        docs = loader.load_all()

    Parameters
    ----------
    raw_dir : str
        原始数据根目录。
    encoding : str
        文本文件编码。

    Returns
    -------
    List[Document]
        LangChain Document 列表。
    """
    return MultiFormatLoader(raw_dir=raw_dir, encoding=encoding).load_all()
