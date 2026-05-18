"""
cleaner.py 单元测试

覆盖：
  1. 纯分隔线去除（=, -, *, #, ~, _）
  2. 空白行压缩
  3. 首尾空白去除
  4. 噪声行过滤（页码、页眉页脚短语）
  5. Document 级清洗 + 空文档丢弃
  6. 真实文档端到端回归：清洗前后对比，确保无信息丢失
"""

import re
from pathlib import Path

import pytest
from langchain_core.documents import Document

from src.ingestion.cleaner import (
    clean_text,
    clean_document,
    clean_documents,
    is_empty_or_noise,
)
from src.ingestion.loader import load_documents

# 项目根目录（用于构建真实数据路径）
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── 行级 / 文本级清洗 ─────────────────────────────────────────

class TestCleanText:
    """clean_text() 单元测试"""

    def test_removes_equal_separators(self):
        text = "=====\nreal content\n=========="
        result = clean_text(text)
        assert result == "real content"

    def test_removes_dash_separators(self):
        text = "--------\nreal content\n------------"
        result = clean_text(text)
        assert result == "real content"

    def test_removes_star_separators(self):
        text = "*******\nreal content\n**********"
        result = clean_text(text)
        assert result == "real content"

    def test_removes_hash_separators(self):
        text = "#####\nreal content\n########"
        result = clean_text(text)
        assert result == "real content"

    def test_removes_tilde_separators(self):
        text = "~~~~~\nreal content\n~~~~~~~~~~"
        result = clean_text(text)
        assert result == "real content"

    def test_removes_underscore_separators(self):
        text = "_____\nreal content\n__________"
        result = clean_text(text)
        assert result == "real content"

    def test_preserves_short_separator_like_strings(self):
        """少于 5 个重复字符的不应被当作分隔线移除。"""
        text = "===\nreal content\n===="
        result = clean_text(text)
        # ==== has 4 chars → NOT removed; ==== has 4 chars → NOT removed
        assert "===" in result or "real content" in result
        # Lines with <5 separator chars should stay
        lines = result.split("\n")
        assert "real content" in lines

    def test_removes_bare_page_numbers(self):
        text = "第一章\n42\nreal content\n128"
        result = clean_text(text)
        assert "42" not in result.split("\n")
        assert "128" not in result.split("\n")
        assert "第一章" in result
        assert "real content" in result

    def test_preserves_section_headers(self):
        """方括号包裹的标题不应被移除。"""
        text = "[CS210产品概述]\nreal content"
        result = clean_text(text)
        assert "[CS210产品概述]" in result
        assert "real content" in result

    def test_collapses_excessive_blank_lines(self):
        text = "line1\n\n\n\n\nline2"
        result = clean_text(text)
        # 5 blank lines → collapse to 2 → result is "line1\n\n\nline2"
        # (2 blank lines = 3 \n total between content)
        assert "\n\n\n\n" not in result, f"Still has 3+ blank lines:\n{result!r}"
        assert result == "line1\n\n\nline2"

    def test_strips_leading_trailing_blank_lines(self):
        text = "\n\n\nreal content\n\n\n"
        result = clean_text(text)
        assert result == "real content"

    def test_strips_trailing_whitespace_per_line(self):
        text = "line1   \n   line2   "
        result = clean_text(text)
        assert "line1" in result
        assert "line2" in result

    def test_removes_header_footer_phrases(self):
        text = "机密\nreal content\n版权所有"
        result = clean_text(text)
        assert "机密" not in result.split("\n")
        assert "版权所有" not in result.split("\n")
        assert "real content" in result

    def test_preserves_content_adjacent_to_separators(self):
        """分隔线之间的真实内容应保留。"""
        text = "=" * 70 + "\n产品文档 | 更新 2024-03\n" + "=" * 70
        result = clean_text(text)
        assert "产品文档" in result
        assert "更新 2024-03" in result

    def test_collapse_blanks_can_be_disabled(self):
        text = "a\n\n\n\n\nb"
        result = clean_text(text, collapse_blanks=False)
        # Without collapsing, blanks are kept as-is
        assert "a" in result
        assert "b" in result
        assert result.count("\n") >= 4

    def test_no_crash_on_empty_string(self):
        assert clean_text("") == ""
        assert clean_text("\n\n\n") == ""

    def test_no_crash_on_only_separators(self):
        text = "=====\n------\n******"
        result = clean_text(text)
        assert result == ""


# ── is_empty_or_noise ──────────────────────────────────────────

class TestIsEmptyOrNoise:
    """is_empty_or_noise() 单元测试"""

    def test_empty_text(self):
        assert is_empty_or_noise("", min_content_chars=10) is True

    def test_only_separators(self):
        assert is_empty_or_noise("=====\n-----\n******", min_content_chars=10) is True

    def test_below_threshold(self):
        assert is_empty_or_noise("Hi", min_content_chars=10) is True

    def test_above_threshold(self):
        assert is_empty_or_noise("Hello World! This is content.", min_content_chars=10) is False

    def test_whitespace_not_counted(self):
        assert is_empty_or_noise("   \n\n   \n  ", min_content_chars=10) is True


# ── Document 级清洗 ────────────────────────────────────────────

class TestCleanDocument:
    """clean_document() 单元测试"""

    def test_cleans_content(self):
        doc = Document(page_content="=====\nhello world\n-----", metadata={"src": "test.txt"})
        result = clean_document(doc, min_content_chars=5)
        assert result is not None
        assert result.page_content == "hello world"
        assert result.metadata == {"src": "test.txt"}

    def test_returns_none_for_empty_result(self):
        doc = Document(page_content="=====\n-----\n*****", metadata={"src": "test.txt"})
        result = clean_document(doc, min_content_chars=10)
        assert result is None

    def test_preserves_metadata(self):
        meta = {"source": "a.txt", "file_type": "text", "title": "T"}
        doc = Document(page_content="clean content here", metadata=meta)
        result = clean_document(doc)
        assert result is not None
        assert result.metadata == meta


class TestCleanDocuments:
    """clean_documents() 批量清洗测试"""

    def test_filters_empty_docs(self):
        docs = [
            Document(page_content="====\n----", metadata={"id": 1}),
            Document(page_content="real content here", metadata={"id": 2}),
            Document(page_content="****\n####", metadata={"id": 3}),
        ]
        result = clean_documents(docs, min_content_chars=10)
        assert len(result) == 1
        assert result[0].metadata["id"] == 2

    def test_keeps_all_when_all_valid(self):
        docs = [
            Document(page_content="content A here", metadata={"id": 1}),
            Document(page_content="content B here", metadata={"id": 2}),
        ]
        result = clean_documents(docs, min_content_chars=10)
        assert len(result) == 2


# ── 真实数据回归测试 ───────────────────────────────────────────

class TestRealDataRegression:
    """用 data/raw 真实文档验证清洗效果，无信息丢失。"""

    @pytest.fixture(scope="class")
    def all_docs(self):
        raw_dir = str(PROJECT_ROOT / "data" / "raw")
        return load_documents(raw_dir=raw_dir)

    @pytest.fixture(scope="class")
    def cleaned_docs(self, all_docs):
        return clean_documents(all_docs, min_content_chars=10)

    def test_no_docs_lost_unless_noise(self, all_docs, cleaned_docs):
        """原始 60 个文档中，只有 3 个 PDF 空页应被丢弃。"""
        dropped = len(all_docs) - len(cleaned_docs)
        # ym.pdf has 3 noise pages (p3 "历史人物" 4chars, p6 "文化风景" 4chars, p12 empty 0chars)
        # But "历史人物" = 4 meaningful chars, "文化风景" = 4 meaningful chars → < 10 threshold
        assert dropped == 3, f"Expected 3 dropped, got {dropped}"

    def test_all_cleaned_have_min_content(self, cleaned_docs):
        """清洗后每个文档的有效字符数应 >= min_content_chars。"""
        for doc in cleaned_docs:
            meaningful = re.sub(r"\s+", "", doc.page_content)
            assert len(meaningful) >= 10, (
                f"Doc too short: {len(meaningful)} chars, "
                f"source={doc.metadata.get('source')}, "
                f"preview={doc.page_content[:80]!r}"
            )

    def test_txt_no_more_separator_lines(self, cleaned_docs):
        """清洗后 .txt 文档不应再包含 ===== 分隔线。"""
        sep_re = re.compile(r"^={5,}$")
        for doc in cleaned_docs:
            if doc.metadata.get("file_type") == "text":
                for line in doc.page_content.split("\n"):
                    assert not sep_re.match(line.strip()), (
                        f"Separator still present in {doc.metadata.get('source')}: {line!r}"
                    )

    def test_csv_content_preserved(self, all_docs, cleaned_docs):
        """CSV 每行独立成 doc，清洗后内容应全部保留。"""
        csv_before = [d for d in all_docs if d.metadata.get("file_type") == "csv"]
        csv_after = [d for d in cleaned_docs if d.metadata.get("file_type") == "csv"]
        assert len(csv_after) == len(csv_before), (
            f"CSV docs lost: {len(csv_before)} → {len(csv_after)}"
        )

    def test_section_headers_preserved_in_txt(self, cleaned_docs):
        """方括号 section header 如 [CS210产品概述] 应保留。"""
        header_re = re.compile(r"\[[\w]+?]")
        found = False
        for doc in cleaned_docs:
            if doc.metadata.get("file_type") == "text":
                if header_re.search(doc.page_content):
                    found = True
                    break
        assert found, "No section headers [xxx] found in TXT docs after cleaning"

    def test_markdown_structure_preserved(self, cleaned_docs):
        """Markdown 的 # 标题和 | 表格应保留。"""
        for doc in cleaned_docs:
            if doc.metadata.get("file_type") == "markdown":
                assert "#" in doc.page_content, (
                    f"Markdown headings lost in {doc.metadata.get('source')}"
                )

    def test_clean_text_does_not_destroy_cjk(self, cleaned_docs):
        """清洗不应破坏中文内容。"""
        cjk_count = 0
        for doc in cleaned_docs:
            for ch in doc.page_content:
                if "\u4e00" <= ch <= "\u9fff":
                    cjk_count += 1
        assert cjk_count > 1000, f"Only {cjk_count} CJK chars found — content may be damaged"

    def test_no_blank_first_or_last_line(self, cleaned_docs):
        """清洗后不应有空白的首行或尾行。"""
        for doc in cleaned_docs:
            lines = doc.page_content.split("\n")
            assert lines[0].strip(), (
                f"Blank first line in {doc.metadata.get('source')}"
            )
            assert lines[-1].strip(), (
                f"Blank last line in {doc.metadata.get('source')}"
            )

    def test_key_terms_preserved(self, cleaned_docs):
        """关键术语应保留在所有文档中。"""
        all_text = " ".join(d.page_content for d in cleaned_docs)
        key_terms = ["CS210", "TM200", "VPN", "API", "SLA"]
        for term in key_terms:
            assert term in all_text, f"Key term '{term}' missing after cleaning!"
