"""
数据探索脚本 (Data Exploration Script)
======================================

回答三个关键问题：
  1. 总共加载了多少文档？每类格式各多少？
  2. 平均文本长度是多少？有没有空文档或超长文档？
  3. 内容里有没有明显的乱码、页眉页脚残留？

用法：
    python scripts/data_explore.py
"""

import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

# 确保项目根在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.loader import load_documents
from langchain_core.documents import Document


# ── 页眉页脚 / 噪声特征模式 ────────────────────────────────────
# 这些正则匹配常见的文档残留噪声
NOISE_PATTERNS: List[Tuple[str, str]] = [
    # PDF 页码残留
    ("bare page number", re.compile(r"^\s*\d{1,4}\s*$")),
    # "第 X 页 / 共 Y 页"
    ("page X of Y (cn)", re.compile(r"第\s*\d+\s*页\s*(/|／|共)\s*\d+\s*页")),
    ("page X of Y (en)", re.compile(r"[Pp]age\s+\d+\s+of\s+\d+", re.IGNORECASE)),
    # 文件系统路径残留（Windows 绝对路径）
    ("filesystem path", re.compile(
        r"[A-Za-z]:\\(?:[\w.-]+\\)+[\w.-]+"          # Windows: C:\a\b.txt
    )),
    # 日期戳（单独的日期行）
    ("date stamp", re.compile(r"^\s*\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?\s*$")),
    # 页眉分隔线残留
    ("header separator", re.compile(r"^[=\-*_#~]{10,}$")),
    # 作者/版本水印行
    ("revision / author", re.compile(r"^\s*(修订|版本|作者|审核|批准|Version|Author|Rev\.?)\s*[:：]", re.IGNORECASE)),
    # "Confidential" / "机密" 标注（单行出现时往往是页眉脚）
    ("confidential mark", re.compile(r"^\s*(机密|内部资料|Confidential|Internal)\s*$", re.IGNORECASE)),
    # 版权声明残留
    ("copyright", re.compile(r"©|Copyright|All Rights Reserved", re.IGNORECASE)),
    # "www.xxx.com" 导航残留
    ("url bare", re.compile(r"www\.[a-zA-Z0-9.-]+\.(com|cn|org|net)")),
]

# URL / API 路径检测（信息性，不作为噪声）
URL_PATTERN = re.compile(
    r"https?://[^\s]+"                          # http(s) URL
    r"|/(?:api|v\d)/[^\s,，。)）]+"               # API 路由: /api/..., /v1/...
)

# 乱码特征：连续的非 CJK/ASCII 可打印字符
GARBLED_PATTERNS: List[Tuple[str, re.Pattern]] = [
    # 控制字符 (除 \n \r \t)
    ("control chars", re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")),
    # 连续 3 个以上的替换字符 �
    ("replacement chars", re.compile(r"�{3,}")),
    # 私有区 / 未分配 Unicode
    ("private-use area", re.compile(r"[\ue000-\uf8ff]{2,}")),
    # 看起来像 Base64 但实际是乱码的长连续字符串
    ("possible base64 noise", re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")),
    # 连续的非语言字符（排除 Markdown 表格/分隔线/常见标点）
    ("symbol soup", re.compile(
        r"[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
        r"|=\-*#~>_.:;!?()\[\]{}'\"，。；：！？）】》…\"\u201c\u201d"
        r"]{10,}"
    )),
]


def load_all_docs() -> List[Document]:
    raw_dir = str(PROJECT_ROOT / "data" / "raw")
    print(f"正在加载所有文档 (raw_dir={raw_dir})...")
    docs = load_documents(raw_dir=raw_dir)
    print(f"加载完成。\n")
    return docs


# ═══════════════════════════════════════════════════════════════
# 问题 1：文档数量与格式分布
# ═══════════════════════════════════════════════════════════════

def analyze_count_and_types(docs: List[Document]) -> Dict:
    type_counts = Counter(d.metadata.get("file_type", "unknown") for d in docs)
    source_counts = Counter(d.metadata.get("source", "unknown") for d in docs)

    print("=" * 60)
    print("📊 问题 1：文档数量与格式分布")
    print("=" * 60)
    print(f"  总文档数: {len(docs)}\n")

    print("  按格式分布:")
    for fmt, count in type_counts.most_common():
        pct = count / len(docs) * 100
        print(f"    {fmt:12s}  {count:4d}  ({pct:5.1f}%)")

    print(f"\n  按文件来源分布 ({len(source_counts)} 个文件):")
    for src, count in source_counts.most_common():
        fname = Path(src).name
        print(f"    {fname:40s}  {count:3d} docs")

    return {"total": len(docs), "type_counts": dict(type_counts)}


# ═══════════════════════════════════════════════════════════════
# 问题 2：文本长度统计
# ═══════════════════════════════════════════════════════════════

def analyze_length(docs: List[Document]) -> Dict:
    lengths = [len(d.page_content) for d in docs]
    chars = sum(lengths)

    if not lengths:
        print("  没有文档！")
        return {}

    avg_len = chars / len(lengths)
    median_len = sorted(lengths)[len(lengths) // 2]
    min_len = min(lengths)
    max_len = max(lengths)

    # 找出空文档（<= 5 字符视为空）
    empty_docs = [
        (d.metadata.get("source", "?"), d.metadata.get("title", "-"), len(d.page_content))
        for d in docs
        if len(d.page_content.strip()) <= 5
    ]

    # 找出超长文档（> avg * 3 或 > 5000 字符）
    long_threshold = max(avg_len * 3, 5000)
    long_docs = sorted(
        [
            (d.metadata.get("source", "?"), d.metadata.get("title", "-"), len(d.page_content))
            for d in docs
            if len(d.page_content) > long_threshold
        ],
        key=lambda x: -x[2],
    )

    # 按格式统计平均长度
    fmt_lengths: Dict[str, List[int]] = {}
    for d in docs:
        fmt = d.metadata.get("file_type", "unknown")
        fmt_lengths.setdefault(fmt, []).append(len(d.page_content))

    print("\n" + "=" * 60)
    print("📏 问题 2：文本长度统计")
    print("=" * 60)
    print(f"  总字符数: {chars:,}")
    print(f"  平均长度: {avg_len:,.0f} chars")
    print(f"  中位数:   {median_len:,} chars")
    print(f"  最短:     {min_len:,} chars")
    print(f"  最长:     {max_len:,} chars")

    print(f"\n  按格式平均长度:")
    for fmt in sorted(fmt_lengths):
        vals = fmt_lengths[fmt]
        print(f"    {fmt:12s}  avg={sum(vals)/len(vals):,.0f}  min={min(vals):,}  max={max(vals):,}  n={len(vals)}")

    print(f"\n  空/接近空文档 (≤5 chars): {len(empty_docs)} 个")
    if empty_docs:
        for src, title, length in empty_docs[:10]:
            print(f"    ⚠ {Path(src).name} | title={title} | len={length}")

    print(f"\n  超长文档 (>{long_threshold:,.0f} chars): {len(long_docs)} 个")
    if long_docs:
        for src, title, length in long_docs[:10]:
            print(f"    ⚠ {Path(src).name} | title={title} | len={length:,}")

    return {
        "avg_len": avg_len,
        "median_len": median_len,
        "min_len": min_len,
        "max_len": max_len,
        "empty_count": len(empty_docs),
        "long_count": len(long_docs),
    }


# ═══════════════════════════════════════════════════════════════
# 问题 3：乱码 / 页眉页脚残留检测
# ═══════════════════════════════════════════════════════════════

def analyze_noise(docs: List[Document]) -> Dict:
    noise_hits: Dict[str, List[Tuple[str, str, str]]] = {}
    garbled_hits: Dict[str, List[Tuple[str, str]]] = {}

    for doc in docs:
        text = doc.page_content
        src = doc.metadata.get("source", "?")
        doc_id = f"{Path(src).name} | {doc.metadata.get('title', '-')}"

        # 检测每一行
        lines = text.split("\n")
        for i, line in enumerate(lines):
            # 页眉页脚噪声
            for name, pattern in NOISE_PATTERNS:
                if pattern.search(line):
                    noise_hits.setdefault(name, []).append(
                        (doc_id, f"L{i+1}", line.strip()[:80])
                    )

        # 全局乱码检测（非逐行）
        for name, pattern in GARBLED_PATTERNS:
            for m in pattern.finditer(text):
                garbled_hits.setdefault(name, []).append(
                    (doc_id, m.group()[:80])
                )

    # ── 针对 PDF 的特殊检测：重复出现的行（疑似页眉页脚） ──
    pdf_docs = [d for d in docs if d.metadata.get("file_type") == "pdf"]
    if pdf_docs:
        pdf_sources = sorted(set(d.metadata.get("source", "") for d in pdf_docs))
        for src in pdf_sources:
            pages = [d for d in pdf_docs if d.metadata.get("source") == src]
            if len(pages) < 2:
                continue
            # 检查每个页面第一行和最后一行的重复情况
            first_lines = [p.page_content.split("\n")[0].strip() if p.page_content.strip() else "" for p in pages]
            last_lines = [p.page_content.split("\n")[-1].strip() if p.page_content.strip() else "" for p in pages]

            # 如果第一行在超过 60% 的页面中出现相同内容 → 页眉
            for label, line_list in [("header (first line)", first_lines), ("footer (last line)", last_lines)]:
                counter = Counter(line_list)
                for line, count in counter.most_common(3):
                    if count >= max(2, len(pages) * 0.6) and len(line) > 1:
                        noise_hits.setdefault(f"pdf-{label}", []).append(
                            (Path(src).name, f"x{count}/{len(pages)}", line[:80])
                        )

    # ── 输出 ──
    print("\n" + "=" * 60)
    print("🔍 问题 3：乱码 / 页眉页脚残留检测")
    print("=" * 60)

    # 噪声
    print(f"\n  页眉页脚 / 文档噪声 (命中 {sum(len(v) for v in noise_hits.values())} 处):")
    if noise_hits:
        for name in sorted(noise_hits):
            hits = noise_hits[name]
            print(f"    [{name}] {len(hits)} hits:")
            seen = set()
            for doc_id, loc, snippet in hits[:5]:
                key = (doc_id, snippet)
                if key not in seen:
                    print(f"      • {doc_id}  @{loc}: {snippet}")
                    seen.add(key)
            if len(hits) > 5:
                print(f"      … 还有 {len(hits) - 5} 处")
    else:
        print("    ✅ 未检测到明显噪声")

    # 乱码
    # ── URL 汇总（信息性，非噪声） ──
    url_hits: List[Tuple[str, str]] = []
    for doc in docs:
        text = doc.page_content
        src = doc.metadata.get("source", "?")
        doc_id = f"{Path(src).name} | {doc.metadata.get('title', '-')}"
        for m in URL_PATTERN.finditer(text):
            url_hits.append((doc_id, m.group()[:80]))

    print(f"\n  内嵌 URL / API 路径 (信息性，{len(url_hits)} 处):")
    if url_hits:
        for doc_id, url in url_hits[:8]:
            print(f"    🔗 {doc_id}: {url}")
        if len(url_hits) > 8:
            print(f"    … 还有 {len(url_hits) - 8} 处")
    else:
        print("    (无)")

    print(f"\n  乱码特征 (命中 {sum(len(v) for v in garbled_hits.values())} 处):")
    if garbled_hits:
        for name in sorted(garbled_hits):
            hits = garbled_hits[name]
            print(f"    [{name}] {len(hits)} hits:")
            for doc_id, snippet in hits[:3]:
                print(f"      • {doc_id}: {snippet}")
            if len(hits) > 3:
                print(f"      … 还有 {len(hits) - 3} 处")
    else:
        print("    ✅ 未检测到乱码")

    return {"noise": {k: len(v) for k, v in noise_hits.items()},
            "garbled": {k: len(v) for k, v in garbled_hits.items()}}


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    docs = load_all_docs()
    analyze_count_and_types(docs)
    analyze_length(docs)
    analyze_noise(docs)
    print("\n" + "=" * 60)
    print("✅ 数据探索完成")


if __name__ == "__main__":
    main()
