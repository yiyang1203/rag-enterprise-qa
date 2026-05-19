"""
交互式问答脚本 (Interactive QA Script)
========================================

串联完整 RAG 流程：用户输入问题 → 检索 → 构建提示词 → 调用 LLM → 打印答案 + 来源。

用法：
    # 交互模式（默认）
    python scripts/test_qa.py

    # 单次问答
    python scripts/test_qa.py --query "CS210 产品的功率是多少？"

    # 指定 Top-K
    python scripts/test_qa.py --top-k 3

    # 指定场景模板
    python scripts/test_qa.py --scenario exact_lookup

    # 批处理模式（从文件读取问题，一行一题）
    python scripts/test_qa.py --batch questions.txt --output answers.json

依赖：
    • 已通过 build_index.py 构建向量索引
    • 已配置 LLM API Key（环境变量或 configs/app_config.yaml）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import create_retriever
from src.generation.prompt_builder import build_prompt, detect_scenario
from src.generation.llm_client import create_llm_client
from src.ingestion.vectorizer import load_vector_store
from src.utils.config_loader import get_config_value

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """配置日志输出。"""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ═════════════════════════════════════════════════════════════════
# 核心问答函数
# ═════════════════════════════════════════════════════════════════

def ask(
    question: str,
    *,
    top_k: Optional[int] = None,
    scenario: Optional[str] = None,
    include_metadata: bool = True,
    verbose: bool = False,
) -> dict:
    """执行一次完整的 RAG 问答。

    Parameters
    ----------
    question : str
        用户问题。
    top_k : int or None
        检索结果数。None 使用配置默认值。
    scenario : str or None
        场景模板。None 时自动检测。
    include_metadata : bool
        是否在上下文中显示来源元数据。
    verbose : bool
        是否输出调试信息。

    Returns
    -------
    dict
        ``{"question", "answer", "sources": [...], "scenario": ..., "chunks_used": N}``
    """
    if top_k is None:
        top_k = get_config_value("retrieval.top_k", 5)

    # ── 1. 检索 ──────────────────────────────────────────────
    if verbose:
        print(f"\n🔍 检索中 (top_k={top_k})...", file=sys.stderr)

    retriever = create_retriever(top_k=top_k)
    chunks = retriever.retrieve(question, top_k=top_k)

    if not chunks:
        if verbose:
            print("⚠ 未检索到相关文档。", file=sys.stderr)
        return {
            "question": question,
            "answer": "未在知识库中找到相关内容。",
            "sources": [],
            "scenario": "default",
            "chunks_used": 0,
        }

    if verbose:
        for i, doc in enumerate(chunks):
            src = doc.metadata.get("source", "?")
            preview = doc.page_content[:80].replace("\n", " ")
            print(f"   #{i+1} [{Path(src).name}] {preview}...", file=sys.stderr)

    # ── 2. 场景检测 ──────────────────────────────────────────
    if scenario is None:
        scenario = detect_scenario(question)
    if verbose:
        print(f"   Scenario: {scenario}", file=sys.stderr)

    # ── 3. 构建提示词 ────────────────────────────────────────
    if verbose:
        print("📝 构建提示词...", file=sys.stderr)

    messages = build_prompt(
        question=question,
        chunks=chunks,
        scenario=scenario,
        include_metadata=include_metadata,
    )

    # ── 4. 调用 LLM ──────────────────────────────────────────
    if verbose:
        print("🤖 调用 LLM...", file=sys.stderr)

    llm = create_llm_client()
    answer = llm.generate_with_messages(messages)

    # ── 5. 汇总来源 ──────────────────────────────────────────
    sources = []
    for doc in chunks:
        src = doc.metadata.get("source", "")
        title = doc.metadata.get("title", "")
        chunk_idx = doc.metadata.get("chunk_index")
        sources.append({
            "source": Path(src).name if src else "",
            "title": title,
            "chunk_index": chunk_idx,
            "preview": doc.page_content[:200].strip(),
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "scenario": scenario,
        "chunks_used": len(chunks),
    }


# ═════════════════════════════════════════════════════════════════
# 输出格式化
# ═════════════════════════════════════════════════════════════════

def print_result(result: dict, verbose: bool = False) -> None:
    """打印问答结果到终端。"""
    print("\n" + "=" * 60)
    print(f"❓ 问题: {result['question']}")
    print("-" * 60)
    print(f"🤖 回答:\n{result['answer']}")
    print("-" * 60)

    if verbose and result.get("sources"):
        print(f"📚 来源 ({result['scenario']}, {result['chunks_used']} chunks):")
        for i, src in enumerate(result["sources"], 1):
            print(f"  [{i}] {src['source'] or '(未知)'} | "
                  f"title={src['title'] or '-'} | "
                  f"chunk={src['chunk_index']}")

    print("=" * 60 + "\n")


# ═════════════════════════════════════════════════════════════════
# 交互模式
# ═════════════════════════════════════════════════════════════════

def interactive_mode(args: argparse.Namespace) -> None:
    """交互式问答循环。"""
    print("=" * 60)
    print("  RAG 企业问答系统 - 交互模式")
    print("  输入问题后按 Enter，输入 /quit 退出")
    print("  命令: /scenario <name>  切换场景模板")
    print(f"  Top-K: {args.top_k or '默认'}  |  Scenario: {args.scenario or '自动检测'}")
    print("=" * 60)

    scenario = args.scenario
    top_k = args.top_k

    while True:
        try:
            user_input = input("\n💬 你的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        if user_input.startswith("/quit"):
            print("👋 再见！")
            break

        if user_input.startswith("/scenario"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                scenario = parts[1].strip()
                print(f"   ✓ 场景切换为: {scenario}")
            else:
                print(f"   当前场景: {scenario or '自动检测'}")
            continue

        if user_input.startswith("/topk"):
            parts = user_input.split(maxsplit=1)
            if len(parts) > 1:
                try:
                    top_k = int(parts[1])
                    print(f"   ✓ Top-K 设置为: {top_k}")
                except ValueError:
                    print("   ✗ 请输入有效数字")
            else:
                print(f"   当前 Top-K: {top_k or '默认'}")
            continue

        # 执行问答
        try:
            result = ask(
                user_input,
                top_k=top_k,
                scenario=scenario,
                verbose=args.verbose,
            )
            print_result(result, verbose=args.verbose)
        except Exception as exc:
            print(f"\n❌ 错误: {exc}", file=sys.stderr)
            if args.verbose:
                import traceback
                traceback.print_exc()


# ═════════════════════════════════════════════════════════════════
# 批处理模式
# ═════════════════════════════════════════════════════════════════

def batch_mode(
    input_file: str,
    output_file: str,
    top_k: Optional[int] = None,
    scenario: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """从文件批量处理问题。"""
    input_path = Path(input_file)
    if not input_path.is_file():
        print(f"❌ 输入文件不存在: {input_path}", file=sys.stderr)
        sys.exit(1)

    # 读取问题（每行一题，跳过空行和 # 注释）
    questions: List[str] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                questions.append(line)

    print(f"📋 读取 {len(questions)} 个问题，开始处理...")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}] {q[:60]}...", file=sys.stderr)
        try:
            result = ask(q, top_k=top_k, scenario=scenario, verbose=verbose)
            results.append(result)
            if verbose:
                print_result(result, verbose=verbose)
        except Exception as exc:
            print(f"  ❌ 处理失败: {exc}", file=sys.stderr)
            results.append({"question": q, "answer": f"ERROR: {exc}", "sources": []})

    # 保存结果
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 结果已保存到 {output_path}")


# ═════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG 企业问答系统 - 交互式问答脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/test_qa.py                           # 交互模式
  python scripts/test_qa.py --query "产品功率？"       # 单次问答
  python scripts/test_qa.py --batch questions.txt     # 批处理
  python scripts/test_qa.py --top-k 3 --scenario exact_lookup
        """,
    )

    parser.add_argument(
        "--query", "-q",
        type=str,
        help="单次问答的问题文本",
    )
    parser.add_argument(
        "--batch", "-b",
        type=str,
        metavar="FILE",
        help="批量处理：从 FILE 读取问题（一行一题），输出为 FILE.json",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="批量模式的输出文件（默认在输入文件名后加 .json）",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        help="检索返回的文档块数量（默认从配置读取）",
    )
    parser.add_argument(
        "--scenario", "-s",
        type=str,
        choices=["default", "exact_lookup", "howto", "fuzzy"],
        help="场景模板；默认自动检测",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="输出调试信息（检索过程、来源详情）",
    )

    args = parser.parse_args()
    setup_logging(verbose=args.verbose)

    # ── 单次问答 ──
    if args.query:
        result = ask(
            args.query,
            top_k=args.top_k,
            scenario=args.scenario,
            verbose=args.verbose,
        )
        print_result(result, verbose=args.verbose)
        return

    # ── 批处理 ──
    if args.batch:
        output_file = args.output or (args.batch + ".json")
        batch_mode(
            input_file=args.batch,
            output_file=output_file,
            top_k=args.top_k,
            scenario=args.scenario,
            verbose=args.verbose,
        )
        return

    # ── 交互模式 ──
    interactive_mode(args)


if __name__ == "__main__":
    main()
