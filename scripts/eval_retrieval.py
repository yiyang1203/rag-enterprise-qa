"""
检索命中率评估 (Retrieval Evaluation)

计算检索器在 golden QA 测试集上的命中率：
  • Hit@K — Top-K 结果中是否包含正确答案所在的文档
  • MRR (Mean Reciprocal Rank) — 第一个正确答案的排名倒数均值

用法：
    python scripts/eval_retrieval.py                          # 使用默认参数
    python scripts/eval_retrieval.py --top-k 3 5 10           # 评估多个 K 值
    python scripts/eval_retrieval.py --output results.json    # 保存详细结果
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import create_retriever, Retriever
from src.utils.config_loader import get_config_value

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════
# 加载金标数据
# ═════════════════════════════════════════════════════════════════

def load_golden_qa(path: Optional[str] = None) -> List[dict]:
    """加载金标 QA 测试集。

    Parameters
    ----------
    path : str or None
        JSON 文件路径。默认 ``tests/golden_qa.json``。

    Returns
    -------
    List[dict]
    """
    if path is None:
        path = str(PROJECT_ROOT / "tests" / "golden_qa.json")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("golden_qa.json 必须是 JSON 数组")

    return data


# ═════════════════════════════════════════════════════════════════
# 命中判断
# ═════════════════════════════════════════════════════════════════

def _doc_matches_relevant(
    doc_meta: dict,
    relevant_docs: List[str],
    doc_index: int,
) -> bool:
    """判断检索到的文档是否与参考答案来源匹配。

    匹配策略：
      1. 文档标题精确匹配 relevant_doc 列表中的任一项
      2. 文件名包含 relevant_doc 关键词

    Parameters
    ----------
    doc_meta : dict
        检索结果的 metadata。
    relevant_docs : List[str]
        金标中标注的相关文档名列表（逗号分隔的也会拆分）。
    doc_index : int
        在 Top-K 结果中的 0-based 排名。

    Returns
    -------
    bool
    """
    title = doc_meta.get("title", "").strip()
    source = doc_meta.get("source", "").strip()
    fname = Path(source).stem if source else ""

    for rd in relevant_docs:
        rd = rd.strip()
        if not rd:
            continue
        # 精确标题匹配
        if rd.lower() == title.lower():
            return True
        # 文件名匹配
        if rd.lower() in fname.lower():
            return True
        # 关键词包含匹配（rd 是 doc 标题的子串 或 反之）
        if rd.lower() in title.lower() or title.lower() in rd.lower():
            return True

    return False


# ═════════════════════════════════════════════════════════════════
# 评估逻辑
# ═════════════════════════════════════════════════════════════════

def evaluate_retrieval(
    retriever: Retriever,
    golden_qa: List[dict],
    top_k_values: List[int] = [3, 5, 10],
    verbose: bool = True,
) -> dict:
    """评估检索命中率。

    Parameters
    ----------
    retriever : Retriever
        已初始化的检索器。
    golden_qa : List[dict]
        金标 QA 列表。
    top_k_values : List[int]
        评估的 K 值列表。
    verbose : bool
        是否打印每个问题的评估详情。

    Returns
    -------
    dict
        {
            "summary": {k: {"hit_rate": float, "mrr": float} ...},
            "per_question": [...]
        }
    """
    max_k = max(top_k_values)

    per_question: List[dict] = []
    hits_by_k: Dict[int, int] = {k: 0 for k in top_k_values}
    reciprocal_ranks: Dict[int, List[float]] = {k: [] for k in top_k_values}
    total = len(golden_qa)

    for item in golden_qa:
        qid = item.get("id", "?")
        question = item["question"]
        relevant_raw = item.get("relevant_doc", "")
        # 拆分逗号分隔的相关文档
        relevant_docs = [r.strip() for r in relevant_raw.split(",") if r.strip()]

        # 检索
        results = retriever.retrieve(question, top_k=max_k)
        # 带分数的结果（用于 MRR 计算）
        results_with_scores = retriever.retrieve_with_scores(question, top_k=max_k)

        # 对每个 K 计算命中
        q_result: dict = {
            "id": qid,
            "question": question,
            "relevant_docs": relevant_docs,
            "k_results": {},
        }

        for k in top_k_values:
            top_k_docs = results[:k]

            # Hit@K
            hit = any(
                _doc_matches_relevant(doc.metadata, relevant_docs, i)
                for i, doc in enumerate(top_k_docs)
            )
            if hit:
                hits_by_k[k] += 1

            # MRR: 第一个匹配的 rank
            rr = 0.0
            for rank, doc in enumerate(top_k_docs, 1):
                if _doc_matches_relevant(doc.metadata, relevant_docs, rank - 1):
                    rr = 1.0 / rank
                    break
            reciprocal_ranks[k].append(rr)

            q_result["k_results"][str(k)] = {
                "hit": hit,
                "reciprocal_rank": rr,
            }

            # 详细结果
            q_result["retrieved"] = [
                {
                    "rank": i + 1,
                    "title": doc.metadata.get("title", ""),
                    "source": Path(doc.metadata.get("source", "")).name,
                    "preview": doc.page_content[:100].strip(),
                }
                for i, doc in enumerate(top_k_docs)
            ]

        per_question.append(q_result)

        if verbose:
            hit_k = ",".join(
                f"Hit@{k}={q_result['k_results'][str(k)]['hit']}"
                for k in top_k_values
            )
            print(f"  [{qid}] {question[:50]}... → {hit_k}")

    # ── 汇总 ──
    summary: Dict[int, dict] = {}
    for k in top_k_values:
        hit_rate = hits_by_k[k] / total if total else 0
        mrr = sum(reciprocal_ranks[k]) / len(reciprocal_ranks[k]) if reciprocal_ranks[k] else 0
        summary[k] = {
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "hits": hits_by_k[k],
            "total": total,
        }

    return {
        "summary": {str(k): v for k, v in summary.items()},
        "per_question": per_question,
    }


# ═════════════════════════════════════════════════════════════════
# 报告
# ═════════════════════════════════════════════════════════════════

def print_report(eval_result: dict) -> None:
    """打印评估报告。"""
    summary = eval_result["summary"]

    print("\n" + "=" * 60)
    print("  检索命中率评估报告")
    print("=" * 60)

    # 表头
    print(f"\n{'K':>6}  {'Hit@K':>10}  {'MRR':>10}  {'Hits':>8}  {'Total':>8}")
    print("-" * 50)

    for k_str, stats in sorted(summary.items(), key=lambda x: int(x[0])):
        k = int(k_str)
        print(
            f"{k:>6}  {stats['hit_rate']:>10.2%}  "
            f"{stats['mrr']:>10.4f}  {stats['hits']:>8}  {stats['total']:>8}"
        )

    # 按类别汇总
    print("\n  按类别:");
    per_q = eval_result["per_question"]
    categories: Dict[str, List[dict]] = {}
    for item in per_q:
        # 从 id 推断类别
        cat = item["id"].rsplit("_", 1)[0] if "_" in item["id"] else "unknown"
        categories.setdefault(cat, []).append(item)

    # 从 golden_qa 获取原始 item 的 category
    # 这里在 per_question 中已有 id，从 original 关联
    print(f"  {'类别':<25} {'数量':>5}  {'Hit@5':>10}")
    print("  " + "-" * 45)
    # 重新计算（从 eval_result 的 per_question）
    cat_hits: Dict[str, dict] = {}
    for q in per_q:
        cat = q["id"].rsplit("_", 1)[0] if "_" in q["id"] else "unknown"
        if cat not in cat_hits:
            cat_hits[cat] = {"total": 0, "hits": 0}
        cat_hits[cat]["total"] += 1
        k5 = q["k_results"].get("5", q["k_results"].get(list(q["k_results"].keys())[0], {}))
        if k5.get("hit", False):
            cat_hits[cat]["hits"] += 1

    for cat, stats in sorted(cat_hits.items()):
        hr = stats["hits"] / stats["total"] if stats["total"] else 0
        print(f"  {cat:<25} {stats['total']:>5}  {hr:>10.2%}")

    print("\n" + "=" * 60)


# ═════════════════════════════════════════════════════════════════
# 主入口
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="检索命中率评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/eval_retrieval.py
  python scripts/eval_retrieval.py --top-k 3 5 10
  python scripts/eval_retrieval.py --output results.json
        """,
    )
    parser.add_argument(
        "--golden-path", "-g",
        type=str,
        help="金标 QA JSON 路径（默认 tests/golden_qa.json）",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        nargs="+",
        default=[3, 5, 10],
        help="评估的 K 值列表（默认 3 5 10）",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="详细结果输出 JSON 路径",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="静默模式（不打印每个问题的详情）",
    )

    args = parser.parse_args()

    # 加载金标数据
    print("📋 加载金标数据...")
    golden_qa = load_golden_qa(args.golden_path)
    print(f"   加载 {len(golden_qa)} 个测试问题")

    # 初始化检索器
    print("🔍 初始化检索器...")
    retriever = create_retriever()
    print(f"   向量库文档数: {retriever.store_count}")

    # 评估
    print(f"\n🚀 开始评估 (Top-K={args.top_k})...\n")
    result = evaluate_retrieval(
        retriever=retriever,
        golden_qa=golden_qa,
        top_k_values=args.top_k,
        verbose=not args.quiet,
    )

    # 报告
    print_report(result)

    # 保存
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2, default=str)
        print(f"\n💾 详细结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
