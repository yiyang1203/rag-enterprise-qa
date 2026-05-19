"""
RAG 企业问答系统 — Web 界面 (Streamlit)

提供极简可演示的问答界面：
  • 输入框输入问题
  • 侧边栏调节 Top-K / 场景 / 重排序开关
  • 展示答案 + 引用来源（可折叠）
  • 检索详情展示

启动：
    streamlit run src/api/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

# 确保项目根在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.retrieval.retriever import create_retriever, Retriever
from src.retrieval.reranker import create_reranker, BaseReranker
from src.generation.prompt_builder import build_prompt, detect_scenario
from src.generation.llm_client import create_llm_client, BaseLLMClient
from src.utils.config_loader import get_config_value

# ═════════════════════════════════════════════════════════════════
# 页面配置
# ═════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="RAG 企业知识库问答",
    page_icon="📚",
    layout="wide",
)

# ═════════════════════════════════════════════════════════════════
# 初始化（缓存，只加载一次）
# ═════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_retriever() -> Retriever:
    """加载检索器（缓存）。"""
    with st.spinner("正在加载向量索引..."):
        return create_retriever()


@st.cache_resource(show_spinner=False)
def load_llm() -> BaseLLMClient:
    """加载 LLM 客户端（缓存）。"""
    return create_llm_client()


@st.cache_resource(show_spinner=False)
def load_reranker() -> Optional[BaseReranker]:
    """加载重排序器（缓存，可能不可用）。"""
    try:
        return create_reranker(backend="bge")
    except Exception:
        st.sidebar.warning("⚠ 重排序模型加载失败，该功能不可用")
        return None


# ═════════════════════════════════════════════════════════════════
# 侧边栏
# ═════════════════════════════════════════════════════════════════

def render_sidebar() -> dict:
    """渲染侧边栏并返回用户设置。"""
    with st.sidebar:
        st.header("⚙ 设置")

        top_k = st.slider(
            "检索 Top-K",
            min_value=1,
            max_value=20,
            value=get_config_value("retrieval.top_k", 5),
            step=1,
            help="从向量库检索的候选文档块数量",
        )

        scenario = st.selectbox(
            "场景模板",
            options=["auto", "default", "exact_lookup", "howto", "fuzzy"],
            index=0,
            help="auto = 根据问题自动检测",
        )

        use_rerank = st.checkbox(
            "启用重排序",
            value=False,
            help="对检索结果用 Cross-Encoder 重排（更精确但更慢）",
        )

        rerank_top_k = None
        if use_rerank:
            rerank_top_k = st.slider(
                "重排序 Top-K",
                min_value=1,
                max_value=top_k,
                value=min(3, top_k),
                step=1,
                help="重排序后保留的文档数",
            )

        show_context = st.checkbox(
            "显示检索上下文",
            value=False,
            help="在答案下方展示检索到的原始文本块",
        )

        st.divider()
        st.caption("📚 RAG Enterprise QA v1.0")
        st.caption(f"向量库后端: {get_config_value('vector_store.backend', 'chroma')}")
        st.caption(f"LLM: {get_config_value('llm.backend', 'openai')}")

    return {
        "top_k": top_k,
        "scenario": None if scenario == "auto" else scenario,
        "use_rerank": use_rerank,
        "rerank_top_k": rerank_top_k,
        "show_context": show_context,
    }


# ═════════════════════════════════════════════════════════════════
# 主界面
# ═════════════════════════════════════════════════════════════════

def main() -> None:
    st.title("📚 RAG 企业知识库问答")
    st.caption("基于检索增强生成（RAG）的企业内部知识库智能问答系统")

    # ── 加载资源 ──
    retriever = load_retriever()
    llm = load_llm()
    reranker = load_reranker()

    # ── 侧边栏 ──
    settings = render_sidebar()

    # ── 对话历史 ──
    if "messages" not in st.session_state:
        st.session_state.messages: List[dict] = []

    # ── 渲染历史 ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📎 查看来源"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.caption(
                            f"**[{i}] {src['source']}** | "
                            f"title={src.get('title', '-')} | "
                            f"chunk={src.get('chunk_index', '-')}"
                        )
                        if settings.get("show_context"):
                            st.text(src["preview"][:300])

    # ── 输入框 ──
    query = st.chat_input("输入你的问题...")

    if not query:
        return

    # ── 用户消息 ──
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # ── 生成回答 ──
    with st.chat_message("assistant"):
        try:
            # 1. 检索
            with st.spinner("🔍 检索中..."):
                chunks = retriever.retrieve(query, top_k=settings["top_k"])

            if not chunks:
                answer = "⚠ 未在知识库中找到相关内容，请尝试换个问法。"
                st.warning(answer)
            else:
                # 2. 重排序（可选）
                if settings["use_rerank"] and reranker and settings.get("rerank_top_k"):
                    with st.spinner("🔄 重排序中..."):
                        chunks = reranker.rerank(
                            query, chunks, top_k=settings["rerank_top_k"]
                        )

                # 3. 场景检测
                scenario = settings["scenario"] or detect_scenario(query)

                # 4. 构建提示词
                with st.spinner("📝 构建提示词..."):
                    messages = build_prompt(
                        question=query,
                        chunks=chunks,
                        scenario=scenario,
                    )

                # 5. 调用 LLM
                with st.spinner("🤖 生成回答..."):
                    answer = llm.generate_with_messages(messages)

                st.markdown(answer)

                # 来源
                sources = [
                    {
                        "source": Path(c.metadata.get("source", "")).name,
                        "title": c.metadata.get("title", ""),
                        "chunk_index": c.metadata.get("chunk_index"),
                        "preview": c.page_content[:300].strip(),
                    }
                    for c in chunks
                ]

                with st.expander(f"📎 查看来源 ({len(sources)} 条)"):
                    for i, src in enumerate(sources, 1):
                        st.caption(
                            f"**[{i}] {src['source']}** | "
                            f"title={src.get('title', '-')} | "
                            f"chunk={src.get('chunk_index', '-')}"
                        )
                        if settings.get("show_context"):
                            st.text(src["preview"])

            # 保存到对话历史
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources if chunks else [],
            })

        except Exception as exc:
            error_msg = f"❌ 处理出错: {str(exc)}"
            st.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "sources": [],
            })

    # ── 清空按钮 ──
    if st.session_state.messages:
        col1, col2, col3 = st.columns([6, 1, 1])
        with col3:
            if st.button("🗑 清空对话"):
                st.session_state.messages = []
                st.rerun()


if __name__ == "__main__":
    main()
