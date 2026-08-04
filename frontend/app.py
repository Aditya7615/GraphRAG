"""Enterprise RAG — Analyst Workspace & Knowledge Ops Console.

Run with:  streamlit run frontend/app.py
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(180.0, connect=10.0)

st.set_page_config(
    page_title="Lengraph — Enterprise RAG",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Enterprise Dark-Mode Design System ───────────────────────────────────────
_DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-tertiary: #1a2236;
    --bg-surface: #1e293b;
    --bg-hover: #253049;
    --border: #2a3650;
    --border-light: #334155;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --accent-glow: rgba(59,130,246,0.15);
    --success: #10b981;
    --success-bg: rgba(16,185,129,0.1);
    --warning: #f59e0b;
    --warning-bg: rgba(245,158,11,0.1);
    --error: #ef4444;
    --error-bg: rgba(239,68,68,0.1);
    --purple: #8b5cf6;
}

.stApp { background: var(--bg-primary) !important; }
.stApp > header { background: transparent !important; }

section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

h1, h2, h3, h4, h5, h6, p, li, span, div, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}
code, pre, .stCode, .stCodeBlock {
    font-family: 'JetBrains Mono', monospace !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-secondary) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-muted) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.75rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom-color: var(--accent) !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

.stChatMessage {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem 1.25rem !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    border-left: 3px solid var(--accent) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    border-left: 3px solid var(--purple) !important;
}

.streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-primary) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}
.streamlit-expanderHeader:hover { background: var(--bg-hover) !important; }
.streamlit-expanderContent {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

.stButton > button {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stButton > button:hover {
    background: var(--bg-hover) !important;
    border-color: var(--accent) !important;
}
.stButton > button[kind="primary"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    color: white !important;
}
.stButton > button[kind="primary"]:hover { background: var(--accent-hover) !important; }
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border-color: var(--border-light) !important;
    color: var(--text-secondary) !important;
}

.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    border-radius: 8px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}

[data-testid="stMetric"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.08em !important;
}

.stAlert {
    background: var(--bg-surface) !important;
    border-radius: 8px !important;
    border-left-width: 4px !important;
}

hr { border-color: var(--border) !important; opacity: 0.5 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 3px; }

.stChatInput > div {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}
.stChatInput > div:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px var(--accent-glow) !important;
}
.stChatInput textarea { color: var(--text-primary) !important; }

.stDataFrame { border: 1px solid var(--border) !important; border-radius: 8px !important; }
</style>
"""
st.markdown(_DARK_CSS, unsafe_allow_html=True)


# ─── API Client ───────────────────────────────────────────────────────────────
def api_get(path: str, **kwargs):
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.get(f"{API_BASE}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()


def api_post(path: str, **kwargs):
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.post(f"{API_BASE}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()


def api_delete(path: str):
    with httpx.Client(timeout=TIMEOUT) as client:
        resp = client.delete(f"{API_BASE}{path}")
        resp.raise_for_status()
        return resp.json()


@st.cache_data(ttl=15)
def cached_stats():
    return api_get("/stats")


def fetch_health():
    try:
        return api_get("/health")
    except (httpx.HTTPError, OSError) as exc:
        return {"status": "unreachable", "error": str(exc)}


# ─── Session State ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_parent" not in st.session_state:
    st.session_state.show_parent = False


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# :blue[◆ Lengraph]")
    st.caption("Enterprise RAG · Hybrid Search · Zero-Hallucination Guardrails")

    # ── System Health ──
    st.markdown("#### System Health")
    status = fetch_health()
    if status.get("status") == "ok":
        st.success("OPERATIONAL")
    elif status.get("status") == "degraded":
        st.warning("DEGRADED")
    else:
        st.error("OFFLINE")
        if "error" in status:
            st.caption(status["error"])

    if "qdrant" in status:
        c1, c2, c3 = st.columns(3)
        q_ok = status["qdrant"] == "up"
        c1.metric("Qdrant", status["qdrant"])
        ds = status.get("docstore", "—").split(":")[0]
        c2.metric("Docstore", ds)
        llm_ok = status["llm"] == "configured"
        c3.metric("LLM", "ready" if llm_ok else "no key")
        if not llm_ok:
            st.error("Set GOOGLE_API_KEY in .env")

    # ── Retrieval Settings ──
    st.markdown("---")
    st.markdown("#### Retrieval Settings")
    search_mode = st.radio(
        "Search mode",
        ["hybrid", "dense", "sparse"],
        horizontal=True,
        help="Hybrid = dense + BM25 with Reciprocal Rank Fusion.",
    )
    top_k = st.slider("Child chunks retrieved", 4, 40, 12)
    max_parents = st.slider("Parent chunks in context", 1, 10, 4)
    st.session_state.show_parent = st.checkbox(
        "Show parent chunks in citations",
        value=st.session_state.show_parent,
    )

    # ── Document Filter ──
    doc_filter: list[str] = []
    try:
        stats = cached_stats()
        docs = stats.get("documents", [])
        if docs:
            labels = {f"{d['source']} ({d['parent_chunks']} chunks)": d["doc_id"] for d in docs}
            picked = st.multiselect("Restrict to documents", list(labels))
            doc_filter = [labels[p] for p in picked]
    except (httpx.HTTPError, KeyError, OSError):
        stats = {}

    # ── Knowledge Base ──
    st.markdown("---")
    st.markdown("#### Knowledge Base")
    if stats:
        m1, m2 = st.columns(2)
        m1.metric("Documents", len(stats.get("documents", [])))
        m2.metric("Vectors", f"{stats.get('vectors', 0):,}")
        st.caption(f"Parents: {stats.get('parent_chunks', 0):,}")
        st.caption(f"Embedding: {stats.get('embedding_model', '-')}")
        st.caption(f"LLM: {stats.get('llm_model', '-')}")

    st.markdown("---")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Refresh", use_container_width=True):
            cached_stats.clear()
            st.rerun()
    with b2:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ─── Helpers ──────────────────────────────────────────────────────────────────
def render_latency(latency: dict) -> None:
    """Render latency breakdown as styled HTML."""
    ret = latency.get("retrieval", 0)
    gen = latency.get("generation", 0)
    tot = latency.get("total", 0)
    html = (
        f"<div style='display:inline-flex;gap:14px;padding:5px 12px;"
        f"background:var(--bg-surface);border:1px solid var(--border);border-radius:8px;"
        f"font-family:JetBrains Mono,monospace;font-size:0.72rem;color:var(--text-secondary);'>"
        f"<span>Retrieval <b style='color:var(--text-primary);'>{ret:.0f}</b>ms</span>"
        f"<span style='color:var(--border-light);'>|</span>"
        f"<span>Generation <b style='color:var(--text-primary);'>{gen:.0f}</b>ms</span>"
        f"<span style='color:var(--border-light);'>|</span>"
        f"<span>Total <b style='color:var(--text-primary);'>{tot:.0f}</b>ms</span>"
        f"</div>"
    )
    st.markdown(html, unsafe_allow_html=True)


def render_grounded_badge(grounded: bool) -> None:
    """Render grounded/ungrounded badge."""
    if grounded:
        html = (
            "<span style='display:inline-block;padding:3px 10px;border-radius:6px;"
            "font-size:0.72rem;font-weight:600;font-family:JetBrains Mono,monospace;"
            "background:rgba(16,185,129,0.12);color:#10b981;"
            "border:1px solid rgba(16,185,129,0.3);'>GROUNDED</span>"
        )
    else:
        html = (
            "<span style='display:inline-block;padding:3px 10px;border-radius:6px;"
            "font-size:0.72rem;font-weight:600;font-family:JetBrains Mono,monospace;"
            "background:rgba(239,68,68,0.12);color:#ef4444;"
            "border:1px solid rgba(239,68,68,0.3);'>UNGROUNDED</span>"
        )
    st.markdown(html, unsafe_allow_html=True)


def render_citations(citations: list, show_parent: bool = False) -> None:
    """Render citation list with expandable source inspectors."""
    if not citations:
        return
    used = [c for c in citations if c.get("used_by_llm")]
    st.info(f"{len(used)} of {len(citations)} sources cited in the answer.")

    for c in citations:
        marker = c.get("marker", "?")
        source = c.get("source", "Unknown")
        section = c.get("section")
        page_num = c.get("page")
        score = c.get("score", 0)
        is_used = c.get("used_by_llm", False)
        snippet = c.get("snippet", "")
        parent_text = c.get("parent_text", "")

        # Build locator parts (plain text, no HTML in expander label)
        locator_parts = [source]
        if section:
            locator_parts.append(f"Section: {section}")
        if page_num:
            locator_parts.append(f"Page {page_num}")
        badge = "CITED" if is_used else "RETRIEVED"
        locator = " | ".join(locator_parts)

        label = f"[{marker}] {locator}  --  {badge}  score {score:.4f}"

        with st.expander(label, expanded=False):
            st.markdown(snippet or "_No snippet available._")

            if show_parent and parent_text:
                st.markdown("---")
                st.markdown("**Full parent chunk sent to the LLM:**")
                st.code(parent_text[:8000], language=None)


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_chat, tab_ingest, tab_debug = st.tabs([
    "Analyst Workspace",
    "Knowledge Ops",
    "Retrieval Inspector",
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: ANALYST WORKSPACE
# ═════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("Ask your documents")
    st.caption("Grounded answers with validated citations from your corpus.")

    # ── Render conversation history ──
    for msg in st.session_state.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if not content:
            continue

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                grounded = msg.get("grounded", True)
                render_grounded_badge(grounded)

                citations = msg.get("citations", [])
                if citations:
                    render_citations(citations, show_parent=st.session_state.show_parent)

                latency = msg.get("latency_ms")
                if latency:
                    render_latency(latency)

                model = msg.get("model", "")
                mode = msg.get("search_mode", "")
                if model or mode:
                    st.caption(f"Model: {model} | Mode: {mode}")

    # ── Chat Input ──
    if question := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving and grounding..."):
                payload = None
                try:
                    payload = api_post(
                        "/chat",
                        json={
                            "question": question,
                            "top_k": top_k,
                            "max_parents": max_parents,
                            "doc_ids": doc_filter or None,
                            "search_mode": search_mode,
                            "include_parent_text": st.session_state.show_parent,
                        },
                    )
                except httpx.HTTPStatusError as exc:
                    detail = ""
                    try:
                        detail = exc.response.json().get("detail", exc.response.text)
                    except (ValueError, KeyError):
                        detail = exc.response.text
                    if exc.response.status_code == 429:
                        st.warning(f"Rate limited — {detail}")
                        st.caption(
                            "Retrieval is unaffected; only generation is blocked. Use the "
                            "Retrieval Inspector tab to keep testing search in the meantime."
                        )
                    elif exc.response.status_code == 503:
                        st.error(f"LLM unavailable — {detail}")
                    else:
                        st.error(f"API error ({exc.response.status_code}): {detail}")
                except (httpx.HTTPError, OSError) as exc:
                    st.error(f"Request failed: {exc}")

            if payload:
                answer_text = payload.get("answer", "")
                st.markdown(answer_text)

                grounded = payload.get("grounded", False)
                render_grounded_badge(grounded)

                citations = payload.get("citations", [])
                if citations:
                    render_citations(citations, show_parent=st.session_state.show_parent)

                latency = payload.get("latency_ms", {})
                if latency:
                    render_latency(latency)

                model = payload.get("model", "")
                mode = payload.get("search_mode", "")
                if model or mode:
                    st.caption(f"Model: {model} | Mode: {mode}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    **payload,
                })


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: KNOWLEDGE OPS
# ═════════════════════════════════════════════════════════════════════════════
with tab_ingest:
    st.subheader("Knowledge Operations Console")
    st.caption("Manage your document corpus. Upload, ingest, and monitor indexing pipelines.")

    col_upload, col_bulk = st.columns(2)

    with col_upload:
        st.markdown("**File Upload**")
        upload = st.file_uploader(
            "Drop files here",
            type=["pdf", "txt", "md", "docx"],
            label_visibility="collapsed",
        )
        if upload and st.button("Ingest File", type="primary", use_container_width=True):
            with st.spinner(f"Indexing {upload.name}..."):
                try:
                    result = api_post(
                        "/ingest/file",
                        files={"file": (upload.name, upload.getvalue(), upload.type)},
                    )
                    st.success(
                        f"{result['parent_chunks']} parents / "
                        f"{result['child_chunks']} children in "
                        f"{result['latency_ms']:.0f}ms"
                    )
                    cached_stats.clear()
                except (httpx.HTTPError, OSError) as exc:
                    st.error(f"Ingestion failed: {exc}")

    with col_bulk:
        st.markdown("**Bulk Ingestion**")
        directory = st.text_input("Directory path", value="data/synthetic")
        if st.button("Ingest Directory", type="primary", use_container_width=True):
            with st.spinner("Chunking, embedding, and indexing..."):
                try:
                    result = api_post("/ingest/directory", params={"directory": directory})
                    st.success(
                        f"{result['documents']} docs / {result['parent_chunks']} parents / "
                        f"{result['child_chunks']} children in {result['latency_ms']:.0f}ms"
                    )
                    if result.get("details"):
                        st.dataframe(result["details"], use_container_width=True)
                    if result.get("skipped"):
                        st.warning("Skipped: " + "; ".join(result["skipped"]))
                    cached_stats.clear()
                except (httpx.HTTPError, OSError) as exc:
                    st.error(f"Ingestion failed: {exc}")

    st.markdown("---")
    st.markdown("**Indexed Corpus**")
    try:
        corpus_stats = cached_stats()
        documents = corpus_stats.get("documents", [])
        if documents:
            sm1, sm2, sm3 = st.columns(3)
            sm1.metric("Documents", len(documents))
            sm2.metric("Vectors", f"{corpus_stats.get('vectors', 0):,}")
            sm3.metric("Parents", f"{corpus_stats.get('parent_chunks', 0):,}")

            st.dataframe(documents, use_container_width=True)

            del_col1, del_col2 = st.columns([3, 1])
            with del_col1:
                target = st.selectbox(
                    "Select document to delete",
                    ["--"] + [d["source"] for d in documents],
                )
            with del_col2:
                if target != "--" and st.button("Delete", type="secondary", use_container_width=True):
                    try:
                        doc_id = next(d["doc_id"] for d in documents if d["source"] == target)
                        api_delete(f"/documents/{doc_id}")
                        cached_stats.clear()
                        st.success(f"Deleted {target}")
                        st.rerun()
                    except StopIteration:
                        st.error("Document not found")
                    except httpx.HTTPStatusError as exc:
                        st.error(f"Delete failed: {exc.response.status_code}")
                    except (httpx.HTTPError, OSError) as exc:
                        st.error(f"Delete failed: {exc}")
        else:
            st.info("No documents indexed. Ingest files or folders to populate the knowledge base.")
    except (httpx.HTTPError, OSError, KeyError) as exc:
        st.error(f"Could not load corpus stats: {exc}")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: RETRIEVAL INSPECTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_debug:
    st.subheader("Retrieval Inspector")
    st.caption("Raw child-chunk hits before the LLM sees them. Compare search modes side-by-side.")

    probe = st.text_input("Search query", key="debug_probe", placeholder="e.g. What is the RTO for tier-1 services?")

    if probe:
        try:
            result = api_post(
                "/search",
                json={
                    "query": probe,
                    "top_k": top_k,
                    "doc_ids": doc_filter or None,
                    "search_mode": search_mode,
                },
            )
            hits = result.get("hits", [])
            latency_ms = result.get("latency_ms", 0)
            mode = result.get("search_mode", search_mode)

            st.info(f"{len(hits)} hits in {latency_ms:.0f}ms ({mode})")

            for i, hit in enumerate(hits, start=1):
                source = hit.get("source", "Unknown")
                section = hit.get("section")
                page_num = hit.get("page")
                score = hit.get("score", 0)
                text = hit.get("text", "")

                locator_parts = [f"{i}. {source}"]
                if section:
                    locator_parts.append(f"Section: {section}")
                if page_num:
                    locator_parts.append(f"Page {page_num}")
                label = " | ".join(locator_parts) + f"  --  score {score:.4f}"

                with st.expander(label, expanded=False):
                    st.markdown(text)
                    parent_id = hit.get("parent_id", "")
                    child_id = hit.get("child_id", "")
                    if parent_id:
                        st.caption(f"parent_id: {parent_id[:20]}... | child_id: {child_id[:20]}...")

        except (httpx.HTTPError, OSError) as exc:
            st.error(f"Search failed: {exc}")
