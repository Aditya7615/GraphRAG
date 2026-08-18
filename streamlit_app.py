"""Streamlit UI for GraphRAG — upload documents and ask questions."""

import os
import time

import streamlit as st

from graphrag import (
    PDF_DIR,
    TEXT_DIR,
    BM25_WEIGHT,
    CHROMA_WEIGHT,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RETRIEVAL_K,
    GraphRAG,
)


# --------------- Custom CSS ---------------
CUSTOM_CSS = """
<style>
    .stApp { background: #0e1117; }
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.5rem; }

    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 { color: #e6edf3; }

    .stChatMessage {
        border-radius: 12px; padding: 0.5rem 1rem;
        margin-bottom: 0.5rem; border: 1px solid #30363d; background: #161b22;
    }
    div[data-testid="stChatMessage"][aria-label="user"] {
        background: #1a2332; border-color: #1f6feb33;
    }
    div[data-testid="stChatMessage"][aria-label="assistant"] {
        background: #161b22; border-color: #3fb95033;
    }

    div[data-testid="stMetric"] {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 10px; padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #8b949e !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e6edf3 !important; }

    .stButton > button {
        border-radius: 8px; border: 1px solid #30363d; transition: all 0.2s ease;
    }
    .stButton > button:hover { border-color: #58a6ff; box-shadow: 0 0 8px #58a6ff33; }
    div[data-testid="stSidebar"] .stButton > button { width: 100%; }

    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        border: 2px dashed #30363d; border-radius: 10px; padding: 8px; transition: border-color 0.2s;
    }
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"]:hover { border-color: #58a6ff; }

    .streamlit-expanderHeader {
        background: #161b22; border: 1px solid #30363d; border-radius: 8px; color: #e6edf3;
    }

    .welcome-card {
        background: linear-gradient(135deg, #161b22 0%, #1a2332 100%);
        border: 1px solid #30363d; border-radius: 16px;
        padding: 2.5rem; text-align: center; margin: 2rem 0;
    }
    .welcome-card h1 { color: #e6edf3; margin-bottom: 0.3rem; font-size: 1.8rem; }
    .welcome-card p { color: #8b949e; font-size: 1rem; margin-bottom: 1.5rem; }

    .example-card {
        background: #0d1117; border: 1px solid #30363d; border-radius: 10px;
        padding: 14px 18px; cursor: pointer; transition: all 0.2s ease;
        text-align: left; color: #c9d1d9; font-size: 0.9rem; line-height: 1.4;
    }
    .example-card:hover { border-color: #58a6ff; background: #161b22; box-shadow: 0 0 12px #58a6ff22; }

    .pipeline-status { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin: 0.5rem 0 1rem 0; }
    .pipeline-step {
        background: #161b22; border: 1px solid #30363d; border-radius: 20px;
        padding: 4px 12px; font-size: 0.75rem; color: #8b949e;
        display: flex; align-items: center; gap: 5px;
    }
    .pipeline-step.active { border-color: #f0883e; color: #f0883e; animation: pulse 1.5s infinite; }
    .pipeline-step.done { border-color: #3fb950; color: #3fb950; }
    .pipeline-arrow { color: #484f58; font-size: 0.7rem; }

    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }

    .footer {
        text-align: center; padding: 1.5rem; color: #484f58;
        font-size: 0.8rem; border-top: 1px solid #21262d; margin-top: 2rem;
    }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
"""


# --------------- Streamlit UI ---------------
st.set_page_config(page_title="GraphRAG", page_icon=":books:", layout="wide")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.markdown("## :books: GraphRAG")
    st.caption("Self-corrective Agentic RAG Engine")
    st.divider()

    st.markdown("### :inbox_tray: Upload")
    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        total_size = sum(uf.size for uf in uploaded_files)
        size_str = f"{total_size / 1024:.1f} KB" if total_size < 1024 * 1024 else f"{total_size / (1024*1024):.1f} MB"
        st.info(f"**{len(uploaded_files)}** file(s) selected ({size_str})")

    c1, c2 = st.columns(2)
    with c1:
        save_btn = st.button(":floppy_disk: Save", use_container_width=True, type="primary",
                             disabled=not uploaded_files)
    with c2:
        clear_btn = st.button(":wastebasket: Clear All", use_container_width=True)

    if save_btn and uploaded_files:
        with st.spinner("Saving..."):
            saved = 0
            for uf in uploaded_files:
                ext = os.path.splitext(uf.name)[1].lower()
                dest_dir = PDF_DIR if ext == ".pdf" else TEXT_DIR
                dest_path = os.path.join(dest_dir, uf.name)
                with open(dest_path, "wb") as f:
                    f.write(uf.getbuffer())
                saved += 1
            st.cache_resource.clear()
        st.success(f"Saved {saved} file(s)")
        if "engine" in st.session_state:
            del st.session_state.engine
        time.sleep(0.5)
        st.rerun()

    if clear_btn:
        count = 0
        for d in [PDF_DIR, TEXT_DIR]:
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f != ".gitkeep":
                        os.remove(os.path.join(d, f))
                        count += 1
        st.cache_resource.clear()
        if "engine" in st.session_state:
            del st.session_state.engine
        if count:
            st.toast(f"Cleared {count} file(s)")
            time.sleep(0.3)
            st.rerun()
        else:
            st.toast("Nothing to clear")

    st.divider()

    st.markdown("### :open_file_folder: Saved Documents")
    saved_pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")] if os.path.isdir(PDF_DIR) else []
    saved_txts = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith((".txt", ".md"))] if os.path.isdir(TEXT_DIR) else []

    m1, m2 = st.columns(2)
    m1.metric("PDFs", len(saved_pdfs))
    m2.metric("Text", len(saved_txts))

    total = len(saved_pdfs) + len(saved_txts)
    if total > 0:
        with st.expander(f"View all {total} file(s)", expanded=False):
            for f in saved_pdfs + saved_txts:
                st.markdown(f"`{f}`")
    else:
        st.caption("No documents saved yet")

    if total > 0:
        st.divider()
        st.markdown("### :bar_chart: Pipeline Info")
        st.markdown(
            f"""
            <div style="font-size:0.8rem; color:#8b949e; line-height:1.6;">
                <b>Retrieval:</b> BM25 ({BM25_WEIGHT:.0%}) + Chroma ({CHROMA_WEIGHT:.0%})<br>
                <b>Chunk size:</b> {CHUNK_SIZE} chars<br>
                <b>Top K:</b> {RETRIEVAL_K} docs<br>
                <b>LLM:</b> Groq / Qwen3.6-27B
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Engine (cached in session, rebuilt only when docs change) ---
has_docs = total > 0

if has_docs:
    # Cache the engine in session_state — only rebuilds when cleared
    if "engine" not in st.session_state or st.session_state.get("_doc_count") != total:
        with st.spinner("Building knowledge base..."):
            st.session_state.engine = GraphRAG()
            st.session_state._doc_count = total
    engine = st.session_state.engine

    st.markdown(
        f"""
        <div class="pipeline-status">
            <span class="pipeline-step done">:white_check_mark: {engine.page_count} pages loaded</span>
            <span class="pipeline-arrow">-></span>
            <span class="pipeline-step done">:white_check_mark: {engine.chunk_count} chunks indexed</span>
            <span class="pipeline-arrow">-></span>
            <span class="pipeline-step done">:white_check_mark: Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Welcome screen
    st.markdown(
        """
        <div class="welcome-card">
            <h1>:books: GraphRAG</h1>
            <p>Upload documents and ask questions answered from their content.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### :rocket: How it works")
    cols = st.columns(3)
    steps = [
        (":inbox_tray: Upload", "Add your PDF, TXT, or Markdown files from the sidebar"),
        (":floppy_disk: Save", "Click **Save** to index them into the knowledge base"),
        (":speech_balloon: Ask", "Type your question below and get grounded answers"),
    ]
    for col, (icon_title, desc) in zip(cols, steps):
        with col:
            st.markdown(
                f"""
                <div style="background:#161b22; border:1px solid #30363d; border-radius:12px;
                            padding:1.2rem; text-align:center; height:130px;">
                    <div style="font-size:1.5rem; margin-bottom:6px;">{icon_title.split()[0]}</div>
                    <b style="color:#e6edf3;">{icon_title.split(' ', 1)[1] if ' ' in icon_title else ''}</b><br>
                    <span style="color:#8b949e; font-size:0.82rem;">{desc}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown("#### :sparkles: Example questions")
    for ex in [
        "What is GraphRAG and how does it work?",
        "Explain hybrid retrieval with BM25 and ChromaDB",
        "What are the key features of self-corrective RAG?",
    ]:
        st.markdown(f'<div class="example-card">:bulb: {ex}</div>', unsafe_allow_html=True)

# --- Chat (always renders, even without docs) ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = ":bust_in_silhouette:" if msg["role"] == "user" else ":robot_face:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f":link: {len(msg['sources'])} source(s)"):
                for src in msg["sources"]:
                    st.markdown(f"`{src}`")
        if msg.get("pipeline_steps"):
            st.markdown(
                '<div class="pipeline-status">'
                + "".join(f'<span class="pipeline-step done">{s}</span>' for s in msg["pipeline_steps"])
                + "</div>",
                unsafe_allow_html=True,
            )

query = st.chat_input("Ask a question about your documents...")
if query:
    if not has_docs:
        st.warning("Upload and save documents first, then ask questions.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=":bust_in_silhouette:"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=":robot_face:"):
        status_placeholder = st.empty()
        status_placeholder.markdown(
            '<div class="pipeline-status">'
            '<span class="pipeline-step active">:mag: Retrieving...</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        start_time = time.time()
        try:
            result = engine.ask(query)
            elapsed = time.time() - start_time
            answer = result["generation"]
            sources = [
                f"[{i+1}] {d.metadata.get('source', 'unknown')}: {d.page_content[:120]}..."
                for i, d in enumerate(result.get("documents", []))
            ]
            pipeline_steps = [
                ":white_check_mark: Retrieved",
                ":white_check_mark: Graded",
                ":white_check_mark: Generated",
                ":white_check_mark: Verified",
                f":stopwatch: {elapsed:.1f}s",
            ]
        except Exception as e:
            answer = f":x: **Error:** {e}"
            sources = []
            pipeline_steps = []

        status_placeholder.empty()
        st.markdown(answer)

        if sources:
            with st.expander(f":link: {len(sources)} source(s)"):
                for s in sources:
                    st.markdown(f"`{s}`")

        if pipeline_steps:
            st.markdown(
                '<div class="pipeline-status">'
                + "".join(f'<span class="pipeline-step done">{s}</span>' for s in pipeline_steps)
                + "</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role": "assistant", "content": answer,
        "sources": sources, "pipeline_steps": pipeline_steps,
    })

st.markdown(
    '<div class="footer">GraphRAG &mdash; Hybrid retrieval (BM25 + Chroma) with self-corrective hallucination checking</div>',
    unsafe_allow_html=True,
)
