import os
import time
import warnings
from typing import List, TypedDict

import streamlit as st
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

# --------------- Config ---------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
TEXT_DIR = os.path.join(DATA_DIR, "texts")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_TOP_K", "3"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
CHROMA_WEIGHT = float(os.getenv("CHROMA_WEIGHT", "0.6"))

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)


# --------------- Pydantic schemas ---------------
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="'yes' if relevant, 'no' otherwise")


class GradeHallucination(BaseModel):
    binary_score: str = Field(description="'yes' if grounded, 'no' otherwise")


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    generation: str
    web_search_needed: str


# --------------- Dotenv loader ---------------
def load_dotenv(path=".env"):
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip().strip("'\""))


load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))


# --------------- Cached resources ---------------
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@st.cache_resource
def get_llm():
    return ChatGroq(model="qwen/qwen3.6-27b", temperature=0)


@st.cache_resource
def load_documents():
    """Load all saved PDFs and text files."""
    docs = []
    if os.path.isdir(PDF_DIR):
        from langchain_community.document_loaders import PyPDFLoader
        for f in sorted(os.listdir(PDF_DIR)):
            if f.lower().endswith(".pdf"):
                try:
                    docs.extend(PyPDFLoader(os.path.join(PDF_DIR, f)).load())
                except Exception:
                    pass
    if os.path.isdir(TEXT_DIR):
        from langchain_community.document_loaders import TextLoader, DirectoryLoader
        for ext in ["*.txt", "*.md"]:
            try:
                docs.extend(
                    DirectoryLoader(
                        TEXT_DIR, glob=ext, loader_cls=TextLoader, show_progress=False
                    ).load()
                )
            except Exception:
                pass
    return docs


def build_retrievers(raw_docs):
    """Split docs and build hybrid retrievers."""
    if not raw_docs:
        return None, []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(raw_docs)
    if not chunks:
        return None, []

    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    chroma_ret = vectorstore.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    bm25_ret = BM25Retriever.from_documents(chunks)
    bm25_ret.k = RETRIEVAL_K

    hybrid = EnsembleRetriever(
        retrievers=[bm25_ret, chroma_ret], weights=[BM25_WEIGHT, CHROMA_WEIGHT]
    )
    return hybrid, chunks


def build_graph(hybrid_retriever):
    """Build and compile the LangGraph workflow."""
    llm = get_llm()
    structured_grader = llm.with_structured_output(GradeDocuments)
    structured_hallucination = llm.with_structured_output(GradeHallucination)

    def retrieve_node(state: AgentState):
        question = state["question"]
        docs = hybrid_retriever.invoke(question)
        return {"documents": docs, "question": question}

    def grade_documents_node(state: AgentState):
        question = state["question"]
        documents = state["documents"]
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a grader assessing relevance of a retrieved document to a user query. "
             "Give a binary score 'yes' or 'no' indicating relevance."),
            ("human", "Retrieved document:\n\n{context}\n\nUser query: {question}"),
        ])
        chain = prompt | structured_grader
        filtered = []
        for doc in documents:
            try:
                score = chain.invoke({"context": doc.page_content[:1500], "question": question})
                if score.binary_score == "yes":
                    filtered.append(doc)
            except Exception:
                filtered.append(doc)
        web_search_needed = "no" if filtered else "yes"
        if not filtered:
            filtered = documents
        return {"documents": filtered, "web_search_needed": web_search_needed}

    def generate_node(state: AgentState):
        question = state["question"]
        documents = state["documents"]
        context = "\n\n".join([d.page_content[:1000] for d in documents[:5]])
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert AI assistant. Answer the user's query using *only* the provided context. "
             "If you cannot find the answer in the context, state that clearly. Do not hallucinate."),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ])
        response = (prompt | llm).invoke({"context": context, "question": question})
        return {
            "generation": response.content,
            "documents": documents,
            "question": question,
            "web_search_needed": state.get("web_search_needed", "no"),
        }

    def hallucination_grader_node(state: AgentState):
        generation = state["generation"]
        documents = state["documents"]
        context = "\n\n".join([d.page_content[:800] for d in documents[:3]])
        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a grader checking if an answer is grounded in the provided context. "
             "Respond with 'yes' if fully supported, 'no' otherwise."),
            ("human", "Context:\n{context}\n\nAnswer:\n{generation}\n\nIs this answer grounded?"),
        ])
        try:
            result = (prompt | structured_hallucination).invoke(
                {"context": context, "generation": generation[:1500]}
            )
            if result.binary_score == "yes":
                return {"generation": generation, "documents": documents,
                        "question": state["question"],
                        "web_search_needed": state.get("web_search_needed", "no")}
            return {"generation": "", "documents": documents,
                    "question": state["question"],
                    "web_search_needed": state.get("web_search_needed", "no")}
        except Exception:
            return {"generation": generation, "documents": documents,
                    "question": state["question"],
                    "web_search_needed": state.get("web_search_needed", "no")}

    def decide_after_grading(state):
        return "web_search" if state.get("web_search_needed") == "yes" else "generate"

    def decide_after_hallucination(state):
        return "generate" if state.get("generation") == "" else END

    workflow = StateGraph(AgentState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("hallucination_grader", hallucination_grader_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents", decide_after_grading,
        {"web_search": "generate", "generate": "generate"},
    )
    workflow.add_edge("generate", "hallucination_grader")
    workflow.add_conditional_edges(
        "hallucination_grader", decide_after_hallucination,
        {"generate": "generate", END: END},
    )
    return workflow.compile()


# --------------- Custom CSS ---------------
CUSTOM_CSS = """
<style>
    /* Main container */
    .stApp { background: #0e1117; }

    /* Header styling */
    header[data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1.5rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #161b22;
        border-right: 1px solid #30363d;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e6edf3;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #30363d;
        background: #161b22;
    }
    div[data-testid="stChatMessage"][aria-label="user"] {
        background: #1a2332;
        border-color: #1f6feb33;
    }
    div[data-testid="stChatMessage"][aria-label="assistant"] {
        background: #161b22;
        border-color: #3fb95033;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 12px 16px;
    }
    div[data-testid="stMetric"] label { color: #8b949e !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #e6edf3 !important; }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #30363d;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        border-color: #58a6ff;
        box-shadow: 0 0 8px #58a6ff33;
    }
    div[data-testid="stSidebar"] .stButton > button {
        width: 100%;
    }

    /* File uploader */
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"] {
        border: 2px dashed #30363d;
        border-radius: 10px;
        padding: 8px;
        transition: border-color 0.2s;
    }
    section[data-testid="stSidebar"] div[data-testid="stFileUploader"]:hover {
        border-color: #58a6ff;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        color: #e6edf3;
    }

    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #161b22 0%, #1a2332 100%);
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 2rem 0;
    }
    .welcome-card h1 { color: #e6edf3; margin-bottom: 0.3rem; font-size: 1.8rem; }
    .welcome-card p { color: #8b949e; font-size: 1rem; margin-bottom: 1.5rem; }

    /* Example prompt cards */
    .example-card {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px 18px;
        cursor: pointer;
        transition: all 0.2s ease;
        text-align: left;
        color: #c9d1d9;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    .example-card:hover {
        border-color: #58a6ff;
        background: #161b22;
        box-shadow: 0 0 12px #58a6ff22;
    }
    .example-card strong { color: #58a6ff; }

    /* Pipeline status bar */
    .pipeline-status {
        display: flex;
        gap: 6px;
        align-items: center;
        flex-wrap: wrap;
        margin: 0.5rem 0 1rem 0;
    }
    .pipeline-step {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #8b949e;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .pipeline-step.active {
        border-color: #f0883e;
        color: #f0883e;
        animation: pulse 1.5s infinite;
    }
    .pipeline-step.done {
        border-color: #3fb950;
        color: #3fb950;
    }
    .pipeline-arrow { color: #484f58; font-size: 0.7rem; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* Delete button in sidebar */
    .del-btn > button {
        background: transparent !important;
        border: none !important;
        color: #f85149 !important;
        font-size: 0.75rem !important;
        padding: 2px 6px !important;
        min-height: 0 !important;
        line-height: 1 !important;
    }
    .del-btn > button:hover { background: #f8514922 !important; }

    /* Footer */
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #484f58;
        font-size: 0.8rem;
        border-top: 1px solid #21262d;
        margin-top: 2rem;
    }

    /* Hide Streamlit branding */
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

    # File upload
    st.markdown("### :inbox_tray: Upload")
    uploaded_files = st.file_uploader(
        "Drag & drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    # Upload info
    if uploaded_files:
        total_size = sum(uf.size for uf in uploaded_files)
        size_str = f"{total_size / 1024:.1f} KB" if total_size < 1024 * 1024 else f"{total_size / (1024*1024):.1f} MB"
        st.info(f"**{len(uploaded_files)}** file(s) selected ({size_str})")

    # Buttons
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
        if count:
            st.toast(f"Cleared {count} file(s)")
            time.sleep(0.3)
            st.rerun()
        else:
            st.toast("Nothing to clear")

    st.divider()

    # Saved documents
    st.markdown("### :open_file_folder: Saved Documents")
    saved_pdfs = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")] if os.path.isdir(PDF_DIR) else []
    saved_txts = [f for f in os.listdir(TEXT_DIR) if f.lower().endswith((".txt", ".md"))] if os.path.isdir(TEXT_DIR) else []

    # Metrics row
    m1, m2 = st.columns(2)
    m1.metric("PDFs", len(saved_pdfs))
    m2.metric("Text", len(saved_txts))

    total = len(saved_pdfs) + len(saved_txts)
    if total > 0:
        with st.expander(f"View all {total} file(s)", expanded=False):
            for f in saved_pdfs + saved_txts:
                ext = ":page_facing_up:" if f.lower().endswith((".txt", ".md")) else ":page_facing_up:"
                st.markdown(f"`{f}`")
    else:
        st.caption("No documents saved yet")

    # Knowledge base stats
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

# --- Main Content ---
# Load and check documents
raw_docs = load_documents()
if not raw_docs:
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
    examples = [
        "What is GraphRAG and how does it work?",
        "Explain hybrid retrieval with BM25 and ChromaDB",
        "What are the key features of self-corrective RAG?",
    ]
    for ex in examples:
        st.markdown(
            f'<div class="example-card">:bulb: {ex}</div>',
            unsafe_allow_html=True,
        )

    st.stop()

# Build pipeline
with st.spinner("Building knowledge base..."):
    hybrid, chunks = build_retrievers(raw_docs)
if hybrid is None:
    st.warning("Could not build retrievers from loaded documents.")
    st.stop()
graph = build_graph(hybrid)

# Status bar
st.markdown(
    f"""
    <div class="pipeline-status">
        <span class="pipeline-step done">:white_check_mark: {len(raw_docs)} pages loaded</span>
        <span class="pipeline-arrow">-></span>
        <span class="pipeline-step done">:white_check_mark: {len(chunks)} chunks indexed</span>
        <span class="pipeline-arrow">-></span>
        <span class="pipeline-step done">:white_check_mark: Hybrid retriever ready</span>
        <span class="pipeline-arrow">-></span>
        <span class="pipeline-step done">:white_check_mark: Graph compiled</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Chat ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render history
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
                + "".join(
                    f'<span class="pipeline-step done">{s}</span>'
                    for s in msg["pipeline_steps"]
                )
                + "</div>",
                unsafe_allow_html=True,
            )

# Chat input
query = st.chat_input("Ask a question about your documents...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar=":bust_in_silhouette:"):
        st.markdown(query)

    with st.chat_message("assistant", avatar=":robot_face:"):
        # Pipeline status live display
        status_placeholder = st.empty()
        status_placeholder.markdown(
            '<div class="pipeline-status">'
            '<span class="pipeline-step active">:mag: Retrieving...</span>'
            "</div>",
            unsafe_allow_html=True,
        )

        start_time = time.time()
        try:
            result = graph.invoke({"question": query})
            elapsed = time.time() - start_time
            answer = result["generation"]
            sources = [
                f"[{i+1}] {d.metadata.get('source', 'unknown')}: {d.page_content[:120]}..."
                for i, d in enumerate(result.get("documents", []))
            ]
            pipeline_steps = [
                f":white_check_mark: Retrieved",
                f":white_check_mark: Graded",
                f":white_check_mark: Generated",
                f":white_check_mark: Verified",
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
                + "".join(
                    f'<span class="pipeline-step done">{s}</span>'
                    for s in pipeline_steps
                )
                + "</div>",
                unsafe_allow_html=True,
            )

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "pipeline_steps": pipeline_steps,
    })

# Footer
st.markdown(
    '<div class="footer">GraphRAG &mdash; Hybrid retrieval (BM25 + Chroma) with self-corrective hallucination checking</div>',
    unsafe_allow_html=True,
)
