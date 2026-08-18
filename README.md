# GraphRAG

**Enterprise-Grade Self-Corrective Agentic RAG Engine**

A production-ready Retrieval-Augmented Generation framework built with LangGraph, LangChain, and Groq — featuring hybrid retrieval, Pydantic-structured grading, self-corrective loops, and a polished Streamlit UI.

## Demo

```
Upload Documents → Ask Questions → Get Grounded Answers
```

## Features

| Feature | Description |
|---------|-------------|
| **Streamlit UI** | Polished web interface with document upload, chat, and real-time pipeline status |
| **Hybrid Retrieval** | BM25 (sparse/keyword) + ChromaDB (dense/semantic) with Reciprocal Rank Fusion |
| **Pydantic Structured Outputs** | Type-safe document grading and hallucination verification via `with_structured_output()` |
| **Self-Corrective Loops** | Automated retry on hallucination detection using LangGraph conditional edges |
| **Web Search Fallback** | Tavily integration triggers when local retrieval finds no relevant documents |
| **Local Embeddings** | HuggingFace BGE-small model — no API key needed for embeddings |
| **Document Ingestion** | PDF, TXT, and Markdown support with drag-and-drop upload |
| **RAGAS Evaluation** | Built-in evaluation metrics for RAG pipeline quality assessment |

## Architecture

```
User Query
    │
    ▼
┌─────────────────┐
│  RETRIEVE        │  Hybrid: BM25 (40%) + Chroma (60%) with RRF
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  GRADE DOCUMENTS │  Pydantic-structured Groq LLM grading (yes/no)
└────────┬────────┘
         │
    ┌────┴────┐
    │ relevant?│
    └────┬────┘
    yes  │  no
    │    └──────► (web search fallback if configured)
    │                    │
    ▼                    ▼
┌─────────────────┐
│  GENERATE        │  Groq Qwen3.6-27B response
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│  HALLUCINATION GRADER│  Pydantic-structured (pass/fail)
└────────┬────────────┘
         │
    ┌────┴────┐
    │ grounded?│
    └────┬────┘
    yes  │  no
    │    └──────► regenerate (loop)
    ▼
  END → Response
```

## Tech Stack

| Category | Library |
|----------|---------|
| **LLM** | [Groq](https://groq.com/) — Qwen3.6-27B (free tier, fast inference) |
| **Embeddings** | [HuggingFace BGE-small](https://huggingface.co/BAAI/bge-small-en-v1.5) (local, no API key) |
| **Orchestration** | [LangGraph](https://github.com/langchain-ai/langgraph) — cyclical state graphs |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) — in-memory semantic search |
| **Keyword Search** | [BM25](https://github.com/dorianbrown/rank_bm25) — sparse lexical retrieval |
| **Structured Outputs** | [Pydantic](https://docs.pydantic.dev/) — type-safe LLM grading |
| **Web Search** | [Tavily](https://tavily.com/) — fallback when local retrieval fails |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) — RAG pipeline quality assessment |
| **UI** | [Streamlit](https://streamlit.io/) — interactive web interface |

## Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/Aditya7615/GraphRAG.git
cd GraphRAG
pip install -r requirements.txt
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env with your keys
```

Required:
- `GROQ_API_KEY` — Get free at [console.groq.com/keys](https://console.groq.com/keys)

Optional:
- `TAVILY_API_KEY` — Get at [tavily.com](https://tavily.com) (for web search fallback)

### 3. Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Opens at **http://localhost:8502** with:
- Drag-and-drop document upload (PDF, TXT, MD)
- Chat interface with real-time pipeline status
- Source attribution with expandable document snippets

### 4. Deploy to Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and select `streamlit_app.py`
4. Add secret in **Manage app → Secrets**:
   ```toml
   GROQ_API_KEY = "your_key_here"
   ```

### 4. (Alternative) Run the Notebook

```bash
jupyter notebook GraphRAG.ipynb
```

Run cells top-to-bottom for the full pipeline with test queries.

## Project Structure

```
GraphRAG/
├── streamlit_app.py         # Streamlit UI — main entry point
├── graphrag.py              # Core RAG pipeline
├── evaluate.py              # RAGAS evaluation module
├── GraphRAG.ipynb           # Jupyter notebook — full pipeline
├── evaluation.ipynb         # Jupyter notebook — RAGAS evaluation
├── requirements.txt         # Python dependencies
├── .env.example             # Environment variable template
├── .env                     # Your API keys (git-ignored)
├── data/
│   ├── pdfs/                # Drop PDF files here (git-ignored)
│   ├── texts/               # Drop .txt/.md files here (git-ignored)
│   └── chroma_db/           # Persisted vector store (git-ignored)
└── README.md
```

## UI Features

### Welcome Screen
When no documents are loaded, the app shows a welcome card with:
- 3-step guide (Upload → Save → Ask)
- Example question prompts

### Sidebar
- **File Upload**: Drag-and-drop PDF, TXT, or MD files
- **Save**: Persists uploaded files to `data/pdfs/` or `data/texts/`
- **Clear All**: Removes all saved documents
- **Document Stats**: PDF/text counts with expandable file lists
- **Pipeline Info**: Retrieval weights, chunk size, LLM model details

### Chat Interface
- User/assistant avatars on messages
- Live pipeline status bar (Retrieving → Graded → Generated → Verified)
- Response timing display
- Expandable source snippets for each answer
- Full conversation history

## Adding Documents

### Via Streamlit UI (Recommended)
1. Open the app at http://localhost:8501
2. Drag files into the sidebar upload area
3. Click **Save**

### Via File System
```bash
cp document.pdf data/pdfs/
cp notes.txt data/texts/
```

### Via Notebook
Edit the `manual_docs` list in Cell 3c of the notebook.

All sources merge automatically into the same vector store.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `TAVILY_API_KEY` | — | Tavily API key (optional) |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | HuggingFace embedding model |
| `CHUNK_SIZE` | `500` | Text splitter chunk size |
| `CHUNK_OVERLAP` | `50` | Text splitter overlap |
| `RETRIEVAL_TOP_K` | `3` | Number of docs to retrieve |
| `BM25_WEIGHT` | `0.4` | BM25 ensemble weight |
| `CHROMA_WEIGHT` | `0.6` | Chroma ensemble weight |

## How It Works

1. **Document Ingestion** — PDFs, text files, and markdown are loaded and split into 500-character chunks with 50-character overlap.

2. **Hybrid Retrieval** — Combines BM25 keyword search (40%) with ChromaDB semantic search (60%) using Reciprocal Rank Fusion for superior ranked results.

3. **Document Grading** — Each retrieved document is graded for relevance using a Pydantic-structured Groq LLM call. Irrelevant docs are filtered out.

4. **Web Search Fallback** — If no documents pass the relevance filter, Tavily web search is triggered to supplement the context (when configured).

5. **Generation** — The LLM generates a response grounded in the filtered context.

6. **Hallucination Grading** — A second Pydantic-structured LLM call verifies the answer is grounded in the provided context. If hallucinated, the generation is retried.

## Resume-Ready Highlights

| Feature | What It Demonstrates |
|---------|---------------------|
| **Hybrid Retrieval (BM25 + RRF)** | Search/recommendation systems, e-commerce relevance |
| **Pydantic Structured Outputs** | Production reliability, type-safe grading pipelines |
| **Self-Corrective Loops** | Fault-tolerant agentic systems, error recovery |
| **Web Search Fallback** | Graceful degradation, external tool integration |
| **Streamlit UI** | User-facing product, interactive data application |
| **LangGraph State Machine** | Complex workflow orchestration, cyclical graphs |

## RAGAS Evaluation

The GraphRAG pipeline includes built-in evaluation using [RAGAS](https://docs.ragas.io/) (Retrieval Augmented Generation Assessment) metrics.

### Metrics

| Metric | Description |
|--------|-------------|
| **Faithfulness** | How grounded the answer is in the provided context |
| **Answer Relevancy** | How relevant the answer is to the user's question |
| **Context Precision** | How precise the retrieved contexts are |
| **Context Recall** | How well the contexts recall relevant information |

### Usage

#### Via Streamlit UI

1. Run the Streamlit app: `streamlit run streamlit_app.py`
2. Scroll down to the **RAGAS Evaluation** section
3. Enter evaluation questions (one per line)
4. Optionally enter ground truth answers
5. Click **Run Evaluation**

#### Via Python

```python
from graphrag import GraphRAG
from evaluate import RAGASEvaluator

engine = GraphRAG()
evaluator = RAGASEvaluator(engine)

questions = [
    "What is GraphRAG?",
    "How does hybrid retrieval work?",
]

result = evaluator.evaluate(questions)

print(f"Faithfulness: {result.faithfulness:.2%}")
print(f"Answer Relevancy: {result.answer_relevancy:.2%}")
print(f"Context Precision: {result.context_precision:.2%}")
print(f"Context Recall: {result.context_recall:.2%}")
print(f"Overall Score: {result.overall_score:.2%}")
```

#### Via Notebook

Run `evaluation.ipynb` for a step-by-step walkthrough with visualization:

```bash
jupyter notebook evaluation.ipynb
```

### Configuration

The evaluation uses:
- **LLM**: Groq (same as main pipeline) via `LangchainLLMWrapper`
- **Embeddings**: HuggingFace BGE-small (`BAAI/bge-small-en-v1.5`) — no API key needed
- **Timeout**: 300s with 5 retries, 4 concurrent workers (avoids Groq rate limits)

To adjust timeout settings, modify `RunConfig` in `evaluate.py`:

```python
from ragas.run_config import RunConfig
run_config = RunConfig(timeout=300, max_retries=5, max_workers=4)
```

## License

MIT
