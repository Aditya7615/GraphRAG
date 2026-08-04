"""FastAPI application: ingestion + retrieval + grounded chat."""

from __future__ import annotations

import logging
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.config import settings
from backend.generation.chain import (
    GeminiLLMUnavailable,
    GeminiRateLimited,
    RAGChain,
    get_chain,
)
from backend.ingestion.docstore import get_docstore
from backend.ingestion.loader import SUPPORTED_SUFFIXES
from backend.ingestion.pipeline import IngestionPipeline, get_pipeline
from backend.ingestion.vector_store import get_vector_store
from backend.models.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    SearchHit,
    SearchRequest,
    SearchResponse,
    StatsResponse,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("rag.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the expensive singletons at boot so the first user query isn't the
    # one that pays for model loading.
    logger.info("Starting RAG API (collection=%s)", settings.qdrant_collection)
    try:
        store = get_vector_store()
        if store.ping():
            store.ensure_collection()
        else:
            logger.warning("Qdrant unreachable at %s - start it with docker-compose.", settings.qdrant_url)
        get_docstore()
    except Exception:
        logger.exception("Startup warm-up failed; API will still serve /health.")
    yield
    logger.info("Shutting down RAG API")


app = FastAPI(
    title="Enterprise RAG API",
    description="Hybrid-search RAG with parent-child chunking and zero-hallucination guardrails.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def pipeline_dep() -> IngestionPipeline:
    return get_pipeline()


def chain_dep() -> RAGChain:
    return get_chain()


@app.exception_handler(GeminiLLMUnavailable)
async def _llm_unavailable(_, exc: GeminiLLMUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(GeminiRateLimited)
async def _llm_rate_limited(_, exc: GeminiRateLimited):
    seconds = exc.retry_after_seconds
    headers = {"Retry-After": str(seconds)} if seconds else None
    return JSONResponse(status_code=429, content={"detail": str(exc)}, headers=headers)


# --- health & stats ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    store = get_vector_store()
    qdrant_ok = store.ping()
    docstore = get_docstore()
    return HealthResponse(
        status="ok" if qdrant_ok else "degraded",
        qdrant="up" if qdrant_ok else "down",
        docstore=f"{docstore.backend}:{'up' if docstore.ping() else 'down'}",
        llm="configured" if settings.google_api_key else "missing GOOGLE_API_KEY",
        collection_exists=store.collection_exists(),
    )


@app.get("/stats", response_model=StatsResponse, tags=["system"])
def stats() -> StatsResponse:
    store = get_vector_store()
    docstore = get_docstore()
    vector_stats = store.stats()
    return StatsResponse(
        collection=vector_stats["collection"],
        vectors=vector_stats["vectors"],
        indexed_vectors=vector_stats.get("indexed_vectors", 0),
        parent_chunks=docstore.count(),
        documents=docstore.list_documents(),
        docstore_backend=docstore.backend,
        embedding_model=settings.embedding_model,
        llm_model=settings.llm_model,
    )


# --- ingestion --------------------------------------------------------------
@app.post("/ingest/file", response_model=IngestResponse, tags=["ingestion"])
async def ingest_file(
    file: UploadFile = File(...),
    doc_type: str | None = Query(None),
    pipeline: IngestionPipeline = Depends(pipeline_dep),
) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Allowed: {sorted(SUPPORTED_SUFFIXES)}")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    target = settings.upload_dir / Path(file.filename or "unnamed").name
    try:
        with target.open("wb") as fh:
            shutil.copyfileobj(file.file, fh)
    finally:
        await file.close()

    try:
        result = pipeline.ingest_file(target, doc_type=doc_type)
    except (FileNotFoundError, ValueError, OSError) as exc:
        logger.exception("Ingestion failed for %s", target.name)
        raise HTTPException(500, f"Ingestion failed: {exc}") from exc

    return IngestResponse(
        status="ok" if result.documents else "failed",
        documents=result.documents,
        parent_chunks=result.parent_chunks,
        child_chunks=result.child_chunks,
        skipped=result.skipped,
        latency_ms=round(result.latency_ms, 2),
        details=result.details,
    )


@app.post("/ingest/directory", response_model=IngestResponse, tags=["ingestion"])
def ingest_directory(
    directory: str = Query(default=None, description="Defaults to the synthetic data folder."),
    doc_type: str | None = Query(None),
    pipeline: IngestionPipeline = Depends(pipeline_dep),
) -> IngestResponse:
    path = Path(directory) if directory else settings.synthetic_data_dir
    if not path.is_dir():
        raise HTTPException(404, f"Directory not found: {path}")

    result = pipeline.ingest_directory(path, doc_type=doc_type)
    return IngestResponse(
        status="ok" if result.documents else "failed",
        documents=result.documents,
        parent_chunks=result.parent_chunks,
        child_chunks=result.child_chunks,
        skipped=result.skipped,
        latency_ms=round(result.latency_ms, 2),
        details=result.details,
    )


@app.delete("/documents/{doc_id}", tags=["ingestion"])
def delete_document(doc_id: str, pipeline: IngestionPipeline = Depends(pipeline_dep)) -> dict:
    removed = pipeline.delete_document(doc_id)
    if not removed:
        raise HTTPException(404, f"No document with id {doc_id}")
    return {"status": "deleted", "doc_id": doc_id, "parent_chunks_removed": removed}


@app.post("/admin/reset", tags=["ingestion"])
def reset_index(
    confirm: bool = Query(False, description="Must be true - this wipes the index."),
    pipeline: IngestionPipeline = Depends(pipeline_dep),
) -> dict:
    if not confirm:
        raise HTTPException(400, "Pass ?confirm=true to wipe the collection and docstore.")
    pipeline.reset()
    return {"status": "reset", "collection": settings.qdrant_collection}


# --- retrieval & chat -------------------------------------------------------
@app.post("/search", response_model=SearchResponse, tags=["retrieval"])
def search(request: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    try:
        hits = get_vector_store().search(
            request.query, top_k=request.top_k, doc_ids=request.doc_ids, mode=request.search_mode
        )
    except (ConnectionError, OSError, ValueError) as exc:
        logger.exception("Search failed")
        raise HTTPException(500, f"Search failed: {exc}") from exc
    return SearchResponse(
        query=request.query,
        hits=[SearchHit(**hit) for hit in hits],
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
        search_mode=request.search_mode,
    )


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest, chain: RAGChain = Depends(chain_dep)) -> ChatResponse:
    try:
        return chain.answer(
            question=request.question,
            top_k=request.top_k,
            max_parents=request.max_parents,
            doc_ids=request.doc_ids,
            mode=request.search_mode,
            include_parent_text=request.include_parent_text,
        )
    except (GeminiLLMUnavailable, GeminiRateLimited):
        raise
    except Exception as exc:
        from google.api_core.exceptions import ResourceExhausted as _ResourceExhausted
        if isinstance(exc, _ResourceExhausted):
            raise HTTPException(429, "Google Gemini daily quota exhausted. Please try again later.") from exc
        logger.exception("Chat failed")
        raise HTTPException(500, f"Generation failed: {exc}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("backend.main:app", host=settings.api_host, port=settings.api_port, reload=True)
