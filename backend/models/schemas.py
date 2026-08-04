"""Pydantic contracts shared by the API, the chain and the Streamlit client."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A single source the LLM was allowed to look at, plus its provenance."""

    marker: str = Field(..., description="Inline marker used in the answer, e.g. 'S1'.")
    parent_id: str
    doc_id: str
    source: str = Field(..., description="Human readable document name.")
    section: str | None = None
    page: int | None = None
    score: float = Field(0.0, description="Fused hybrid relevance score.")
    used_by_llm: bool = Field(
        False, description="True when the generated answer actually cites this marker."
    )
    snippet: str = ""
    parent_text: str = ""


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=50)
    max_parents: int | None = Field(None, ge=1, le=12)
    doc_ids: list[str] | None = Field(None, description="Restrict search to these documents.")
    search_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    include_parent_text: bool = False


class ChatResponse(BaseModel):
    answer: str
    grounded: bool = Field(..., description="False when the guardrail fallback was returned.")
    citations: list[Citation] = []
    question: str
    latency_ms: dict[str, float] = {}
    model: str
    search_mode: str


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(10, ge=1, le=100)
    doc_ids: list[str] | None = None
    search_mode: Literal["hybrid", "dense", "sparse"] = "hybrid"


class SearchHit(BaseModel):
    child_id: str
    parent_id: str
    doc_id: str
    source: str
    section: str | None = None
    page: int | None = None
    score: float
    text: str


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    latency_ms: float
    search_mode: str


class IngestResponse(BaseModel):
    status: str
    documents: int
    parent_chunks: int
    child_chunks: int
    skipped: list[str] = []
    latency_ms: float
    details: list[dict[str, Any]] = []


class DocumentSummary(BaseModel):
    doc_id: str
    source: str
    parent_chunks: int
    doc_type: str | None = None


class StatsResponse(BaseModel):
    collection: str
    vectors: int
    indexed_vectors: int
    parent_chunks: int
    documents: list[DocumentSummary]
    docstore_backend: str
    embedding_model: str
    llm_model: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    docstore: str
    llm: str
    collection_exists: bool
