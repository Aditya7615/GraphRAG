"""Parent-document retriever.

Search over precise child vectors, then hand the LLM the full parent chunk each
winning child belongs to. Parents are ranked by the best child they contain and
deduplicated, so four adjacent child hits from one section cost one context slot
instead of four.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from backend.config import settings
from backend.ingestion.chunker import count_tokens
from backend.ingestion.docstore import BaseDocStore, get_docstore
from backend.ingestion.vector_store import (
    HybridVectorStore,
    SearchMode,
    get_vector_store,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievedParent:
    marker: str
    parent_id: str
    doc_id: str
    source: str
    text: str
    section: str | None
    page: int | None
    score: float
    snippet: str
    child_hits: int = 1

    def to_context_block(self) -> str:
        locator = " | ".join(
            part
            for part in (
                f"source: {self.source}",
                f"section: {self.section}" if self.section else "",
                f"page: {self.page}" if self.page else "",
            )
            if part
        )
        return f"[{self.marker}] ({locator})\n{self.text.strip()}"


@dataclass
class RetrievalResult:
    parents: list[RetrievedParent]
    child_hits: list[dict[str, Any]]
    context: str
    latency_ms: float

    @property
    def is_empty(self) -> bool:
        return not self.parents


class ParentDocumentRetriever:
    def __init__(
        self,
        vector_store: HybridVectorStore | None = None,
        docstore: BaseDocStore | None = None,
    ) -> None:
        self.vector_store = vector_store or get_vector_store()
        self.docstore = docstore or get_docstore()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        max_parents: int | None = None,
        doc_ids: Sequence[str] | None = None,
        mode: SearchMode = "hybrid",
    ) -> RetrievalResult:
        started = time.perf_counter()
        top_k = top_k or settings.retrieval_top_k
        max_parents = max_parents or settings.max_parents_in_context

        hits = self.vector_store.search(query, top_k=top_k, doc_ids=doc_ids, mode=mode)
        hits = [h for h in hits if h["score"] >= settings.min_relevance_score]
        if not hits:
            return RetrievalResult([], [], "", (time.perf_counter() - started) * 1000)

        # Collapse children into their parents, keeping the best child score.
        ranked: dict[str, dict[str, Any]] = {}
        for hit in hits:
            parent_id = hit["parent_id"]
            if not parent_id:
                continue
            entry = ranked.get(parent_id)
            if entry is None:
                ranked[parent_id] = {"hit": hit, "score": hit["score"], "count": 1}
            else:
                entry["count"] += 1
                if hit["score"] > entry["score"]:
                    entry["score"] = hit["score"]
                    entry["hit"] = hit

        order = sorted(ranked.items(), key=lambda kv: kv[1]["score"], reverse=True)[:max_parents]
        records = self.docstore.mget([pid for pid, _ in order])

        parents: list[RetrievedParent] = []
        used_tokens = 0
        for pid, meta in order:
            record = records.get(pid)
            hit = meta["hit"]
            # Missing parent (docstore drift) degrades to the child text rather
            # than dropping a genuinely relevant result.
            text = record["text"] if record else hit["text"]
            tokens = count_tokens(text)
            if used_tokens + tokens > settings.max_context_tokens and parents:
                logger.debug("Context budget reached at %s parents", len(parents))
                break
            used_tokens += tokens

            parents.append(
                RetrievedParent(
                    marker=f"S{len(parents) + 1}",
                    parent_id=pid,
                    doc_id=(record or hit)["doc_id"],
                    source=(record or hit)["source"],
                    text=text,
                    # Locate the citation at the *child* that actually matched -
                    # a parent spans ~2000 tokens and several sections, so its
                    # own section/page would point the reader too coarsely.
                    section=hit.get("section") or (record or hit).get("section"),
                    page=hit.get("page") or (record or hit).get("page"),
                    score=round(meta["score"], 6),
                    snippet=hit["text"][:320].strip(),
                    child_hits=meta["count"],
                )
            )

        context = "\n\n---\n\n".join(p.to_context_block() for p in parents)
        return RetrievalResult(
            parents=parents,
            child_hits=hits,
            context=context,
            latency_ms=(time.perf_counter() - started) * 1000,
        )


_retriever: ParentDocumentRetriever | None = None
_retriever_lock = threading.Lock()


def get_retriever() -> ParentDocumentRetriever:
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                _retriever = ParentDocumentRetriever()
    return _retriever
