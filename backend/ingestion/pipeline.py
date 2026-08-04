"""Ingestion orchestration: file -> parents (docstore) + children (Qdrant)."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.ingestion.chunker import ParentChildChunker, SourceDocument
from backend.ingestion.docstore import get_docstore
from backend.ingestion.loader import load_directory, load_document
from backend.ingestion.vector_store import get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class IngestionResult:
    documents: int = 0
    parent_chunks: int = 0
    child_chunks: int = 0
    skipped: list[str] = field(default_factory=list)
    details: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: float = 0.0


class IngestionPipeline:
    def __init__(self, chunker: ParentChildChunker | None = None) -> None:
        self.chunker = chunker or ParentChildChunker()
        self.docstore = get_docstore()
        self.vector_store = get_vector_store()

    def ingest_documents(self, docs: Sequence[SourceDocument]) -> IngestionResult:
        started = time.perf_counter()
        result = IngestionResult()
        if not docs:
            return result

        self.vector_store.ensure_collection()

        for doc in docs:
            try:
                parents, children = self.chunker.split(doc)
                if not parents:
                    result.skipped.append(f"{doc.source}: no content after chunking")
                    continue

                # Parents first: a child hit whose parent is missing is worse
                # than a child that isn't searchable yet.
                self.docstore.upsert(parents)
                written = self.vector_store.upsert_children(children)

                result.documents += 1
                result.parent_chunks += len(parents)
                result.child_chunks += written
                result.details.append(
                    {
                        "doc_id": doc.doc_id,
                        "source": doc.source,
                        "pages": len(doc.pages),
                        "parent_chunks": len(parents),
                        "child_chunks": written,
                        "doc_type": doc.metadata.get("doc_type"),
                    }
                )
                logger.info(
                    "Ingested %s -> %s parents / %s children",
                    doc.source,
                    len(parents),
                    written,
                )
            except Exception as exc:
                logger.exception("Ingestion failed for %s", doc.source)
                result.skipped.append(f"{doc.source}: {exc}")

        result.latency_ms = (time.perf_counter() - started) * 1000
        return result

    def ingest_file(self, path: str | Path, doc_type: str | None = None) -> IngestionResult:
        return self.ingest_documents([load_document(path, doc_type=doc_type)])

    def ingest_directory(self, directory: str | Path, doc_type: str | None = None) -> IngestionResult:
        return self.ingest_documents(load_directory(directory, doc_type=doc_type))

    def delete_document(self, doc_id: str) -> int:
        self.vector_store.delete_document(doc_id)
        return self.docstore.delete_document(doc_id)

    def reset(self) -> None:
        self.vector_store.recreate()
        self.docstore.clear()


_pipeline: IngestionPipeline | None = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> IngestionPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                _pipeline = IngestionPipeline()
    return _pipeline
