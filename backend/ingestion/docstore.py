"""Parent document store.

Parents are never vectorised - they're plain key/value blobs fetched by ID after
retrieval. Postgres is the production backend; the JSON store is a zero-infra
fallback so the pipeline runs end-to-end before anyone starts a container.
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from backend.config import settings
from backend.ingestion.chunker import ParentChunk

logger = logging.getLogger(__name__)


class BaseDocStore(ABC):
    backend: str = "base"

    @abstractmethod
    def upsert(self, parents: Iterable[ParentChunk]) -> int: ...

    @abstractmethod
    def get(self, parent_id: str) -> dict[str, Any] | None: ...

    @abstractmethod
    def mget(self, parent_ids: Iterable[str]) -> dict[str, dict[str, Any]]: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def list_documents(self) -> list[dict[str, Any]]: ...

    @abstractmethod
    def delete_document(self, doc_id: str) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    def ping(self) -> bool:
        try:
            self.count()
            return True
        except (ConnectionError, OSError):
            return False


class JSONDocStore(BaseDocStore):
    """Thread-safe, atomically-written JSON store. Fine up to ~1M parents."""

    backend = "json"

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or settings.json_docstore_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._data: dict[str, dict[str, Any]] = self._read()

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error("Corrupt docstore at %s (%s); starting empty.", self.path, exc)
            return {}

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)  # atomic on POSIX

    def upsert(self, parents: Iterable[ParentChunk]) -> int:
        with self._lock:
            n = 0
            for parent in parents:
                self._data[parent.parent_id] = parent.to_record()
                n += 1
            self._flush()
            return n

    def get(self, parent_id: str) -> dict[str, Any] | None:
        return self._data.get(parent_id)

    def mget(self, parent_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        return {pid: self._data[pid] for pid in parent_ids if pid in self._data}

    def count(self) -> int:
        return len(self._data)

    def list_documents(self) -> list[dict[str, Any]]:
        docs: dict[str, dict[str, Any]] = {}
        for record in self._data.values():
            entry = docs.setdefault(
                record["doc_id"],
                {
                    "doc_id": record["doc_id"],
                    "source": record["source"],
                    "parent_chunks": 0,
                    "doc_type": (record.get("metadata") or {}).get("doc_type"),
                },
            )
            entry["parent_chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"])

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            targets = [k for k, v in self._data.items() if v["doc_id"] == doc_id]
            for key in targets:
                del self._data[key]
            self._flush()
            return len(targets)

    def clear(self) -> None:
        with self._lock:
            self._data = {}
            self._flush()


class PostgresDocStore(BaseDocStore):
    """JSONB-backed parent store with an index on doc_id."""

    backend = "postgres"

    def __init__(self, dsn: str | None = None) -> None:
        from sqlalchemy import create_engine, text

        self._text = text
        self.engine = create_engine(
            dsn or settings.postgres_dsn, pool_pre_ping=True, pool_size=5, max_overflow=10
        )
        self._create_schema()

    def _create_schema(self) -> None:
        ddl = """
        CREATE TABLE IF NOT EXISTS parent_chunks (
            parent_id   TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            source      TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            page        INTEGER,
            section     TEXT,
            token_count INTEGER,
            text        TEXT NOT NULL,
            metadata    JSONB DEFAULT '{}'::jsonb,
            created_at  TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS idx_parent_doc_id ON parent_chunks (doc_id);
        """
        with self.engine.begin() as conn:
            for statement in filter(None, (s.strip() for s in ddl.split(";"))):
                conn.execute(self._text(statement))

    def upsert(self, parents: Iterable[ParentChunk]) -> int:
        rows = [
            {
                "parent_id": p.parent_id,
                "doc_id": p.doc_id,
                "source": p.source,
                "chunk_index": p.index,
                "page": p.page,
                "section": p.section,
                "token_count": p.token_count,
                "text": p.text,
                "metadata": json.dumps(p.metadata),
            }
            for p in parents
        ]
        if not rows:
            return 0
        sql = self._text(
            """
            INSERT INTO parent_chunks
                (parent_id, doc_id, source, chunk_index, page, section,
                 token_count, text, metadata)
            VALUES
                (:parent_id, :doc_id, :source, :chunk_index, :page, :section,
                 :token_count, :text, CAST(:metadata AS JSONB))
            ON CONFLICT (parent_id) DO UPDATE SET
                text = EXCLUDED.text,
                section = EXCLUDED.section,
                page = EXCLUDED.page,
                token_count = EXCLUDED.token_count,
                metadata = EXCLUDED.metadata
            """
        )
        with self.engine.begin() as conn:
            conn.execute(sql, rows)
        return len(rows)

    @staticmethod
    def _row_to_record(row) -> dict[str, Any]:
        meta = row.metadata
        if isinstance(meta, str):
            meta = json.loads(meta)
        return {
            "parent_id": row.parent_id,
            "doc_id": row.doc_id,
            "source": row.source,
            "index": row.chunk_index,
            "page": row.page,
            "section": row.section,
            "token_count": row.token_count,
            "text": row.text,
            "metadata": meta or {},
        }

    def get(self, parent_id: str) -> dict[str, Any] | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                self._text("SELECT * FROM parent_chunks WHERE parent_id = :pid"),
                {"pid": parent_id},
            ).fetchone()
        return self._row_to_record(row) if row else None

    def mget(self, parent_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
        ids = list(parent_ids)
        if not ids:
            return {}
        with self.engine.connect() as conn:
            rows = conn.execute(
                self._text("SELECT * FROM parent_chunks WHERE parent_id = ANY(:ids)"),
                {"ids": ids},
            ).fetchall()
        return {r.parent_id: self._row_to_record(r) for r in rows}

    def count(self) -> int:
        with self.engine.connect() as conn:
            return int(conn.execute(self._text("SELECT COUNT(*) FROM parent_chunks")).scalar() or 0)

    def list_documents(self) -> list[dict[str, Any]]:
        sql = self._text(
            """
            SELECT doc_id,
                   MIN(source) AS source,
                   COUNT(*)    AS parent_chunks,
                   MIN(metadata ->> 'doc_type') AS doc_type
            FROM parent_chunks
            GROUP BY doc_id
            ORDER BY source
            """
        )
        with self.engine.connect() as conn:
            rows = conn.execute(sql).fetchall()
        return [
            {
                "doc_id": r.doc_id,
                "source": r.source,
                "parent_chunks": int(r.parent_chunks),
                "doc_type": r.doc_type,
            }
            for r in rows
        ]

    def delete_document(self, doc_id: str) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                self._text("DELETE FROM parent_chunks WHERE doc_id = :doc_id"), {"doc_id": doc_id}
            )
        return result.rowcount or 0

    def clear(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(self._text("TRUNCATE parent_chunks"))


_store: BaseDocStore | None = None
_store_lock = threading.Lock()


def get_docstore() -> BaseDocStore:
    """Singleton resolver honouring `DOCSTORE_BACKEND` (auto | postgres | json)."""
    global _store
    if _store is not None:
        return _store
    with _store_lock:
        if _store is not None:
            return _store
        backend = settings.docstore_backend.lower()
        if backend in {"postgres", "auto"}:
            try:
                _store = PostgresDocStore()
                logger.info("Docstore backend: postgres")
                return _store
            except (ConnectionError, OSError, Exception) as exc:
                if backend == "postgres":
                    raise
                logger.warning("Postgres unavailable (%s); falling back to JSON docstore.", exc)
        _store = JSONDocStore()
        logger.info("Docstore backend: json (%s)", _store.path)
        return _store


def reset_docstore_singleton() -> None:
    global _store
    with _store_lock:
        _store = None
