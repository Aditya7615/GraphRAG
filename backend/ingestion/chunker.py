"""Parent-child chunking.

Retrieval precision and generation context have opposite ideal chunk sizes:
small chunks embed cleanly (one idea per vector) but read badly, large chunks
read well but embed into mush. So we do both:

    Document -> Parent chunks (~2000 tokens, stored in the doc store)
             -> Child chunks  (~400 tokens, embedded into Qdrant)

Only children are vectorised. At query time a child hit is swapped for its
parent, so the LLM always sees the full surrounding context.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import settings

# Stable namespace so re-ingesting the same document yields the same point IDs
# (idempotent upserts instead of duplicates).
_UUID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")

# Headings we recognise: markdown, numbered ("3.1 Revenue"), or SHOUTED lines.
_HEADING_PATTERNS = (
    re.compile(r"^\s{0,3}#{1,6}\s+(?P<title>.+?)\s*$"),
    re.compile(r"^\s{0,3}(?P<num>\d+(?:\.\d+)*\.?)\s+(?P<title>[A-Z][^\n]{2,80})\s*$"),
    re.compile(r"^\s{0,3}(?P<title>[A-Z][A-Z0-9 &/\-,'()]{5,80})\s*$"),
)


@lru_cache(maxsize=4)
def _encoder(name: str):
    return tiktoken.get_encoding(name)


def count_tokens(text: str) -> int:
    return len(_encoder(settings.tokenizer_encoding).encode(text, disallowed_special=()))


def stable_id(*parts: str) -> str:
    """Deterministic UUIDv5 - Qdrant only accepts UUIDs or unsigned ints as IDs."""
    return str(uuid.uuid5(_UUID_NAMESPACE, "::".join(parts)))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass(slots=True)
class Page:
    """One physical page (PDF) or logical block (txt/md) of a source document."""

    number: int
    text: str


@dataclass(slots=True)
class SourceDocument:
    doc_id: str
    source: str
    pages: list[Page]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages)


@dataclass(slots=True)
class ParentChunk:
    parent_id: str
    doc_id: str
    source: str
    text: str
    index: int
    page: int | None = None
    section: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "index": self.index,
            "page": self.page,
            "section": self.section,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ChildChunk:
    child_id: str
    parent_id: str
    doc_id: str
    source: str
    text: str
    index: int
    page: int | None = None
    section: str | None = None
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        return {
            "child_id": self.child_id,
            "parent_id": self.parent_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "index": self.index,
            "page": self.page,
            "section": self.section,
            "token_count": self.token_count,
            **self.metadata,
        }


class _OffsetMap:
    """Maps a character offset in the concatenated text back to page + section."""

    def __init__(self, doc: SourceDocument) -> None:
        self._page_starts: list[tuple[int, int]] = []
        self._headings: list[tuple[int, str]] = []

        cursor = 0
        for page in doc.pages:
            self._page_starts.append((cursor, page.number))
            self._scan_headings(page.text, cursor)
            cursor += len(page.text) + 2  # the "\n\n" join

    def _scan_headings(self, text: str, base: int) -> None:
        offset = base
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            if 3 <= len(stripped) <= 90:
                for pattern in _HEADING_PATTERNS:
                    m = pattern.match(stripped)
                    if m:
                        title = m.group("title").strip(" .:-")
                        if title and not title.endswith((".", ",", ";")):
                            self._headings.append((offset, title))
                        break
            offset += len(line)

    def page_at(self, offset: int) -> int | None:
        found = None
        for start, number in self._page_starts:
            if start <= offset:
                found = number
            else:
                break
        return found

    def section_at(self, offset: int) -> str | None:
        found = None
        for start, title in self._headings:
            if start <= offset:
                found = title
            else:
                break
        return found


class ParentChildChunker:
    """Splits a `SourceDocument` into parents and their children."""

    def __init__(
        self,
        parent_tokens: int | None = None,
        parent_overlap: int | None = None,
        child_tokens: int | None = None,
        child_overlap: int | None = None,
    ) -> None:
        self.parent_tokens = parent_tokens or settings.parent_chunk_tokens
        self.parent_overlap = parent_overlap or settings.parent_chunk_overlap_tokens
        self.child_tokens = child_tokens or settings.child_chunk_tokens
        self.child_overlap = child_overlap or settings.child_chunk_overlap_tokens

        # Separator order matters: prefer breaking on section/paragraph edges
        # so a chunk rarely straddles two unrelated topics.
        separators = ["\n\n\n", "\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
        self._parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=settings.tokenizer_encoding,
            chunk_size=self.parent_tokens,
            chunk_overlap=self.parent_overlap,
            separators=separators,
            keep_separator=True,
        )
        self._child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name=settings.tokenizer_encoding,
            chunk_size=self.child_tokens,
            chunk_overlap=self.child_overlap,
            separators=separators,
            keep_separator=True,
        )

    def split(self, doc: SourceDocument) -> tuple[list[ParentChunk], list[ChildChunk]]:
        text = doc.full_text
        if not text.strip():
            return [], []

        offsets = _OffsetMap(doc)
        parents: list[ParentChunk] = []
        children: list[ChildChunk] = []

        cursor = 0
        for p_idx, parent_text in enumerate(self._parent_splitter.split_text(text)):
            start = text.find(parent_text[:80], cursor)
            if start == -1:
                start = cursor
            cursor = start + max(len(parent_text) - self.parent_overlap, 1)

            parent = ParentChunk(
                parent_id=stable_id(doc.doc_id, "parent", str(p_idx)),
                doc_id=doc.doc_id,
                source=doc.source,
                text=parent_text,
                index=p_idx,
                page=offsets.page_at(start),
                section=offsets.section_at(start),
                token_count=count_tokens(parent_text),
                metadata=dict(doc.metadata),
            )
            parents.append(parent)

            local = 0
            for c_idx, child_text in enumerate(self._child_splitter.split_text(parent_text)):
                if not child_text.strip():
                    continue
                local_start = parent_text.find(child_text[:60], local)
                if local_start == -1:
                    local_start = local
                local = local_start + max(len(child_text) - self.child_overlap, 1)
                abs_start = start + local_start

                children.append(
                    ChildChunk(
                        child_id=stable_id(doc.doc_id, "child", str(p_idx), str(c_idx)),
                        parent_id=parent.parent_id,
                        doc_id=doc.doc_id,
                        source=doc.source,
                        text=child_text,
                        index=c_idx,
                        page=offsets.page_at(abs_start) or parent.page,
                        section=offsets.section_at(abs_start) or parent.section,
                        token_count=count_tokens(child_text),
                        metadata=dict(doc.metadata),
                    )
                )

        return parents, children

    def split_many(
        self, docs: Iterable[SourceDocument]
    ) -> tuple[list[ParentChunk], list[ChildChunk]]:
        all_parents: list[ParentChunk] = []
        all_children: list[ChildChunk] = []
        for doc in docs:
            parents, children = self.split(doc)
            all_parents.extend(parents)
            all_children.extend(children)
        return all_parents, all_children
