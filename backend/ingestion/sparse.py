"""BM25 sparse encoder for the lexical half of hybrid search.

Dense vectors miss exact identifiers - ticker symbols, error codes, policy
numbers, `s3:PutObject`. BM25 nails those and misses paraphrase. Running both
and fusing is why hybrid beats either alone.

Qdrant stores the term-frequency component and applies IDF server-side
(`Modifier.IDF`), so corpus statistics stay correct as documents are added
without any client-side reindexing.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import threading
from collections import Counter
from pathlib import Path

from backend.config import settings

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[._\-/][a-z0-9]+)*")

_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "if", "in", "into", "is", "it", "its", "of",
    "on", "or", "our", "she", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "to", "was", "were", "what", "when", "where", "which",
    "who", "will", "with", "would", "you", "your", "we", "us", "i",
])

# Conservative suffix folding: enough to match plural/tense variants, not so
# aggressive that "billing" collapses into "bill".
_ES_STEM_ENDINGS = ("s", "x", "z", "ch", "sh")

MAX_INDEX = 2**31 - 1


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        if len(raw) < 2 or raw in _STOPWORDS:
            continue
        tokens.append(_fold(raw))
    return tokens


def _fold(token: str) -> str:
    """Light stemmer. Correctness matters less than applying it identically to
    documents and queries - but the plural rules below are ordered so that
    "employees" -> "employee" rather than "employe"."""
    if len(token) <= 3 or not token.isalpha():
        return token

    if token.endswith("ies") and len(token) >= 5:
        return token[:-3] + "y"
    # "boxes"/"matches" drop "es"; "employees"/"files" only drop "s".
    if token.endswith("es") and token[:-2].endswith(_ES_STEM_ENDINGS):
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    for suffix in ("ing", "ed"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def term_index(token: str) -> int:
    """Stable 31-bit index. Collisions are ~0 at enterprise vocab sizes."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % MAX_INDEX


class BM25SparseEncoder:
    """k1/b-parameterised BM25. Document side carries length normalisation."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, avgdl: float = 180.0) -> None:
        self.k1 = k1
        self.b = b
        self.avgdl = avgdl
        self._total_len = 0
        self._doc_count = 0
        self._lock = threading.Lock()
        self._path = Path(settings.bm25_stats_path)
        self._load()

    # --- corpus statistics --------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.avgdl = float(data.get("avgdl", self.avgdl)) or self.avgdl
            self._total_len = int(data.get("total_len", 0))
            self._doc_count = int(data.get("doc_count", 0))
        except (json.JSONDecodeError, OSError, ValueError):
            pass

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "avgdl": self.avgdl,
            "total_len": self._total_len,
            "doc_count": self._doc_count,
            "k1": self.k1,
            "b": self.b,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self._path)

    def observe(self, texts: list[str]) -> None:
        """Fold a freshly ingested batch into the running average doc length."""
        if not texts:
            return
        with self._lock:
            for text in texts:
                self._total_len += len(tokenize(text))
                self._doc_count += 1
            if self._doc_count:
                self.avgdl = max(self._total_len / self._doc_count, 1.0)
            self._save()

    def reset(self) -> None:
        with self._lock:
            self._total_len = 0
            self._doc_count = 0
            self.avgdl = 180.0
            self._save()

    # --- encoding -----------------------------------------------------------
    def encode_document(self, text: str) -> tuple[list[int], list[float]]:
        tokens = tokenize(text)
        if not tokens:
            return [], []
        counts = Counter(tokens)
        norm = self.k1 * (1.0 - self.b + self.b * len(tokens) / self.avgdl)

        indices: list[int] = []
        values: list[float] = []
        for token, tf in counts.items():
            indices.append(term_index(token))
            values.append(round(tf * (self.k1 + 1.0) / (tf + norm), 6))
        return indices, values

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """No length normalisation on queries - Qdrant supplies the IDF weight."""
        tokens = tokenize(text)
        if not tokens:
            return [], []
        counts = Counter(tokens)
        indices, values = [], []
        for token, tf in counts.items():
            indices.append(term_index(token))
            # Damped query TF keeps a repeated word from dominating the query.
            values.append(round(1.0 + math.log(tf), 6))
        return indices, values


_encoder: BM25SparseEncoder | None = None
_lock = threading.Lock()


def get_sparse_encoder() -> BM25SparseEncoder:
    global _encoder
    if _encoder is None:
        with _lock:
            if _encoder is None:
                _encoder = BM25SparseEncoder()
    return _encoder
