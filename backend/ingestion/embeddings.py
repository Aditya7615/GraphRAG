"""Dense embeddings via sentence-transformers (local, no API key, no rate limit).

Loaded lazily and cached process-wide: the model costs ~2s and ~130MB once, then
nothing. `bge` models need an instruction prefix on queries but not on passages -
getting this backwards silently degrades recall, so the two paths are separate.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from backend.config import settings

logger = logging.getLogger(__name__)


class DenseEmbedder:
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._model = None
        self._lock = threading.Lock()

    @property
    def model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model %s on %s", self.model_name, self.device)
                    self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        try:
            return int(self.model.get_sentence_embedding_dimension())
        except (AttributeError, TypeError, ValueError):
            return settings.embedding_dim

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self.model.encode(
            list(texts),
            batch_size=settings.embedding_batch_size,
            normalize_embeddings=True,  # cosine == dot product once normalised
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        prefix = settings.embedding_query_prefix if "bge" in self.model_name.lower() else ""
        vector = self.model.encode(
            prefix + text, normalize_embeddings=True, convert_to_numpy=True
        )
        return vector.tolist()


_embedder: DenseEmbedder | None = None
_lock = threading.Lock()


def get_embedder() -> DenseEmbedder:
    global _embedder
    if _embedder is None:
        with _lock:
            if _embedder is None:
                _embedder = DenseEmbedder()
    return _embedder
