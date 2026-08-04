"""Qdrant hybrid vector store.

Collection layout (named vectors, one point per *child* chunk):

    "dense"  -> 384-d cosine vector from bge-small
    "sparse" -> BM25 term frequencies, IDF applied server-side

Search runs both arms as `Prefetch` branches and fuses them with Reciprocal Rank
Fusion inside Qdrant, so only one network round trip happens per query.

Scale notes for the 100k+ PDF target:
  * scalar quantization (int8) cuts the vector RAM footprint ~4x with rescoring
    from the original vectors, so recall stays intact
  * payloads live on disk; only the HNSW graph and quantized vectors stay hot
  * keyword indexes on doc_id/source/doc_type keep metadata filters sub-linear
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from backend.config import settings
from backend.ingestion.chunker import ChildChunk
from backend.ingestion.embeddings import get_embedder
from backend.ingestion.sparse import get_sparse_encoder

logger = logging.getLogger(__name__)

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "sparse"

SearchMode = Literal["hybrid", "dense", "sparse"]


class HybridVectorStore:
    def __init__(self, collection: str | None = None, client: QdrantClient | None = None) -> None:
        self.collection = collection or settings.qdrant_collection
        self.client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key or None,
            timeout=settings.qdrant_timeout,
            prefer_grpc=False,
        )
        self.embedder = get_embedder()
        self.sparse = get_sparse_encoder()

    # --- lifecycle ----------------------------------------------------------
    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except (UnexpectedResponse, ConnectionError, OSError) as exc:
            logger.debug("Qdrant ping failed: %s", exc)
            return False

    def collection_exists(self) -> bool:
        try:
            return self.client.collection_exists(self.collection)
        except (UnexpectedResponse, ConnectionError, OSError):
            return False

    def ensure_collection(self, recreate: bool = False) -> None:
        if recreate and self.collection_exists():
            self.client.delete_collection(self.collection)
        if self.collection_exists():
            return

        quantization = (
            models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            )
            if settings.qdrant_use_scalar_quantization
            else None
        )

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=settings.embedding_dim,
                    distance=models.Distance.COSINE,
                    on_disk=True,
                )
            },
            sparse_vectors_config={
                # IDF is computed by Qdrant across the live collection, so BM25
                # weights stay correct as documents stream in.
                SPARSE_VECTOR: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            hnsw_config=models.HnswConfigDiff(
                m=settings.qdrant_hnsw_m, ef_construct=settings.qdrant_hnsw_ef_construct
            ),
            quantization_config=quantization,
            optimizers_config=models.OptimizersConfigDiff(default_segment_number=4),
            on_disk_payload=settings.qdrant_on_disk_payload,
        )
        self._create_payload_indexes()
        logger.info("Created Qdrant collection '%s'", self.collection)

    def _create_payload_indexes(self) -> None:
        for field in ("doc_id", "parent_id", "source", "doc_type"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except UnexpectedResponse as exc:  # already exists
                logger.debug("Payload index %s: %s", field, exc)

    def recreate(self) -> None:
        self.ensure_collection(recreate=True)
        self.sparse.reset()

    # --- writes -------------------------------------------------------------
    def upsert_children(self, children: Sequence[ChildChunk], batch_size: int = 64) -> int:
        if not children:
            return 0
        self.ensure_collection()

        texts = [c.text for c in children]
        self.sparse.observe(texts)  # keep avgdl honest before encoding

        written = 0
        for start in range(0, len(children), batch_size):
            batch = children[start : start + batch_size]
            dense_vectors = self.embedder.embed_documents([c.text for c in batch])

            points: list[models.PointStruct] = []
            for child, dense in zip(batch, dense_vectors, strict=False):
                indices, values = self.sparse.encode_document(child.text)
                points.append(
                    models.PointStruct(
                        id=child.child_id,
                        vector={
                            DENSE_VECTOR: dense,
                            SPARSE_VECTOR: models.SparseVector(indices=indices, values=values),
                        },
                        payload=child.to_payload(),
                    )
                )
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
            written += len(points)
            logger.debug("Upserted %s/%s child chunks", written, len(children))
        return written

    def delete_document(self, doc_id: str) -> None:
        if not self.collection_exists():
            return
        self.client.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=self._doc_filter([doc_id])),
            wait=True,
        )

    # --- reads --------------------------------------------------------------
    @staticmethod
    def _doc_filter(doc_ids: Iterable[str] | None) -> models.Filter | None:
        ids = [d for d in (doc_ids or []) if d]
        if not ids:
            return None
        return models.Filter(
            must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=ids))]
        )

    def search(
        self,
        query: str,
        top_k: int | None = None,
        doc_ids: Sequence[str] | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[dict[str, Any]]:
        """Return fused child-chunk hits, best first."""
        if not self.collection_exists():
            return []

        top_k = top_k or settings.retrieval_top_k
        query_filter = self._doc_filter(doc_ids)
        search_params = models.SearchParams(
            hnsw_ef=settings.qdrant_search_hnsw_ef,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0),
        )

        if mode == "dense":
            response = self.client.query_points(
                collection_name=self.collection,
                query=self.embedder.embed_query(query),
                using=DENSE_VECTOR,
                limit=top_k,
                query_filter=query_filter,
                search_params=search_params,
                with_payload=True,
            )
        elif mode == "sparse":
            indices, values = self.sparse.encode_query(query)
            if not indices:
                return []
            response = self.client.query_points(
                collection_name=self.collection,
                query=models.SparseVector(indices=indices, values=values),
                using=SPARSE_VECTOR,
                limit=top_k,
                query_filter=query_filter,
                with_payload=True,
            )
        else:
            response = self.client.query_points(
                collection_name=self.collection,
                prefetch=self._hybrid_prefetch(query, query_filter, search_params),
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_k,
                with_payload=True,
            )

        return [self._to_hit(point) for point in response.points]

    def _hybrid_prefetch(
        self,
        query: str,
        query_filter: models.Filter | None,
        search_params: models.SearchParams,
    ) -> list[models.Prefetch]:
        branches = [
            models.Prefetch(
                query=self.embedder.embed_query(query),
                using=DENSE_VECTOR,
                limit=settings.dense_prefetch_limit,
                filter=query_filter,
                params=search_params,
            )
        ]
        indices, values = self.sparse.encode_query(query)
        if indices:
            branches.append(
                models.Prefetch(
                    query=models.SparseVector(indices=indices, values=values),
                    using=SPARSE_VECTOR,
                    limit=settings.sparse_prefetch_limit,
                    filter=query_filter,
                )
            )
        return branches

    @staticmethod
    def _to_hit(point) -> dict[str, Any]:
        payload = point.payload or {}
        return {
            "child_id": str(point.id),
            "parent_id": payload.get("parent_id", ""),
            "doc_id": payload.get("doc_id", ""),
            "source": payload.get("source", "unknown"),
            "section": payload.get("section"),
            "page": payload.get("page"),
            "text": payload.get("text", ""),
            "score": float(point.score or 0.0),
        }

    def stats(self) -> dict[str, Any]:
        if not self.collection_exists():
            return {"collection": self.collection, "vectors": 0, "indexed_vectors": 0}
        info = self.client.get_collection(self.collection)
        return {
            "collection": self.collection,
            "vectors": int(info.points_count or 0),
            "indexed_vectors": int(info.indexed_vectors_count or 0),
            "status": str(info.status),
        }


_store: HybridVectorStore | None = None
_store_lock = threading.Lock()


def get_vector_store() -> HybridVectorStore:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                _store = HybridVectorStore()
    return _store
