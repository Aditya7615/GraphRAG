"""Unit tests that run without Qdrant, Postgres or a Gemini key."""

from __future__ import annotations

import pytest

from backend.generation.prompts import FALLBACK_ANSWER
from backend.ingestion.chunker import (
    Page,
    ParentChildChunker,
    SourceDocument,
    count_tokens,
)
from backend.ingestion.sparse import BM25SparseEncoder, term_index, tokenize


def make_doc(paragraph_count: int = 60) -> SourceDocument:
    body = "\n\n".join(
        f"Section {i // 10 + 1}. Policy Detail\n"
        f"Employees accrue {20 + i % 5} days of leave under clause {i}. "
        "This paragraph exists to provide enough tokens for the parent splitter "
        "to produce more than a single chunk when the document is long."
        for i in range(paragraph_count)
    )
    return SourceDocument(
        doc_id="doc-test",
        source="policy.pdf",
        pages=[Page(number=1, text=body[: len(body) // 2]), Page(number=2, text=body[len(body) // 2 :])],
        metadata={"doc_type": "hr"},
    )


class TestChunker:
    def test_children_are_smaller_than_parents(self):
        parents, children = ParentChildChunker().split(make_doc())
        assert parents and children
        assert len(children) > len(parents)
        assert max(p.token_count for p in parents) <= 2000 * 1.1
        assert max(c.token_count for c in children) <= 400 * 1.1

    def test_every_child_maps_to_a_real_parent(self):
        parents, children = ParentChildChunker().split(make_doc())
        parent_ids = {p.parent_id for p in parents}
        assert all(c.parent_id in parent_ids for c in children)

    def test_ids_are_deterministic_across_runs(self):
        a_parents, a_children = ParentChildChunker().split(make_doc())
        b_parents, b_children = ParentChildChunker().split(make_doc())
        assert [p.parent_id for p in a_parents] == [p.parent_id for p in b_parents]
        assert [c.child_id for c in a_children] == [c.child_id for c in b_children]

    def test_page_numbers_are_propagated(self):
        parents, _ = ParentChildChunker().split(make_doc())
        assert {p.page for p in parents} <= {1, 2}
        assert all(p.page is not None for p in parents)

    def test_empty_document_yields_nothing(self):
        doc = SourceDocument(doc_id="d", source="empty.txt", pages=[Page(1, "   ")])
        assert ParentChildChunker().split(doc) == ([], [])

    def test_token_counting(self):
        assert count_tokens("hello world") > 0
        assert count_tokens("") == 0


class TestSparseEncoder:
    def test_tokenizer_drops_stopwords_and_folds_suffixes(self):
        tokens = tokenize("The employees are reviewing the POLICIES")
        assert "the" not in tokens and "are" not in tokens
        assert "employee" in tokens

    def test_dotted_identifiers_survive(self):
        assert "s3.putobject" in tokenize("Grant s3.PutObject to the role")

    def test_term_index_is_stable_and_in_range(self):
        assert term_index("aurora") == term_index("aurora")
        assert 0 <= term_index("aurora") < 2**31 - 1

    def test_document_encoding_is_sparse_and_positive(self, tmp_path, monkeypatch):
        from backend.config import settings

        monkeypatch.setattr(settings, "bm25_stats_path", tmp_path / "bm25.json")
        encoder = BM25SparseEncoder()
        indices, values = encoder.encode_document("connection pool exhaustion on the aurora writer")
        assert len(indices) == len(values) == len(set(indices))
        assert all(v > 0 for v in values)

    def test_query_encoding_has_no_length_normalisation(self, tmp_path, monkeypatch):
        from backend.config import settings

        monkeypatch.setattr(settings, "bm25_stats_path", tmp_path / "bm25.json")
        encoder = BM25SparseEncoder()
        short = encoder.encode_query("aurora")
        assert short[1] == [1.0]

    def test_empty_text_returns_empty_vector(self, tmp_path, monkeypatch):
        from backend.config import settings

        monkeypatch.setattr(settings, "bm25_stats_path", tmp_path / "bm25.json")
        assert BM25SparseEncoder().encode_document("the of and") == ([], [])


class TestDocStore:
    def test_json_roundtrip(self, tmp_path):
        from backend.ingestion.docstore import JSONDocStore

        parents, _ = ParentChildChunker().split(make_doc())
        store = JSONDocStore(tmp_path / "parents.json")
        assert store.upsert(parents) == len(parents)
        assert store.count() == len(parents)
        assert store.get(parents[0].parent_id)["text"] == parents[0].text
        assert len(store.mget([p.parent_id for p in parents])) == len(parents)

        docs = store.list_documents()
        assert docs[0]["source"] == "policy.pdf"
        assert store.delete_document("doc-test") == len(parents)
        assert store.count() == 0

    def test_upsert_is_idempotent(self, tmp_path):
        from backend.ingestion.docstore import JSONDocStore

        parents, _ = ParentChildChunker().split(make_doc())
        store = JSONDocStore(tmp_path / "parents.json")
        store.upsert(parents)
        store.upsert(parents)
        assert store.count() == len(parents)


class TestGuardrails:
    """The post-generation guardrail is what actually enforces zero hallucination."""

    @pytest.fixture
    def chain(self):
        from backend.generation.chain import RAGChain

        return RAGChain.__new__(RAGChain)  # no LLM, no retriever needed

    @pytest.fixture
    def parents(self):
        from backend.generation.retriever import RetrievedParent

        return [
            RetrievedParent(
                marker=f"S{i}",
                parent_id=f"p{i}",
                doc_id="d1",
                source="handbook.pdf",
                text="Primary caregivers receive 20 weeks of fully paid parental leave.",
                section="3. Parental Leave",
                page=2,
                score=0.9 - i * 0.1,
                snippet="Primary caregivers receive 20 weeks...",
            )
            for i in (1, 2)
        ]

    def test_valid_citation_is_grounded(self, chain, parents):
        _, grounded, cited = chain._enforce_guardrails(
            "Primary caregivers receive 20 weeks of paid leave [S1].", parents
        )
        assert grounded and cited == {"S1"}

    def test_answer_without_citations_is_rejected(self, chain, parents):
        answer, grounded, cited = chain._enforce_guardrails(
            "Primary caregivers receive 26 weeks of paid leave.", parents
        )
        assert not grounded and answer == FALLBACK_ANSWER and not cited

    def test_hallucinated_marker_is_stripped(self, chain, parents):
        answer, grounded, cited = chain._enforce_guardrails(
            "Leave is 20 weeks [S1]. Bonus is 30% [S9].", parents
        )
        assert grounded and cited == {"S1"} and "[S9]" not in answer

    def test_answer_with_only_invalid_markers_falls_back(self, chain, parents):
        answer, grounded, _ = chain._enforce_guardrails("Bonus is 30% [S7].", parents)
        assert not grounded and answer == FALLBACK_ANSWER

    def test_refusal_phrasing_is_normalised_to_the_exact_fallback(self, chain, parents):
        answer, grounded, _ = chain._enforce_guardrails(
            "The context does not contain information about the bonus rate.", parents
        )
        assert answer == FALLBACK_ANSWER and not grounded

    def test_empty_generation_falls_back(self, chain, parents):
        assert chain._enforce_guardrails("", parents)[0] == FALLBACK_ANSWER

    def test_citations_are_empty_when_ungrounded(self, chain, parents):
        assert chain._build_citations(parents, set(), grounded=False, include_parent_text=False) == []

    def test_cited_sources_sort_first(self, chain, parents):
        citations = chain._build_citations(parents, {"S2"}, True, include_parent_text=False)
        assert citations[0].marker == "S2" and citations[0].used_by_llm
        assert not citations[1].used_by_llm


class TestRateLimitMapping:
    """A provider 429 must surface as a clean 429, not a 500 with a JSON dump."""

    @staticmethod
    def _groq_429() -> Exception:
        exc = Exception(
            "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
            "`llama-3.3-70b-versatile` on tokens per day (TPD): Limit 100000, Used 97363. "
            "Please try again in 51m22.752s.', 'code': 'rate_limit_exceeded'}}"
        )
        return exc

    def test_detects_daily_quota_exhaustion(self):
        from backend.generation.chain import _as_rate_limit

        err = _as_rate_limit(self._groq_429())
        assert err is not None
        assert "daily token quota" in str(err)
        assert "51m22.752s.." not in str(err)  # no doubled full stop

    def test_retry_after_converts_to_integer_seconds(self):
        from backend.generation.chain import _as_rate_limit

        err = _as_rate_limit(self._groq_429())
        assert err.retry_after_seconds == 3082  # 51*60 + 22

    def test_unrelated_errors_are_not_swallowed(self):
        from backend.generation.chain import _as_rate_limit

        assert _as_rate_limit(ValueError("connection reset by peer")) is None

    def test_missing_retry_hint_yields_no_header_value(self):
        from backend.generation.chain import _as_rate_limit

        err = _as_rate_limit(Exception("Error code: 429 - rate_limit_exceeded"))
        assert err is not None and err.retry_after_seconds is None
