"""RAG chain: retrieve -> ground -> generate -> verify.

The LLM is only half of the anti-hallucination story. The other half is the
post-generation guardrail in `_enforce_guardrails`, which rejects answers that
cite nothing, cite markers that were never retrieved, or arrive when retrieval
came back empty. A rejected answer becomes the fallback string rather than
reaching the user.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from collections.abc import Sequence

from langchain_core.output_parsers import StrOutputParser

from backend.config import settings
from backend.generation.prompts import FALLBACK_ANSWER, RAG_PROMPT
from backend.generation.retriever import (
    ParentDocumentRetriever,
    RetrievedParent,
    get_retriever,
)
from backend.ingestion.vector_store import SearchMode
from backend.models.schemas import ChatResponse, Citation

logger = logging.getLogger(__name__)

_MARKER_RE = re.compile(r"\[(S\d+)\]")
# Phrases that mean "I don't know" but aren't the exact fallback string.
_REFUSAL_HINTS = (
    "cannot find the answer",
    "not find the answer",
    "does not contain",
    "no information",
    "not mentioned in the",
    "not provided in the",
    "insufficient information",
)


class GeminiLLMUnavailable(RuntimeError):
    pass


class GeminiRateLimited(RuntimeError):
    """Google Gemini quota exhausted. Carries the provider's retry hint when it gives one."""

    def __init__(self, message: str, retry_after: str | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after

    @property
    def retry_after_seconds(self) -> int | None:
        """The HTTP Retry-After header must be integer seconds, not '51m22.7s'."""
        if not self.retry_after:
            return None
        total = 0.0
        for value, unit in re.findall(r"([0-9.]+)([hms])", self.retry_after):
            total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
        return int(total) or None


_RETRY_AFTER_RE = re.compile(r"try again in ([0-9][0-9hms.]*)")


def _as_rate_limit(exc: Exception) -> GeminiRateLimited | None:
    """Turn a provider 429 into a clean, user-facing error.

    The raw Gemini payload is a wall of JSON; the API should not forward that to a
    browser as an HTTP 500."""
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None
    )
    text = str(exc)

    # google.api_core.exceptions.ResourceExhausted is the gRPC form of 429
    exc_name = type(exc).__name__
    is_resource_exhausted = exc_name == "ResourceExhausted"

    if status != 429 and "rate_limit_exceeded" not in text and "Rate limit reached" not in text and "Resource has been exhausted" not in text and not is_resource_exhausted:
        return None

    retry_after = None
    if match := _RETRY_AFTER_RE.search(text):
        retry_after = match.group(1).rstrip(".")

    detail = "Google Gemini rate limit reached for the configured model."
    if "tokens per day" in text or "TPD" in text or "daily" in text.lower() or "PerDay" in text:
        detail = "Google Gemini daily token quota exhausted for this API key."
    elif "tokens per minute" in text or "TPM" in text or "per minute" in text.lower() or "PerMinute" in text:
        detail = "Google Gemini per-minute token limit (TPM) reached."
    if retry_after:
        detail += f" Retry in {retry_after}."
    return GeminiRateLimited(detail, retry_after=retry_after)


class RAGChain:
    def __init__(self, retriever: ParentDocumentRetriever | None = None) -> None:
        self.retriever = retriever or get_retriever()
        self._llm = None

    # --- LLM ----------------------------------------------------------------
    @property
    def llm(self):
        if self._llm is None:
            if not settings.google_api_key:
                raise GeminiLLMUnavailable(
                    "GOOGLE_API_KEY is not set. Add it to .env to enable generation."
                )
            from langchain_google_genai import ChatGoogleGenerativeAI

            self._llm = ChatGoogleGenerativeAI(
                google_api_key=settings.google_api_key,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                max_output_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_seconds,
                max_retries=2,
            )
        return self._llm

    def llm_ready(self) -> bool:
        try:
            _ = self.llm
            return True
        except (RuntimeError, OSError):
            return False

    # --- main entrypoint ----------------------------------------------------
    def answer(
        self,
        question: str,
        top_k: int | None = None,
        max_parents: int | None = None,
        doc_ids: Sequence[str] | None = None,
        mode: SearchMode = "hybrid",
        include_parent_text: bool = False,
    ) -> ChatResponse:
        total_started = time.perf_counter()

        retrieval = self.retriever.retrieve(
            question, top_k=top_k, max_parents=max_parents, doc_ids=doc_ids, mode=mode
        )

        # Nothing retrieved -> refuse without burning an LLM call.
        if retrieval.is_empty:
            return ChatResponse(
                answer=FALLBACK_ANSWER,
                grounded=False,
                citations=[],
                question=question,
                latency_ms={
                    "retrieval": round(retrieval.latency_ms, 2),
                    "generation": 0.0,
                    "total": round((time.perf_counter() - total_started) * 1000, 2),
                },
                model=settings.llm_model,
                search_mode=mode,
            )

        gen_started = time.perf_counter()
        chain = RAG_PROMPT | self.llm | StrOutputParser()
        try:
            raw = chain.invoke(
                {
                    "context": retrieval.context,
                    "question": question,
                    "fallback": FALLBACK_ANSWER,
                }
            )
        except Exception as exc:
            if rate_limited := _as_rate_limit(exc):
                logger.warning("Gemini rate limit: %s", rate_limited)
                raise rate_limited from exc
            raise
        gen_ms = (time.perf_counter() - gen_started) * 1000

        answer, grounded, used_markers = self._enforce_guardrails(raw, retrieval.parents)
        citations = self._build_citations(
            retrieval.parents, used_markers, grounded, include_parent_text
        )

        return ChatResponse(
            answer=answer,
            grounded=grounded,
            citations=citations,
            question=question,
            latency_ms={
                "retrieval": round(retrieval.latency_ms, 2),
                "generation": round(gen_ms, 2),
                "total": round((time.perf_counter() - total_started) * 1000, 2),
            },
            model=settings.llm_model,
            search_mode=mode,
        )

    # --- guardrails ---------------------------------------------------------
    def _enforce_guardrails(
        self, raw: str, parents: list[RetrievedParent]
    ) -> tuple[str, bool, set[str]]:
        """Returns (answer, grounded, markers actually cited)."""
        answer = (raw or "").strip()
        if not answer:
            return FALLBACK_ANSWER, False, set()

        lowered = answer.lower()
        if FALLBACK_ANSWER.lower() in lowered or any(h in lowered for h in _REFUSAL_HINTS):
            return FALLBACK_ANSWER, False, set()

        valid = {p.marker for p in parents}
        cited = set(_MARKER_RE.findall(answer))

        # Hallucinated markers: strip them, then re-check that something real is left.
        invalid = cited - valid
        if invalid:
            logger.warning("Dropping invalid citation markers: %s", sorted(invalid))
            for marker in invalid:
                answer = answer.replace(f"[{marker}]", "")
            answer = re.sub(r"[ \t]{2,}", " ", answer).strip()
            cited &= valid

        # An answer with zero valid citations is, by this system's definition,
        # ungrounded - regardless of how plausible it reads.
        if not cited:
            logger.warning("Answer had no valid citations; returning fallback.")
            return FALLBACK_ANSWER, False, set()

        return answer, True, cited

    @staticmethod
    def _build_citations(
        parents: list[RetrievedParent],
        used_markers: set[str],
        grounded: bool,
        include_parent_text: bool,
    ) -> list[Citation]:
        if not grounded:
            return []
        citations = [
            Citation(
                marker=p.marker,
                parent_id=p.parent_id,
                doc_id=p.doc_id,
                source=p.source,
                section=p.section,
                page=p.page,
                score=p.score,
                used_by_llm=p.marker in used_markers,
                snippet=p.snippet,
                parent_text=p.text if include_parent_text else "",
            )
            for p in parents
        ]
        # Sources the model actually leaned on float to the top of the UI.
        citations.sort(key=lambda c: (not c.used_by_llm, -c.score))
        return citations


_chain: RAGChain | None = None
_chain_lock = threading.Lock()


def get_chain() -> RAGChain:
    global _chain
    if _chain is None:
        with _chain_lock:
            if _chain is None:
                _chain = RAGChain()
    return _chain
