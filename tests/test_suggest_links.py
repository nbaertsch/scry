"""Tests for scry.suggest — AI batch link suggestion engine (Wave 5b).

Covers:
- Fake LLM provider implementing the LLMProvider Protocol
- Candidate selection: only unlinked anchor pairs are surfaced
- Pair generation: top-K embedding neighbors are picked
- LLM response parsing: valid + malformed JSON (graceful via LLMJSONModeError)
- Min-confidence filter
- Idempotency: re-running with same state proposes 0 new suggestions
- batch_llm_evaluate: token-warn threshold; per-batch error recovery
- run_suggest_links: end-to-end async orchestration
- scry suggest-links CLI: --json, --apply, --apply --yes, --min-confidence
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import subprocess
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from scry.cli import main
from scry.embed import StubEmbedder
from scry.llm import LLMJSONModeError, LLMRequest, LLMResponse
from scry.models import (
    Anchor,
    AnchorType,
    Link,
    LinkId,
    LinkOp,
    LinkRecord,
    LinkType,
    new_event_id,
    new_link_id,
)
from scry.retrieve import SearchResult
from scry.store.db import ScryDB
from scry.store.links import LinkStore
from scry.suggest import (
    DEFAULT_MIN_CONFIDENCE,
    LinkSuggestion,
    SuggestConfig,
    _parse_llm_batch,
    batch_llm_evaluate,
    run_suggest_links,
    select_candidate_pairs,
)

# ─── Fake LLM providers ───────────────────────────────────────────────────────


@dataclass
class FakeProvider:
    """Minimal LLMProvider implementation for tests.

    Satisfies the :class:`~scry.llm.LLMProvider` Protocol.
    Cycles through *responses* (JSON strings or bare text).
    """

    name: str = "fake"
    responses: list[str] = field(default_factory=list)
    _idx: int = field(default=0, init=False, repr=False)

    async def complete(self, req: LLMRequest) -> LLMResponse:
        if not self.responses:
            raise AssertionError("FakeProvider has no responses configured")
        text = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        return LLMResponse(
            text=text,
            model="fake-model",
            provider="fake",
            usage={"prompt": 100, "completion": 50, "total": 150},
            finish_reason="stop",
        )


@dataclass
class ErrorProvider:
    """Provider that raises LLMJSONModeError on every call."""

    name: str = "error-fake"

    async def complete(self, req: LLMRequest) -> LLMResponse:
        raise LLMJSONModeError("error-fake: not valid JSON. Response head: 'nope'")


class MixedProvider:
    """Provider that returns canned responses, then raises on subsequent calls.

    Used to test partial-batch-failure graceful-degradation: the first
    N batches succeed, the rest raise.
    """

    name: str = "mixed-fake"

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._call_count = 0

    async def complete(self, req: LLMRequest) -> LLMResponse:
        idx = self._call_count
        self._call_count += 1
        if idx < len(self._responses):
            return LLMResponse(
                text=self._responses[idx],
                model="mixed-fake",
                provider="mixed-fake",
                usage=None,
                finish_reason="stop",
            )
        raise LLMJSONModeError(
            f"mixed-fake: simulated failure on call {idx}. Response head: 'nope'"
        )


# ─── Anchor / DB helpers ──────────────────────────────────────────────────────


def _make_anchor(
    anchor_id: str,
    anchor_type: AnchorType,
    path: str,
    content: str,
) -> Anchor:
    """Construct a minimal Anchor with a correct sha256 content_hash."""
    h = hashlib.sha256(content.encode()).hexdigest()
    return Anchor(
        id=anchor_id,
        type=anchor_type,
        path=path,
        content_text=content,
        content_hash=f"sha256:{h}",
        fingerprint_simhash=0,
    )


CODE_ID = "src/main.py::symbol::process"
DOC_ID = "docs/spec.md::section::overview"
CODE_CONTENT = "def process(x: int) -> int: return x + 1"
DOC_CONTENT = "# Overview\n\nThe process function increments its input."


@pytest.fixture
def scry_db(tmp_path: Path) -> Generator[ScryDB, None, None]:
    """A ScryDB with schema + one code anchor + one doc anchor."""
    (tmp_path / ".scry").mkdir(exist_ok=True)
    db = ScryDB(tmp_path)
    db.init_schema(embedding_dimensions=4)
    db.upsert_anchor(_make_anchor(CODE_ID, AnchorType.CODE, "src/main.py", CODE_CONTENT))
    db.upsert_anchor(_make_anchor(DOC_ID, AnchorType.SECTION, "docs/spec.md", DOC_CONTENT))
    yield db
    db.close()


@pytest.fixture
def link_store(tmp_path: Path) -> LinkStore:
    """An empty LinkStore for the tmp_path repo."""
    return LinkStore(tmp_path)


@pytest.fixture
def stub_embedder() -> StubEmbedder:
    return StubEmbedder(dimensions=4)


def _make_search_result(parent_id: str, score: float = 0.8) -> SearchResult:
    return SearchResult(
        parent_anchor_id=parent_id,
        score=score,
        best_chunk_rowid_vec=None,
        best_chunk_rowid_bm25=None,
        parent_rank_in_vec=None,
        parent_rank_in_bm25=None,
    )


def _good_llm_response(
    pair_id: str,
    link_type: str = "mirrors",
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "suggestions": [
                {
                    "pair_id": pair_id,
                    "should_link": True,
                    "link_type": link_type,
                    "confidence": confidence,
                    "reason": "They describe the same concept.",
                }
            ]
        }
    )


# ─── _parse_llm_batch unit tests ─────────────────────────────────────────────


class TestParseLlmBatch:
    """Unit tests for the internal _parse_llm_batch helper."""

    def test_valid_response_parsed(self) -> None:
        raw = json.dumps(
            {
                "suggestions": [
                    {
                        "pair_id": "p_0",
                        "should_link": True,
                        "link_type": "mirrors",
                        "confidence": 0.92,
                        "reason": "Same concept.",
                    }
                ]
            }
        )
        result = _parse_llm_batch(raw, ["p_0"])
        assert len(result) == 1
        assert result[0]["link_type"] == "mirrors"
        assert result[0]["confidence"] == pytest.approx(0.92)
        assert result[0]["reason"] == "Same concept."

    def test_should_link_false_dropped(self) -> None:
        raw = json.dumps(
            {
                "suggestions": [
                    {
                        "pair_id": "p_0",
                        "should_link": False,
                        "link_type": "mirrors",
                        "confidence": 0.2,
                        "reason": "Unrelated.",
                    }
                ]
            }
        )
        assert _parse_llm_batch(raw, ["p_0"]) == []

    def test_unknown_pair_id_dropped(self) -> None:
        raw = json.dumps(
            {
                "suggestions": [
                    {
                        "pair_id": "p_999",
                        "should_link": True,
                        "link_type": "implements",
                        "confidence": 0.8,
                        "reason": "x",
                    }
                ]
            }
        )
        assert _parse_llm_batch(raw, ["p_0"]) == []

    def test_invalid_link_type_dropped(self) -> None:
        raw = json.dumps(
            {
                "suggestions": [
                    {
                        "pair_id": "p_0",
                        "should_link": True,
                        "link_type": "supersedes",  # not in _VALID_LINK_TYPES
                        "confidence": 0.9,
                        "reason": "x",
                    }
                ]
            }
        )
        assert _parse_llm_batch(raw, ["p_0"]) == []

    def test_malformed_json_returns_empty(self) -> None:
        assert _parse_llm_batch("not json at all!!!", ["p_0"]) == []
        assert _parse_llm_batch("{broken", ["p_0"]) == []

    def test_non_dict_root_returns_empty(self) -> None:
        assert _parse_llm_batch(json.dumps([1, 2, 3]), ["p_0"]) == []

    def test_multiple_items_all_valid(self) -> None:
        raw = json.dumps(
            {
                "suggestions": [
                    {
                        "pair_id": "p_0",
                        "should_link": True,
                        "link_type": "mirrors",
                        "confidence": 0.9,
                        "reason": "a",
                    },
                    {
                        "pair_id": "p_1",
                        "should_link": True,
                        "link_type": "implements",
                        "confidence": 0.7,
                        "reason": "b",
                    },
                ]
            }
        )
        result = _parse_llm_batch(raw, ["p_0", "p_1"])
        assert len(result) == 2


# ─── select_candidate_pairs unit tests ───────────────────────────────────────


class TestSelectCandidatePairs:
    """Tests for candidate pair selection logic."""

    def test_unlinked_pair_is_surfaced(
        self,
        scry_db: ScryDB,
        stub_embedder: StubEmbedder,
    ) -> None:
        """A (code, doc) pair with no existing link is returned as a candidate."""
        config = SuggestConfig(source="code", top_k_neighbors=1)
        search_results = [_make_search_result(DOC_ID, 0.85)]

        with patch("scry.suggest.hybrid_search", return_value=search_results):
            pairs = select_candidate_pairs(
                db=scry_db,
                active_links={},
                embedder=stub_embedder,
                config=config,
            )

        assert len(pairs) == 1
        code, doc = pairs[0]
        assert code.id == CODE_ID
        assert doc.id == DOC_ID

    def test_idempotency_existing_link_excluded(
        self,
        scry_db: ScryDB,
        stub_embedder: StubEmbedder,
    ) -> None:
        """A pair that already has a link is NOT returned as a candidate."""
        link_id = new_link_id()
        evt_id = new_event_id()
        code_anchor = scry_db.get_anchor(CODE_ID)
        doc_anchor = scry_db.get_anchor(DOC_ID)
        assert code_anchor is not None and doc_anchor is not None

        existing_link = Link(
            link_id=link_id,
            from_id=CODE_ID,
            from_type=AnchorType.CODE,
            to_id=DOC_ID,
            to_type=AnchorType.SECTION,
            type=LinkType.MIRRORS,
            from_content_hash=code_anchor.content_hash,
            to_content_hash=doc_anchor.content_hash,
            last_event_id=evt_id,
        )
        active_links: dict[LinkId, Link] = {link_id: existing_link}

        config = SuggestConfig(source="code", top_k_neighbors=1)
        search_results = [_make_search_result(DOC_ID, 0.85)]

        with patch("scry.suggest.hybrid_search", return_value=search_results):
            pairs = select_candidate_pairs(
                db=scry_db,
                active_links=active_links,
                embedder=stub_embedder,
                config=config,
            )

        assert pairs == []

    def test_reversed_existing_link_excluded(
        self,
        scry_db: ScryDB,
        stub_embedder: StubEmbedder,
    ) -> None:
        """A pair is excluded even when the existing link is in the reversed direction."""
        link_id = new_link_id()
        evt_id = new_event_id()
        code_anchor = scry_db.get_anchor(CODE_ID)
        doc_anchor = scry_db.get_anchor(DOC_ID)
        assert code_anchor is not None and doc_anchor is not None

        # doc → code direction (reversed)
        existing_link = Link(
            link_id=link_id,
            from_id=DOC_ID,
            from_type=AnchorType.SECTION,
            to_id=CODE_ID,
            to_type=AnchorType.CODE,
            type=LinkType.MIRRORS,
            from_content_hash=doc_anchor.content_hash,
            to_content_hash=code_anchor.content_hash,
            last_event_id=evt_id,
        )
        active_links: dict[LinkId, Link] = {link_id: existing_link}

        config = SuggestConfig(source="code", top_k_neighbors=1)
        search_results = [_make_search_result(DOC_ID, 0.85)]

        with patch("scry.suggest.hybrid_search", return_value=search_results):
            pairs = select_candidate_pairs(
                db=scry_db,
                active_links=active_links,
                embedder=stub_embedder,
                config=config,
            )

        assert pairs == []

    def test_limit_caps_pairs(
        self,
        scry_db: ScryDB,
        stub_embedder: StubEmbedder,
    ) -> None:
        """limit=0 returns no candidate pairs."""
        config = SuggestConfig(source="code", top_k_neighbors=5, limit=0)
        search_results = [_make_search_result(DOC_ID, 0.85)]

        with patch("scry.suggest.hybrid_search", return_value=search_results):
            pairs = select_candidate_pairs(
                db=scry_db,
                active_links={},
                embedder=stub_embedder,
                config=config,
            )

        assert pairs == []

    def test_scope_filters_by_path_prefix(
        self,
        scry_db: ScryDB,
        stub_embedder: StubEmbedder,
    ) -> None:
        """--scope excludes anchors whose path does not start with the prefix."""
        # "other/" matches neither src/main.py nor docs/spec.md
        config = SuggestConfig(source="code", scope="other/", top_k_neighbors=1)
        search_results = [_make_search_result(DOC_ID, 0.85)]

        with patch("scry.suggest.hybrid_search", return_value=search_results):
            pairs = select_candidate_pairs(
                db=scry_db,
                active_links={},
                embedder=stub_embedder,
                config=config,
            )

        assert pairs == []

    def test_top_k_neighbors_respected(
        self,
        scry_db: ScryDB,
        stub_embedder: StubEmbedder,
    ) -> None:
        """hybrid_search is called with the configured top_k_neighbors value."""
        config = SuggestConfig(source="code", top_k_neighbors=3)
        captured_top_k: list[int] = []

        def _mock_search(*args: Any, **kwargs: Any) -> list[SearchResult]:
            captured_top_k.append(kwargs.get("top_k", -1))
            return []

        with patch("scry.suggest.hybrid_search", side_effect=_mock_search):
            select_candidate_pairs(
                db=scry_db,
                active_links={},
                embedder=stub_embedder,
                config=config,
            )

        assert all(k == 3 for k in captured_top_k)

    def test_duplicate_pairs_deduplicated(
        self,
        scry_db: ScryDB,
        stub_embedder: StubEmbedder,
    ) -> None:
        """Pairs from both scan directions are de-duplicated to one entry."""
        config = SuggestConfig(source="both", top_k_neighbors=1)

        def _mock_search(*args: Any, **kwargs: Any) -> list[SearchResult]:
            anchor_types = kwargs.get("anchor_types", [])
            # doc-side scan → return DOC_ID; code-side scan → return CODE_ID
            if any(t in (AnchorType.SECTION, AnchorType.CODE_IN_DOC) for t in anchor_types):
                return [_make_search_result(DOC_ID)]
            return [_make_search_result(CODE_ID)]

        with patch("scry.suggest.hybrid_search", side_effect=_mock_search):
            pairs = select_candidate_pairs(
                db=scry_db,
                active_links={},
                embedder=stub_embedder,
                config=config,
            )

        # Both scanning directions produce the same (CODE_ID, DOC_ID) pair.
        assert len(pairs) == 1
        assert pairs[0][0].id == CODE_ID
        assert pairs[0][1].id == DOC_ID

    def test_hybrid_search_exception_skipped(
        self,
        scry_db: ScryDB,
        stub_embedder: StubEmbedder,
    ) -> None:
        """A hybrid_search failure is logged and skipped; no exception propagates."""
        config = SuggestConfig(source="code", top_k_neighbors=1)

        with patch("scry.suggest.hybrid_search", side_effect=RuntimeError("db error")):
            pairs = select_candidate_pairs(
                db=scry_db,
                active_links={},
                embedder=stub_embedder,
                config=config,
            )

        assert pairs == []


# ─── batch_llm_evaluate unit tests ───────────────────────────────────────────


class TestBatchLlmEvaluate:
    """Tests for the async LLM batch evaluation function."""

    def test_happy_path_returns_suggestion(self, scry_db: ScryDB) -> None:
        """A valid LLM response produces a LinkSuggestion."""
        code = scry_db.get_anchor(CODE_ID)
        doc = scry_db.get_anchor(DOC_ID)
        assert code is not None and doc is not None

        provider = FakeProvider(responses=[_good_llm_response("p_0", "mirrors", 0.91)])
        pairs: list[tuple[Anchor, Anchor]] = [(code, doc)]

        suggestions = asyncio.run(batch_llm_evaluate(pairs, provider=provider, batch_size=10))

        assert len(suggestions) == 1
        assert suggestions[0].from_id == CODE_ID
        assert suggestions[0].to_id == DOC_ID
        assert suggestions[0].link_type == "mirrors"
        assert suggestions[0].confidence == pytest.approx(0.91)

    def test_llm_json_mode_error_all_batches_raises(self, scry_db: ScryDB) -> None:
        """When ALL batches fail with LLMError, the error is re-raised so the CLI
        can distinguish "no suggestions" from "provider broken" (review-w5b HIGH).
        """
        from scry.llm import LLMJSONModeError

        code = scry_db.get_anchor(CODE_ID)
        doc = scry_db.get_anchor(DOC_ID)
        assert code is not None and doc is not None

        provider = ErrorProvider()
        pairs: list[tuple[Anchor, Anchor]] = [(code, doc)]

        with pytest.raises(LLMJSONModeError):
            asyncio.run(batch_llm_evaluate(pairs, provider=provider, batch_size=10))

    def test_llm_partial_batch_failure_returns_partial(self, scry_db: ScryDB) -> None:
        """When SOME batches fail and SOME succeed, return the successes — no raise.

        Demonstrates the graceful-degradation contract: failure is fatal only
        when ALL batches fail (review-w5b HIGH).
        """
        code = scry_db.get_anchor(CODE_ID)
        doc = scry_db.get_anchor(DOC_ID)
        assert code is not None and doc is not None

        # Two pairs across two batches (batch_size=1).
        pairs: list[tuple[Anchor, Anchor]] = [(code, doc), (code, doc)]

        # First batch returns valid JSON suggesting a link; second batch errors.
        good_resp = json.dumps(
            {
                "suggestions": [
                    {
                        "pair_id": "p_0",
                        "should_link": True,
                        "link_type": "mirrors",
                        "confidence": 0.9,
                        "reason": "ok",
                    }
                ]
            }
        )
        provider = MixedProvider(responses=[good_resp])  # 2nd call raises
        suggestions = asyncio.run(batch_llm_evaluate(pairs, provider=provider, batch_size=1))
        assert len(suggestions) == 1
        assert suggestions[0].confidence == pytest.approx(0.9)

    def test_garbage_text_response_skipped(self, scry_db: ScryDB) -> None:
        """A non-JSON response text is parsed as empty without crashing."""
        code = scry_db.get_anchor(CODE_ID)
        doc = scry_db.get_anchor(DOC_ID)
        assert code is not None and doc is not None

        provider = FakeProvider(responses=["not json at all!!!"])
        pairs: list[tuple[Anchor, Anchor]] = [(code, doc)]

        suggestions = asyncio.run(batch_llm_evaluate(pairs, provider=provider, batch_size=10))

        assert suggestions == []

    def test_token_warn_emits_warning(
        self, scry_db: ScryDB, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Token usage above threshold emits a logger.warning."""
        import logging

        code = scry_db.get_anchor(CODE_ID)
        doc = scry_db.get_anchor(DOC_ID)
        assert code is not None and doc is not None

        provider = FakeProvider(responses=[_good_llm_response("p_0")])
        pairs: list[tuple[Anchor, Anchor]] = [(code, doc)]

        with caplog.at_level(logging.WARNING, logger="scry.suggest"):
            asyncio.run(
                batch_llm_evaluate(
                    pairs,
                    provider=provider,
                    batch_size=10,
                    token_warn_threshold=1,  # very low → always triggers
                )
            )

        assert any("token" in r.message.lower() for r in caplog.records)

    def test_batching_splits_across_llm_calls(self, scry_db: ScryDB) -> None:
        """3 pairs with batch_size=2 → 2 LLM calls (ceil(3/2))."""
        code = scry_db.get_anchor(CODE_ID)
        doc = scry_db.get_anchor(DOC_ID)
        assert code is not None and doc is not None

        call_count = 0

        @dataclass
        class CountingProvider:
            name: str = "counting"

            async def complete(self, req: LLMRequest) -> LLMResponse:
                nonlocal call_count
                call_count += 1
                return LLMResponse(
                    text=json.dumps({"suggestions": []}),
                    model="counting",
                    provider="counting",
                    usage={"prompt": 10, "completion": 5, "total": 15},
                    finish_reason="stop",
                )

        pairs: list[tuple[Anchor, Anchor]] = [(code, doc)] * 3
        asyncio.run(batch_llm_evaluate(pairs, provider=CountingProvider(), batch_size=2))

        assert call_count == 2

    def test_min_confidence_not_filtered_here(self, scry_db: ScryDB) -> None:
        """batch_llm_evaluate returns all should_link=True entries regardless of confidence."""
        code = scry_db.get_anchor(CODE_ID)
        doc = scry_db.get_anchor(DOC_ID)
        assert code is not None and doc is not None

        provider = FakeProvider(responses=[_good_llm_response("p_0", "references", 0.1)])
        pairs: list[tuple[Anchor, Anchor]] = [(code, doc)]

        suggestions = asyncio.run(batch_llm_evaluate(pairs, provider=provider, batch_size=10))

        # min_confidence filtering is run_suggest_links' responsibility, not ours.
        assert len(suggestions) == 1
        assert suggestions[0].confidence == pytest.approx(0.1)


# ─── run_suggest_links integration tests ─────────────────────────────────────


class TestRunSuggestLinks:
    """End-to-end tests for the async run_suggest_links orchestrator."""

    def test_happy_path(
        self,
        scry_db: ScryDB,
        link_store: LinkStore,
        stub_embedder: StubEmbedder,
    ) -> None:
        """run_suggest_links returns a filtered, sorted suggestion list."""
        provider = FakeProvider(responses=[_good_llm_response("p_0", "mirrors", 0.88)])
        config = SuggestConfig(source="code", top_k_neighbors=1, min_confidence=0.5)

        with patch(
            "scry.suggest.hybrid_search",
            return_value=[_make_search_result(DOC_ID)],
        ):
            results = asyncio.run(
                run_suggest_links(
                    db=scry_db,
                    link_store=link_store,
                    embedder=stub_embedder,
                    provider=provider,
                    config=config,
                )
            )

        assert len(results) == 1
        assert results[0].from_id == CODE_ID
        assert results[0].to_id == DOC_ID
        assert results[0].link_type == "mirrors"

    def test_min_confidence_filter(
        self,
        scry_db: ScryDB,
        link_store: LinkStore,
        stub_embedder: StubEmbedder,
    ) -> None:
        """Suggestions below min_confidence are filtered out."""
        provider = FakeProvider(responses=[_good_llm_response("p_0", "references", 0.3)])
        config = SuggestConfig(source="code", top_k_neighbors=1, min_confidence=0.7)

        with patch(
            "scry.suggest.hybrid_search",
            return_value=[_make_search_result(DOC_ID)],
        ):
            results = asyncio.run(
                run_suggest_links(
                    db=scry_db,
                    link_store=link_store,
                    embedder=stub_embedder,
                    provider=provider,
                    config=config,
                )
            )

        assert results == []

    def test_idempotency_already_linked(
        self,
        scry_db: ScryDB,
        link_store: LinkStore,
        stub_embedder: StubEmbedder,
    ) -> None:
        """Re-running when the link already exists returns 0 suggestions."""
        code_anchor = scry_db.get_anchor(CODE_ID)
        doc_anchor = scry_db.get_anchor(DOC_ID)
        assert code_anchor is not None and doc_anchor is not None

        link_id = new_link_id()
        record = LinkRecord.model_validate(
            {
                "op": LinkOp.UPSERT,
                "link_id": link_id,
                "event_id": new_event_id(),
                "from": CODE_ID,
                "from_type": AnchorType.CODE,
                "to": DOC_ID,
                "to_type": AnchorType.SECTION,
                "type": LinkType.MIRRORS,
                "from_content_hash": code_anchor.content_hash,
                "to_content_hash": doc_anchor.content_hash,
            }
        )
        link_store.append_baseline(record)

        provider = FakeProvider(responses=[_good_llm_response("p_0")])
        config = SuggestConfig(source="code", top_k_neighbors=1)

        with patch(
            "scry.suggest.hybrid_search",
            return_value=[_make_search_result(DOC_ID)],
        ):
            results = asyncio.run(
                run_suggest_links(
                    db=scry_db,
                    link_store=link_store,
                    embedder=stub_embedder,
                    provider=provider,
                    config=config,
                )
            )

        assert results == []

    def test_no_pairs_returns_empty(
        self,
        scry_db: ScryDB,
        link_store: LinkStore,
        stub_embedder: StubEmbedder,
    ) -> None:
        """run_suggest_links returns [] when no candidate pairs are found."""
        provider = FakeProvider(responses=[])
        config = SuggestConfig(source="code", top_k_neighbors=1)

        with patch("scry.suggest.hybrid_search", return_value=[]):
            results = asyncio.run(
                run_suggest_links(
                    db=scry_db,
                    link_store=link_store,
                    embedder=stub_embedder,
                    provider=provider,
                    config=config,
                )
            )

        assert results == []

    def test_sorted_by_confidence_descending(
        self,
        scry_db: ScryDB,
        link_store: LinkStore,
        stub_embedder: StubEmbedder,
    ) -> None:
        """Multiple suggestions are sorted by confidence descending."""
        doc2_id = "docs/api.md::section::api"
        doc2_hash = hashlib.sha256(b"API docs content").hexdigest()
        scry_db.upsert_anchor(
            Anchor(
                id=doc2_id,
                type=AnchorType.SECTION,
                path="docs/api.md",
                content_text="API docs content",
                content_hash=f"sha256:{doc2_hash}",
                fingerprint_simhash=0,
            )
        )

        response_text = json.dumps(
            {
                "suggestions": [
                    {
                        "pair_id": "p_0",
                        "should_link": True,
                        "link_type": "mirrors",
                        "confidence": 0.75,
                        "reason": "good match",
                    },
                    {
                        "pair_id": "p_1",
                        "should_link": True,
                        "link_type": "implements",
                        "confidence": 0.95,
                        "reason": "strong match",
                    },
                ]
            }
        )
        provider = FakeProvider(responses=[response_text])
        config = SuggestConfig(source="code", top_k_neighbors=2, min_confidence=0.5)

        search_results = [
            _make_search_result(DOC_ID),
            _make_search_result(doc2_id, 0.7),
        ]
        with patch("scry.suggest.hybrid_search", return_value=search_results):
            results = asyncio.run(
                run_suggest_links(
                    db=scry_db,
                    link_store=link_store,
                    embedder=stub_embedder,
                    provider=provider,
                    config=config,
                )
            )

        assert len(results) == 2
        assert results[0].confidence >= results[1].confidence


# ─── CLI integration tests ────────────────────────────────────────────────────


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def cli_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """Minimal git repo with .scry/config.yaml + vectors.db with 2 anchors."""
    _git(tmp_path, "init")
    _git(tmp_path, "config", "user.email", "ci@test.local")
    _git(tmp_path, "config", "user.name", "CI Test")
    _git(tmp_path, "commit", "--allow-empty", "-m", "init")

    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir(exist_ok=True)
    (scry_dir / "config.yaml").write_text(
        "include:\n  - '**/*.py'\n  - '**/*.md'\n", encoding="utf-8"
    )

    db = ScryDB(tmp_path)
    db.init_schema(embedding_dimensions=4)
    db.upsert_anchor(_make_anchor(CODE_ID, AnchorType.CODE, "src/main.py", CODE_CONTENT))
    db.upsert_anchor(_make_anchor(DOC_ID, AnchorType.SECTION, "docs/spec.md", DOC_CONTENT))
    db.close()

    yield tmp_path


_STUB_ENV = {"SCRY_EMBEDDER": "stub"}


def _run_cli(
    runner: CliRunner,
    args: list[str],
    *,
    repo: Path,
    input: str | None = None,
    env_extra: dict[str, str] | None = None,
) -> Any:
    env = {**_STUB_ENV, **(env_extra or {})}
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        return runner.invoke(main, args, catch_exceptions=False, env=env, input=input)
    finally:
        os.chdir(old_cwd)


def _mock_run(suggestions: list[LinkSuggestion]) -> Any:
    """Return an async callable that always returns *suggestions*."""

    async def _fake(**kwargs: Any) -> list[LinkSuggestion]:
        return suggestions

    return _fake


class TestSuggestLinksCLI:
    """CLI integration tests for scry suggest-links.

    Patches scry.suggest.run_suggest_links so the CLI's local import picks
    up the mock (the function is imported inside the command body via
    ``from scry.suggest import run_suggest_links``).
    """

    @pytest.fixture
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_json_output_valid(self, runner: CliRunner, cli_repo: Path) -> None:
        """--json produces valid JSON with a suggestions key."""
        suggestion = LinkSuggestion(
            from_id=CODE_ID,
            to_id=DOC_ID,
            link_type="mirrors",
            confidence=0.9,
            reason="They match.",
        )

        with patch("scry.suggest.run_suggest_links", _mock_run([suggestion])):
            result = _run_cli(runner, ["suggest-links", "--json"], repo=cli_repo)

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert "suggestions" in data
        assert len(data["suggestions"]) == 1
        assert data["suggestions"][0]["from_id"] == CODE_ID
        assert data["suggestions"][0]["link_type"] == "mirrors"

    def test_json_empty_when_no_suggestions(self, runner: CliRunner, cli_repo: Path) -> None:
        """--json outputs {suggestions: []} when no suggestions are found."""
        with patch("scry.suggest.run_suggest_links", _mock_run([])):
            result = _run_cli(runner, ["suggest-links", "--json"], repo=cli_repo)

        assert result.exit_code == 0, result.output
        data = json.loads(result.output)
        assert data == {"suggestions": []}

    def test_table_output_contains_ids(self, runner: CliRunner, cli_repo: Path) -> None:
        """Default table output shows anchor IDs and link type."""
        suggestion = LinkSuggestion(
            from_id=CODE_ID,
            to_id=DOC_ID,
            link_type="implements",
            confidence=0.85,
            reason="Implements spec.",
        )

        with patch("scry.suggest.run_suggest_links", _mock_run([suggestion])):
            result = _run_cli(runner, ["suggest-links"], repo=cli_repo)

        assert result.exit_code == 0, result.output
        assert CODE_ID in result.output
        assert DOC_ID in result.output
        assert "implements" in result.output

    def test_no_suggestions_message(self, runner: CliRunner, cli_repo: Path) -> None:
        """When no suggestions, prints a helpful message."""
        with patch("scry.suggest.run_suggest_links", _mock_run([])):
            result = _run_cli(runner, ["suggest-links"], repo=cli_repo)

        assert result.exit_code == 0, result.output
        assert "No link suggestions" in result.output

    def test_apply_yes_writes_link(self, runner: CliRunner, cli_repo: Path) -> None:
        """--apply --yes writes the suggested link to the overlay without prompting."""
        suggestion = LinkSuggestion(
            from_id=CODE_ID,
            to_id=DOC_ID,
            link_type="mirrors",
            confidence=0.9,
            reason="Same concept.",
        )

        with patch("scry.suggest.run_suggest_links", _mock_run([suggestion])):
            result = _run_cli(runner, ["suggest-links", "--apply", "--yes"], repo=cli_repo)

        assert result.exit_code == 0, result.output
        assert "Wrote 1 link" in result.output

    def test_accept_all_implies_apply_yes(self, runner: CliRunner, cli_repo: Path) -> None:
        """--accept-all is equivalent to --apply --yes."""
        suggestion = LinkSuggestion(
            from_id=CODE_ID,
            to_id=DOC_ID,
            link_type="mirrors",
            confidence=0.9,
            reason="Same concept.",
        )

        with patch("scry.suggest.run_suggest_links", _mock_run([suggestion])):
            result = _run_cli(runner, ["suggest-links", "--accept-all"], repo=cli_repo)

        assert result.exit_code == 0, result.output
        assert "Wrote 1 link" in result.output

    def test_apply_prompts_and_aborts_on_no(self, runner: CliRunner, cli_repo: Path) -> None:
        """--apply without --yes prompts; 'n' aborts without writing."""
        suggestion = LinkSuggestion(
            from_id=CODE_ID,
            to_id=DOC_ID,
            link_type="mirrors",
            confidence=0.9,
            reason="Test.",
        )

        with patch("scry.suggest.run_suggest_links", _mock_run([suggestion])):
            result = _run_cli(runner, ["suggest-links", "--apply"], repo=cli_repo, input="n\n")

        assert result.exit_code == 0, result.output
        assert "Aborted" in result.output

    def test_min_confidence_flag_passed_to_config(self, runner: CliRunner, cli_repo: Path) -> None:
        """--min-confidence is forwarded to SuggestConfig.min_confidence."""
        captured: dict[str, Any] = {}

        async def _capture(**kwargs: Any) -> list[LinkSuggestion]:
            captured["config"] = kwargs.get("config")
            return []

        with patch("scry.suggest.run_suggest_links", _capture):
            result = _run_cli(
                runner,
                ["suggest-links", "--min-confidence", "0.85"],
                repo=cli_repo,
            )

        assert result.exit_code == 0, result.output
        assert captured.get("config") is not None
        assert captured["config"].min_confidence == pytest.approx(0.85)

    def test_limit_flag_passed_to_config(self, runner: CliRunner, cli_repo: Path) -> None:
        """--limit is forwarded to SuggestConfig.limit."""
        captured: dict[str, Any] = {}

        async def _capture(**kwargs: Any) -> list[LinkSuggestion]:
            captured["config"] = kwargs.get("config")
            return []

        with patch("scry.suggest.run_suggest_links", _capture):
            result = _run_cli(runner, ["suggest-links", "--limit", "5"], repo=cli_repo)

        assert result.exit_code == 0, result.output
        assert captured.get("config") is not None
        assert captured["config"].limit == 5

    def test_no_db_exits_1(self, runner: CliRunner, tmp_path: Path) -> None:
        """Error exit when vectors.db is missing."""
        _git(tmp_path, "init")
        _git(tmp_path, "config", "user.email", "ci@test.local")
        _git(tmp_path, "config", "user.name", "CI Test")
        _git(tmp_path, "commit", "--allow-empty", "-m", "init")
        (tmp_path / ".scry").mkdir(exist_ok=True)
        (tmp_path / ".scry" / "config.yaml").write_text(
            "include:\n  - '**/*.py'\n", encoding="utf-8"
        )

        result = _run_cli(runner, ["suggest-links"], repo=tmp_path)
        assert result.exit_code == 1
        assert "vectors.db" in result.output

    def test_default_min_confidence_is_reasonable(self) -> None:
        """DEFAULT_MIN_CONFIDENCE is between 0 and 1 (sanity check)."""
        assert 0.0 < DEFAULT_MIN_CONFIDENCE < 1.0
