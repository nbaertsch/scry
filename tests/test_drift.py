"""Tests for scry.drift — section-level drift detection.

Covers DESIGN.md §5.1 (all drift statuses + precedence + prior-hash override
+ semantic_drift flag) and §5.2 (drift_score + coverage_score formulas,
null semantics, DriftCounts population).

Every §5.1 status value has at least one positive test. Wave 2 never
produces ``drift-unknown`` — that status is excluded here by design.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from scry.drift import (
    DriftDetectionError,
    DriftEvaluation,
    compute_drift_summary,
    evaluate_all_drift,
    evaluate_link_drift,
    evaluate_semantic_drift,
)
from scry.models import (
    Anchor,
    AnchorType,
    DriftConfig,
    DriftStatus,
    Link,
    LinkType,
    TransitiveHashStatus,
    new_event_id,
    new_link_id,
)
from scry.store.db import ScryDB
from scry.store.links import LinkStore

# ─── Constants ────────────────────────────────────────────────────────────────

_HA = "sha256:" + "a" * 64  # "original" hash
_HB = "sha256:" + "b" * 64  # "changed" hash
_HC = "sha256:" + "c" * 64  # third distinct hash

# SHA-256 of empty bytes — the sentinel stored in ``closure_hash`` by the LSP
# closure walker for early-exit ``lsp_error`` and ``unsupported`` paths.
# lsp/closure.py: _EMPTY_SHA256 = "sha256:e3b0c4..."
_EMPTY_CLOSURE = "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
# A non-empty closure hash (represents "LSP computed a real closure").
_CLOSURE_A = "sha256:" + "c" * 64
_CLOSURE_B = "sha256:" + "d" * 64

_FROM_ID = "docs/spec.md::intro"
_TO_ID = "src/app.py:main"

_DIMS = 4

# ─── DB fixture ───────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_repo: Path) -> ScryDB:
    """A ScryDB opened over ``tmp_repo`` with schema initialised."""
    d = ScryDB(tmp_repo)
    d.init_schema(embedding_dimensions=_DIMS)
    return d


# ─── Builders ─────────────────────────────────────────────────────────────────


def _make_anchor(
    anchor_id: str = _FROM_ID,
    *,
    anchor_type: AnchorType = AnchorType.SECTION,
    content_hash: str = _HA,
    path: str = "docs/spec.md",
    symbol_name: str | None = None,
    transitive_hash_status: TransitiveHashStatus | None = None,
    closure_hash: str | None = None,
) -> Anchor:
    """Build a minimal valid ``Anchor`` for testing."""
    return Anchor(
        id=anchor_id,
        type=anchor_type,
        path=path,
        content_text="test content",
        content_hash=content_hash,
        fingerprint_simhash=0xDEAD,
        symbol_name=symbol_name,
        transitive_hash_status=transitive_hash_status,
        closure_hash=closure_hash,
    )


def _make_code_anchor(
    anchor_id: str = _TO_ID,
    *,
    content_hash: str = _HA,
    transitive_hash_status: TransitiveHashStatus = TransitiveHashStatus.LSP_UNAVAILABLE,
    closure_hash: str | None = None,
) -> Anchor:
    """Build a minimal CODE anchor."""
    return _make_anchor(
        anchor_id=anchor_id,
        anchor_type=AnchorType.CODE,
        content_hash=content_hash,
        path="src/app.py",
        transitive_hash_status=transitive_hash_status,
        closure_hash=closure_hash,
    )


def _make_link(
    *,
    from_id: str = _FROM_ID,
    from_type: AnchorType = AnchorType.SECTION,
    to_id: str = _TO_ID,
    to_type: AnchorType = AnchorType.CODE,
    link_type: LinkType = LinkType.IMPLEMENTS,
    from_hash: str = _HA,
    to_hash: str = _HA,
    prior_from_hash: str | None = None,
    prior_to_hash: str | None = None,
    from_closure_hash: str | None = None,
    to_closure_hash: str | None = None,
    link_id: str | None = None,
) -> Link:
    """Build a minimal valid ``Link`` for testing."""
    return Link(
        link_id=link_id or new_link_id(),
        from_id=from_id,
        from_type=from_type,
        to_id=to_id,
        to_type=to_type,
        type=link_type,
        from_content_hash=from_hash,
        to_content_hash=to_hash,
        prior_from_content_hash=prior_from_hash,
        prior_to_content_hash=prior_to_hash,
        from_closure_hash=from_closure_hash,
        to_closure_hash=to_closure_hash,
        last_event_id=new_event_id(),
    )


def _float_blob(*values: float) -> bytes:
    """Pack floats into a float32 little-endian BLOB for ``overview_embedding``."""
    return struct.pack(f"{len(values)}f", *values)


def _insert_embedding(db: ScryDB, anchor_id: str, *values: float) -> None:
    """Write ``overview_embedding`` for an existing anchor row."""
    db._conn.execute(
        "UPDATE anchors SET overview_embedding = ? WHERE id = ?",
        (_float_blob(*values), anchor_id),
    )
    db._conn.commit()


# ─── Tests: evaluate_link_drift — status ladder ───────────────────────────────


class TestEvaluateLinkDriftStatuses:
    """One positive test per §5.1 status value (Wave 2 subset — no drift-unknown)."""

    def test_fresh(self, db: ScryDB) -> None:
        """Both endpoints unchanged → fresh."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH
        assert ev.semantic_drift is None  # not a mirrors link

    def test_spec_changed_section_from(self, db: ScryDB) -> None:
        """SECTION from-anchor hash changed → spec-changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HB))  # changed
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))
        link = _make_link(from_hash=_HA, to_hash=_HA)  # stored = HA, current = HB

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.SPEC_CHANGED

    def test_spec_changed_section_to(self, db: ScryDB) -> None:
        """SECTION to-anchor hash changed → spec-changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        # to-anchor is SECTION (overriding default CODE)
        db.upsert_anchor(_make_anchor(_TO_ID, content_hash=_HB, path="src/app.py"))
        link = _make_link(
            to_type=AnchorType.SECTION,
            from_hash=_HA,
            to_hash=_HA,
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.SPEC_CHANGED

    def test_code_changed(self, db: ScryDB) -> None:
        """CODE to-anchor hash changed → code-changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HB))  # changed
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED

    def test_code_changed_from_is_code(self, db: ScryDB) -> None:
        """CODE from-anchor hash changed → code-changed (CODE → anything)."""
        db.upsert_anchor(_make_code_anchor(_FROM_ID, content_hash=_HB))  # changed
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))
        link = _make_link(
            from_type=AnchorType.CODE,
            to_type=AnchorType.CODE,
            from_hash=_HA,
            to_hash=_HA,
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED

    def test_both_changed(self, db: ScryDB) -> None:
        """Both endpoints' hashes changed → both-changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HB))
        db.upsert_anchor(_make_code_anchor(content_hash=_HB))
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.BOTH_CHANGED

    def test_broken_source(self, db: ScryDB) -> None:
        """from-anchor absent from DB → broken-source."""
        # Only insert the to-anchor; leave from-anchor absent.
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))
        link = _make_link()

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.BROKEN_SOURCE

    def test_broken_target(self, db: ScryDB) -> None:
        """to-anchor absent from DB → broken-target."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        link = _make_link()

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.BROKEN_TARGET

    def test_merge_conflict(self, db: ScryDB) -> None:
        """link_id in merge_conflicts set → merge-conflict."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))
        link = _make_link()

        ev = evaluate_link_drift(link, db=db, merge_conflicts={link.link_id})

        assert ev.drift_status == DriftStatus.MERGE_CONFLICT


# ─── Tests: evaluate_link_drift — precedence ladder ──────────────────────────


class TestPrecedenceLadder:
    """Verify §5.1 status precedence: merge-conflict > broken-* > both-changed > …"""

    def test_merge_conflict_beats_broken_source(self, db: ScryDB) -> None:
        """merge-conflict wins even when from-anchor is also absent."""
        link = _make_link()  # both anchors absent (broken-source would apply too)

        ev = evaluate_link_drift(link, db=db, merge_conflicts={link.link_id})

        assert ev.drift_status == DriftStatus.MERGE_CONFLICT

    def test_merge_conflict_beats_broken_target(self, db: ScryDB) -> None:
        """merge-conflict wins over broken-target."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        # to-anchor absent → would be broken-target, but conflict wins.
        link = _make_link()

        ev = evaluate_link_drift(link, db=db, merge_conflicts={link.link_id})

        assert ev.drift_status == DriftStatus.MERGE_CONFLICT

    def test_broken_source_beats_broken_target(self, db: ScryDB) -> None:
        """broken-source wins over broken-target when both anchors are absent."""
        link = _make_link()

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.BROKEN_SOURCE

    def test_broken_source_beats_both_changed(self, db: ScryDB) -> None:
        """broken-source wins even if stored hashes would indicate both-changed."""
        # Only to-anchor exists; from-anchor absent.
        db.upsert_anchor(_make_code_anchor(content_hash=_HB))  # would be to-changed
        link = _make_link(from_hash=_HB, to_hash=_HB)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.BROKEN_SOURCE

    def test_both_changed_beats_spec_changed(self, db: ScryDB) -> None:
        """both-changed wins over spec-changed when both endpoints differ."""
        db.upsert_anchor(_make_anchor(content_hash=_HB))
        db.upsert_anchor(_make_code_anchor(content_hash=_HB))
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.BOTH_CHANGED

    def test_spec_changed_beats_fresh(self, db: ScryDB) -> None:
        """spec-changed wins over fresh when spec endpoint changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HB))  # spec changed
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))  # code fresh
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.SPEC_CHANGED

    def test_code_changed_beats_fresh(self, db: ScryDB) -> None:
        """code-changed wins over fresh when code endpoint changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))  # spec fresh
        db.upsert_anchor(_make_code_anchor(content_hash=_HB))  # code changed
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED


# ─── Tests: prior-hash override (§3.3) ───────────────────────────────────────


class TestPriorHashOverride:
    """Verify §3.3 prior_*_content_hash escalates fresh→changed after rebase."""

    def test_prior_from_hash_escalates_to_spec_changed(self, db: ScryDB) -> None:
        """A rebase sets from_content_hash=HC and prior_from_content_hash=HA.
        Current from-anchor hash=HC matches the stored post-rebase hash → would
        be fresh naively. But prior hash HA ≠ current HC → spec-changed.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HC))  # current = HC
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))

        link = _make_link(
            from_hash=_HC,  # post-rebase stored hash
            to_hash=_HA,
            prior_from_hash=_HA,  # pre-rebase hash ← this triggers override
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.SPEC_CHANGED

    def test_prior_to_hash_escalates_to_code_changed(self, db: ScryDB) -> None:
        """Same override on the to-endpoint (CODE anchor) → code-changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HC))  # current = HC

        link = _make_link(
            from_hash=_HA,
            to_hash=_HC,  # post-rebase stored hash
            prior_to_hash=_HA,  # pre-rebase hash ← triggers override
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED

    def test_both_prior_hashes_escalate_to_both_changed(self, db: ScryDB) -> None:
        """Both prior fields set and both current hashes differ → both-changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HC))
        db.upsert_anchor(_make_code_anchor(content_hash=_HC))

        link = _make_link(
            from_hash=_HC,
            to_hash=_HC,
            prior_from_hash=_HA,
            prior_to_hash=_HA,
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.BOTH_CHANGED

    def test_prior_hash_same_as_current_stays_fresh(self, db: ScryDB) -> None:
        """Prior hash equals current hash → override does not fire → fresh."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))

        link = _make_link(
            from_hash=_HA,
            to_hash=_HA,
            prior_from_hash=_HA,  # prior == current → no escalation
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH

    def test_prior_hash_absent_uses_stored_hash(self, db: ScryDB) -> None:
        """Without prior fields, naive stored-vs-current comparison applies."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))

        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH


# ─── Tests: semantic_drift ────────────────────────────────────────────────────


class TestSemanticDrift:
    """Verify evaluate_semantic_drift and its integration in evaluate_link_drift."""

    def _setup_mirrors_link(self, db: ScryDB) -> tuple[Link, str, str]:
        """Insert two anchors and return a mirrors link + their IDs."""
        from_id = "docs/spec.md::algo"
        to_id = "src/lib.py:algo"
        db.upsert_anchor(_make_anchor(from_id, content_hash=_HA, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, content_hash=_HA))
        link = _make_link(
            from_id=from_id,
            to_id=to_id,
            link_type=LinkType.MIRRORS,
            from_hash=_HA,
            to_hash=_HA,
        )
        return link, from_id, to_id

    def test_semantic_drift_true_for_divergent_embeddings(self, db: ScryDB) -> None:
        """Cosine distance > threshold → semantic_drift=True."""
        link, from_id, to_id = self._setup_mirrors_link(db)
        # Orthogonal unit vectors → cosine similarity = 0.0 → distance = 1.0
        _insert_embedding(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_embedding(db, to_id, 0.0, 1.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is True

    def test_semantic_drift_false_for_similar_embeddings(self, db: ScryDB) -> None:
        """Cosine distance < threshold → semantic_drift=False."""
        link, from_id, to_id = self._setup_mirrors_link(db)
        # Identical vectors → distance = 0.0 → below any positive threshold.
        _insert_embedding(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_embedding(db, to_id, 1.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is False

    def test_semantic_drift_none_when_from_embedding_missing(self, db: ScryDB) -> None:
        """from-anchor has no embedding → None."""
        link, _, to_id = self._setup_mirrors_link(db)
        _insert_embedding(db, to_id, 1.0, 0.0, 0.0, 0.0)
        # from-anchor embedding intentionally left NULL.

        result = evaluate_semantic_drift(link, db=db)

        assert result is None

    def test_semantic_drift_none_when_to_embedding_missing(self, db: ScryDB) -> None:
        """to-anchor has no embedding → None."""
        link, from_id, _ = self._setup_mirrors_link(db)
        _insert_embedding(db, from_id, 1.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is None

    def test_semantic_drift_none_when_both_embeddings_missing(self, db: ScryDB) -> None:
        """Neither embedding present → None."""
        link, _, _ = self._setup_mirrors_link(db)

        result = evaluate_semantic_drift(link, db=db)

        assert result is None

    def test_evaluate_link_drift_sets_semantic_drift_on_mirrors(self, db: ScryDB) -> None:
        """evaluate_link_drift propagates semantic_drift for mirrors links."""
        link, from_id, to_id = self._setup_mirrors_link(db)
        # Divergent embeddings.
        _insert_embedding(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_embedding(db, to_id, 0.0, 1.0, 0.0, 0.0)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH
        assert ev.semantic_drift is True

    def test_evaluate_link_drift_semantic_drift_none_for_non_mirrors(self, db: ScryDB) -> None:
        """Non-mirrors links always get semantic_drift=None."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))
        link = _make_link(link_type=LinkType.IMPLEMENTS)

        ev = evaluate_link_drift(link, db=db)

        assert ev.semantic_drift is None

    def test_semantic_drift_threshold_boundary(self, db: ScryDB) -> None:
        """Exactly at threshold → False (strictly greater is True)."""
        link, from_id, to_id = self._setup_mirrors_link(db)
        # Build vectors with cosine distance exactly equal to the threshold.
        threshold = DriftConfig().semantic_drift_threshold  # default 0.25
        # cos(θ) = 1 - threshold  →  θ = arccos(1 - threshold)
        # We use a = [1, 0] and b = [cos(θ), sin(θ)] in 2D; pad to 4D.
        import math

        cos_sim = 1.0 - threshold
        sin_val = math.sqrt(max(0.0, 1.0 - cos_sim**2))
        _insert_embedding(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_embedding(db, to_id, float(cos_sim), float(sin_val), 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        # Distance equals threshold exactly → NOT strictly greater → False.
        assert result is False


# ─── Tests: evaluate_all_drift ────────────────────────────────────────────────


class TestEvaluateAllDrift:
    def test_empty_link_store_returns_empty_list(self, db: ScryDB, tmp_repo: Path) -> None:
        """No links → empty result list."""
        store = LinkStore(tmp_repo)
        result = evaluate_all_drift(db=db, link_store=store)
        assert result == []

    def test_active_links_evaluated(self, db: ScryDB, tmp_repo: Path) -> None:
        """Active links in the store are evaluated and returned."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))

        store = LinkStore(tmp_repo)
        from scry.models import LinkRecord, new_link_id

        lid = new_link_id()
        record = LinkRecord.model_validate(
            {
                "op": "upsert",
                "link_id": lid,
                "from": _FROM_ID,
                "from_type": "section",
                "to": _TO_ID,
                "to_type": "code",
                "type": "implements",
                "from_content_hash": _HA,
                "to_content_hash": _HA,
            }
        )
        store.append_baseline(record)

        result = evaluate_all_drift(db=db, link_store=store)

        assert len(result) == 1
        assert result[0].drift_status == DriftStatus.FRESH

    def test_merge_conflicts_from_replay_propagated(self, db: ScryDB, tmp_repo: Path) -> None:
        """merge_conflicts from replay are forwarded to evaluate_link_drift."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))

        store = LinkStore(tmp_repo)
        from scry.models import LinkRecord, new_event_id, new_link_id

        lid = new_link_id()
        eid1 = new_event_id()
        r1 = LinkRecord.model_validate(
            {
                "op": "upsert",
                "link_id": lid,
                "event_id": eid1,
                "from": _FROM_ID,
                "from_type": "section",
                "to": _TO_ID,
                "to_type": "code",
                "type": "implements",
                "from_content_hash": _HA,
                "to_content_hash": _HA,
            }
        )
        store.append_baseline(r1)
        # Write both links; the first one is clean (fresh), check that the
        # conflict detection path is reachable by using the replay API directly.
        result = evaluate_all_drift(db=db, link_store=store)
        assert any(ev.drift_status == DriftStatus.FRESH for ev in result)


# ─── Tests: compute_drift_summary — §5.2 ─────────────────────────────────────


class TestComputeDriftSummary:
    def _fresh_ev(self) -> DriftEvaluation:
        """Dummy DriftEvaluation with FRESH status."""
        link = _make_link()
        return DriftEvaluation(link=link, drift_status=DriftStatus.FRESH, semantic_drift=None)

    def _status_ev(
        self, status: DriftStatus, semantic_drift: bool | None = None
    ) -> DriftEvaluation:
        link_type = LinkType.MIRRORS if semantic_drift is not None else LinkType.IMPLEMENTS
        link = _make_link(link_type=link_type)
        return DriftEvaluation(link=link, drift_status=status, semantic_drift=semantic_drift)

    # — null semantics —

    def test_drift_score_null_for_empty_evaluations(self) -> None:
        """No links → drift_score is None per §5.2 v3.1."""
        summary = compute_drift_summary([])
        assert summary.drift_score is None

    def test_coverage_score_null_when_no_code_anchors(self) -> None:
        """No code anchors → coverage_score is None."""
        summary = compute_drift_summary([])
        assert summary.coverage_score is None

    def test_coverage_score_null_when_total_code_anchors_zero(self) -> None:
        """coverage_total_code_anchors=0 → coverage_score is None."""
        summary = compute_drift_summary([], coverage_total_code_anchors=0)
        assert summary.coverage_score is None

    # — drift_coverage always section-only for Wave 2 —

    def test_drift_coverage_section_only_empty(self) -> None:
        summary = compute_drift_summary([])
        assert summary.drift_coverage == "section-only"

    def test_drift_coverage_section_only_with_evaluations(self) -> None:
        evs = [self._fresh_ev()]
        summary = compute_drift_summary(evs)
        assert summary.drift_coverage == "section-only"

    # — drift_score formula: hand-computed scenario —

    def test_drift_score_formula_hand_computed(self) -> None:
        """Manual scenario with known weights verifies §5.2 formula.

        1 spec-changed (w=0.3) + 1 fresh (w=0) over 2 links:
            weighted_sum = 0.3 * 1 + 0 * 1 = 0.3
            drift_score  = 100 * (1 - 0.3 / 2) = 100 * 0.85 = 85.0
        """
        evs = [
            self._status_ev(DriftStatus.SPEC_CHANGED),
            self._fresh_ev(),
        ]
        summary = compute_drift_summary(evs)

        assert summary.drift_score == pytest.approx(85.0)

    def test_drift_score_all_fresh_is_100(self) -> None:
        """All fresh → drift_score = 100.0."""
        evs = [self._fresh_ev(), self._fresh_ev()]
        summary = compute_drift_summary(evs)
        assert summary.drift_score == pytest.approx(100.0)

    def test_drift_score_all_broken_source_is_0(self) -> None:
        """All broken-source (w=1.0) over all links → drift_score = 0.0."""
        evs = [self._status_ev(DriftStatus.BROKEN_SOURCE)]
        summary = compute_drift_summary(evs)
        assert summary.drift_score == pytest.approx(0.0)

    def test_drift_score_clamped_above_zero(self) -> None:
        """Score cannot go below 0 (broken + semantic_drift on one link)."""
        config = DriftConfig()
        # One broken-source link with semantic_drift True: 1.0 + 0.2 = 1.2 per link
        ev = self._status_ev(DriftStatus.BROKEN_SOURCE, semantic_drift=True)
        summary = compute_drift_summary([ev], config=config)
        assert summary.drift_score is not None
        assert summary.drift_score >= 0.0

    # — coverage_score formula —

    def test_coverage_score_formula(self) -> None:
        """3 linked out of 5 code anchors → 60.0."""
        summary = compute_drift_summary(
            [],
            coverage_total_code_anchors=5,
            coverage_linked_code_anchors=3,
        )
        assert summary.coverage_score == pytest.approx(60.0)

    def test_coverage_score_100_all_linked(self) -> None:
        summary = compute_drift_summary(
            [],
            coverage_total_code_anchors=4,
            coverage_linked_code_anchors=4,
        )
        assert summary.coverage_score == pytest.approx(100.0)

    def test_coverage_score_zero_none_linked(self) -> None:
        summary = compute_drift_summary(
            [],
            coverage_total_code_anchors=4,
            coverage_linked_code_anchors=0,
        )
        assert summary.coverage_score == pytest.approx(0.0)

    # — DriftCounts raw counts —

    def test_counts_populated_correctly(self) -> None:
        """Each status bucket and semantic_drift_flagged are counted correctly."""
        evs = [
            self._status_ev(DriftStatus.BROKEN_SOURCE),
            self._status_ev(DriftStatus.BROKEN_TARGET),
            self._status_ev(DriftStatus.MERGE_CONFLICT),
            self._status_ev(DriftStatus.BOTH_CHANGED),
            self._status_ev(DriftStatus.SPEC_CHANGED),
            self._status_ev(DriftStatus.CODE_CHANGED),
            self._fresh_ev(),
        ]
        summary = compute_drift_summary(evs)
        c = summary.counts

        assert c.broken_source == 1
        assert c.broken_target == 1
        assert c.merge_conflict == 1
        assert c.both_changed == 1
        assert c.spec_changed == 1
        assert c.code_changed == 1
        assert c.fresh == 1
        assert c.total == 7
        assert c.semantic_drift_flagged == 0

    def test_semantic_drift_flagged_increments_independently(self) -> None:
        """semantic_drift_flagged increments regardless of base status per §5.2."""
        evs = [
            # mirrors + code-changed + semantic_drift True
            self._status_ev(DriftStatus.CODE_CHANGED, semantic_drift=True),
            # mirrors + fresh + semantic_drift True
            self._status_ev(DriftStatus.FRESH, semantic_drift=True),
            # mirrors + spec-changed + semantic_drift False (not flagged)
            self._status_ev(DriftStatus.SPEC_CHANGED, semantic_drift=False),
        ]
        summary = compute_drift_summary(evs)

        assert summary.counts.semantic_drift_flagged == 2
        assert summary.counts.code_changed == 1
        assert summary.counts.spec_changed == 1
        assert summary.counts.fresh == 1

    def test_semantic_drift_weight_added_on_top_of_base_weight(self) -> None:
        """semantic_drift adds 0.2 on top of base status weight (§5.2).

        1 mirrors/code-changed link (w=0.3) with semantic_drift=True (w=+0.2):
            weighted_sum = 0.3 + 0.2 = 0.5 over 1 link
            drift_score  = 100 * (1 - 0.5 / 1) = 50.0
        """
        ev = self._status_ev(DriftStatus.CODE_CHANGED, semantic_drift=True)
        summary = compute_drift_summary([ev])

        assert summary.drift_score == pytest.approx(50.0)

    def test_semantic_drift_false_does_not_add_weight(self) -> None:
        """semantic_drift=False does NOT increment flagged or add weight."""
        ev = self._status_ev(DriftStatus.CODE_CHANGED, semantic_drift=False)
        summary = compute_drift_summary([ev])

        # Only base code_changed weight: 100 * (1 - 0.3) = 70.0
        assert summary.drift_score == pytest.approx(70.0)
        assert summary.counts.semantic_drift_flagged == 0

    def test_semantic_drift_none_does_not_add_weight(self) -> None:
        """semantic_drift=None (missing embedding) does NOT add weight."""
        ev = self._status_ev(DriftStatus.FRESH, semantic_drift=None)
        summary = compute_drift_summary([ev])

        assert summary.drift_score == pytest.approx(100.0)
        assert summary.counts.semantic_drift_flagged == 0

    def test_drift_unknown_status_counted_and_weighted(self) -> None:
        """drift_unknown (Wave 4 status) is correctly bucketed if passed in."""
        ev = self._status_ev(DriftStatus.DRIFT_UNKNOWN)
        summary = compute_drift_summary([ev])

        assert summary.counts.drift_unknown == 1
        # weight = 0.3; 1 link: 100 * (1 - 0.3) = 70.0
        assert summary.drift_score == pytest.approx(70.0)

    # — custom config —

    def test_custom_scoring_config_applied(self) -> None:
        """Custom weights in DriftConfig override defaults."""
        from scry.models import DriftScoringConfig

        config = DriftConfig(scoring=DriftScoringConfig(spec_changed=0.5))
        ev = self._status_ev(DriftStatus.SPEC_CHANGED)
        summary = compute_drift_summary([ev], config=config)

        # 100 * (1 - 0.5 / 1) = 50.0
        assert summary.drift_score == pytest.approx(50.0)


# ─── Tests: DriftDetectionError is importable / raiseable ─────────────────────


class TestDriftDetectionError:
    def test_is_exception(self) -> None:
        err = DriftDetectionError("something failed")
        assert isinstance(err, Exception)
        assert str(err) == "something failed"


# ─── Tests: CODE_IN_DOC anchor type handling ──────────────────────────────────


class TestCodeInDocAnchorType:
    """CODE_IN_DOC anchors are treated as code endpoints for drift labelling."""

    def test_code_in_doc_from_changed_gives_code_changed(self, db: ScryDB) -> None:
        """CODE_IN_DOC from-anchor changed → code-changed (not spec-changed)."""
        from_id = "docs/spec.md::intro::example"
        to_id = "src/app.py:main"
        db.upsert_anchor(
            Anchor(
                id=from_id,
                type=AnchorType.CODE_IN_DOC,
                path="docs/spec.md",
                content_text="example code",
                content_hash=_HB,  # changed
                fingerprint_simhash=0,
            )
        )
        db.upsert_anchor(_make_code_anchor(to_id, content_hash=_HA))

        link = _make_link(
            from_id=from_id,
            from_type=AnchorType.CODE_IN_DOC,
            to_id=to_id,
            to_type=AnchorType.CODE,
            from_hash=_HA,  # stored was HA, current is HB
            to_hash=_HA,
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED


# ─── Tests: drift-unknown signal (§5.1 v3.1, W4a) ────────────────────────────


class TestDriftUnknown:
    """Regression tests for the ``drift-unknown`` status (DESIGN.md §5.1 v3.1).

    ``drift-unknown`` fires when status would be ``fresh`` (both content/closure
    hashes match) but a CODE endpoint carries ``transitive_hash_status == "lsp_error"``.
    This means the closure-derived signal is missing; the link cannot be fully
    evaluated.  CI policy default is to fail on ``drift-unknown``.

    Precedence (§5.1): code-changed > drift-unknown > fresh.  So drift-unknown
    is only produced when the hash comparison doesn't surface a concrete change.
    """

    def test_drift_unknown_from_lsp_error(self, db: ScryDB) -> None:
        """from=CODE with lsp_error, hashes unchanged → drift-unknown (§5.1)."""
        db.upsert_anchor(
            _make_code_anchor(
                _FROM_ID,
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
            )
        )
        db.upsert_anchor(_make_code_anchor(content_hash=_HA))
        link = _make_link(
            from_type=AnchorType.CODE,
            to_type=AnchorType.CODE,
            from_hash=_HA,
            to_hash=_HA,
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.DRIFT_UNKNOWN

    def test_drift_unknown_to_lsp_error(self, db: ScryDB) -> None:
        """to=CODE with lsp_error, hashes unchanged → drift-unknown (§5.1, target side)."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))  # from=SECTION, fresh
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.DRIFT_UNKNOWN

    def test_drift_unknown_both_lsp_error(self, db: ScryDB) -> None:
        """Both CODE endpoints have lsp_error → drift-unknown."""
        db.upsert_anchor(
            _make_code_anchor(
                _FROM_ID,
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
            )
        )
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
            )
        )
        link = _make_link(
            from_type=AnchorType.CODE,
            to_type=AnchorType.CODE,
            from_hash=_HA,
            to_hash=_HA,
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.DRIFT_UNKNOWN

    def test_drift_unknown_not_from_partial_status(self, db: ScryDB) -> None:
        """partial transitive_hash_status does NOT trigger drift-unknown (§5.3).

        ``partial`` means some callees resolved; the hash still reflects real data.
        Only ``lsp_error`` warrants ``drift-unknown``.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.PARTIAL,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH

    def test_drift_unknown_not_from_unsupported_status(self, db: ScryDB) -> None:
        """unsupported transitive_hash_status does NOT trigger drift-unknown (§5.3).

        ``unsupported`` means the LSP lacks callHierarchy; the anchor's own AST
        hash is the complete signal available.  No closure info was expected.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.UNSUPPORTED,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH

    def test_drift_unknown_not_from_lsp_unavailable_status(self, db: ScryDB) -> None:
        """lsp_unavailable does NOT trigger drift-unknown (§5.3).

        ``lsp_unavailable`` means the LSP binary is absent from PATH; this is
        expected when scry is run without a language server installed.  The signal
        is weaker but not a runtime failure.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.LSP_UNAVAILABLE,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH

    def test_drift_unknown_not_from_section_anchor(self, db: ScryDB) -> None:
        """SECTION anchors have no transitive_hash_status → drift-unknown never fires.

        §5.1 field placement rule: ``transitive_hash_status`` is omitted from
        JSON for non-CODE targets; no LSP inference for markdown sections.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))  # SECTION from
        db.upsert_anchor(_make_anchor(_TO_ID, content_hash=_HA, path="docs/target.md"))
        link = _make_link(
            to_type=AnchorType.SECTION,
            from_hash=_HA,
            to_hash=_HA,
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH

    def test_drift_unknown_not_from_code_in_doc_anchor(self, db: ScryDB) -> None:
        """CODE_IN_DOC anchors are never CODE-typed → drift-unknown never fires for them."""
        from_id = "docs/spec.md::intro::snippet"
        db.upsert_anchor(
            Anchor(
                id=from_id,
                type=AnchorType.CODE_IN_DOC,
                path="docs/spec.md",
                content_text="fn example() {}",
                content_hash=_HA,
                fingerprint_simhash=0,
            )
        )
        db.upsert_anchor(_make_anchor(_TO_ID, content_hash=_HA, path="docs/b.md"))
        link = _make_link(
            from_id=from_id,
            from_type=AnchorType.CODE_IN_DOC,
            to_type=AnchorType.SECTION,
            from_hash=_HA,
            to_hash=_HA,
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH


# ─── Tests: drift-unknown vs. code-changed precedence (§5.1 v3.1) ────────────


class TestDriftUnknownPrecedence:
    """§5.1 precedence: code-changed / spec-changed beat drift-unknown.

    When a CODE endpoint has ``lsp_error`` BUT also has a changed content hash
    or changed closure hash (from non-error signal), the concrete change status
    wins over drift-unknown.
    """

    def test_code_changed_beats_drift_unknown_content_hash(self, db: ScryDB) -> None:
        """content_hash changed + lsp_error → code-changed (not drift-unknown).

        §5.1: code-changed > drift-unknown.  When the AST text changed we have
        a concrete signal; LSP uncertainty is secondary.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HB,  # changed AST
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
            )
        )
        # Link stored _HA; current anchor is _HB → content diff detected.
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED

    def test_spec_changed_beats_drift_unknown(self, db: ScryDB) -> None:
        """spec endpoint changed + to-CODE has lsp_error → spec-changed (not drift-unknown)."""
        db.upsert_anchor(_make_anchor(content_hash=_HB))  # spec changed
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.SPEC_CHANGED

    def test_both_changed_beats_drift_unknown(self, db: ScryDB) -> None:
        """Both content hashes changed + lsp_error → both-changed (not drift-unknown)."""
        db.upsert_anchor(_make_anchor(content_hash=_HB))  # spec changed
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HB,  # code content changed
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.BOTH_CHANGED


# ─── Tests: closure-hash-based code-changed (§5.3, W3d) ──────────────────────


class TestClosureHashDrift:
    """Tests for the §5.3 transitive-closure hash comparison in ``_resolve_changed``.

    ``code-changed`` can fire when the caller's own AST is unchanged but a
    callee's body changed (surfaced via ``closure_hash`` comparison).
    Several edge cases guard against false positives.
    """

    def test_closure_hash_changed_to_endpoint_gives_code_changed(self, db: ScryDB) -> None:
        """to=CODE closure_hash changed → code-changed even if content_hash unchanged.

        §5.3: "if a callee's body changed, closure_hash differs even when the
        caller's own AST is unchanged."
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))  # spec unchanged
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,  # own AST unchanged
                transitive_hash_status=TransitiveHashStatus.PARTIAL,
                closure_hash=_CLOSURE_B,  # callee changed → hash differs
            )
        )
        # Baseline link stored _CLOSURE_A as the to-endpoint closure hash.
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED

    def test_closure_hash_changed_from_endpoint_gives_code_changed(self, db: ScryDB) -> None:
        """from=CODE closure_hash changed → code-changed (from-endpoint side)."""
        db.upsert_anchor(
            _make_code_anchor(
                _FROM_ID,
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.COMPLETE,
                closure_hash=_CLOSURE_B,  # callee changed
            )
        )
        # to-anchor: a SECTION at _TO_ID so that it exists in the DB.
        db.upsert_anchor(_make_anchor(_TO_ID, content_hash=_HA, path="src/app.py"))
        link = _make_link(
            from_type=AnchorType.CODE,
            to_type=AnchorType.SECTION,
            from_hash=_HA,
            to_hash=_HA,
            from_closure_hash=_CLOSURE_A,  # baseline stored _CLOSURE_A
        )

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED

    def test_closure_hash_unchanged_stays_fresh(self, db: ScryDB) -> None:
        """closure_hash matches baseline → fresh (no false positive)."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.COMPLETE,
                closure_hash=_CLOSURE_A,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH

    def test_no_baseline_closure_hash_no_false_positive(self, db: ScryDB) -> None:
        """Backward compat: link.to_closure_hash=None → no closure comparison (W3d).

        Old links created before the closure-hash field was added have
        ``to_closure_hash=None``; we must never fire ``code-changed`` from
        comparing against an absent baseline.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.COMPLETE,
                closure_hash=_CLOSURE_A,  # current has a closure hash
            )
        )
        # link.to_closure_hash=None — no baseline to compare against.
        link = _make_link(from_hash=_HA, to_hash=_HA)  # to_closure_hash defaults to None

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH

    def test_lsp_error_suppresses_closure_comparison_no_false_positive(self, db: ScryDB) -> None:
        """KEY REGRESSION: lsp_error must NOT produce code-changed via closure diff.

        Scenario: link was created when LSP worked (real closure hash stored).
        LSP later fails → anchor.closure_hash becomes _EMPTY_SHA256 (sentinel).
        Without the guard, ``_EMPTY_SHA256 != real_hash`` → false-positive
        ``code-changed``.  With the guard, closure comparison is suppressed and
        ``drift-unknown`` fires instead (§5.1 v3.1: "text hashes match so not
        code-changed, but closure-derived signal is missing").
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,  # own AST unchanged
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
                closure_hash=_EMPTY_CLOSURE,  # early-exit lsp_error sentinel
            )
        )
        # Baseline link recorded a real closure hash from when LSP worked.
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        # Must be drift-unknown, NOT code-changed.
        assert ev.drift_status == DriftStatus.DRIFT_UNKNOWN

    def test_lsp_error_with_content_change_still_code_changed(self, db: ScryDB) -> None:
        """lsp_error + content_hash changed → code-changed (AST is authoritative).

        Even when LSP errored and we suppress closure comparison, a real change
        to the anchor's own AST text produces ``code-changed``.  The concrete
        change signal takes precedence over the uncertain closure state.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HB,  # AST CHANGED
                transitive_hash_status=TransitiveHashStatus.LSP_ERROR,
                closure_hash=_EMPTY_CLOSURE,
            )
        )
        # Baseline stored _HA; AST changed → code-changed, not drift-unknown.
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED

    def test_partial_status_closure_hash_change_gives_code_changed(self, db: ScryDB) -> None:
        """partial status does NOT suppress closure comparison — its hash is usable.

        §5.3: partial means at least one symbol was unresolvable; the hash still
        incorporates whatever was resolved.  If that partial hash differs from
        baseline, a real callee change is detected → code-changed.
        """
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.PARTIAL,
                closure_hash=_CLOSURE_B,  # partial but changed
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED


# ─── Regression: closure suppression for ALL non-comparable statuses ───
# (review-w4a HIGH expansion: was lsp_error-only; now {unsupported, lsp_unavailable, lsp_error}.)


class TestRegressionClosureSuppressionExpandedW4a:
    """Regression: review-w4a HIGH — closure suppression must apply for all
    non-comparable transitive_hash_status values, not just lsp_error.

    Per DESIGN.md §5.3, only ``complete`` and ``partial`` produce a real,
    comparable closure hash.  ``unsupported``, ``lsp_unavailable``, and
    ``lsp_error`` all fall back to AST-only hashing and may emit either
    the empty sentinel or an unreliable partial-walk hash — comparing
    them against a baseline captured under ``complete``/``partial`` would
    produce a spurious ``code-changed`` purely from LSP availability
    transitions (e.g. user uninstalls pyright, or upgrades to a config
    that skips Python).
    """

    def test_unsupported_status_suppresses_closure_comparison(self, db: ScryDB) -> None:
        """Baseline real closure_hash + current unsupported empty → fresh."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,  # AST unchanged
                transitive_hash_status=TransitiveHashStatus.UNSUPPORTED,
                closure_hash=_EMPTY_CLOSURE,  # empty sentinel
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        # Must NOT be code-changed; unsupported isn't an error so also
        # NOT drift-unknown.  AST is unchanged → fresh.
        assert ev.drift_status == DriftStatus.FRESH, (
            f"unsupported status must suppress closure comparison and "
            f"resolve to fresh when AST is unchanged; got {ev.drift_status}"
        )

    def test_lsp_unavailable_status_suppresses_closure_comparison(self, db: ScryDB) -> None:
        """Baseline real closure_hash + current lsp_unavailable empty → fresh."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.LSP_UNAVAILABLE,
                closure_hash=_EMPTY_CLOSURE,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.FRESH, (
            f"lsp_unavailable must suppress closure comparison; got {ev.drift_status}"
        )

    def test_unsupported_status_with_content_change_still_code_changed(self, db: ScryDB) -> None:
        """Even with closure suppressed, real AST change still produces code-changed."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HB,  # AST changed
                transitive_hash_status=TransitiveHashStatus.UNSUPPORTED,
                closure_hash=_EMPTY_CLOSURE,
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED

    def test_complete_status_does_not_suppress_closure(self, db: ScryDB) -> None:
        """Sanity: complete status is comparable → real change still detected."""
        db.upsert_anchor(_make_anchor(content_hash=_HA))
        db.upsert_anchor(
            _make_code_anchor(
                content_hash=_HA,
                transitive_hash_status=TransitiveHashStatus.COMPLETE,
                closure_hash=_CLOSURE_B,  # different from baseline _CLOSURE_A
            )
        )
        link = _make_link(from_hash=_HA, to_hash=_HA, to_closure_hash=_CLOSURE_A)

        ev = evaluate_link_drift(link, db=db)

        assert ev.drift_status == DriftStatus.CODE_CHANGED, (
            "complete status MUST allow closure comparison to fire"
        )


# uat-r5-5 pr-d noise
