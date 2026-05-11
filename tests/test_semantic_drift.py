"""Tests for W4b — semantic_drift cosine flag on mirrors links.

Covers DESIGN.md §5.1 v3.1:
  - ``cosine_similarity`` math correctness (parallel / orthogonal / opposite).
  - ``evaluate_semantic_drift`` same-content → False, diverged → True.
  - Cross-language regression: SECTION→CODE links (doc→python, doc→typescript)
    correctly return a bool (not null) because SECTION anchors are
    language-neutral and share the same embedding space.
  - Cross-language CODE↔CODE (python→typescript) emits ``None`` + warning
    when ``cross_language_threshold`` is unconfigured; uses the custom
    threshold when it is set.
  - No-embedding anchor → ``None`` (graceful, not an exception).
  - Zero-norm embedding → conservative ``True`` (treated as fully diverged).
  - Threshold boundary: exactly at threshold → False; just above → True.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import pytest

from scry.drift import (
    _infer_language,  # type: ignore[attr-defined]
    cosine_similarity,
    evaluate_semantic_drift,
)
from scry.models import (
    Anchor,
    AnchorType,
    DriftConfig,
    Link,
    LinkType,
    TransitiveHashStatus,
    new_event_id,
    new_link_id,
)
from scry.store.db import ScryDB

# ─── Constants ─────────────────────────────────────────────────────────────────

_HA = "sha256:" + "a" * 64
_DIMS = 4


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def db(tmp_path: Path) -> ScryDB:
    """A fresh ScryDB with schema initialised."""
    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir()
    d = ScryDB(tmp_path)
    d.init_schema(embedding_dimensions=_DIMS)
    return d


# ─── Builders ──────────────────────────────────────────────────────────────────


def _make_anchor(
    anchor_id: str,
    *,
    anchor_type: AnchorType = AnchorType.SECTION,
    path: str = "docs/spec.md",
    transitive_hash_status: TransitiveHashStatus | None = None,
) -> Anchor:
    return Anchor(
        id=anchor_id,
        type=anchor_type,
        path=path,
        content_text="content",
        content_hash=_HA,
        fingerprint_simhash=0xDEAD,
        transitive_hash_status=transitive_hash_status,
    )


def _make_code_anchor(anchor_id: str, *, path: str = "src/lib.py") -> Anchor:
    return _make_anchor(
        anchor_id,
        anchor_type=AnchorType.CODE,
        path=path,
        transitive_hash_status=TransitiveHashStatus.LSP_UNAVAILABLE,
    )


def _make_mirrors_link(from_id: str, to_id: str) -> Link:
    return Link(
        link_id=new_link_id(),
        from_id=from_id,
        from_type=AnchorType.SECTION,
        to_id=to_id,
        to_type=AnchorType.CODE,
        type=LinkType.MIRRORS,
        from_content_hash=_HA,
        to_content_hash=_HA,
        last_event_id=new_event_id(),
    )


def _make_code_to_code_link(
    from_id: str,
    to_id: str,
    *,
    from_type: AnchorType = AnchorType.CODE,
    to_type: AnchorType = AnchorType.CODE,
) -> Link:
    return Link(
        link_id=new_link_id(),
        from_id=from_id,
        from_type=from_type,
        to_id=to_id,
        to_type=to_type,
        type=LinkType.MIRRORS,
        from_content_hash=_HA,
        to_content_hash=_HA,
        last_event_id=new_event_id(),
    )


def _float_blob(*values: float) -> bytes:
    return struct.pack(f"{len(values)}f", *values)


def _insert_emb(db: ScryDB, anchor_id: str, *values: float) -> None:
    db._conn.execute(
        "UPDATE anchors SET overview_embedding = ? WHERE id = ?",
        (_float_blob(*values), anchor_id),
    )
    db._conn.commit()


# ─── Tests: cosine_similarity math ─────────────────────────────────────────────


class TestCosineSimilarityMath:
    """Unit-test the cosine_similarity helper function."""

    def test_parallel_vectors_return_one(self) -> None:
        """Identical direction → similarity = 1.0."""
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_parallel_scaled_vectors_return_one(self) -> None:
        """Scalar-scaled same direction → similarity = 1.0."""
        assert cosine_similarity([2.0, 0.0], [5.0, 0.0]) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self) -> None:
        """Orthogonal unit vectors → similarity = 0.0."""
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors_return_minus_one(self) -> None:
        """Opposite directions → similarity = -1.0."""
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_general_vectors(self) -> None:
        """45° apart → similarity = cos(45°) ≈ 0.707."""
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        expected = 1.0 / math.sqrt(2.0)
        assert cosine_similarity(a, b) == pytest.approx(expected, rel=1e-5)

    def test_zero_vector_a_returns_zero(self) -> None:
        """Zero-norm first vector → 0.0 (no exception)."""
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)

    def test_zero_vector_b_returns_zero(self) -> None:
        """Zero-norm second vector → 0.0 (no exception)."""
        assert cosine_similarity([1.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)

    def test_both_zero_vectors_return_zero(self) -> None:
        """Both vectors zero-norm → 0.0 (no exception)."""
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == pytest.approx(0.0)

    def test_length_mismatch_returns_zero(self) -> None:
        """Vectors of different lengths → 0.0 (not an exception)."""
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(0.0)

    def test_empty_vectors_return_zero(self) -> None:
        """Empty vectors → 0.0."""
        assert cosine_similarity([], []) == pytest.approx(0.0)

    def test_four_dimensional_vectors(self) -> None:
        """Sanity check in the dimension used by most tests (_DIMS=4)."""
        a = [1.0, 0.0, 0.0, 0.0]
        b = [0.0, 0.0, 1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)


# ─── Tests: evaluate_semantic_drift — same vs. diverged ────────────────────────


class TestEvaluateSemanticDriftBasic:
    """Core True/False/None behaviour of evaluate_semantic_drift."""

    def _setup(self, db: ScryDB) -> tuple[Link, str, str]:
        from_id = "docs/spec.md::intro"
        to_id = "src/lib.py:main"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/lib.py"))
        link = _make_mirrors_link(from_id, to_id)
        return link, from_id, to_id

    def test_same_content_embeddings_return_false(self, db: ScryDB) -> None:
        """Identical embeddings → distance=0 → semantic_drift=False."""
        link, from_id, to_id = self._setup(db)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 1.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is False

    def test_diverged_embeddings_return_true(self, db: ScryDB) -> None:
        """Orthogonal unit vectors → distance=1.0 > 0.25 → semantic_drift=True."""
        link, from_id, to_id = self._setup(db)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 1.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is True

    def test_no_from_embedding_returns_none(self, db: ScryDB) -> None:
        """from-anchor embedding NULL → None."""
        link, _, to_id = self._setup(db)
        _insert_emb(db, to_id, 1.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is None

    def test_no_to_embedding_returns_none(self, db: ScryDB) -> None:
        """to-anchor embedding NULL → None."""
        link, from_id, _ = self._setup(db)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is None

    def test_both_embeddings_missing_returns_none(self, db: ScryDB) -> None:
        """Neither embedding present → None."""
        link, _, _ = self._setup(db)

        result = evaluate_semantic_drift(link, db=db)

        assert result is None


# ─── Tests: zero-norm embedding ────────────────────────────────────────────────


class TestZeroNormEmbedding:
    """Zero-magnitude embeddings handled without exceptions (conservative True)."""

    def test_zero_norm_from_embedding_is_conservative(self, db: ScryDB) -> None:
        """from-anchor all-zeros → undefined cosine → conservative True."""
        from_id = "docs/spec.md::intro"
        to_id = "src/lib.py:main"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/lib.py"))
        link = _make_mirrors_link(from_id, to_id)
        _insert_emb(db, from_id, 0.0, 0.0, 0.0, 0.0)  # zero-norm
        _insert_emb(db, to_id, 1.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        # Zero-norm → _cosine_distance returns 1.0 → 1.0 > 0.25 → True
        assert result is True

    def test_zero_norm_to_embedding_is_conservative(self, db: ScryDB) -> None:
        """to-anchor all-zeros → undefined cosine → conservative True."""
        from_id = "docs/spec.md::intro"
        to_id = "src/lib.py:main"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/lib.py"))
        link = _make_mirrors_link(from_id, to_id)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 0.0, 0.0, 0.0)  # zero-norm

        result = evaluate_semantic_drift(link, db=db)

        assert result is True

    def test_both_zero_norm_embeddings_is_conservative(self, db: ScryDB) -> None:
        """Both all-zeros → undefined cosine → conservative True."""
        from_id = "docs/spec.md::intro"
        to_id = "src/lib.py:main"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/lib.py"))
        link = _make_mirrors_link(from_id, to_id)
        _insert_emb(db, from_id, 0.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is True


# ─── Tests: threshold boundary ─────────────────────────────────────────────────


class TestThresholdBoundary:
    """Strictly-greater-than semantics for the threshold gate."""

    def _setup(self, db: ScryDB) -> tuple[Link, str, str]:
        from_id = "docs/spec.md::algo"
        to_id = "src/lib.py:algo"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/lib.py"))
        link = _make_mirrors_link(from_id, to_id)
        return link, from_id, to_id

    def test_exactly_at_threshold_returns_false(self, db: ScryDB) -> None:
        """Distance == threshold → NOT strictly greater → False."""
        link, from_id, to_id = self._setup(db)
        threshold = DriftConfig().semantic_drift_threshold  # default 0.25
        # Build b so that cosine_similarity(a, b) == 1 - threshold,
        # i.e. cosine distance == threshold.
        cos_sim = 1.0 - threshold
        sin_val = math.sqrt(max(0.0, 1.0 - cos_sim**2))
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, float(cos_sim), float(sin_val), 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is False

    def test_just_above_threshold_returns_true(self, db: ScryDB) -> None:
        """Distance slightly > threshold → True."""
        link, from_id, to_id = self._setup(db)
        threshold = DriftConfig().semantic_drift_threshold  # 0.25
        # Slightly above threshold: distance = threshold + epsilon
        epsilon = 0.01
        cos_sim = 1.0 - (threshold + epsilon)
        sin_val = math.sqrt(max(0.0, 1.0 - cos_sim**2))
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, float(cos_sim), float(sin_val), 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is True

    def test_just_below_threshold_returns_false(self, db: ScryDB) -> None:
        """Distance slightly < threshold → False."""
        link, from_id, to_id = self._setup(db)
        threshold = DriftConfig().semantic_drift_threshold  # 0.25
        epsilon = 0.01
        cos_sim = 1.0 - (threshold - epsilon)
        sin_val = math.sqrt(max(0.0, 1.0 - cos_sim**2))
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, float(cos_sim), float(sin_val), 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is False

    def test_custom_threshold_respected(self, db: ScryDB) -> None:
        """Custom semantic_drift_threshold overrides default 0.25."""
        link, from_id, to_id = self._setup(db)
        # Use threshold=0.5; set distance=0.4 (below 0.5 → should be False).
        custom_config = DriftConfig(semantic_drift_threshold=0.5)
        cos_sim = 1.0 - 0.4
        sin_val = math.sqrt(max(0.0, 1.0 - cos_sim**2))
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, float(cos_sim), float(sin_val), 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db, config=custom_config)

        # distance=0.4 < threshold=0.5 → False
        assert result is False


# ─── Tests: cross-language — doc→code links (regression) ──────────────────────


class TestCrossLanguageDocToCode:
    """SECTION→CODE mirrors links evaluate correctly for any target language.

    This is the key W4b regression: all anchor types share one embedding space
    (DESIGN.md §3.1), so a language-neutral doc anchor linked to Python or
    TypeScript code should produce a bool result, never null.
    """

    def test_doc_to_python_returns_bool(self, db: ScryDB) -> None:
        """SECTION (doc) → CODE (.py) mirrors link → bool result (not None)."""
        from_id = "docs/spec.md::auth"
        to_id = "src/auth.py:authenticate"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/auth.py"))
        link = _make_mirrors_link(from_id, to_id)
        # Divergent embeddings to force True.
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 1.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        # Must be a bool, not None — SECTION is language-neutral.
        assert isinstance(result, bool)
        assert result is True

    def test_doc_to_python_fresh_returns_false(self, db: ScryDB) -> None:
        """SECTION→Python with identical embeddings → False (not None)."""
        from_id = "docs/spec.md::auth"
        to_id = "src/auth.py:authenticate"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/auth.py"))
        link = _make_mirrors_link(from_id, to_id)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 1.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is False

    def test_doc_to_typescript_returns_bool(self, db: ScryDB) -> None:
        """SECTION (doc) → CODE (.ts) mirrors link → bool result (not None)."""
        from_id = "docs/spec.md::ui"
        to_id = "src/ui.ts:render"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/ui.ts"))
        link = _make_mirrors_link(from_id, to_id)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 1.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert isinstance(result, bool)
        assert result is True

    def test_doc_to_typescript_fresh_returns_false(self, db: ScryDB) -> None:
        """SECTION→TypeScript with identical embeddings → False."""
        from_id = "docs/spec.md::ui"
        to_id = "src/ui.ts:render"
        db.upsert_anchor(_make_anchor(from_id, path="docs/spec.md"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/ui.ts"))
        link = _make_mirrors_link(from_id, to_id)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 1.0, 0.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is False


# ─── Tests: cross-language CODE↔CODE pairs ─────────────────────────────────────


class TestCrossLanguageCodeToCode:
    """CODE↔CODE mirrors links in different languages gate correctly (§5.1 v3.1)."""

    def _setup_py_to_ts(self, db: ScryDB) -> tuple[Link, str, str]:
        from_id = "src/auth.py:hash_password"
        to_id = "src/auth.ts:hashPassword"
        db.upsert_anchor(_make_code_anchor(from_id, path="src/auth.py"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/auth.ts"))
        link = _make_code_to_code_link(from_id, to_id)
        return link, from_id, to_id

    def test_python_to_typescript_without_threshold_returns_none(
        self, db: ScryDB, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Python→TypeScript CODE↔CODE without cross_language_threshold → None + warning."""
        link, from_id, to_id = self._setup_py_to_ts(db)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 1.0, 0.0, 0.0)

        import logging

        with caplog.at_level(logging.WARNING, logger="scry.drift"):
            result = evaluate_semantic_drift(link, db=db)

        assert result is None
        assert "cross-language" in caplog.text.lower()

    def test_python_to_typescript_with_threshold_returns_bool(self, db: ScryDB) -> None:
        """Python→TypeScript with cross_language_threshold set → uses that threshold."""
        link, from_id, to_id = self._setup_py_to_ts(db)
        # Divergent embeddings (orthogonal → distance=1.0).
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 1.0, 0.0, 0.0)
        config = DriftConfig(cross_language_threshold=0.5)

        result = evaluate_semantic_drift(link, db=db, config=config)

        # distance=1.0 > cross_language_threshold=0.5 → True
        assert result is True

    def test_python_to_typescript_with_threshold_fresh_returns_false(self, db: ScryDB) -> None:
        """Python→TypeScript with configured threshold, identical embeddings → False."""
        link, from_id, to_id = self._setup_py_to_ts(db)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 1.0, 0.0, 0.0, 0.0)
        config = DriftConfig(cross_language_threshold=0.5)

        result = evaluate_semantic_drift(link, db=db, config=config)

        # distance=0.0 < 0.5 → False
        assert result is False

    def test_same_language_code_to_code_uses_default_threshold(self, db: ScryDB) -> None:
        """Python→Python CODE link is NOT cross-language → uses semantic_drift_threshold."""
        from_id = "src/auth.py:hash_password"
        to_id = "src/auth.py:verify_password"
        db.upsert_anchor(_make_code_anchor(from_id, path="src/auth.py"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/auth.py"))
        link = _make_code_to_code_link(from_id, to_id)
        # Orthogonal → diverged.
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 1.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        # Same language → default threshold applies; distance=1.0 > 0.25 → True.
        assert result is True

    def test_go_to_rust_without_threshold_returns_none(self, db: ScryDB) -> None:
        """Go→Rust CODE↔CODE without cross_language_threshold → None."""
        from_id = "pkg/auth.go:HashPassword"
        to_id = "src/auth.rs:hash_password"
        db.upsert_anchor(_make_code_anchor(from_id, path="pkg/auth.go"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/auth.rs"))
        link = _make_code_to_code_link(from_id, to_id)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 1.0, 0.0, 0.0)

        result = evaluate_semantic_drift(link, db=db)

        assert result is None

    def test_unknown_extension_pair_uses_default_threshold(self, db: ScryDB) -> None:
        """CODE anchors with unrecognised extensions → language unknown → default threshold."""
        from_id = "src/auth.scry:hash"
        to_id = "src/verify.scry:verify"
        db.upsert_anchor(_make_code_anchor(from_id, path="src/auth.scry"))
        db.upsert_anchor(_make_code_anchor(to_id, path="src/verify.scry"))
        link = _make_code_to_code_link(from_id, to_id)
        _insert_emb(db, from_id, 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, to_id, 0.0, 1.0, 0.0, 0.0)

        # Unknown extensions → _infer_language returns None → no cross-lang gate.
        result = evaluate_semantic_drift(link, db=db)

        assert result is True  # distance=1.0 > 0.25


# ─── Tests: _infer_language helper ─────────────────────────────────────────────


class TestInferLanguage:
    """Unit-test the private _infer_language helper used in cross-lang detection."""

    def test_python(self) -> None:
        assert _infer_language("src/lib.py") == "python"

    def test_typescript(self) -> None:
        assert _infer_language("src/ui.ts") == "typescript"

    def test_tsx(self) -> None:
        assert _infer_language("src/App.tsx") == "typescript"

    def test_javascript(self) -> None:
        assert _infer_language("src/app.js") == "javascript"

    def test_jsx(self) -> None:
        assert _infer_language("src/App.jsx") == "javascript"

    def test_go(self) -> None:
        assert _infer_language("pkg/auth.go") == "go"

    def test_rust(self) -> None:
        assert _infer_language("src/auth.rs") == "rust"

    def test_java(self) -> None:
        assert _infer_language("src/Main.java") == "java"

    def test_markdown_returns_none(self) -> None:
        """Markdown is language-neutral → None."""
        assert _infer_language("docs/spec.md") is None

    def test_no_extension_returns_none(self) -> None:
        assert _infer_language("Makefile") is None

    def test_unknown_extension_returns_none(self) -> None:
        assert _infer_language("src/lib.scry") is None

    def test_case_insensitive(self) -> None:
        """Extension matching is case-insensitive."""
        assert _infer_language("src/Lib.PY") == "python"
        assert _infer_language("src/ui.TS") == "typescript"


# ─── Regression: config plumbed through evaluate_link_drift (review-w4b BLOCKING) ──


class TestRegressionConfigPlumbingW4b:
    """Regression: review-w4b BLOCKING — drift config thresholds must reach
    ``evaluate_semantic_drift`` from the public ``evaluate_link_drift`` entry
    point.  Without this, CLI/MCP callers always got default thresholds.
    """

    def test_evaluate_link_drift_passes_config_to_semantic(self, db: ScryDB) -> None:
        """Calling ``evaluate_link_drift(..., config=cfg)`` honours the
        explicit cross_language_threshold for code↔code mirrors links.
        """
        from scry.drift import evaluate_link_drift

        # Two code anchors in different languages with diverged embeddings.
        from_a = _make_code_anchor("py-anchor", path="src/main.py")
        to_a = _make_code_anchor("ts-anchor", path="src/main.ts")
        db.upsert_anchor(from_a)
        db.upsert_anchor(to_a)
        # Diverged embeddings (cosine distance ≈ 1.0)
        _insert_emb(db, "py-anchor", 1.0, 0.0, 0.0, 0.0)
        _insert_emb(db, "ts-anchor", 0.0, 1.0, 0.0, 0.0)
        link = _make_code_to_code_link("py-anchor", "ts-anchor")

        # 1) WITHOUT explicit cross_language_threshold → semantic_drift = None
        cfg_no = DriftConfig(cross_language_threshold=None)
        eval_no = evaluate_link_drift(link, db=db, config=cfg_no)
        assert eval_no.semantic_drift is None, (
            "without cross_language_threshold cross-language code↔code must be null"
        )

        # 2) WITH explicit cross_language_threshold → semantic_drift evaluated
        cfg_yes = DriftConfig(cross_language_threshold=0.25)
        eval_yes = evaluate_link_drift(link, db=db, config=cfg_yes)
        assert eval_yes.semantic_drift is True, (
            "with cross_language_threshold the cross-language code↔code "
            "evaluation must run; orthogonal vectors → distance 1.0 > 0.25"
        )

    def test_evaluate_all_drift_signature_accepts_config(self) -> None:
        """``evaluate_all_drift`` MUST accept a ``config`` kwarg.

        Functional plumbing of config -> link evaluation is covered by
        ``test_evaluate_link_drift_passes_config_to_semantic`` since
        ``evaluate_all_drift`` is a thin wrapper that just forwards
        each link to ``evaluate_link_drift(..., config=config)``.
        """
        import inspect

        from scry.drift import evaluate_all_drift

        sig = inspect.signature(evaluate_all_drift)
        assert "config" in sig.parameters, (
            "evaluate_all_drift must accept a config kwarg so user "
            "drift thresholds reach link evaluation"
        )
        # Default must be optional (None) for backward compat.
        assert sig.parameters["config"].default is None


# ─── Regression: Zig in language inference (review-w4b HIGH) ──────────


class TestRegressionZigLanguageInferenceW4b:
    """Regression: review-w4b HIGH — Zig was missing from _EXT_TO_LANGUAGE,
    causing supported zig anchors to be treated as language-unknown.
    """

    def test_zig_extension_inferred(self) -> None:
        assert _infer_language("src/main.zig") == "zig"

    def test_zon_extension_inferred(self) -> None:
        assert _infer_language("build.zon") == "zig"

    def test_pyi_extension_inferred(self) -> None:
        """Python interface stubs also recognized."""
        assert _infer_language("foo.pyi") == "python"

    def test_typescript_variants_inferred(self) -> None:
        for ext in ("mts", "cts"):
            assert _infer_language(f"src/x.{ext}") == "typescript", (
                f".{ext} should resolve to typescript"
            )

    def test_javascript_variants_inferred(self) -> None:
        for ext in ("mjs", "cjs"):
            assert _infer_language(f"src/x.{ext}") == "javascript", (
                f".{ext} should resolve to javascript"
            )

# uat-r5-5 pr-d noise
