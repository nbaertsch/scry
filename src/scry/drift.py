"""Section-level drift detection for scry.

Implements DESIGN.md §5.1 (drift signals, post-v3.1 with prior-hash override
and ``semantic_drift`` as a separate boolean flag) and §5.2 (drift score +
coverage score with null semantics) for Wave 2.

Wave 2 scope and contracts:
  - Section-level drift only. CODE-typed anchors use own ``content_hash``
    comparison; transitive LSP closure drift (§5.3) is produced by W3d.
  - ``drift-unknown`` is produced (W3d) when a CODE endpoint has
    ``transitive_hash_status == "lsp_error"`` (DESIGN.md §5.1).
  - ``drift_coverage`` is always ``"section-only"`` for Wave 2 outputs.
  - Cross-language CODE↔CODE ``semantic_drift`` detection implemented in W4b:
    emits ``None`` + warning when ``cross_language_threshold`` is not set.
"""

from __future__ import annotations

import logging
import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass

from scry.models import (
    AnchorType,
    ContentHash,
    DriftConfig,
    DriftCounts,
    DriftStatus,
    DriftSummary,
    Link,
    LinkId,
    LinkType,
    TransitiveHashStatus,
)
from scry.store.db import ScryDB
from scry.store.links import LinkStore

__all__ = [
    "DriftDetectionError",
    "DriftEvaluation",
    "compute_drift_summary",
    "cosine_similarity",
    "evaluate_all_drift",
    "evaluate_link_drift",
    "evaluate_semantic_drift",
]

logger = logging.getLogger(__name__)


class DriftDetectionError(Exception):
    """Raised for unrecoverable errors during drift detection."""


@dataclass(frozen=True)
class DriftEvaluation:
    """Per-link drift result — one row in the ``evaluate_all_drift`` output.

    ``semantic_drift`` is computed only for ``mirrors`` links where both
    endpoints have overview embeddings available. For cross-language pairs or
    missing embeddings it is ``None``.
    """

    link: Link
    drift_status: DriftStatus
    semantic_drift: bool | None
    """Only set on ``mirrors`` links; None for cross-language pairs or missing embeddings."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_overview_embedding(db: ScryDB, anchor_id: str) -> list[float] | None:
    """Return the ``overview_embedding`` float vector for an anchor row.

    ``ScryDB.get_anchor()`` does not expose ``overview_embedding``; we query
    the underlying connection directly — the DB file belongs exclusively to
    this process and no other module interposes on this column.

    Returns ``None`` if the anchor row is absent or its embedding is NULL.
    """
    row = db._conn.execute(
        "SELECT overview_embedding FROM anchors WHERE id = ?",
        (anchor_id,),
    ).fetchone()
    if row is None or row[0] is None:
        return None
    blob: bytes = row[0]
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """Compute cosine distance (1 - similarity) between two vectors.

    Returns ``1.0`` (maximum distance) if either vector has zero magnitude or
    the lengths differ, so undefined similarity is treated as full divergence.
    """
    if len(a) != len(b) or not a:
        return 1.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 1.0
    return 1.0 - dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Cross-language detection helpers (§5.1 v3.1 W4b)
# ---------------------------------------------------------------------------

# File-extension → normalised language tag.  Only extensions for languages
# commonly handled by tree-sitter + LSP adapters are listed; unlisted
# extensions return None (treated as "language unknown" — no cross-language
# gate is applied).
_EXT_TO_LANGUAGE: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
    ".go": "go",
    ".rs": "rust",
    ".zig": "zig",
    ".zon": "zig",
    ".java": "java",
    ".rb": "ruby",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin",
    ".kts": "kotlin",
}


def _infer_language(path: str) -> str | None:
    """Return a normalised language tag for *path*'s file extension.

    Returns ``None`` for markdown, plain-text, or any extension not in the
    known map — these are treated as language-neutral and never trigger the
    cross-language gate in :func:`evaluate_semantic_drift`.

    Args:
        path: Repo-relative file path (e.g. ``"src/lib.py"``).

    Returns:
        A lowercase language tag (e.g. ``"python"``, ``"typescript"``), or
        ``None`` when the extension is unknown.
    """
    dot = path.rfind(".")
    if dot == -1:
        return None
    return _EXT_TO_LANGUAGE.get(path[dot:].lower())


def _resolve_changed(
    link: Link,
    *,
    current_from_hash: ContentHash,
    current_to_hash: ContentHash,
    current_from_closure_hash: str | None = None,
    current_to_closure_hash: str | None = None,
    from_closure_comparable: bool = True,
    to_closure_comparable: bool = True,
) -> tuple[bool, bool]:
    """Return ``(from_changed, to_changed)`` honouring the prior-hash override (§3.3).

    After an inline rebase, ``from_content_hash`` / ``to_content_hash`` are
    updated to the post-rename hashes. If ``prior_from_content_hash`` is set,
    we compare ``current_from_hash`` against the pre-rename hash, so a
    rename-plus-edit is correctly seen as changed rather than fresh.

    For CODE anchors, the transitive closure hash is also compared (W3d §5.3):
    if a callee's body changed, ``closure_hash`` differs even when the caller's
    own AST is unchanged.  Backward compat: a ``None`` baseline closure hash
    means "no closure comparison available" and never causes a false positive.

    When ``from_closure_comparable`` (or ``to_closure_comparable``) is
    ``False``, the closure-hash comparison for that endpoint is suppressed.
    Per DESIGN.md §5.3, only the ``complete`` and ``partial`` statuses produce
    a closure hash that is real and comparable; ``unsupported``,
    ``lsp_unavailable``, and ``lsp_error`` all fall back to AST-only hashing
    and emit either ``_EMPTY_SHA256`` or an unreliable partial-walk hash.
    Comparing those values against a baseline that was captured under
    ``complete``/``partial`` would produce a spurious ``code-changed`` purely
    from LSP availability transitions (review-w4a HIGH fix expanding the
    original lsp_error-only suppression).  §5.1 v3.1 mandates that the
    *content* hash (own AST) is still compared normally; only the
    closure-derived signal is skipped.  The caller's ``drift-unknown`` check
    (step 8 in ``evaluate_link_drift``) then surfaces the uncertainty when
    the trigger was specifically ``lsp_error``.
    """
    from_ref: ContentHash = link.prior_from_content_hash or link.from_content_hash
    to_ref: ContentHash = link.prior_to_content_hash or link.to_content_hash

    from_content_changed = current_from_hash != from_ref
    to_content_changed = current_to_hash != to_ref

    # Closure hash comparison fires only when:
    #   1. Both baseline and current closure hashes are available (None
    #      baseline never false-positives — backward compat for old links)
    #   2. The current anchor's transitive_hash_status indicates a real
    #      closure value (``complete`` or ``partial`` per §5.3); ``unsupported``,
    #      ``lsp_unavailable``, and ``lsp_error`` all produce unreliable
    #      hashes that must not be compared (review-w4a HIGH fix).
    from_closure_changed = (
        from_closure_comparable
        and link.from_closure_hash is not None
        and current_from_closure_hash is not None
        and current_from_closure_hash != link.from_closure_hash
    )
    to_closure_changed = (
        to_closure_comparable
        and link.to_closure_hash is not None
        and current_to_closure_hash is not None
        and current_to_closure_hash != link.to_closure_hash
    )

    return (from_content_changed or from_closure_changed), (
        to_content_changed or to_closure_changed
    )


def _changed_to_status(
    from_changed: bool,
    to_changed: bool,
    from_type: AnchorType,
    to_type: AnchorType,
) -> DriftStatus:
    """Map ``(from_changed, to_changed)`` to a ``DriftStatus`` for live endpoints.

    Wave 2 endpoint-type mapping (§5.1 functional spec):
      - ``SECTION`` endpoint changed → ``spec-changed``
      - ``CODE`` or ``CODE_IN_DOC`` endpoint changed → ``code-changed``
      - Both endpoints changed → ``both-changed`` (regardless of types)
      - Neither changed → ``fresh``
    """
    if from_changed and to_changed:
        return DriftStatus.BOTH_CHANGED
    if from_changed:
        return (
            DriftStatus.SPEC_CHANGED
            if from_type == AnchorType.SECTION
            else DriftStatus.CODE_CHANGED
        )
    if to_changed:
        return (
            DriftStatus.SPEC_CHANGED if to_type == AnchorType.SECTION else DriftStatus.CODE_CHANGED
        )
    return DriftStatus.FRESH


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Compute cosine similarity between two float vectors.

    Returns a value in ``[-1.0, 1.0]``:
        - ``1.0``  — identical direction (parallel vectors)
        - ``0.0``  — orthogonal
        - ``-1.0`` — opposite directions

    Returns ``0.0`` when either vector has zero magnitude or the vectors
    have different lengths, so the degenerate cases never raise an exception.

    This function operates on plain Python sequences; see
    :func:`scry.embed.cosine_similarity` for the serialised-blob variant
    used in retrieval scoring.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in ``[-1.0, 1.0]``.
    """
    if len(a) != len(b) or not a:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


def evaluate_semantic_drift(
    link: Link,
    *,
    db: ScryDB,
    config: DriftConfig | None = None,
) -> bool | None:
    """Compute the ``semantic_drift`` flag for a ``mirrors`` link (§5.1 v3.1).

    Compares the **current** embeddings of both endpoint anchors in the shared
    vector space (DESIGN.md §3.1 — all three anchor types share one embedding
    space, so doc-to-code ``mirrors`` links evaluate correctly without special
    treatment).

    **Cross-language CODE↔CODE pairs** (W4b): when both endpoints are ``CODE``
    anchors whose file extensions resolve to different programming languages,
    the 0.25 threshold is not calibrated for the cross-language embedding
    distance distribution (§5.1 v3.1).  Scry handles this case as follows:

    * If ``config.cross_language_threshold`` is ``None`` (default): return
      ``None`` and emit a ``WARNING``-level log so ``scry doctor`` can surface
      it.  The flag will appear as ``null`` in the anchor packet.
    * If ``config.cross_language_threshold`` is set: use that threshold
      instead of the default, enabling cross-language drift checking for
      users who have calibrated their threshold.

    ``SECTION`` and ``CODE_IN_DOC`` anchors are language-neutral (they carry
    natural-language text rather than source code), so a ``SECTION``→``CODE``
    ``mirrors`` link is never treated as cross-language.

    Returns:
        ``True``  — cosine distance exceeds the applicable threshold.
        ``False`` — within threshold.
        ``None``  — one or both embeddings are unavailable (missing anchor or
                    NULL in DB), OR a cross-language CODE↔CODE pair without a
                    configured ``cross_language_threshold``.
    """
    if config is None:
        config = DriftConfig()

    from_emb = _get_overview_embedding(db, link.from_id)
    to_emb = _get_overview_embedding(db, link.to_id)
    if from_emb is None or to_emb is None:
        return None

    # Cross-language CODE↔CODE detection (§5.1 v3.1 W4b).
    # Fetch anchors to check endpoint types; SECTION / CODE_IN_DOC are
    # language-neutral and never trigger this gate.
    threshold = config.semantic_drift_threshold
    from_anchor = db.get_anchor(link.from_id)
    to_anchor = db.get_anchor(link.to_id)
    if (
        from_anchor is not None
        and to_anchor is not None
        and from_anchor.type == AnchorType.CODE.value
        and to_anchor.type == AnchorType.CODE.value
    ):
        from_lang = _infer_language(from_anchor.path)
        to_lang = _infer_language(to_anchor.path)
        if from_lang is not None and to_lang is not None and from_lang != to_lang:
            if config.cross_language_threshold is None:
                logger.warning(
                    "semantic_drift: cross-language mirrors pair (%s [%s] ↔ %s [%s]); "
                    "threshold is not calibrated for cross-language pairs — emitting "
                    "semantic_drift=null. Configure drift.cross_language_threshold to "
                    "enable cross-language semantic drift checking.",
                    link.from_id,
                    from_lang,
                    link.to_id,
                    to_lang,
                )
                return None
            threshold = config.cross_language_threshold

    dist = _cosine_distance(from_emb, to_emb)
    return dist > threshold


def evaluate_link_drift(
    link: Link,
    *,
    db: ScryDB,
    merge_conflicts: set[LinkId] | None = None,
    config: DriftConfig | None = None,
) -> DriftEvaluation:
    """Compute the drift status for a single link (DESIGN.md §5.1, post-v3.1).

    Precedence ladder (highest wins):
        1. ``merge-conflict``  — ``link.link_id`` in ``merge_conflicts``.
        2. ``broken-source``   — ``db.get_anchor(link.from_id)`` is None.
        3. ``broken-target``   — ``db.get_anchor(link.to_id)`` is None.
        4. Prior-hash override (§3.3): if ``prior_from/to_content_hash`` is
           set on the link and the current hash differs from the prior hash,
           treat the endpoint as changed. Both prior fields set and both
           endpoints changed → ``both-changed``.
        5. ``both-changed``    — both endpoints' hashes differ.
        6. ``spec-changed``    — spec/doc endpoint (SECTION or CODE_IN_DOC) hash changed.
        7. ``code-changed``    — code endpoint (CODE) hash changed.
        8. ``drift-unknown``   — either CODE endpoint has ``transitive_hash_status
                                 == "lsp_error"`` (DESIGN.md §5.1 / W3d).
        9. ``fresh``           — both endpoints' hashes match.

    ``semantic_drift`` is computed for ``mirrors`` links when both endpoints
    have ``overview_embedding`` stored in the DB; ``None`` otherwise.

    The optional *config* threads ``DriftConfig`` (e.g.
    ``semantic_drift_threshold``, ``cross_language_threshold``) through to
    ``evaluate_semantic_drift`` so user overrides actually apply
    (review-w4b BLOCKING fix).  When omitted, defaults from
    :class:`~scry.models.DriftConfig` are used.
    """
    if merge_conflicts is None:
        merge_conflicts = set()

    # UAT-R5-5 PR-C simulation: injected comment to trigger code_changed drift.
    # Step 1 — merge-conflict (highest precedence).
    if link.link_id in merge_conflicts:
        return DriftEvaluation(
            link=link,
            drift_status=DriftStatus.MERGE_CONFLICT,
            semantic_drift=None,
        )

    # Step 2 — broken-source.
    from_anchor = db.get_anchor(link.from_id)
    if from_anchor is None:
        return DriftEvaluation(
            link=link,
            drift_status=DriftStatus.BROKEN_SOURCE,
            semantic_drift=None,
        )

    # Step 3 — broken-target.
    to_anchor = db.get_anchor(link.to_id)
    if to_anchor is None:
        return DriftEvaluation(
            link=link,
            drift_status=DriftStatus.BROKEN_TARGET,
            semantic_drift=None,
        )

    # Steps 4-7 -- hash comparison with prior-hash override (§3.3).
    from_type = AnchorType(from_anchor.type)
    to_type = AnchorType(to_anchor.type)

    # Pre-compute per-endpoint flags driving §5.1 v3.1 logic:
    #   * ``*_closure_comparable``: True only when the current
    #     ``transitive_hash_status`` is one of {complete, partial} (§5.3 —
    #     these are the statuses that produce a real, comparable closure
    #     hash).  ``unsupported``, ``lsp_unavailable``, and ``lsp_error``
    #     all emit either the empty sentinel or an unreliable partial-walk
    #     hash, which must not be compared against a baseline captured
    #     under a different status (review-w4a HIGH expansion).
    #   * ``*_lsp_error``: True when the endpoint's transitive status is
    #     specifically ``lsp_error`` — used to escalate fresh→drift-unknown
    #     in step 8.
    # Non-CODE anchors don't carry transitive_hash_status; they are
    # treated as comparable (no closure to suppress) and never trigger
    # drift-unknown.
    _COMPARABLE_STATUSES = {
        TransitiveHashStatus.COMPLETE,
        TransitiveHashStatus.PARTIAL,
    }
    from_closure_comparable = (
        from_type != AnchorType.CODE or from_anchor.transitive_hash_status in _COMPARABLE_STATUSES
    )
    to_closure_comparable = (
        to_type != AnchorType.CODE or to_anchor.transitive_hash_status in _COMPARABLE_STATUSES
    )
    from_lsp_error = (
        from_type == AnchorType.CODE
        and from_anchor.transitive_hash_status == TransitiveHashStatus.LSP_ERROR
    )
    to_lsp_error = (
        to_type == AnchorType.CODE
        and to_anchor.transitive_hash_status == TransitiveHashStatus.LSP_ERROR
    )

    from_changed, to_changed = _resolve_changed(
        link,
        current_from_hash=from_anchor.content_hash,
        current_to_hash=to_anchor.content_hash,
        current_from_closure_hash=from_anchor.closure_hash,
        current_to_closure_hash=to_anchor.closure_hash,
        from_closure_comparable=from_closure_comparable,
        to_closure_comparable=to_closure_comparable,
    )
    status = _changed_to_status(from_changed, to_changed, from_type, to_type)

    # Step 8 — drift-unknown when an LSP error prevented reliable closure
    # computation on a CODE endpoint (DESIGN.md §5.1 / §5.3).  Only applies
    # when neither endpoint has already triggered a change status (steps 5-7),
    # because those signals are more actionable (§5.1 precedence:
    # code-changed > drift-unknown).
    if status == DriftStatus.FRESH and (from_lsp_error or to_lsp_error):
        status = DriftStatus.DRIFT_UNKNOWN

    # Semantic drift (independent of status ladder, mirrors links only — §5.1).
    semantic_drift: bool | None = None
    if link.type == LinkType.MIRRORS:
        semantic_drift = evaluate_semantic_drift(link, db=db, config=config)

    return DriftEvaluation(link=link, drift_status=status, semantic_drift=semantic_drift)


def evaluate_all_drift(
    *,
    db: ScryDB,
    link_store: LinkStore,
    config: DriftConfig | None = None,
) -> list[DriftEvaluation]:
    """Replay baseline ⊕ current-branch overlay, then evaluate every active link.

    Calls ``link_store.replay()`` to compute the merged active link table and
    surface merge-conflict link IDs, then delegates each link to
    ``evaluate_link_drift``.

    The optional *config* is forwarded to ``evaluate_link_drift`` so user
    drift thresholds are honoured end-to-end (review-w4b BLOCKING fix).

    Returns a ``DriftEvaluation`` for every link in the active table.
    """
    result = link_store.replay()
    conflicts: set[LinkId] = set(result.merge_conflicts)
    return [
        evaluate_link_drift(link, db=db, merge_conflicts=conflicts, config=config)
        for link in result.active_links.values()
    ]


def compute_drift_summary(
    evaluations: list[DriftEvaluation],
    *,
    config: DriftConfig | None = None,
    coverage_total_code_anchors: int | None = None,
    coverage_linked_code_anchors: int | None = None,
) -> DriftSummary:
    """Compute §5.2 v3.1 drift_score + coverage_score + raw counts.

    drift_score:
        ``100 * (1 - sum(weight_c * count_c) / max(1, total_links))``
        Returns ``None`` when ``total_links == 0`` (per §5.2 v3.1).

    coverage_score:
        ``100 * (linked_code_anchors / max(1, total_code_anchors))``
        Returns ``None`` when ``total_code_anchors == 0``.

    drift_coverage:
        Always ``"section-only"`` for Wave 2 outputs.

    ``semantic_drift_flagged`` is incremented for every ``mirrors`` link where
    ``evaluation.semantic_drift is True``. Per §5.2 v3.1, this adds
    ``config.scoring.semantic_drift`` (default 0.2) on top of the base-status
    weight in the weighted sum — so a ``mirrors/code-changed`` link with
    semantic drift flagged contributes ``0.3 + 0.2 = 0.5`` to the sum.
    """
    if config is None:
        config = DriftConfig()
    scoring = config.scoring

    counts = DriftCounts()
    weighted_sum = 0.0

    for ev in evaluations:
        counts.total += 1
        status = ev.drift_status

        if status == DriftStatus.BROKEN_SOURCE:
            counts.broken_source += 1
            weighted_sum += scoring.broken
        elif status == DriftStatus.BROKEN_TARGET:
            counts.broken_target += 1
            weighted_sum += scoring.broken
        elif status == DriftStatus.MERGE_CONFLICT:
            counts.merge_conflict += 1
            weighted_sum += scoring.merge_conflict
        elif status == DriftStatus.BOTH_CHANGED:
            counts.both_changed += 1
            weighted_sum += scoring.both_changed
        elif status == DriftStatus.SPEC_CHANGED:
            counts.spec_changed += 1
            weighted_sum += scoring.spec_changed
        elif status == DriftStatus.CODE_CHANGED:
            counts.code_changed += 1
            weighted_sum += scoring.code_changed
        elif status == DriftStatus.DRIFT_UNKNOWN:
            counts.drift_unknown += 1
            weighted_sum += scoring.drift_unknown
        elif status == DriftStatus.FRESH:
            counts.fresh += 1
        # (No else needed — StrEnum is exhaustive.)

        # semantic_drift_flagged adds its own weight on top (§5.2 v3.1).
        if ev.semantic_drift is True:
            counts.semantic_drift_flagged += 1
            weighted_sum += scoring.semantic_drift

    # drift_score: null when empty (§5.2 v3.1).
    drift_score: float | None
    if counts.total == 0:
        drift_score = None
    else:
        raw = 100.0 * (1.0 - weighted_sum / max(1, counts.total))
        drift_score = max(0.0, min(100.0, raw))

    # coverage_score: null when no code anchors exist (§5.2 v3.1).
    coverage_score: float | None
    if not coverage_total_code_anchors:
        coverage_score = None
    else:
        linked = coverage_linked_code_anchors or 0
        coverage_score = 100.0 * (linked / max(1, coverage_total_code_anchors))

    return DriftSummary(
        drift_score=drift_score,
        coverage_score=coverage_score,
        counts=counts,
        drift_coverage="section-only",
    )
