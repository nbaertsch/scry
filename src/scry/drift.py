"""Section-level drift detection for scry.

Implements DESIGN.md §5.1 (drift signals, post-v3.1 with prior-hash override
and ``semantic_drift`` as a separate boolean flag) and §5.2 (drift score +
coverage score with null semantics) for Wave 2.

Wave 2 scope and contracts:
  - Section-level drift only. CODE-typed anchors use own ``content_hash``
    comparison; transitive LSP closure drift (§5.3) is NOT included (W4a).
  - ``drift-unknown`` is never produced by Wave 2.
  - ``drift_coverage`` is always ``"section-only"`` for Wave 2 outputs.
  - Cross-language ``semantic_drift`` detection is deferred to Wave 6.
"""

from __future__ import annotations

import logging
import math
import struct
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
)
from scry.store.db import ScryDB
from scry.store.links import LinkStore

__all__ = [
    "DriftDetectionError",
    "DriftEvaluation",
    "compute_drift_summary",
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


def _resolve_changed(
    link: Link,
    *,
    current_from_hash: ContentHash,
    current_to_hash: ContentHash,
) -> tuple[bool, bool]:
    """Return ``(from_changed, to_changed)`` honouring the prior-hash override (§3.3).

    After an inline rebase, ``from_content_hash`` / ``to_content_hash`` are
    updated to the post-rename hashes. If ``prior_from_content_hash`` is set,
    we compare ``current_from_hash`` against the pre-rename hash, so a
    rename-plus-edit is correctly seen as changed rather than fresh.
    """
    from_ref: ContentHash = link.prior_from_content_hash or link.from_content_hash
    to_ref: ContentHash = link.prior_to_content_hash or link.to_content_hash
    return current_from_hash != from_ref, current_to_hash != to_ref


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


def evaluate_semantic_drift(
    link: Link,
    *,
    db: ScryDB,
    config: DriftConfig | None = None,
) -> bool | None:
    """Compute the ``semantic_drift`` flag for a ``mirrors`` link (§5.1 v3.1).

    Returns:
        ``True``  — cosine distance between endpoint embeddings exceeds
                    ``config.semantic_drift_threshold`` (default 0.25).
        ``False`` — within threshold.
        ``None``  — one or both embeddings are unavailable (missing anchor or
                    NULL in DB). Cross-language detection is deferred to Wave 6
                    (TODO W6): the ``Anchor`` model does not expose a language
                    field, so all pairs with available embeddings receive a
                    scalar result regardless of language.
    """
    if config is None:
        config = DriftConfig()

    from_emb = _get_overview_embedding(db, link.from_id)
    to_emb = _get_overview_embedding(db, link.to_id)
    if from_emb is None or to_emb is None:
        return None

    # TODO(W6): detect cross-language mirrors pairs and emit None + warning
    # when endpoint anchor types resolve to different source languages.
    # Currently the Anchor model carries no language field, so we cannot
    # distinguish same-language from cross-language at Wave 2.

    dist = _cosine_distance(from_emb, to_emb)
    return dist > config.semantic_drift_threshold


def evaluate_link_drift(
    link: Link,
    *,
    db: ScryDB,
    merge_conflicts: set[LinkId] | None = None,
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
        8. (``drift-unknown``  — never produced by Wave 2; reserved for W4a.)
        9. ``fresh``           — both endpoints' hashes match.

    Wave 2 does NOT compute transitive closure drift (§5.3) — all CODE
    anchors' ``transitive_hash_status`` values are treated as irrelevant.

    ``semantic_drift`` is computed for ``mirrors`` links when both endpoints
    have ``overview_embedding`` stored in the DB; ``None`` otherwise.
    """
    if merge_conflicts is None:
        merge_conflicts = set()

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

    # Steps 4-9 -- hash comparison with prior-hash override (§3.3).
    from_type = AnchorType(from_anchor.type)
    to_type = AnchorType(to_anchor.type)
    from_changed, to_changed = _resolve_changed(
        link,
        current_from_hash=from_anchor.content_hash,
        current_to_hash=to_anchor.content_hash,
    )
    status = _changed_to_status(from_changed, to_changed, from_type, to_type)

    # Semantic drift (independent of status ladder, mirrors links only — §5.1).
    semantic_drift: bool | None = None
    if link.type == LinkType.MIRRORS:
        semantic_drift = evaluate_semantic_drift(link, db=db)

    return DriftEvaluation(link=link, drift_status=status, semantic_drift=semantic_drift)


def evaluate_all_drift(
    *,
    db: ScryDB,
    link_store: LinkStore,
) -> list[DriftEvaluation]:
    """Replay baseline ⊕ current-branch overlay, then evaluate every active link.

    Calls ``link_store.replay()`` to compute the merged active link table and
    surface merge-conflict link IDs, then delegates each link to
    ``evaluate_link_drift``.

    Returns a ``DriftEvaluation`` for every link in the active table.
    """
    result = link_store.replay()
    conflicts: set[LinkId] = set(result.merge_conflicts)
    return [
        evaluate_link_drift(link, db=db, merge_conflicts=conflicts)
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
