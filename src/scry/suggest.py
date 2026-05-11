"""scry suggest-links — AI-augmented batch link suggestion engine (Wave 5b).

DESIGN.md §9: ``scry suggest-links [--scope <path>] [--accept-all]``
AI-augmented batch link suggestions (opt-in; requires LLM provider).

Pipeline
--------
1. Replay the active link table (baseline + current branch overlay).
2. List all anchors from the vector DB; apply ``scope`` path-prefix filter.
3. Select candidate ``(code, doc)`` pairs via embedding similarity
   (``hybrid_search``).  Pairs already present in the active link table
   are excluded (idempotency guarantee).
4. Batch-evaluate pairs via the configured LLM provider (``json_mode=True``).
5. Apply ``min_confidence`` threshold; sort by confidence descending.
6. Return the suggestion list (caller handles output / ``--apply`` writes).

Idempotency
-----------
Re-running on unchanged state produces 0 new suggestions: step 3 always
excludes pairs already present in the active link table, and the LLM batch
will receive no input, returning an empty list immediately.

Cost control
------------
Token usage is summed across all LLM batches; a ``logger.warning`` is
emitted when the cumulative total exceeds ``token_warn_threshold``.
:class:`~scry.llm.LLMJSONModeError` and other
:class:`~scry.llm.LLMError` subclasses are caught per-batch — a failed
batch is skipped with a warning (graceful degradation, not a fatal error).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from scry.embed import Embedder
from scry.llm import LLMError, LLMJSONModeError, LLMNetworkError, LLMProvider, LLMRequest
from scry.models import Anchor, AnchorType, Link, LinkId, RetrievalConfig
from scry.retrieve import hybrid_search
from scry.store.db import ScryDB
from scry.store.links import LinkStore

__all__ = [
    "DEFAULT_MIN_CONFIDENCE",
    "LinkSuggestion",
    "SuggestConfig",
    "batch_llm_evaluate",
    "run_suggest_links",
    "select_candidate_pairs",
]

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

DEFAULT_MIN_CONFIDENCE: float = 0.7
_DEFAULT_TOP_K_NEIGHBORS: int = 5
_DEFAULT_BATCH_SIZE: int = 20
_DEFAULT_TOKEN_WARN_THRESHOLD: int = 50_000

_VALID_LINK_TYPES: frozenset[str] = frozenset({"mirrors", "implements", "references"})

_SYSTEM_PROMPT = """\
You are a documentation linker for a software project.
Given pairs of (code anchor, doc anchor), decide if they describe the same concept
and should be linked. Respond with valid JSON only — no markdown, no extra text.

Link types (use exactly one of these strings):
- "mirrors"    — The code directly implements what the doc describes (strongest match)
- "implements" — The code implements a specific requirement from the doc
- "references" — One mentions the other but is not the primary implementation

Required JSON output schema:
{
  "suggestions": [
    {
      "pair_id": "<string from input>",
      "should_link": true,
      "link_type": "mirrors",
      "confidence": 0.9,
      "reason": "one sentence explanation"
    }
  ]
}

Include one entry per pair. Set should_link=false when the pair should not be
linked. When should_link is false, confidence should be below 0.5."""


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class LinkSuggestion:
    """A proposed link between a code anchor and a doc anchor.

    Produced by :func:`batch_llm_evaluate`; returned by :func:`run_suggest_links`.
    """

    from_id: str
    to_id: str
    link_type: str  # "mirrors" | "implements" | "references"
    confidence: float  # 0.0 .. 1.0
    reason: str  # short LLM-generated explanation


@dataclass
class SuggestConfig:
    """Tuning parameters for the suggest-links pipeline."""

    min_confidence: float = DEFAULT_MIN_CONFIDENCE
    limit: int | None = None
    top_k_neighbors: int = _DEFAULT_TOP_K_NEIGHBORS
    batch_size: int = _DEFAULT_BATCH_SIZE
    source: Literal["code", "doc", "both"] = "both"
    scope: str | None = None
    token_warn_threshold: int = _DEFAULT_TOKEN_WARN_THRESHOLD


# ─── Candidate pair selection ─────────────────────────────────────────────────


def _existing_pairs(active_links: dict[LinkId, Link]) -> set[tuple[str, str]]:
    """Build the set of directed ``(from_id, to_id)`` edges already present."""
    return {(lnk.from_id, lnk.to_id) for lnk in active_links.values()}


def _is_doc_type(anchor_type: str) -> bool:
    return anchor_type in (AnchorType.SECTION.value, AnchorType.CODE_IN_DOC.value)


def _is_code_type(anchor_type: str) -> bool:
    return anchor_type == AnchorType.CODE.value


def select_candidate_pairs(
    *,
    db: ScryDB,
    active_links: dict[LinkId, Link],
    embedder: Embedder,
    config: SuggestConfig,
) -> list[tuple[Anchor, Anchor]]:
    """Select ``(code_anchor, doc_anchor)`` candidate pairs via embedding similarity.

    For each anchor on the scanned side (determined by ``config.source``),
    :func:`~scry.retrieve.hybrid_search` surfaces the top-K nearest neighbors
    of the opposite type.  Pairs that already exist in ``active_links`` (in
    either direction) are excluded.  Duplicates arising from bidirectional
    scanning are de-duplicated.

    Applies ``config.scope`` as a path-prefix filter to **both** sides of
    every pair.  Caps the result at ``config.limit`` when set.

    Args:
        db:           Read-only :class:`~scry.store.db.ScryDB` connection.
        active_links: Replayed active link table (baseline ⊕ overlay).
        embedder:     Embedding backend used by ``hybrid_search``.
        config:       Suggest-links tuning parameters.

    Returns:
        List of ``(code_anchor, doc_anchor)`` pairs (de-duplicated, filtered).
    """
    all_anchors = db.list_anchors()
    if config.scope:
        all_anchors = [a for a in all_anchors if a.path.startswith(config.scope)]

    code_anchors = [a for a in all_anchors if _is_code_type(a.type)]
    doc_anchors = [a for a in all_anchors if _is_doc_type(a.type)]

    existing = _existing_pairs(active_links)
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[Anchor, Anchor]] = []
    retrieval_cfg = RetrievalConfig()

    def _maybe_add(code: Anchor, doc: Anchor) -> None:
        # Defense-in-depth scope filter (review-w5b MEDIUM): hybrid_search
        # neighbours can return anchors outside the scoped corpus; recheck
        # both endpoints' paths so suggestions stay within --scope.
        if config.scope and (
            not code.path.startswith(config.scope) or not doc.path.startswith(config.scope)
        ):
            return
        key = (code.id, doc.id)
        if key in seen:
            return
        # Exclude both directions so we don't re-propose a link that exists reversed.
        if key in existing or (doc.id, code.id) in existing:
            return
        seen.add(key)
        pairs.append((code, doc))

    if config.source in ("code", "both"):
        for code_anchor in code_anchors:
            query = code_anchor.content_text[:1000]
            try:
                results = hybrid_search(
                    query,
                    db=db,
                    embedder=embedder,
                    config=retrieval_cfg,
                    top_k=config.top_k_neighbors,
                    anchor_types=[AnchorType.SECTION, AnchorType.CODE_IN_DOC],
                )
            except Exception:
                logger.warning(
                    "hybrid_search failed for code anchor %s", code_anchor.id, exc_info=True
                )
                continue
            for r in results:
                doc_anchor = db.get_anchor(r.parent_anchor_id)
                if doc_anchor is not None and _is_doc_type(doc_anchor.type):
                    _maybe_add(code_anchor, doc_anchor)

    if config.source in ("doc", "both"):
        for doc_anchor in doc_anchors:
            query = doc_anchor.content_text[:1000]
            try:
                results = hybrid_search(
                    query,
                    db=db,
                    embedder=embedder,
                    config=retrieval_cfg,
                    top_k=config.top_k_neighbors,
                    anchor_types=[AnchorType.CODE],
                )
            except Exception:
                logger.warning(
                    "hybrid_search failed for doc anchor %s", doc_anchor.id, exc_info=True
                )
                continue
            for r in results:
                ca = db.get_anchor(r.parent_anchor_id)
                if ca is not None and _is_code_type(ca.type):
                    _maybe_add(ca, doc_anchor)

    if config.limit is not None:
        pairs = pairs[: config.limit]

    return pairs


# ─── LLM batch evaluation ─────────────────────────────────────────────────────


def _anchor_payload(anchor: Anchor) -> dict[str, Any]:
    """Compact JSON-serializable representation of *anchor* for the LLM prompt."""
    payload: dict[str, Any] = {
        "id": anchor.id,
        "path": anchor.path,
        "content": anchor.content_text[:500],
    }
    if anchor.symbol_name:
        payload["symbol"] = anchor.symbol_name
    if anchor.heading_path:
        payload["heading"] = " > ".join(anchor.heading_path)
    return payload


def _parse_llm_batch(
    raw_text: str,
    pair_ids: list[str],
) -> list[dict[str, Any]]:
    """Parse the LLM's JSON response into valid suggestion dicts.

    Tolerates extra fields and silently drops:
    - Entries with an unknown ``pair_id``.
    - Entries with ``should_link=false``.
    - Entries with an unrecognised ``link_type``.

    Returns an empty list on top-level JSON parse errors (caller logs).
    """
    try:
        data: Any = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    raw_suggestions = data.get("suggestions", [])
    if not isinstance(raw_suggestions, list):
        return []

    pair_id_set = set(pair_ids)
    valid: list[dict[str, Any]] = []
    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue
        pair_id = item.get("pair_id")
        if pair_id not in pair_id_set:
            continue
        if not bool(item.get("should_link", False)):
            continue
        link_type = str(item.get("link_type", ""))
        if link_type not in _VALID_LINK_TYPES:
            continue
        try:
            confidence = float(item.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        # Drop hallucinated out-of-range values (review-w5b MEDIUM):
        # an LLM that returns confidence=1.5 or -0.3 would otherwise
        # bypass --min-confidence thresholding.
        if not (0.0 <= confidence <= 1.0):
            continue
        reason = str(item.get("reason", ""))
        valid.append(
            {
                "pair_id": str(pair_id),
                "link_type": link_type,
                "confidence": confidence,
                "reason": reason,
            }
        )
    return valid


async def batch_llm_evaluate(
    pairs: list[tuple[Anchor, Anchor]],
    *,
    provider: LLMProvider,
    batch_size: int = _DEFAULT_BATCH_SIZE,
    token_warn_threshold: int = _DEFAULT_TOKEN_WARN_THRESHOLD,
) -> list[LinkSuggestion]:
    """Evaluate *pairs* in batches via the LLM provider.

    Each batch is a single chat completion with ``json_mode=True``.
    :class:`~scry.llm.LLMJSONModeError` and other
    :class:`~scry.llm.LLMError` subclasses are caught per-batch — a
    failed batch is skipped with a warning (graceful degradation).

    If EVERY batch fails (e.g. provider unreachable / misconfigured),
    the LAST :class:`~scry.llm.LLMError` is re-raised so the CLI can
    distinguish "no suggestions" from "provider failure" and surface
    actionable guidance to the user (review-w5b HIGH fix).

    Token usage is accumulated; *token_warn_threshold* triggers a
    ``logger.warning`` when exceeded.

    Args:
        pairs:                ``(code_anchor, doc_anchor)`` pairs to evaluate.
        provider:             Configured :class:`~scry.llm.LLMProvider`.
        batch_size:           Maximum pairs per LLM request.
        token_warn_threshold: Cumulative token count that triggers a warning.

    Returns:
        List of :class:`LinkSuggestion` objects (``should_link=True`` entries
        with recognised ``link_type`` values).

    Raises:
        LLMError: when EVERY batch failed (no suggestions could be produced
            because the provider was unreachable / misconfigured).  The most
            recent provider exception is re-raised verbatim so the CLI can
            print its message.
    """
    pair_map: dict[str, tuple[Anchor, Anchor]] = {f"p_{i}": pair for i, pair in enumerate(pairs)}
    items = list(pair_map.items())  # [(pair_id, (code, doc)), ...]

    suggestions: list[LinkSuggestion] = []
    total_tokens = 0
    batch_count = 0
    failed_batches = 0
    last_error: LLMError | None = None

    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start : batch_start + batch_size]
        batch_count += 1
        batch_pair_ids = [pid for pid, _ in batch]

        user_content = json.dumps(
            {
                "pairs": [
                    {
                        "pair_id": pid,
                        "code": _anchor_payload(code),
                        "doc": _anchor_payload(doc),
                    }
                    for pid, (code, doc) in batch
                ]
            }
        )

        req = LLMRequest(
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            json_mode=True,
        )

        try:
            resp = await provider.complete(req)
        except LLMNetworkError:
            # UAT-2: connection errors mean the LLM is unreachable
            # (down / wrong port / firewalled).  Iterating the remaining
            # 800+ candidate pairs printing the same error per batch is
            # a 5-minute time-sink with zero useful output.  Abort
            # IMMEDIATELY on the first network failure; the CLI owns the
            # user-facing error message via its `except LLMError` clause.
            # Other LLMError subclasses (rate limits, bad JSON) remain
            # per-batch since those can be transient.
            raise
        except LLMJSONModeError as exc:
            logger.warning(
                "suggest-links: LLM returned non-JSON for batch at index %d: %s",
                batch_start,
                exc,
            )
            failed_batches += 1
            last_error = exc
            continue
        except LLMError as exc:
            logger.warning(
                "suggest-links: LLM error for batch at index %d: %s",
                batch_start,
                exc,
            )
            failed_batches += 1
            last_error = exc
            continue

        if resp.usage:
            total_tokens += resp.usage.get("total", 0)

        parsed = _parse_llm_batch(resp.text, batch_pair_ids)
        for item in parsed:
            code_anchor, doc_anchor = pair_map[item["pair_id"]]
            suggestions.append(
                LinkSuggestion(
                    from_id=code_anchor.id,
                    to_id=doc_anchor.id,
                    link_type=item["link_type"],
                    confidence=item["confidence"],
                    reason=item["reason"],
                )
            )

    if total_tokens > token_warn_threshold:
        logger.warning(
            "suggest-links: consumed %d tokens across LLM calls (threshold: %d). "
            "Consider reducing --limit or --source.",
            total_tokens,
            token_warn_threshold,
        )

    # All-batches-failed: re-raise the last error so the CLI can surface it
    # as a clear "provider unavailable" message instead of an ambiguous
    # "no suggestions found" (review-w5b HIGH fix).  Only escalate when at
    # least one batch was attempted (no pairs at all is not an error).
    if batch_count > 0 and failed_batches == batch_count and last_error is not None:
        raise last_error

    return suggestions


async def run_suggest_links(
    *,
    db: ScryDB,
    link_store: LinkStore,
    embedder: Embedder,
    provider: LLMProvider,
    config: SuggestConfig,
) -> list[LinkSuggestion]:
    """Orchestrate the full suggest-links pipeline.

    Steps:

    1. Replay active link table (baseline + current branch overlay).
    2. Select candidate ``(code, doc)`` anchor pairs via embedding similarity.
    3. Batch-evaluate pairs via LLM.
    4. Apply ``config.min_confidence`` threshold.
    5. Return sorted by confidence descending.

    Idempotent: pairs already present in the active link table are excluded
    in step 2, so re-running on unchanged state returns an empty list.

    Args:
        db:         Read-only :class:`~scry.store.db.ScryDB` connection.
        link_store: :class:`~scry.store.links.LinkStore` for replaying the
                    active link table.
        embedder:   Embedding backend.
        provider:   LLM provider for batch evaluation.
        config:     Suggest-links tuning parameters.

    Returns:
        Filtered, sorted :class:`LinkSuggestion` list.
    """
    replay = link_store.replay()
    active_links = replay.active_links

    pairs = select_candidate_pairs(
        db=db,
        active_links=active_links,
        embedder=embedder,
        config=config,
    )

    if not pairs:
        return []

    all_suggestions = await batch_llm_evaluate(
        pairs,
        provider=provider,
        batch_size=config.batch_size,
        token_warn_threshold=config.token_warn_threshold,
    )

    filtered = [s for s in all_suggestions if s.confidence >= config.min_confidence]
    filtered.sort(key=lambda s: s.confidence, reverse=True)
    return filtered
