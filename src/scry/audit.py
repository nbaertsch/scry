"""Semantic documentation audit for scry.

Compares doc anchors against related code anchors using an LLM to identify
semantic drift — cases where documentation no longer accurately describes
the actual code behavior.

Unlike hash-based drift (§5.1), semantic audit catches drift that occurred
*before* scry was initialized: docs that were never updated when code evolved.

Pipeline:
    1. Load all doc anchors (SECTION type) from the index.
    2. For each doc anchor, find related code anchors via:
       a. Existing links (strongest signal — known relationships)
       b. Embedding similarity (fallback — discovers unlabeled relationships)
    3. Batch-send (doc_content, code_content) pairs to LLM for comparison.
    4. Parse structured responses into AuditFinding objects with severity.
    5. Return findings sorted by severity (HIGH → MEDIUM → LOW).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from scry.embed import Embedder
from scry.llm import LLMError, LLMNetworkError, LLMProvider, LLMRequest
from scry.models import Anchor, AnchorType, Link, LinkId, RetrievalConfig
from scry.retrieve import hybrid_search
from scry.store.db import ScryDB
from scry.store.links import LinkStore

__all__ = [
    "AuditConfig",
    "AuditFinding",
    "AuditResult",
    "build_audit_payload",
    "parse_agent_audit",
    "run_audit",
]

logger = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────

_DEFAULT_BATCH_SIZE: int = 5
_DEFAULT_TOP_K: int = 3
_DEFAULT_TOKEN_WARN_THRESHOLD: int = 100_000

_AUDIT_SYSTEM_PROMPT = """\
You are a documentation accuracy auditor for a software project.
Given a documentation section and related code snippets, determine whether
the documentation accurately describes what the code actually does.

Focus on:
- Factual claims in the doc that contradict the code
- Missing critical behavior that the code implements but the doc omits
- Outdated references (wrong function names, wrong parameters, wrong paths)
- Incorrect architectural claims (wrong dependencies, wrong data flow)

Do NOT flag:
- Minor wording preferences or style issues
- Code that implements MORE than the doc describes (docs can be summaries)
- Implementation details the doc reasonably omits for clarity

Respond with valid JSON only — no markdown, no extra text.

Required JSON output schema:
{
  "findings": [
    {
      "pair_id": "<string from input>",
      "is_drifted": true,
      "severity": "HIGH",
      "doc_claim": "what the doc says (brief quote or paraphrase)",
      "code_reality": "what the code actually does",
      "suggestion": "one sentence fix suggestion"
    }
  ]
}

Severity levels:
- "HIGH": Doc makes a factually incorrect claim that would mislead a developer
- "MEDIUM": Doc omits important behavior or has stale references
- "LOW": Minor inaccuracy unlikely to cause confusion

Include one entry per pair. Set is_drifted=false when the doc is accurate.
When is_drifted=false, omit severity/doc_claim/code_reality/suggestion."""


# ─── Data types ───────────────────────────────────────────────────────────────


@dataclass
class AuditFinding:
    """A single semantic drift finding from the audit."""

    doc_anchor_id: str
    doc_path: str
    doc_section: str
    code_anchor_id: str
    code_path: str
    severity: Literal["HIGH", "MEDIUM", "LOW"]
    doc_claim: str
    code_reality: str
    suggestion: str


@dataclass
class AuditConfig:
    """Tuning parameters for the audit pipeline."""

    scope: str | None = None
    """Restrict audit to doc anchors under this path prefix."""

    top_k: int = _DEFAULT_TOP_K
    """Number of code neighbors to retrieve per doc anchor."""

    batch_size: int = _DEFAULT_BATCH_SIZE
    """Doc anchors per LLM request."""

    token_warn_threshold: int = _DEFAULT_TOKEN_WARN_THRESHOLD
    """Cumulative token count that triggers a warning."""

    max_doc_chars: int = 1500
    """Maximum characters of doc content to send per anchor."""

    max_code_chars: int = 2000
    """Maximum characters of code content to send per anchor."""

    limit: int | None = None
    """Maximum doc anchors to audit (None = all)."""


@dataclass
class AuditResult:
    """Complete result of an audit run."""

    findings: list[AuditFinding] = field(default_factory=list)
    docs_audited: int = 0
    docs_skipped: int = 0
    total_tokens: int = 0
    errors: int = 0


# ─── Pair selection ───────────────────────────────────────────────────────────


def _get_linked_code_anchors(
    doc_anchor: Anchor,
    *,
    db: ScryDB,
    active_links: dict[LinkId, Link],
) -> list[Anchor]:
    """Find code anchors linked to this doc anchor (either direction)."""
    results: list[Anchor] = []
    for link in active_links.values():
        target_id: str | None = None
        if link.from_id == doc_anchor.id:
            target_id = link.to_id
        elif link.to_id == doc_anchor.id:
            target_id = link.from_id
        if target_id:
            anchor = db.get_anchor(target_id)
            if anchor and anchor.type == AnchorType.CODE.value:
                results.append(anchor)
    return results


def _get_similar_code_anchors(
    doc_anchor: Anchor,
    *,
    db: ScryDB,
    embedder: Embedder,
    top_k: int,
) -> list[Anchor]:
    """Find code anchors via embedding similarity."""
    query = doc_anchor.content_text[:1000]
    try:
        results = hybrid_search(
            query,
            db=db,
            embedder=embedder,
            config=RetrievalConfig(),
            top_k=top_k,
            anchor_types=[AnchorType.CODE],
        )
    except Exception:
        logger.warning("hybrid_search failed for %s", doc_anchor.id, exc_info=True)
        return []

    anchors: list[Anchor] = []
    for r in results:
        a = db.get_anchor(r.parent_anchor_id)
        if a and a.type == AnchorType.CODE.value:
            anchors.append(a)
    return anchors


def select_audit_pairs(
    *,
    db: ScryDB,
    active_links: dict[LinkId, Link],
    embedder: Embedder,
    config: AuditConfig,
) -> list[tuple[Anchor, list[Anchor]]]:
    """Select (doc_anchor, [code_anchors]) pairs for audit.

    Priority: use existing links first, then fall back to embedding similarity.
    Only returns pairs where at least one code anchor was found.
    """
    all_anchors = db.list_anchors()

    # Filter to doc anchors
    doc_anchors = [
        a for a in all_anchors
        if a.type in (AnchorType.SECTION.value, AnchorType.CODE_IN_DOC.value)
    ]

    if config.scope:
        doc_anchors = [a for a in doc_anchors if a.path.startswith(config.scope)]

    if config.limit:
        doc_anchors = doc_anchors[: config.limit]

    pairs: list[tuple[Anchor, list[Anchor]]] = []
    for doc in doc_anchors:
        # Try links first
        code_anchors = _get_linked_code_anchors(doc, db=db, active_links=active_links)

        # Fall back to similarity
        if not code_anchors:
            code_anchors = _get_similar_code_anchors(
                doc, db=db, embedder=embedder, top_k=config.top_k
            )

        if code_anchors:
            pairs.append((doc, code_anchors))

    return pairs


# ─── LLM evaluation ──────────────────────────────────────────────────────────


def _doc_payload(anchor: Anchor, max_chars: int) -> dict[str, Any]:
    """Compact representation of a doc anchor for the LLM."""
    payload: dict[str, Any] = {
        "path": anchor.path,
        "content": anchor.content_text[:max_chars],
    }
    if anchor.heading_path:
        payload["section"] = " > ".join(anchor.heading_path)
    return payload


def _code_payload(anchor: Anchor, max_chars: int) -> dict[str, Any]:
    """Compact representation of a code anchor for the LLM."""
    payload: dict[str, Any] = {
        "path": anchor.path,
        "content": anchor.content_text[:max_chars],
    }
    if anchor.symbol_name:
        payload["symbol"] = anchor.symbol_name
    return payload


def build_audit_payload(
    pairs: list[tuple[Anchor, list[Anchor]]],
    config: AuditConfig,
) -> dict[str, Any]:
    """Build the agent-driven audit payload (no LLM call by scry).

    Returns the system prompt, schema, and pair payloads for an external
    agent (Claude/Copilot) to evaluate directly.
    """
    pair_payloads: list[dict[str, Any]] = []
    for i, (doc, code_anchors) in enumerate(pairs):
        pair_payloads.append({
            "pair_id": f"a_{i}",
            "doc": _doc_payload(doc, config.max_doc_chars),
            "code": [_code_payload(c, config.max_code_chars) for c in code_anchors],
        })
    return {
        "system_prompt": _AUDIT_SYSTEM_PROMPT,
        "schema": {
            "findings": [{
                "pair_id": "<string from input>",
                "is_drifted": "<boolean>",
                "severity": "<HIGH|MEDIUM|LOW>",
                "doc_claim": "<what the doc says>",
                "code_reality": "<what the code does>",
                "suggestion": "<fix suggestion>",
            }]
        },
        "pairs": pair_payloads,
        "_count": len(pair_payloads),
    }


def _parse_audit_batch(
    raw_text: str,
    pair_ids: list[str],
) -> list[dict[str, Any]]:
    """Parse the LLM's JSON response into valid finding dicts."""
    try:
        data: Any = json.loads(raw_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    raw_findings = data.get("findings", [])
    if not isinstance(raw_findings, list):
        return []

    pair_id_set = set(pair_ids)
    valid_severities = {"HIGH", "MEDIUM", "LOW"}
    valid: list[dict[str, Any]] = []
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        pair_id = item.get("pair_id")
        if pair_id not in pair_id_set:
            continue
        if not bool(item.get("is_drifted", False)):
            continue
        severity = str(item.get("severity", "")).upper()
        if severity not in valid_severities:
            continue
        valid.append({
            "pair_id": str(pair_id),
            "severity": severity,
            "doc_claim": str(item.get("doc_claim", "")),
            "code_reality": str(item.get("code_reality", "")),
            "suggestion": str(item.get("suggestion", "")),
        })
    return valid


def parse_agent_audit(
    raw: dict[str, Any] | str,
    *,
    pairs: list[tuple[Anchor, list[Anchor]]],
) -> list[AuditFinding]:
    """Validate an agent-supplied audit response.

    Accepts parsed dict or JSON string. Returns validated AuditFinding list.
    """
    if isinstance(raw, str):
        try:
            data: Any = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"agent audit response is not valid JSON: {exc}") from exc
    else:
        data = raw

    pair_id_to_pair: dict[str, tuple[Anchor, list[Anchor]]] = {
        f"a_{i}": pair for i, pair in enumerate(pairs)
    }
    parsed = _parse_audit_batch(json.dumps(data), list(pair_id_to_pair.keys()))

    findings: list[AuditFinding] = []
    for item in parsed:
        pair = pair_id_to_pair.get(item["pair_id"])
        if pair is None:
            continue
        doc, code_anchors = pair
        # Use first code anchor as representative
        code_repr = code_anchors[0] if code_anchors else None
        findings.append(AuditFinding(
            doc_anchor_id=doc.id,
            doc_path=doc.path,
            doc_section=" > ".join(doc.heading_path) if doc.heading_path else doc.path,
            code_anchor_id=code_repr.id if code_repr else "",
            code_path=code_repr.path if code_repr else "",
            severity=item["severity"],
            doc_claim=item["doc_claim"],
            code_reality=item["code_reality"],
            suggestion=item["suggestion"],
        ))

    # Sort by severity: HIGH first
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    findings.sort(key=lambda f: severity_order.get(f.severity, 3))
    return findings


async def run_audit(
    *,
    db: ScryDB,
    active_links: dict[LinkId, Link],
    embedder: Embedder,
    provider: LLMProvider,
    config: AuditConfig,
) -> AuditResult:
    """Run the full semantic audit pipeline with scry's own LLM provider.

    For agent-driven (no scry LLM) flow, use build_audit_payload() +
    parse_agent_audit() instead.
    """
    pairs = select_audit_pairs(
        db=db, active_links=active_links, embedder=embedder, config=config
    )

    result = AuditResult(docs_audited=len(pairs))

    if not pairs:
        return result

    # Build pair_id mapping
    pair_map: dict[str, tuple[Anchor, list[Anchor]]] = {
        f"a_{i}": pair for i, pair in enumerate(pairs)
    }
    items = list(pair_map.items())

    for batch_start in range(0, len(items), config.batch_size):
        batch = items[batch_start: batch_start + config.batch_size]
        batch_pair_ids = [pid for pid, _ in batch]

        user_content = json.dumps({
            "pairs": [
                {
                    "pair_id": pid,
                    "doc": _doc_payload(doc, config.max_doc_chars),
                    "code": [_code_payload(c, config.max_code_chars) for c in codes],
                }
                for pid, (doc, codes) in batch
            ]
        })

        req = LLMRequest(
            system=_AUDIT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.0,
            json_mode=True,
        )

        try:
            resp = await provider.complete(req)
        except LLMNetworkError:
            raise
        except LLMError as exc:
            logger.warning("audit: LLM error for batch at index %d: %s", batch_start, exc)
            result.errors += 1
            continue

        if resp.usage:
            result.total_tokens += resp.usage.get("total", 0)

        parsed = _parse_audit_batch(resp.text, batch_pair_ids)
        for item in parsed:
            pair = pair_map.get(item["pair_id"])
            if not pair:
                continue
            doc, code_anchors = pair
            code_repr = code_anchors[0] if code_anchors else None
            result.findings.append(AuditFinding(
                doc_anchor_id=doc.id,
                doc_path=doc.path,
                doc_section=" > ".join(doc.heading_path) if doc.heading_path else doc.path,
                code_anchor_id=code_repr.id if code_repr else "",
                code_path=code_repr.path if code_repr else "",
                severity=item["severity"],
                doc_claim=item["doc_claim"],
                code_reality=item["code_reality"],
                suggestion=item["suggestion"],
            ))

    if result.total_tokens > config.token_warn_threshold:
        logger.warning(
            "audit: consumed %d tokens (threshold: %d). Consider reducing --limit.",
            result.total_tokens,
            config.token_warn_threshold,
        )

    # Sort findings by severity
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    result.findings.sort(key=lambda f: severity_order.get(f.severity, 3))
    return result
