"""Core data models for scry.

Per workstream W0b in `plan.md`, this module defines the Pydantic v2
models used throughout the codebase. The shapes are derived directly
from DESIGN.md v3.1 (commit 7f1e217). Section references are inline.

Conventions:
    - All models use `model_config = ConfigDict(frozen=True)` where
      they represent immutable values (records, identifiers).
    - Optional fields default to `None` rather than empty containers.
    - Field aliases match the JSON wire format used in JSONL records
      and MCP responses (see DESIGN.md §3.5.1, §4.2).
"""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

# ───── Identifiers and primitives ────────────────────────────────────

# DESIGN.md §3.2 — anchor primary IDs are path-derived strings with `::`
# separators between path / heading-slug / declaration-slug.
AnchorId = Annotated[str, StringConstraints(min_length=1, max_length=2048)]

# DESIGN.md §3.5 — link_id is the logical edge identity (stable across
# the link's history). Format: `lnk_<base32-ulid-or-uuid7>`; we use
# UUIDv7 for monotonic sortability under the prefix.
LinkId = Annotated[str, StringConstraints(pattern=r"^lnk_[A-Za-z0-9_-]+$")]

# DESIGN.md §3.5 — event_id is the immutable per-record identity.
# Format: `evt_<uuidv7>`.
EventId = Annotated[str, StringConstraints(pattern=r"^evt_[A-Za-z0-9_-]+$")]

# DESIGN.md §5.4 — content_hash is sha256 over canonicalized content.
ContentHash = Annotated[str, StringConstraints(pattern=r"^sha256:[0-9a-f]{64}$")]

# Idempotency token for IPC writes (DESIGN.md §10.3). Format: `tok_<uuidv7>`.
IdempotencyToken = Annotated[str, StringConstraints(pattern=r"^tok_[A-Za-z0-9_-]+$")]


def new_link_id() -> LinkId:
    """Mint a new logical link_id."""
    return f"lnk_{uuid.uuid4().hex}"


def new_event_id() -> EventId:
    """Mint a new immutable per-record event_id (UUIDv7-shaped via uuid4 fallback).

    DESIGN.md §3.5 specifies UUIDv7 for monotonic sortability; when
    Python's stdlib gains uuid7() we'll switch. Until then, uuid4 is
    sufficient (uniqueness is guaranteed; sortability is incidental).
    """
    return f"evt_{uuid.uuid4().hex}"


def new_idempotency_token() -> IdempotencyToken:
    """Mint a new idempotency token for an IPC write call."""
    return f"tok_{uuid.uuid4().hex}"


# ───── Anchor types ──────────────────────────────────────────────────


# DESIGN.md §3.1 — three anchor types share one embedding space.
class AnchorType(StrEnum):
    SECTION = "section"
    CODE_IN_DOC = "code_in_doc"
    CODE = "code"


# DESIGN.md §5.3 — closure-quality enum surfaced on every code anchor.
class TransitiveHashStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    LSP_UNAVAILABLE = "lsp_unavailable"
    LSP_ERROR = "lsp_error"


class Anchor(BaseModel):
    """Smallest addressable, embeddable, linkable unit (DESIGN.md §3.1).

    `content_text` is the canonicalized anchor content (per §5.4) stored
    so that `--reembed` does not require re-reading the source file.
    `transitive_hash_status` is set only on `CODE` anchors (omitted from
    JSON for other types — see model_dump exclusion logic).
    """

    model_config = ConfigDict(frozen=True, use_enum_values=True)

    id: AnchorId
    type: AnchorType
    path: str
    """Repo-relative path to the source file."""

    heading_path: list[str] | None = None
    """Heading breadcrumb (sections / code_in_doc only)."""

    symbol_name: str | None = None
    """Symbol name (code anchors only)."""

    content_text: str
    """Canonicalized content per §5.4 — what the hash and embedding cover."""

    content_hash: ContentHash
    """sha256 over `content_text`."""

    fingerprint_simhash: int
    """64-bit SimHash for fuzzy fingerprint matching (§3.3)."""

    transitive_hash_status: TransitiveHashStatus | None = None
    """Set on CODE anchors only; omitted from JSON for SECTION/CODE_IN_DOC."""

    is_test: bool = False
    """SR5-6: True when the anchor lives in a test file (per filename
    heuristic) OR is itself a test-framework construct (e.g. a Jest
    ``describe`` / ``it`` anchor produced by the SR5-5 test-call
    walker).  Always False for SECTION (markdown) anchors — there is
    no test/prod distinction for documentation.  Callers that want to
    suppress test results can pass ``exclude_tests=True`` to
    :func:`~scry.retrieve.hybrid_search` or the MCP ``search`` tool.
    """

    closure_hash: str | None = None
    """SHA-256 over transitive callee content hashes (CODE anchors only).

    Populated during W3d LSP enrichment via ``compute_closure``.  ``None``
    when LSP is unavailable, the language is skipped, or an error occurred
    (see ``transitive_hash_status`` for the reason).  Not persisted to the
    DB for the current wave; stored separately from ``content_hash`` so
    callers can distinguish AST drift from transitive-closure drift.
    """

    def_line: int | None = None
    """0-based line of the symbol's definition position (CODE anchors only).

    Set by the code extractor from the tree-sitter node's ``start_point``.
    Used by the W3d LSP enrichment pass to call
    ``textDocument/prepareCallHierarchy`` at the correct position.  NOT
    persisted to the DB (None after a DB round-trip).
    """

    def_char: int | None = None
    """0-based character offset within ``def_line`` (CODE anchors only).

    See ``def_line`` for full semantics.
    """

    @field_validator("path")
    @classmethod
    def _path_is_relative(cls, v: str) -> str:
        # Cross-platform absolute-path detection: pathlib.Path.is_absolute is
        # platform-specific (POSIX vs NT), so check both common shapes.
        if Path(v).is_absolute() or v.startswith(("/", "\\")) or (len(v) >= 2 and v[1] == ":"):
            raise ValueError(f"anchor path must be repo-relative, got absolute: {v}")
        return v.replace("\\", "/")

    @model_validator(mode="after")
    def _status_only_on_code(self) -> Anchor:
        if self.type != AnchorType.CODE.value:
            if self.transitive_hash_status is not None:
                raise ValueError("transitive_hash_status may only be set on CODE anchors")
            if self.closure_hash is not None:
                raise ValueError("closure_hash may only be set on CODE anchors")
        return self


# ───── Sub-chunks (internal retrieval cache) ─────────────────────────


class SubChunk(BaseModel):
    """An internal retrieval fragment of a parent anchor (DESIGN.md §3.4).

    Sub-chunks are NOT separate drift units and NEVER serve as link
    targets. Each row carries the parent's `content_hash` so that the
    read-side hash-equality filter (§7.3) can drop stale orphans.
    """

    model_config = ConfigDict(frozen=True)

    parent_id: AnchorId
    chunk_index: int = Field(ge=0)
    text: str
    parent_content_hash: ContentHash
    overlap_with_prev: int = Field(default=0, ge=0)


# ───── Links ─────────────────────────────────────────────────────────


# DESIGN.md §3.6 — canonical-direction link types.
class LinkType(StrEnum):
    IMPLEMENTS = "implements"
    TESTS = "tests"
    EXAMPLES = "examples"
    MIRRORS = "mirrors"
    DERIVES_FROM = "derives-from"
    REFERENCES = "references"


class LinkOp(StrEnum):
    UPSERT = "upsert"
    DELETE = "delete"


# DESIGN.md §5.1 — drift status taxonomy (post-v3.1).
class DriftStatus(StrEnum):
    FRESH = "fresh"
    SPEC_CHANGED = "spec-changed"
    CODE_CHANGED = "code-changed"
    BOTH_CHANGED = "both-changed"
    BROKEN_SOURCE = "broken-source"
    BROKEN_TARGET = "broken-target"
    MERGE_CONFLICT = "merge-conflict"
    DRIFT_UNKNOWN = "drift-unknown"


# DESIGN.md §5.1 — precedence ladder (highest → lowest), v3.1.
DRIFT_PRECEDENCE: tuple[DriftStatus, ...] = (
    DriftStatus.MERGE_CONFLICT,
    DriftStatus.BROKEN_SOURCE,
    DriftStatus.BROKEN_TARGET,
    DriftStatus.BOTH_CHANGED,
    DriftStatus.SPEC_CHANGED,
    DriftStatus.CODE_CHANGED,
    DriftStatus.DRIFT_UNKNOWN,
    DriftStatus.FRESH,
)


def drift_winner(*statuses: DriftStatus) -> DriftStatus:
    """Return the highest-precedence status from the inputs (§5.1)."""
    if not statuses:
        return DriftStatus.FRESH
    rank = {s: i for i, s in enumerate(DRIFT_PRECEDENCE)}
    return min(statuses, key=lambda s: rank[s])


class LinkRecord(BaseModel):
    """One append-only event record in `links.jsonl` or an overlay file.

    Schema per DESIGN.md §3.5.1 (post-v3.1: per-endpoint prior_*_content_hash,
    immutable event_id distinct from logical link_id).

    Two record shapes are validated:
        - upsert: requires from/to/type/hashes
        - delete: requires only link_id + reason

    Validation enforced at write time per §3.5.2:
        - `supersedes` REQUIRED if link_id already exists in baseline ⊕ overlay
        - tombstones absorbing within same file (validated by reader, not here)
    """

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    op: LinkOp
    link_id: LinkId
    event_id: EventId = Field(default_factory=new_event_id)
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Required for upsert; forbidden for delete.
    from_: str | None = Field(default=None, alias="from")
    from_type: AnchorType | None = None
    to: str | None = None
    to_type: AnchorType | None = None
    type: LinkType | None = None
    from_content_hash: ContentHash | None = None
    to_content_hash: ContentHash | None = None
    from_closure_hash: str | None = None
    """Transitive closure hash of the source anchor at link-proposal time (W3d).

    Stored alongside ``from_content_hash`` so drift detection can fire
    ``code-changed`` when a callee body changes even if the anchor's own
    AST is unchanged (DESIGN.md §5.3).  ``None`` for non-CODE source
    anchors or when LSP enrichment was unavailable.  Missing from
    pre-W3d baseline records — treated as "no closure comparison" (no
    false positive on baseline upgrade).
    """
    to_closure_hash: str | None = None
    """Transitive closure hash of the target anchor at link-proposal time (W3d).

    Same semantics as ``from_closure_hash`` for the target endpoint.
    """
    prior_from_content_hash: ContentHash | None = None
    """Set on rebased upserts (§3.3); consulted by §5.1 prior-hash override."""
    prior_to_content_hash: ContentHash | None = None
    """Set on rebased upserts (§3.3); consulted by §5.1 prior-hash override."""
    commit_sha: str | None = None
    """git HEAD at upsert time."""
    worktree_dirty: bool = False
    evidence: str | None = None
    supersedes: EventId | None = None
    """REQUIRED if link_id exists in baseline ⊕ overlay (§3.5.2 rule 5);
    references the prior record's event_id, NOT the link_id."""

    # Only valid for delete.
    reason: str | None = None

    # UAT-10: identity of the user/session that authored this event.
    # Auto-populated from ``SCRY_USER`` env or ``git config user.email``
    # at creation time (propose_link, unlink, CLI scry link).  None for
    # legacy baseline records (pre-UAT-10) — these parse cleanly since
    # the field has a default.  Promoted records (commit_links) preserve
    # the original created_by rather than re-stamping the promoter.
    created_by: str | None = None

    @model_validator(mode="after")
    def _shape_check(self) -> LinkRecord:
        if self.op == LinkOp.UPSERT.value:
            required = (
                "from_",
                "from_type",
                "to",
                "to_type",
                "type",
                "from_content_hash",
                "to_content_hash",
            )
            missing = [name for name in required if getattr(self, name) is None]
            if missing:
                raise ValueError(f"upsert record missing required fields: {missing}")
            if self.reason is not None:
                raise ValueError("upsert record may not carry `reason`")
        elif self.op == LinkOp.DELETE.value:
            forbidden = (
                "from_",
                "from_type",
                "to",
                "to_type",
                "type",
                "from_content_hash",
                "to_content_hash",
                "prior_from_content_hash",
                "prior_to_content_hash",
                "evidence",
            )
            present = [name for name in forbidden if getattr(self, name) is not None]
            if present:
                raise ValueError(f"delete record must not carry: {present}")
        return self


class Link(BaseModel):
    """A typed, directed edge as seen by retrieval after replay (DESIGN.md §3.5).

    This is the *active* shape — what `replay(baseline) ⊕ replay(overlay)`
    produces. Distinct from `LinkRecord` (the on-disk event).
    """

    model_config = ConfigDict(use_enum_values=True)

    link_id: LinkId
    from_id: AnchorId
    from_type: AnchorType
    to_id: AnchorId
    to_type: AnchorType
    type: LinkType
    from_content_hash: ContentHash
    to_content_hash: ContentHash
    from_closure_hash: str | None = None
    """Transitive closure hash of the source anchor at link-proposal time.

    Used by drift detection to detect callee-body changes (DESIGN.md §5.3).
    ``None`` for non-CODE source endpoints or when LSP enrichment was
    unavailable.  Pre-W3d baselines carry ``None`` — drift treats ``None``
    as "no closure comparison available" (no false positive).
    """
    to_closure_hash: str | None = None
    """Transitive closure hash of the target anchor at link-proposal time."""
    prior_from_content_hash: ContentHash | None = None
    prior_to_content_hash: ContentHash | None = None
    commit_sha: str | None = None
    worktree_dirty: bool = False
    evidence: str | None = None
    last_event_id: EventId
    """The event_id of the most recent record establishing this link."""

    # UAT-10: identity of the user/session that created this logical link.
    created_by: str | None = None


# ───── Anchor packet (search result envelope) ────────────────────────


class IndexState(StrEnum):
    """DESIGN.md §7.2 — values returned in MCP `index_state` field (post-v3.1).

    `STALE_WARNED` is set when reconcile would exceed
    `index.auto_reconcile_max_changed_files` (default 500); user must
    explicitly run `scry index`.
    """

    FRESH = "fresh"
    STALE_RECONCILING = "stale-reconciling"
    STALE_NO_WRITE_LOCK = "stale-no-write-lock"
    STALE_WARNED = "stale-warned"


class AnchorLinkProjection(BaseModel):
    """A link as projected into an anchor packet (DESIGN.md §4.2).

    `transitive_hash_status` is OMITTED from the JSON when the target
    is not a CODE anchor (§5.3 field-placement rule, post-v3.1) so
    agents can use `"transitive_hash_status" in link` as a presence
    check.
    """

    model_config = ConfigDict(use_enum_values=True)

    to: AnchorId
    to_type: AnchorType
    type: LinkType
    drift_status: DriftStatus
    semantic_drift: bool | None = None
    """For `mirrors` links only. True = embeddings diverge beyond
    `drift.semantic_drift_threshold`. None = cross-language pair where
    threshold isn't calibrated."""
    transitive_hash_status: TransitiveHashStatus | None = None
    """Set ONLY when `to_type == CODE`; omitted from JSON otherwise."""

    def serialize(self) -> dict[str, Any]:
        """Serialize with `transitive_hash_status` omitted for non-CODE targets."""
        d = self.model_dump()
        if self.to_type != AnchorType.CODE.value:
            d.pop("transitive_hash_status", None)
        return d


class AnchorPacket(BaseModel):
    """The envelope returned by `search` for each result (DESIGN.md §4.2).

    `content` is bounded by `retrieval.content_preview_tokens` (default
    500); when truncated, `content_truncated=True` and full content is
    available via `get_anchor(id)`.
    """

    model_config = ConfigDict(use_enum_values=True)

    anchor: Anchor
    score: float
    evidence_excerpt: str | None = None
    match_offset: int | None = None
    """Character offset of ``evidence_excerpt`` within the anchor's ``content_text``.

    Only present when ``evidence_excerpt`` is a TRUE sub-string of the anchor
    content (i.e. was NOT dropped because it was identical to the displayed
    ``content_text``).  ``None`` when ``evidence_excerpt`` is absent, when
    it is not a substring of ``content_text`` (e.g. generated chunks,
    overlap windows), or when the offset could not be determined.  Allows
    callers to reconstruct the surrounding context window (UAT-R5-9).

    Note: this is a Python ``str`` character offset, NOT a UTF-8 byte
    offset.  Consistent with other character-based positions in
    AnchorPacket.  (Clarified in review-r6abc-3.)
    """
    links: list[AnchorLinkProjection] = Field(default_factory=list)
    index_state: IndexState = IndexState.FRESH
    content_truncated: bool = False


# ───── Drift summary (output of `scry check`) ────────────────────────


class DriftCounts(BaseModel):
    """Raw counts emitted alongside scores (DESIGN.md §5.2)."""

    model_config = ConfigDict(extra="forbid")

    broken_source: int = 0
    broken_target: int = 0
    merge_conflict: int = 0
    both_changed: int = 0
    spec_changed: int = 0
    code_changed: int = 0
    drift_unknown: int = 0
    semantic_drift_flagged: int = 0
    fresh: int = 0
    total: int = 0


class DriftSummary(BaseModel):
    """Output of `scry check` (DESIGN.md §5.2, post-v3.1)."""

    model_config = ConfigDict(extra="forbid")

    drift_score: float | None
    """0-100 normalized; null when `total_links == 0`."""
    coverage_score: float | None
    """0-100; null when no code anchors."""
    counts: DriftCounts
    drift_coverage: Literal["section-only", "full"] | None = None
    """Wave 2 emits "section-only"; Wave 4+ emits "full" or omits."""


# ───── Index provenance metadata ─────────────────────────────────────


class IndexMetadata(BaseModel):
    """The single `index_metadata` row in `vectors.db` (DESIGN.md §7.1)."""

    model_config = ConfigDict(extra="forbid")

    indexed_git_head: str
    indexed_git_tree_hash: str | None = None
    indexed_branch: str
    indexed_file_manifest: dict[str, str] = Field(default_factory=dict)
    config_hash: str
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    tokenizer_version: str | None = None


# ───── Configuration models (.scry/config.yaml) ──────────────────────


class ClassifyEntry(BaseModel):
    """One entry in the ordered classify list (DESIGN.md §6)."""

    model_config = ConfigDict(extra="forbid")

    glob: str
    type: Literal["spec", "doc"]


class SectionsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_heading_depth: int = Field(default=4, ge=1, le=6)
    max_tokens: int = Field(default=600, gt=0)
    overlap_tokens: int = Field(default=50, ge=0)
    min_section_tokens: int = Field(default=0, ge=0)


class CodeAnchorsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    granularity: Literal["symbol", "file"] = "symbol"
    symbol_kinds: dict[str, list[str]] = Field(default_factory=dict)
    languages: dict[str, Literal["lsp", "skip"]] = Field(default_factory=dict)
    transitive_resolution: Literal["call_only", "full"] = "call_only"
    lsp: dict[str, dict[str, Any]] = Field(default_factory=dict)
    transitive_max_depth: int = Field(
        default=32,
        ge=1,
        le=256,
        description=(
            "Maximum BFS depth for the transitive call-closure walk "
            "(DESIGN.md §6 / §11).  Default 32."
        ),
    )


class CodeAnchorsExtraConfig(BaseModel):
    """Defensive caps for the LSP closure walk (DESIGN.md §5.3, post-v3.1).

    .. deprecated::
        ``code_anchors_extra.transitive_max_depth`` is superseded by
        ``code_anchors.transitive_max_depth`` (DESIGN.md §6 / §11).  The
        field remains here for backward compatibility — a deprecation warning
        is emitted when the key is present in ``.scry/config.yaml``.  It
        will be removed in a future release.
    """

    model_config = ConfigDict(extra="forbid")

    transitive_max_depth: int = Field(default=32, ge=1, le=256)


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: Literal["local", "openai", "voyage", "custom"] = "local"
    model: str = "BAAI/bge-small-en-v1.5"
    dimensions: int = Field(default=384, gt=0)


class BM25Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    index_table_cells: bool = True


class LinksPerResultConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outgoing: int = Field(default=5, ge=0)
    incoming: int = Field(default=5, ge=0)


class RetrievalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fusion_rrf_k: int = Field(default=60, gt=0)
    bm25: BM25Config = Field(default_factory=BM25Config)
    links_per_result: LinksPerResultConfig = Field(default_factory=LinksPerResultConfig)
    content_preview_tokens: int = Field(default=500, gt=0)


class DriftScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    broken: float = 1.0
    merge_conflict: float = 1.0
    both_changed: float = 0.5
    spec_changed: float = 0.3
    code_changed: float = 0.3
    drift_unknown: float = 0.3
    semantic_drift: float = 0.2


class DriftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic_drift_threshold: float = Field(default=0.25, ge=0.0, le=2.0)
    cross_language_threshold: float | None = None
    scoring: DriftScoringConfig = Field(default_factory=DriftScoringConfig)


class FuzzyMatchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    embedding_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    simhash_jaccard_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    simhash_jaccard_threshold_migration: float = Field(default=0.85, ge=0.0, le=1.0)


class IndexConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cross_file_rebase: bool = False
    fuzzy_match: FuzzyMatchConfig = Field(default_factory=FuzzyMatchConfig)
    max_file_size_bytes: int = Field(default=5_242_880, gt=0)
    head_poll_interval_seconds: int = Field(default=30, ge=0)
    poll_dirty: bool = True
    auto_reconcile_max_changed_files: int = Field(default=500, gt=0)


class IPCTimeoutsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    short: float = 5.0
    long_heartbeat_interval: float = 10.0
    long_heartbeat_max_lapse: float = 30.0


class IPCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeouts: IPCTimeoutsConfig = Field(default_factory=IPCTimeoutsConfig)
    idempotency_cache_size: int = Field(default=10_000, gt=0)


class LLMConfig(BaseModel):
    """Configuration for the LLM provider (DESIGN.md §11, Wave 5).

    API keys are NEVER stored here — read from environment variables at
    provider construction time (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).
    The default provider is ``'ollama'`` (local-first: no API key, no cost).
    """

    model_config = ConfigDict(extra="forbid")

    provider: Literal["ollama", "openai", "anthropic", "litellm"] = "ollama"
    """LLM backend.  ``'ollama'`` (default) requires a running Ollama daemon."""

    model: str = "llama3.2"
    """Model identifier interpreted by the provider (e.g. ``'gpt-4o-mini'``,
    ``'claude-3-5-haiku-20241022'``, ``'llama3.2'``)."""

    base_url: str | None = None
    """Override the provider's default API base URL.  ``None`` uses the
    provider's built-in default (e.g. ``http://localhost:11434`` for Ollama)."""

    timeout: float = Field(default=60.0, gt=0.0)
    """HTTP request timeout in seconds."""


class Config(BaseModel):
    """Top-level `.scry/config.yaml` shape (DESIGN.md §6)."""

    model_config = ConfigDict(extra="forbid")

    include: list[str] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    classify: list[ClassifyEntry] = Field(default_factory=list)
    sections: SectionsConfig = Field(default_factory=SectionsConfig)
    code_anchors: CodeAnchorsConfig = Field(default_factory=CodeAnchorsConfig)
    code_anchors_extra: CodeAnchorsExtraConfig = Field(default_factory=CodeAnchorsExtraConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    index: IndexConfig = Field(default_factory=IndexConfig)
    ipc: IPCConfig = Field(default_factory=IPCConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)

    @model_validator(mode="before")
    @classmethod
    def _warn_deprecated_code_anchors_extra(cls, data: object) -> object:
        """Emit a deprecation warning when code_anchors_extra is in the input.

        The warning fires only when the caller explicitly provides the key,
        not when ``Config`` is default-constructed (which uses
        ``default_factory=CodeAnchorsExtraConfig``).
        """
        import logging as _logging
        import warnings

        if isinstance(data, dict) and "code_anchors_extra" in data:
            _logging.getLogger(__name__).warning(
                "DEPRECATED: 'code_anchors_extra.transitive_max_depth' is superseded by "
                "'code_anchors.transitive_max_depth' (DESIGN.md §6 / §11). "
                "Migrate your .scry/config.yaml and remove the code_anchors_extra section."
            )
            warnings.warn(
                "code_anchors_extra.transitive_max_depth is deprecated; "
                "use code_anchors.transitive_max_depth instead.",
                DeprecationWarning,
                stacklevel=4,
            )
        return data

    @field_validator("classify")
    @classmethod
    def _classify_must_be_list(cls, v: list[ClassifyEntry]) -> list[ClassifyEntry]:
        # Pydantic already enforces list-ness; this docstring captures the
        # spec's "non-list = error" rule explicitly. (DESIGN.md §6 comment.)
        return v


# ───── Frontmatter overrides (per-file, optional) ────────────────────


class Frontmatter(BaseModel):
    """Per-file scry frontmatter overrides (DESIGN.md §6.1, post-v3.1).

    Field name is `skip` (renamed from `exclude` in v3 to disambiguate
    from the global config exclude glob list).
    """

    model_config = ConfigDict(extra="ignore")

    type: Literal["spec", "doc"] | None = None
    id: str | None = None
    skip: bool = False


# Re-exports for convenience.
_ANCHOR_ID_RE = re.compile(r"^[A-Za-z0-9_./@#%:-]+$")


def is_well_formed_anchor_id(s: str) -> bool:
    """Lightweight sanity check for anchor IDs."""
    return bool(_ANCHOR_ID_RE.match(s))


__all__ = [
    "DRIFT_PRECEDENCE",
    "Anchor",
    "AnchorId",
    "AnchorLinkProjection",
    "AnchorPacket",
    "AnchorType",
    "BM25Config",
    "ClassifyEntry",
    "CodeAnchorsConfig",
    "CodeAnchorsExtraConfig",
    "Config",
    "ContentHash",
    "DriftConfig",
    "DriftCounts",
    "DriftScoringConfig",
    "DriftStatus",
    "DriftSummary",
    "EmbeddingsConfig",
    "EventId",
    "Frontmatter",
    "FuzzyMatchConfig",
    "IPCConfig",
    "IPCTimeoutsConfig",
    "IdempotencyToken",
    "IndexConfig",
    "IndexMetadata",
    "IndexState",
    "LLMConfig",
    "Link",
    "LinkId",
    "LinkOp",
    "LinkRecord",
    "LinkType",
    "LinksPerResultConfig",
    "RetrievalConfig",
    "SectionsConfig",
    "SubChunk",
    "TransitiveHashStatus",
    "drift_winner",
    "is_well_formed_anchor_id",
    "new_event_id",
    "new_idempotency_token",
    "new_link_id",
]
