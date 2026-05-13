"""Markdown extraction for scry — workstream W1a.

Implements ``extract_markdown`` per DESIGN.md §3.1, §3.4, §5.4, §6.1,
§15.1, §15.2, and §15.4.

The public helper ``split_section_text`` is used by the indexer
(workstream W2k) to sub-chunk long sections (DESIGN.md §3.4).

Section references
------------------
§3.1   Anchor types and primary IDs
§3.4   Two-tier embedding and sub-chunk split priority
§5.4   Content-hash canonicalization
§6.1   Per-file frontmatter overrides
§15.1  Heading extraction rules
§15.2  Code block extraction rules
§15.3  ID derivation edge cases (scry-id override, sibling collisions)
§15.4  File-level filters
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from markdown_it import MarkdownIt
from markdown_it.token import Token

from scry.anchor_id import (
    canonicalize_content,
    content_hash,
    derive_section_id,
    disambiguate_siblings,
    fingerprint_simhash,
    parse_html_comment_id,
    slugify,
)
from scry.config import parse_frontmatter
from scry.models import Anchor, AnchorType, Frontmatter, SectionsConfig

__all__ = [
    "DEFAULT_MAX_FILE_SIZE_BYTES",
    "MarkdownDiagnostics",
    "extract_markdown",
    "extract_markdown_with_diagnostics",
    "split_section_text",
    "validate_markdown_file",
]

logger = logging.getLogger(__name__)

# ── File-encoding sentinels (§15.4) ──────────────────────────────────────────

_UTF8_BOM: bytes = b"\xef\xbb\xbf"
_UTF16_BE_BOM: bytes = b"\xfe\xff"
_UTF16_LE_BOM: bytes = b"\xff\xfe"

#: Default maximum file size in bytes per DESIGN.md §15.4 (``index.max_file_size_bytes``).
DEFAULT_MAX_FILE_SIZE_BYTES: int = 5_242_880  # 5 MiB

# ── Declaration detection regex (§15.2 heuristic — no AST) ───────────────────

# Matches the *first* named declaration encountered in a code block.
# Named groups give the declaration name; the caller takes the first non-None
# group value.  Covers Python, JS/TS, Rust, Zig, and common binding patterns.
_DECL_RE = re.compile(
    r"""(?mx)
    ^[ \t]*                                              # optional indent
    (?:
        # Rust / Zig: fn, pub fn, pub(crate) fn, async fn
        (?:pub(?:\([\w:]+\))?\s+)?(?:async\s+)?fn\s+(?P<fn>\w+)
        |
        # JS / TS / Go: function, async function, export function …
        (?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+(?P<function>\w+)
        |
        # Python: def, async def
        (?:async\s+)?def\s+(?P<def_>\w+)
        |
        # class (Python, JS, TS, Java, Rust struct-like, …)
        (?:export\s+)?(?:default\s+)?(?:abstract\s+)?class\s+(?P<class_>\w+)
        |
        # TypeScript / Flow interface
        (?:export\s+)?(?:default\s+)?interface\s+(?P<interface>\w+)
        |
        # TypeScript / Flow type alias: type NAME = …
        (?:export\s+)?type\s+(?P<type_alias>\w+)\s*=
        |
        # const / let / var binding: NAME = or NAME : (typed)
        (?:export\s+)?(?:pub\s+)?(?:const|let|var)\s+(?P<binding>\w+)\s*[=:]
    )
    """,
    re.MULTILINE | re.VERBOSE,
)

# ── Shared MarkdownIt instance ────────────────────────────────────────────────

# CommonMark-compliant parser; shared across calls (stateless after parse()).
_MD = MarkdownIt()


# ── Internal data classes ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarkdownDiagnostics:
    """SR3-6: extractor-side diagnostics surfaced by
    :func:`extract_markdown_with_diagnostics`.

    Attributes:
        validation_errors: Count of §15.3 validation problems detected
            in the file (currently: duplicate ``scry-id`` occurrences
            after the first).  Indexers / CLI use this to decide
            whether to exit non-zero.
    """

    validation_errors: int = 0


@dataclass
class _HeadingEntry:
    """One parsed heading extracted from the markdown token stream."""

    level: int
    """ATX/setext heading level 1-6."""
    text: str
    """Human-readable heading text (for ``Anchor.heading_path``)."""
    base_slug: str
    """Slug *before* sibling-collision resolution (may be from scry-id override)."""
    start_line: int
    """0-indexed body line where the heading starts."""


@dataclass
class _FenceEntry:
    """One parsed fenced code block."""

    info: str
    """Language tag (empty string when absent)."""
    content: str
    """Raw fence body, excluding the opening/closing ``` markers."""
    start_line: int
    """0-indexed body line of the opening fence marker."""
    end_line: int
    """Exclusive end line (the line immediately after the closing marker)."""


@dataclass
class _SectionRecord:
    """Resolved section anchor descriptor, ready for Anchor construction."""

    section_id: str
    heading_path: list[str]
    start_line: int
    end_line: int


# ── Private helpers ───────────────────────────────────────────────────────────


def _extract_inline_text(tok: Token) -> str:
    """Return plain text from a markdown-it ``inline`` token.

    Strips markup (bold, italic, links, images) while preserving text
    content and code-span literals.  Used to populate ``Anchor.heading_path``
    with human-readable heading labels.
    """
    if not tok.children:
        return tok.content
    parts: list[str] = []
    for child in tok.children:
        if child.type in {"text", "code_inline"}:
            parts.append(child.content)
        elif child.type == "softbreak":
            parts.append(" ")
    return "".join(parts)


def _find_declaration(code: str) -> str | None:
    """Find the first named declaration in a code block (heuristic, §15.2).

    Uses ``_DECL_RE`` to detect ``def``, ``class``, ``function``, ``fn``,
    ``interface``, ``type``, ``const``, ``let``, and ``var`` declarations.
    Does **not** invoke tree-sitter — that is W1b's domain.

    Args:
        code: Raw fence body content.

    Returns:
        The bare declaration name (unslugged) or ``None`` when nothing matches.
    """
    m = _DECL_RE.search(code)
    if not m:
        return None
    return next((v for v in m.groupdict().values() if v is not None), None)


def _approx_tokens(text: str) -> int:
    """Approximate token count (1.4 tokens per whitespace-separated word)."""
    return max(1, int(len(text.split()) * 1.4))


def _collect_tokens(
    tokens: list[Token],
) -> tuple[list[_HeadingEntry], list[_FenceEntry]]:
    """Walk the markdown-it token stream collecting headings and fenced blocks.

    Heading level, display text, and start-line are captured.  The §3.2
    scry-id HTML-comment escape hatch is resolved here:

    * An inline ``<!-- scry-id: foo -->`` comment within the heading text.
    * An ``html_block`` token immediately following the ``heading_close``.

    The slug derived from the pinned value replaces the normal heading slug.
    Depth filtering (``max_heading_depth``) is NOT applied here.

    Args:
        tokens: Token list from ``MarkdownIt.parse()``.

    Returns:
        ``(headings, fences)`` — both lists in document order.
    """
    headings: list[_HeadingEntry] = []
    fences: list[_FenceEntry] = []
    n = len(tokens)

    for i, tok in enumerate(tokens):
        if tok.type == "heading_open":
            level = int(tok.tag[1])
            start_line = tok.map[0] if tok.map else 0

            # Inline token sits at i+1.
            inline_tok: Token | None = tokens[i + 1] if i + 1 < n else None
            raw_content = (
                inline_tok.content if inline_tok is not None and inline_tok.type == "inline" else ""
            )
            display_text = (
                _extract_inline_text(inline_tok).strip()
                if inline_tok is not None
                else raw_content.strip()
            )

            # §3.2 scry-id override: check inline content, then html_block after close.
            pinned = parse_html_comment_id(raw_content)
            if pinned is None and i + 3 < n and tokens[i + 3].type == "html_block":
                pinned = parse_html_comment_id(tokens[i + 3].content)

            # Slug from override (already trimmed by parse_html_comment_id) or heading text.
            base_slug = slugify(pinned if pinned is not None else raw_content)

            headings.append(
                _HeadingEntry(
                    level=level,
                    text=display_text,
                    base_slug=base_slug,
                    start_line=start_line,
                )
            )

        elif tok.type == "fence":
            start_line = tok.map[0] if tok.map else 0
            end_line = tok.map[1] if tok.map else start_line + 1
            fences.append(
                _FenceEntry(
                    info=tok.info.strip(),
                    content=tok.content,
                    start_line=start_line,
                    end_line=end_line,
                )
            )

    return headings, fences


def _check_duplicate_scry_ids(tokens: list[Token], path: Path) -> int:
    """Count duplicate scry-id occurrences in one document (§15.3).

    Duplicate ``scry-id`` values within a single document are a validation
    error per DESIGN.md §15.3.  We warn but continue indexing.

    Args:
        tokens: Token list from ``MarkdownIt.parse()``.
        path:   Source file path (for the warning message).

    Returns:
        SR3-6: total count of OFFENDING occurrences (duplicates after the
        first occurrence of each slug).  E.g. a doc with three ``scry-id:
        foo`` headings counts as 2 validation errors.  Callers can treat
        a non-zero count as a §15.3 violation and exit non-zero.
    """
    seen: dict[str, int] = {}
    duplicate_count = 0
    n = len(tokens)
    for i, tok in enumerate(tokens):
        if tok.type != "heading_open":
            continue
        inline_tok = tokens[i + 1] if i + 1 < n else None
        raw = inline_tok.content if inline_tok and inline_tok.type == "inline" else ""
        pinned = parse_html_comment_id(raw)
        if pinned is None and i + 3 < n and tokens[i + 3].type == "html_block":
            pinned = parse_html_comment_id(tokens[i + 3].content)
        if pinned is not None:
            slug = slugify(pinned)
            count = seen.get(slug, 0) + 1
            seen[slug] = count
            if count >= 2:
                # SR3-6: every occurrence after the first counts.
                duplicate_count += 1
                if count == 2:
                    # Emit the warning once per duplicated slug to keep
                    # log noise bounded.
                    logger.warning(
                        "extract_markdown: duplicate scry-id %r in %s "
                        "(§15.3 index-time validation error)",
                        slug,
                        path,
                    )
    return duplicate_count


def _build_section_records(
    headings: list[_HeadingEntry],
    body_lines: list[str],
    id_base: str,
    max_heading_depth: int,
) -> list[_SectionRecord]:
    """Build resolved section records with unique IDs.

    Only headings at level ≤ *max_heading_depth* become anchors (§15.1).
    Headings deeper than this limit are ignored here; they remain as text
    within their nearest ancestor anchor.

    Sibling-slug collisions are resolved per §15.3 by delegating to
    :func:`scry.anchor_id.disambiguate_siblings` — which is collision-aware
    AND skips past any literal sibling slug that already matches the
    suffixed form (e.g. an "Examples 2" heading followed by two "Examples"
    siblings does NOT produce duplicate ``examples-2`` IDs).  Implementing
    this locally with a naïve counter is a recurring source of duplicate
    Anchor IDs (caught by review-w1a as a BLOCKING bug).

    Section content spans from the heading's start line to the start line of
    the next heading at the same or shallower level (or EOF).

    Args:
        headings:          All headings collected by ``_collect_tokens``.
        body_lines:        Canonicalized body split by ``'\\n'``.
        id_base:           Base for ID construction (repo-relative path or
                           frontmatter ``id`` override).
        max_heading_depth: Maximum heading level that becomes its own anchor.

    Returns:
        List of :class:`_SectionRecord` in document order.
    """
    anchor_headings = [h for h in headings if h.level <= max_heading_depth]

    # ---- Pass 1: walk to determine each heading's parent context -----
    # Each entry is (parent_ctx_tuple_of_resolved_slugs, base_slug).  We
    # resolve final slugs in a second pass via disambiguate_siblings so
    # that a literal heading like "Examples 2" pre-empts the auto-suffix
    # that would otherwise collide.

    # First, do a structural walk that just tracks the parent breadcrumb of
    # *base* slugs (since we can't pick final slugs until we've seen all
    # siblings under the same parent).
    base_slug_stack: list[tuple[int, str]] = []  # (level, base_slug)
    parent_base_ctx_per_heading: list[tuple[str, ...]] = []
    base_slug_per_heading: list[str] = []
    for h in anchor_headings:
        while base_slug_stack and base_slug_stack[-1][0] >= h.level:
            base_slug_stack.pop()
        parent_base_ctx_per_heading.append(tuple(s for _, s in base_slug_stack))
        base_slug_per_heading.append(h.base_slug)
        base_slug_stack.append((h.level, h.base_slug))

    # Group base_slugs by their (parent_base_ctx) and order-of-appearance.
    # Each group's resolved tail slugs are computed by disambiguate_siblings
    # so that pre-existing literal siblings consume their auto-suffix slot.
    group_indices: dict[tuple[str, ...], list[int]] = {}
    for idx, ctx in enumerate(parent_base_ctx_per_heading):
        group_indices.setdefault(ctx, []).append(idx)

    final_slug_per_heading: list[str] = [""] * len(anchor_headings)
    for _ctx, indices in group_indices.items():
        bases = [base_slug_per_heading[i] for i in indices]
        resolved = disambiguate_siblings(bases, suffix="-")
        for idx, slug in zip(indices, resolved, strict=True):
            final_slug_per_heading[idx] = slug

    # ---- Pass 2: build records using the resolved slugs ---------------
    final_slug_stack: list[tuple[int, str, str]] = []  # (level, final_slug, display_text)
    records: list[_SectionRecord] = []
    for j, h in enumerate(anchor_headings):
        while final_slug_stack and final_slug_stack[-1][0] >= h.level:
            final_slug_stack.pop()
        final_slug = final_slug_per_heading[j]
        final_slug_stack.append((h.level, final_slug, h.text))

        slug_path = [s for _, s, _ in final_slug_stack]
        section_id = derive_section_id(id_base, slug_path)
        heading_path = [t for _, _, t in final_slug_stack]

        end_line = len(body_lines)
        for k in range(j + 1, len(anchor_headings)):
            if anchor_headings[k].level <= h.level:
                end_line = anchor_headings[k].start_line
                break

        records.append(
            _SectionRecord(
                section_id=section_id,
                heading_path=heading_path,
                start_line=h.start_line,
                end_line=end_line,
            )
        )

    return records


def _section_text(body_lines: list[str], start: int, end: int) -> str:
    """Extract and re-canonicalize a section slice from *body_lines*.

    The body is already canonicalized (§5.4); this function only ensures
    the EOF trailing-newline rule for the extracted slice.

    Args:
        body_lines: Body text split on ``'\\n'``.
        start:      Inclusive start line (0-indexed).
        end:        Exclusive end line.

    Returns:
        Section text ending with exactly one ``'\\n'``, or ``''`` if the
        slice is empty or whitespace-only.
    """
    lines = body_lines[start:end]
    joined = "\n".join(lines).rstrip("\n")
    return joined + "\n" if joined else ""


def _find_parent_section(
    fence_start: int,
    section_records: list[_SectionRecord],
) -> _SectionRecord | None:
    """Return the latest section record whose ``start_line`` ≤ *fence_start*.

    Args:
        fence_start:     0-indexed start line of the fence token.
        section_records: Section records in document order (monotonic start_line).

    Returns:
        The nearest ancestor section, or ``None`` when the fence precedes
        all sections (e.g. a code block before the first heading).
    """
    parent: _SectionRecord | None = None
    for rec in section_records:
        if rec.start_line <= fence_start:
            parent = rec
        else:
            break  # records are in document order; no later one can qualify
    return parent


def _make_anchor(
    *,
    anchor_id: str,
    anchor_type: AnchorType,
    path: str,
    heading_path: list[str] | None,
    content_text: str,
) -> Anchor:
    """Construct an :class:`~scry.models.Anchor` from pre-computed fields.

    Args:
        anchor_id:    Layer 1 primary ID string.
        anchor_type:  ``SECTION`` or ``CODE_IN_DOC``.
        path:         Repo-relative file path (forward slashes).
        heading_path: Human-readable heading breadcrumb, or ``None``.
        content_text: Canonicalized content (§5.4) for hashing and embedding.

    Returns:
        A frozen :class:`~scry.models.Anchor` instance.
    """
    c_hash = content_hash(content_text)
    sim = fingerprint_simhash(content_text)
    return Anchor(
        id=anchor_id,
        type=anchor_type,
        path=path,
        heading_path=heading_path if heading_path else None,
        content_text=content_text,
        content_hash=c_hash,
        fingerprint_simhash=sim,
    )


# ── Sub-chunk splitting helper (W2k API) ──────────────────────────────────────


def _split_paragraphs_respecting_fences(text: str) -> list[str]:
    """Split *text* by blank lines, never splitting inside a fenced block.

    A fenced block is detected by lines starting with ` ``` ` or `~~~`.
    The closing fence is the next line with the same marker and nothing
    else (up to trailing whitespace).

    Args:
        text: Section content text.

    Returns:
        Non-empty paragraph strings in document order.
    """
    lines = text.split("\n")
    paragraphs: list[str] = []
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in lines:
        stripped = line.lstrip()
        if not in_fence:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = True
                fence_marker = stripped[:3]
                current.append(line)
            elif stripped == "" and current:
                paragraphs.append("\n".join(current))
                current = []
            else:
                current.append(line)
        else:
            current.append(line)
            # Closing fence: same marker, nothing after but optional spaces.
            if stripped.startswith(fence_marker) and stripped[len(fence_marker) :].strip() == "":
                in_fence = False
                fence_marker = ""

    if current:
        paragraphs.append("\n".join(current))

    return [p for p in paragraphs if p.strip()]


def _split_sentences(text: str) -> list[str]:
    """Rough sentence split on ``. ``, ``! ``, ``? `` boundaries.

    Args:
        text: A paragraph of prose.

    Returns:
        List of non-empty sentence strings.
    """
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def split_section_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int = 50,
) -> list[str]:
    """Split section text into overlapping sub-chunks for embedding (§3.4).

    Implements a SUBSET of the §3.4 split-priority order:

    2. Fenced-code-block boundaries (never split *inside* a code block).
    3. Paragraph boundaries (blank lines).
    4. Sentence boundaries (final fallback).

    Priority 1 — splitting on sub-headings deeper than the section's own
    level — is **NOT yet implemented**.  Adding it requires re-parsing the
    section text via markdown-it, which depends on integration choices that
    will be locked when the indexer (W2k) lands.  Until then,
    sub-heading-rich sections that exceed ``max_tokens`` fall through to
    paragraph/sentence splitting.  Tracked as a follow-up to W2k.

    Token count is approximated as ``word_count * 1.4``.  Each chunk
    (except the first) is prefixed with the last *overlap_tokens* tokens'
    worth of words from the previous chunk.

    This function is exposed for the indexer (workstream W2k) to call after
    receiving fully-formed ``SECTION`` anchors.  It is **not** called by
    ``extract_markdown`` itself — sub-chunking is W2k's responsibility.

    Args:
        text:           Section ``content_text`` (canonicalized markdown).
        max_tokens:     Maximum token budget per sub-chunk.
        overlap_tokens: Approximate token overlap with the preceding chunk.

    Returns:
        List of text chunks.  Never empty; returns ``[text]`` when the
        section fits within *max_tokens*.
    """
    if not text.strip():
        return [text] if text else [""]

    if _approx_tokens(text) <= max_tokens:
        return [text]

    paragraphs = _split_paragraphs_respecting_fences(text)
    if not paragraphs:
        return [text]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0
    overlap_words: list[str] = []

    def _flush() -> None:
        nonlocal current_parts, current_tokens, overlap_words
        if current_parts:
            chunk = "\n\n".join(current_parts)
            chunks.append(chunk)
            all_words = chunk.split()
            n_ov = max(0, int(overlap_tokens / 1.4))
            overlap_words = all_words[-n_ov:] if n_ov else []
            current_parts = []
            current_tokens = int(len(overlap_words) * 1.4) if overlap_words else 0

    for para in paragraphs:
        p_tokens = _approx_tokens(para)
        if current_tokens + p_tokens > max_tokens and current_parts:
            _flush()
        if p_tokens > max_tokens:
            # Single paragraph too long — split by sentences.
            for sent in _split_sentences(para):
                s_tokens = _approx_tokens(sent)
                if current_tokens + s_tokens > max_tokens and current_parts:
                    _flush()
                if overlap_words and not current_parts:
                    current_parts.append(" ".join(overlap_words))
                current_parts.append(sent)
                current_tokens += s_tokens
        else:
            if overlap_words and not current_parts:
                current_parts.append(" ".join(overlap_words))
            current_parts.append(para)
            current_tokens += p_tokens

    _flush()
    return chunks or [text]


# ── Main extraction function ───────────────────────────────────────────────────


def extract_markdown(
    path: Path,
    repo_root: Path,
    *,
    frontmatter: Frontmatter | None = None,
    config: SectionsConfig | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> list[Anchor]:
    """Extract ``SECTION`` and ``CODE_IN_DOC`` anchors from a markdown file.

    Convenience wrapper around :func:`extract_markdown_with_diagnostics`
    that returns only the anchor list — preserves the historical signature
    so existing callers (tests, MCP write helpers) don't need updates.

    Use :func:`extract_markdown_with_diagnostics` when you need access to
    extractor-side validation diagnostics (e.g. to surface §15.3 duplicate
    ``scry-id`` violations to the CLI / MCP response).
    """
    anchors, _diag = extract_markdown_with_diagnostics(
        path,
        repo_root,
        frontmatter=frontmatter,
        config=config,
        max_file_size_bytes=max_file_size_bytes,
    )
    return anchors


def extract_markdown_with_diagnostics(
    path: Path,
    repo_root: Path,
    *,
    frontmatter: Frontmatter | None = None,
    config: SectionsConfig | None = None,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> tuple[list[Anchor], MarkdownDiagnostics]:
    """Extract anchors AND surface §15.3 validation diagnostics (SR3-6).

    Returns a ``(anchors, diagnostics)`` pair.  ``diagnostics.validation_errors``
    counts duplicate ``scry-id`` occurrences after the first (each duplicate =
    +1).  Indexers thread this into ``IndexResult.validation_errors`` so the
    CLI can exit non-zero per DESIGN.md §15.3.

    Implements §15.1 (heading extraction), §15.2 (code block extraction),
    and §15.4 (file-level filters) from DESIGN.md.

    **Not produced here:** ``CODE`` anchors (workstream W1b) and
    sub-chunking (workstream W2k — use :func:`split_section_text`).

    Args:
        path:                 Absolute path to the markdown file.
        repo_root:            Absolute path to the repository root.  Used to
                              derive the repo-relative path for anchor IDs.
        frontmatter:          Optional pre-parsed :class:`~scry.models.Frontmatter`
                              override supplied by the caller (e.g. the
                              indexer).  When provided, its ``skip`` and ``id``
                              fields take precedence over the file's own YAML
                              frontmatter.
        config:               Sections extraction configuration.  ``None``
                              uses defaults: ``max_heading_depth=4``,
                              ``max_tokens=600``.
        max_file_size_bytes:  Files larger than this are skipped (§15.4).
                              Defaults to ``DEFAULT_MAX_FILE_SIZE_BYTES``
                              (5 MiB).

    Returns:
        Ordered list of :class:`~scry.models.Anchor` objects (``SECTION``
        and ``CODE_IN_DOC`` only).  Returns an empty list for every skipped
        or empty-file case defined in §15.4.
    """
    cfg = config or SectionsConfig()
    _empty_diag = MarkdownDiagnostics()

    # ── §15.4: file-level guards ──────────────────────────────────────────────

    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        logger.warning("extract_markdown: cannot read %s: %s", path, exc)
        return [], _empty_diag

    if len(raw_bytes) > max_file_size_bytes:
        logger.warning(
            "extract_markdown: skipping %s (size %d > limit %d bytes)",
            path,
            len(raw_bytes),
            max_file_size_bytes,
        )
        return [], _empty_diag

    # UTF-16 BOM check must precede the UTF-8 decode attempt (§15.4).
    if raw_bytes.startswith(_UTF16_BE_BOM) or raw_bytes.startswith(_UTF16_LE_BOM):
        logger.warning(
            "extract_markdown: skipping %s — UTF-16 encoded; transcode to UTF-8 to index",
            path,
        )
        return [], _empty_diag

    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("extract_markdown: skipping %s — binary file (UTF-8 decode failed)", path)
        return [], _empty_diag

    # §5.4 canonicalization (also strips the UTF-8 BOM U+FEFF if present).
    canonical_text = canonicalize_content(raw_text)

    # §6.1 frontmatter parse.  parse_frontmatter handles CRLF and warns on bad YAML.
    file_fm, body = parse_frontmatter(canonical_text)

    # Caller-supplied frontmatter override wins over the file's own frontmatter.
    active_fm: Frontmatter | None = frontmatter if frontmatter is not None else file_fm

    if active_fm is not None and active_fm.skip:
        return [], _empty_diag

    # §15.4: empty body (or frontmatter-only file).
    if not body.strip():
        return [], _empty_diag

    # ── ID base ───────────────────────────────────────────────────────────────

    rel_path = str(path.relative_to(repo_root)).replace("\\", "/")
    id_base: str = (active_fm.id or rel_path) if active_fm is not None else rel_path

    # ── Parse markdown ────────────────────────────────────────────────────────

    tokens = _MD.parse(body)
    body_lines = body.split("\n")

    headings, fences = _collect_tokens(tokens)

    # SR3-6: §15.3 duplicate scry-id validation.  Count is surfaced via
    # the MarkdownDiagnostics return so the indexer / CLI can exit
    # non-zero per the spec; the warning is still logged for humans.
    duplicate_scry_id_count = _check_duplicate_scry_ids(tokens, path)
    diagnostics = MarkdownDiagnostics(validation_errors=duplicate_scry_id_count)

    # ── §15.1: build SECTION anchors ─────────────────────────────────────────

    anchors: list[Anchor] = []
    anchor_headings = [h for h in headings if h.level <= cfg.max_heading_depth]

    if not anchor_headings:
        # §15.1: no headings → whole file becomes one SECTION anchor.
        if body.strip():
            anchors.append(
                _make_anchor(
                    anchor_id=id_base,
                    anchor_type=AnchorType.SECTION,
                    path=rel_path,
                    heading_path=None,
                    content_text=body,
                )
            )
        section_records: list[_SectionRecord] = (
            [_SectionRecord(id_base, [], 0, len(body_lines))] if anchors else []
        )
    else:
        section_records = _build_section_records(
            headings=headings,
            body_lines=body_lines,
            id_base=id_base,
            max_heading_depth=cfg.max_heading_depth,
        )
        for rec in section_records:
            sec_text = _section_text(body_lines, rec.start_line, rec.end_line)
            if not sec_text.strip():
                continue
            anchors.append(
                _make_anchor(
                    anchor_id=rec.section_id,
                    anchor_type=AnchorType.SECTION,
                    path=rel_path,
                    heading_path=rec.heading_path,
                    content_text=sec_text,
                )
            )

    # ── §15.2: build CODE_IN_DOC anchors ─────────────────────────────────────
    #
    # Per-section sibling slugs are batched and resolved through
    # disambiguate_siblings(suffix="@") to match the resolution semantics
    # of §15.3 — same collision-aware behavior used for section IDs above,
    # so a literal declaration named ``Config@2`` cannot accidentally
    # collide with the auto-suffix assigned to two ``Config`` siblings.

    # First pass: collect per-section (slug, fence) tuples in document order.
    by_section: dict[str, list[tuple[str, _FenceEntry]]] = {}
    code_in_doc_jobs: list[tuple[_SectionRecord, str, _FenceEntry]] = []

    for fence in fences:
        if not fence.info:
            # No language tag → skip (§15.2).
            continue

        decl_name = _find_declaration(fence.content)
        if decl_name is None:
            # Language tag present but no named declaration → skip (§15.2).
            continue

        parent = _find_parent_section(fence.start_line, section_records)
        if parent is None:
            # Fence precedes all anchor-level headings → no parent; skip.
            continue

        decl_slug = slugify(decl_name)
        by_section.setdefault(parent.section_id, []).append((decl_slug, fence))
        code_in_doc_jobs.append((parent, decl_slug, fence))

    # Second pass: per-section collision resolution.
    resolved_per_section: dict[str, list[str]] = {
        sid: disambiguate_siblings([slug for slug, _ in entries], suffix="@")
        for sid, entries in by_section.items()
    }
    # Index into each section's resolved list as we walk jobs in order.
    cursor: dict[str, int] = dict.fromkeys(by_section, 0)

    for parent, _decl_slug, fence in code_in_doc_jobs:
        sid = parent.section_id
        idx = cursor[sid]
        cursor[sid] += 1
        resolved_slug = resolved_per_section[sid][idx]
        # derive_code_in_doc_id slugifies the declaration name internally,
        # so we pass our already-resolved slug verbatim and reconstruct the
        # exact format ourselves to match.  (Calling derive_code_in_doc_id
        # would re-slugify and lose the @N suffix.)
        cid = f"{parent.section_id}::{resolved_slug}"

        code_text = canonicalize_content(fence.content)
        if not code_text.strip():
            continue

        anchors.append(
            _make_anchor(
                anchor_id=cid,
                anchor_type=AnchorType.CODE_IN_DOC,
                path=rel_path,
                heading_path=parent.heading_path if parent.heading_path else None,
                content_text=code_text,
            )
        )

    # SR3-5: surface frontmatter-only / empty-body files via a debug log
    # so an operator can correlate "files_processed counted this but 0
    # anchors emitted" with a clear cause.  Stays at debug level to
    # avoid noise in normal indexing.
    if not anchors:
        logger.debug(
            "extract_markdown: %s produced no anchors (empty body or frontmatter-only)",
            path,
        )

    return anchors, diagnostics


def validate_markdown_file(
    path: Path,
    *,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
) -> int:
    """SR3-6: cheap §15.3 validation (duplicate scry-id) without full extraction.

    Used by the indexer to re-validate UNCHANGED markdown files on every
    incremental run — otherwise an invalid file indexed once would
    silently drop out of the validation_errors total on subsequent runs
    (the file's hash is unchanged, so it would be skipped).

    Returns:
        Count of duplicate scry-id occurrences after the first.  Files
        that can't be read, exceed the size limit, or fail UTF-8 decode
        are treated as 0 errors (the extractor will surface those
        separately).
    """
    try:
        raw_bytes = path.read_bytes()
    except OSError:
        return 0
    if len(raw_bytes) > max_file_size_bytes:
        return 0
    if raw_bytes.startswith(_UTF16_BE_BOM) or raw_bytes.startswith(_UTF16_LE_BOM):
        return 0
    try:
        raw_text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return 0
    canonical_text = canonicalize_content(raw_text)
    _file_fm, body = parse_frontmatter(canonical_text)
    if not body.strip():
        return 0
    tokens = _MD.parse(body)
    return _check_duplicate_scry_ids(tokens, path)
