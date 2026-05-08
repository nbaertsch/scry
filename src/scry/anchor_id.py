"""Anchor ID derivation, slugification, and content-hash canonicalization.

Pure functions for workstream W1c — no I/O, no Pydantic model construction.

DESIGN.md references
--------------------
§3.2  — Anchor identity (Layer 1 + Layer 2): ID formats.
§3.3  — Inline rebase on re-index: fingerprints & SimHash usage.
§5.4  — Content-hash canonicalization steps.
§15.3 — ID derivation edge-case table.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Literal

from simhash import Simhash

__all__ = [
    "canonicalize_content",
    "content_hash",
    "derive_anonymous_code_in_doc_id",
    "derive_code_id",
    "derive_code_in_doc_id",
    "derive_section_id",
    "disambiguate_siblings",
    "fingerprint_simhash",
    "parse_html_comment_id",
    "short_content_hash",
    "slugify",
]

# Pre-compiled patterns.
_NONALPHA_RE = re.compile(r"[^a-z0-9]+")
_SCRY_ID_RE = re.compile(r"<!--\s*scry-id:\s*(.*?)\s*-->")


# ──────────────────────────────────────────────────────────────────────
# Content canonicalization (§5.4)
# ──────────────────────────────────────────────────────────────────────


def canonicalize_content(text: str) -> str:
    """Apply §5.4 canonicalization.

    Steps (in order):

    1. Strip UTF-8 BOM (U+FEFF) if present.
    2. Normalize line endings: ``\\r\\n`` → ``\\n``, then bare ``\\r`` → ``\\n``.
    3. Trim trailing whitespace per line (``line.rstrip()``).
    4. Collapse multiple trailing newlines at EOF to a single ``\\n``.
       If the original had no trailing newline, that state is preserved.

    Deliberately **not** applied:

    - No paragraph reflow.
    - No collapse of internal whitespace runs.
    - No Unicode NFC normalization.
    """
    # 1. Strip UTF-8 BOM.
    text = text.removeprefix("\ufeff")
    # 2. Normalize line endings (CRLF first, then bare CR).
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    # 3. Trim trailing whitespace per line.
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    # 4. Collapse trailing newlines to at most one.
    ends_with_newline = text.endswith("\n")
    text = text.rstrip("\n")
    if ends_with_newline:
        text += "\n"
    return text


def content_hash(text: str) -> str:
    """SHA-256 over ``canonicalize_content(text)``.

    Returns ``'sha256:<64 lowercase hex digits>'``, matching the
    ``ContentHash`` constraint in DESIGN.md §5.4 and ``models.ContentHash``.
    """
    canonical = canonicalize_content(text)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def short_content_hash(text: str, *, length: int = 8) -> str:
    """First *length* hex characters of SHA-256 over canonicalized content.

    Default length is 8, matching the anonymous block-ID format (§15.3):
    ``block-<short-content-hash>``.
    """
    canonical = canonicalize_content(text)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return digest[:length]


def fingerprint_simhash(text: str) -> int:
    """64-bit SimHash of canonicalized content (§3.3).

    Computed via the ``simhash`` package (≥ 2.1). Used for fuzzy
    fingerprint matching during inline rebase. Deterministic: identical
    text always produces the same integer.
    """
    canonical = canonicalize_content(text)
    return int(Simhash(canonical).value)


# ──────────────────────────────────────────────────────────────────────
# Slugification (§15.3)
# ──────────────────────────────────────────────────────────────────────


def slugify(text: str, *, fallback_prefix: str = "section") -> str:
    """Slug a heading or symbol name (§15.3).

    Rules (applied in order):

    1. Decompose via NFKD and discard non-ASCII code points — strips
       combining diacritics (``Café`` → ``Cafe``) and causes emoji and
       other pure non-ASCII sequences to vanish.
    2. Lowercase the resulting ASCII string.
    3. Replace every run of non-alphanumeric characters with ``'-'``.
    4. Strip leading and trailing ``'-'``.
    5. If the result is empty (empty input, punctuation-only, emoji-only,
       or any all-non-ASCII input) → ``'<fallback_prefix>-<8-char hash>'``.
    """
    normalized = unicodedata.normalize("NFKD", text)
    ascii_only = "".join(c for c in normalized if ord(c) < 128)
    lowered = ascii_only.lower()
    slugged = _NONALPHA_RE.sub("-", lowered).strip("-")
    if not slugged:
        return f"{fallback_prefix}-{short_content_hash(text)}"
    return slugged


# ──────────────────────────────────────────────────────────────────────
# ID derivation (§3.2, §15.3)
# ──────────────────────────────────────────────────────────────────────


def derive_section_id(path: str, heading_path: list[str]) -> str:
    """Build the Layer 1 section anchor ID (§3.2).

    Format: ``<path>::<slug1>::<slug2>::...``

    Each element of *heading_path* is slugified independently.  Sibling
    disambiguation (``-N`` suffix for positional collision resolution) is
    a caller responsibility — apply :func:`disambiguate_siblings` to the
    batch of sibling IDs after derivation.

    Example::

        derive_section_id(
            "docs/POLICY_ENGINE.md",
            ["Policy Engine", "Rule Structure"],
        )
        # → "docs/POLICY_ENGINE.md::policy-engine::rule-structure"
    """
    parts = [path] + [slugify(h) for h in heading_path]
    return "::".join(parts)


def derive_code_in_doc_id(section_id: str, declaration_name: str) -> str:
    """Build the Layer 1 code-in-doc anchor ID (§3.2).

    Format: ``<section-id>::<slugified-declaration-name>``

    Sibling disambiguation (``@N`` suffix for same-name collisions within
    the same section) is a caller responsibility — apply
    :func:`disambiguate_siblings` with ``suffix="@"`` to the batch of
    sibling IDs after derivation.
    """
    return f"{section_id}::{slugify(declaration_name)}"


def derive_code_id(
    path: str,
    symbol_path: list[str],
    *,
    signature_hash: str | None = None,
) -> str:
    """Build the Layer 1 code anchor ID (§3.2).

    Format: ``<path>:<sym1>.<sym2>[.<...>][@<sig-hash[:6]>]``

    *signature_hash* is supplied only for overloaded symbols.  When
    present, the first six hex characters are appended as ``@<hash6>``
    to disambiguate overloads sharing the same qualified name.

    Examples::

        derive_code_id("python/policy.py", ["OuterClass", "method"])
        # → "python/policy.py:OuterClass.method"

        derive_code_id("api.ts", ["f"], signature_hash="abc123def456")
        # → "api.ts:f@abc123"

    Raises:
        ValueError: if *symbol_path* is empty (would produce a
            malformed ``<path>:`` ID).
    """
    if not symbol_path:
        raise ValueError("derive_code_id requires a non-empty symbol_path")
    qualified = ".".join(symbol_path)
    base = f"{path}:{qualified}"
    if signature_hash is not None:
        base += f"@{signature_hash[:6]}"
    return base


def derive_anonymous_code_in_doc_id(section_id: str, body: str) -> str:
    r"""Build the ID for an anonymous fenced code block (§15.3 row 4).

    Anonymous blocks are fenced code blocks that have no extractable
    declaration name (e.g. ``\`\`\`python\nprint("hi")\n\`\`\```).
    Per §15.3 they get an ID of the form
    ``<section-id>::block-<short-content-hash>`` where the hash is an
    8-hex prefix of the body's canonicalized SHA-256.  This is the
    canonical helper for that pattern; callers must NOT reuse
    :func:`slugify`'s ``<fallback_prefix>-<hash>`` form because the
    spec mandates the literal ``block-`` prefix.

    The body should be passed BEFORE canonicalization — the helper
    canonicalizes internally for hash stability.

    Example::

        derive_anonymous_code_in_doc_id(
            "docs/spec.md::examples", "print('hi')"
        )
        # → "docs/spec.md::examples::block-2c26b46b"
    """
    return f"{section_id}::block-{short_content_hash(body)}"


def disambiguate_siblings(
    ids: list[str],
    *,
    suffix: Literal["-", "@"] = "-",
) -> list[str]:
    """Assign deterministic positional suffixes to duplicate IDs.

    Returns a same-length list. The first occurrence of each ID is kept
    bare; the second gets ``<suffix>2``, the third ``<suffix>3``, etc.

    The function GUARANTEES uniqueness in the output even when the
    input already contains a value matching the suffixed form: in
    that case, the assigned counter is incremented past the colliding
    value until a free slot is found.  This protects callers from a
    subtle source of duplicate ``AnchorId``s when a literal heading
    such as "Examples 2" coexists with two siblings named "Examples"
    (which would slugify to ``examples-2``).

    Args:
        ids:    Ordered list of (possibly duplicate) IDs reflecting
                document order.
        suffix: ``"-"`` for section IDs (default); ``"@"`` for
                code-in-doc IDs per §15.3.

    Examples::

        disambiguate_siblings(["examples", "intro", "examples", "examples"])
        # → ["examples", "intro", "examples-2", "examples-3"]

        disambiguate_siblings(["Config", "Config"], suffix="@")
        # → ["Config", "Config@2"]

        # Pre-existing suffix form is detected and skipped:
        disambiguate_siblings(["examples", "examples-2", "examples"])
        # → ["examples", "examples-2", "examples-3"]
    """
    # Pre-compute the multiset of inputs so we can detect "the bare ID
    # already appears more than once AND a suffixed variant also exists"
    # situations without re-scanning.
    input_set: set[str] = set(ids)
    counts: dict[str, int] = {}
    emitted: set[str] = set()
    result: list[str] = []
    for raw_id in ids:
        n = counts.get(raw_id, 0) + 1
        if n == 1 and raw_id not in emitted:
            chosen = raw_id
        else:
            # Find the smallest counter ≥ n whose suffixed form does
            # not collide with another input ID OR a previously emitted
            # ID. n is monotonic per raw_id so we never go backwards.
            counter = n
            while True:
                candidate = f"{raw_id}{suffix}{counter}"
                if candidate not in input_set and candidate not in emitted:
                    break
                counter += 1
            chosen = candidate
            # Persist the bumped counter so the *next* occurrence of
            # raw_id starts past the collision.
            n = counter
        counts[raw_id] = n
        emitted.add(chosen)
        result.append(chosen)
    return result


# ──────────────────────────────────────────────────────────────────────
# HTML-comment ID extraction (§3.2 escape hatch)
# ──────────────────────────────────────────────────────────────────────


def parse_html_comment_id(line: str) -> str | None:
    """Extract the user-pinned slug from a ``<!-- scry-id: ... -->`` comment.

    The returned string is the EXACT text between ``scry-id:`` and ``-->``
    with only surrounding whitespace trimmed — no case-folding, no
    slugification — preserving round-trip fidelity.  The caller decides
    whether to slugify the returned value.

    Returns ``None`` when the pattern is absent on *line*.

    Examples::

        parse_html_comment_id("<!-- scry-id: rule-structure -->")
        # → "rule-structure"

        parse_html_comment_id("## Heading")
        # → None
    """
    m = _SCRY_ID_RE.search(line)
    if m:
        slug = m.group(1)
        return slug if slug else None
    return None
