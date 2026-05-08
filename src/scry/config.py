"""Config loader for scry.

Implements loading and working with ``.scry/config.yaml`` per DESIGN.md §6,
§6.1, §6.2, and provides helpers for path classification, glob matching, and
per-file frontmatter parsing used by the indexer and extractor workstreams.

References
----------
DESIGN.md §6    — Configuration shape and classify list
DESIGN.md §6.1  — Per-file frontmatter overrides; safety-exclude precedence
DESIGN.md §6.2  — LSP binary allowlist
DESIGN.md §13   — config_hash canonicalization (question 12)
DESIGN.md §15.4 — File-level filter edge cases

Public surface
--------------
load_config          — Parse and validate ``.scry/config.yaml`` → Config
compute_config_hash  — Stable SHA-256 over canonicalized config (§13 q.12)
parse_frontmatter    — Extract YAML frontmatter from file text (§6.1, §15.4)
matches_globs        — Shell glob matching with ``**`` support
classify_path        — First-match-wins classify list lookup (§6)
is_safety_excluded   — Hard safety-exclude check (§6.1)
should_index         — Combined include/exclude/frontmatter decision
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from scry.models import ClassifyEntry, Config, Frontmatter

__all__ = [
    "ConfigError",
    "classify_path",
    "compute_config_hash",
    "is_safety_excluded",
    "load_config",
    "matches_globs",
    "parse_frontmatter",
    "should_index",
]

logger = logging.getLogger(__name__)


# ─── Exception ────────────────────────────────────────────────────────


class ConfigError(Exception):
    """Raised for missing, malformed, or schema-invalid ``.scry/config.yaml``."""


# ─── load_config ──────────────────────────────────────────────────────


def load_config(repo_root: Path) -> Config:
    """Load ``.scry/config.yaml`` from *repo_root* and return a validated Config.

    Validation is performed by the :class:`~scry.models.Config` Pydantic model
    (DESIGN.md §6).  All three failure modes raise :class:`ConfigError` with an
    actionable message.

    Args:
        repo_root: Absolute path to the repository root that contains
            ``.scry/config.yaml``.

    Returns:
        A fully-validated :class:`~scry.models.Config` instance.

    Raises:
        ConfigError: If the config file is missing, the YAML is malformed,
            or the parsed mapping fails Pydantic schema validation.
    """
    config_path = repo_root / ".scry" / "config.yaml"
    if not config_path.exists():
        raise ConfigError(
            f"No .scry/config.yaml found at {config_path}; run `scry init` to create one"
        )

    try:
        raw = config_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Failed to read .scry/config.yaml: {exc}") from exc

    try:
        data: Any = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse .scry/config.yaml: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError("Failed to parse .scry/config.yaml: top-level value must be a mapping")

    try:
        return Config(**data)
    except ValidationError as exc:
        raise ConfigError(f"Config validation error: {exc}") from exc


# ─── compute_config_hash ──────────────────────────────────────────────


def _canonicalize(obj: Any) -> Any:
    """Recursively sort dict keys so JSON serialisation is deterministic."""
    if isinstance(obj, dict):
        return {k: _canonicalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_canonicalize(item) for item in obj]
    return obj


def compute_config_hash(config: Config) -> str:
    """Return a stable SHA-256 over the parsed-and-canonicalized config.

    Per DESIGN.md §13 question 12: the hash is computed over the
    Pydantic-validated, JSON-serialised representation (not raw YAML bytes).
    Sorted keys and a minimal JSON encoder ensure that cosmetic YAML
    reformatting (different key order, added comments, trailing newlines)
    produces **the same hash** and therefore does NOT trigger a full reindex.

    Args:
        config: A validated :class:`~scry.models.Config` instance.

    Returns:
        ``'sha256:<64-hex-digits>'``.
    """
    raw = config.model_dump(mode="json")
    canonical = _canonicalize(raw)
    serialised = json.dumps(canonical, separators=(",", ":"), sort_keys=True)
    digest = hashlib.sha256(serialised.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


# ─── parse_frontmatter ────────────────────────────────────────────────

# Matches a YAML frontmatter block at the very beginning of the file.
# Group 1 captures the YAML content between the two --- delimiters.
_FM_RE = re.compile(r"^---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|$)", re.DOTALL)


def parse_frontmatter(text: str) -> tuple[Frontmatter | None, str]:
    """Parse top-of-file YAML frontmatter (``---``/yaml/``---``).

    Returns ``(Frontmatter, body)`` when a valid ``scry:`` block is found.
    Returns ``(None, original_text)`` in every failure path and
    ``(None, body)`` when frontmatter is valid YAML but lacks a ``scry:`` key.
    A :mod:`logging` ``WARNING`` is emitted when YAML is malformed (§15.4).

    Args:
        text: Full file content as a string (any line-ending style).

    Returns:
        ``(Frontmatter | None, body_str)`` where *body_str* is the text
        that follows the frontmatter delimiter (or the full *text* when no
        block matched or parsing failed).

    Note:
        A leading UTF-8 BOM (``\\ufeff``) is stripped before delimiter
        matching so that files saved by Windows Notepad (which writes
        UTF-8 with BOM by default) are parsed correctly.  Matches the
        BOM-handling convention in :func:`scry.anchor_id.canonicalize_content`.
    """
    if text.startswith("\ufeff"):
        text = text.removeprefix("\ufeff")

    match = _FM_RE.match(text)
    if match is None:
        return None, text

    fm_raw = match.group(1)
    body = text[match.end() :]

    try:
        fm_data: Any = yaml.safe_load(fm_raw)
    except yaml.YAMLError as exc:
        logger.warning("Malformed YAML frontmatter (proceeding without it): %s", exc)
        return None, text

    if not isinstance(fm_data, dict):
        logger.warning(
            "Malformed frontmatter: top-level must be a mapping, got %s",
            type(fm_data).__name__,
        )
        return None, text

    scry_block = fm_data.get("scry")
    if scry_block is None:
        # Valid YAML frontmatter but no scry: key — return body without the block.
        return None, body

    if not isinstance(scry_block, dict):
        logger.warning(
            "Malformed frontmatter: 'scry' must be a mapping, got %s",
            type(scry_block).__name__,
        )
        return None, text

    try:
        frontmatter = Frontmatter(**scry_block)
    except (ValidationError, TypeError) as exc:
        logger.warning("Invalid scry frontmatter fields (proceeding without it): %s", exc)
        return None, text

    return frontmatter, body


# ─── matches_globs ────────────────────────────────────────────────────


def _glob_to_re(pattern: str) -> re.Pattern[str]:
    """Compile a shell glob pattern (with ``**`` support) to a :mod:`re` pattern.

    Conversion rules:

    * ``**`` → matches any sequence of characters **including** ``/`` (and
      when ``**/`` is seen, the trailing ``/`` is made optional so that
      ``**/*.md`` also matches top-level files like ``README.md``).
    * ``*``  → matches any sequence of characters **excluding** ``/``.
    * ``?``  → matches a single character **excluding** ``/``.
    * All other characters are regex-escaped and matched literally.
    """
    parts: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == "*" and i + 1 < n and pattern[i + 1] == "*":
            # Double-star: match anything including directory separators.
            parts.append(".*")
            i += 2
            # Consume optional '/' so '**/' makes the slash optional,
            # allowing '**/*.md' to match top-level files.
            if i < n and pattern[i] == "/":
                parts.append("/?")
                i += 1
        elif c == "*":
            # Single-star: match anything except '/'.
            parts.append("[^/]*")
            i += 1
        elif c == "?":
            parts.append("[^/]")
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    return re.compile("^" + "".join(parts) + "$")


def matches_globs(path: str, globs: list[str]) -> bool:
    """Return ``True`` if *path* matches any pattern in *globs*.

    Uses shell-style glob semantics extended with ``**`` (cross-directory
    wildcard).  Both *path* and each pattern are normalised to forward-slash
    form before comparison so Windows paths work transparently.

    Args:
        path:  Repo-relative file path (forward- or back-slash separated).
        globs: List of glob patterns; may be empty.

    Returns:
        ``True`` if *any* pattern matches; ``False`` when the list is empty
        or no pattern matches.
    """
    if not globs:
        return False
    norm = path.replace("\\", "/")
    for pattern in globs:
        norm_pat = pattern.replace("\\", "/")
        # Fast path: no wildcards — direct equality check.
        if not any(c in norm_pat for c in "*?["):
            if norm == norm_pat:
                return True
            continue
        # Patterns containing '**' need our regex engine.
        if "**" in norm_pat:
            if _glob_to_re(norm_pat).match(norm):
                return True
        else:
            # Plain fnmatch handles single-star and '?' patterns fine
            # because fnmatch's '*' already matches '/'.  Use fnmatchcase
            # to keep semantics consistent across OSes — fnmatch.fnmatch
            # would call os.path.normcase which is case-INSENSITIVE on
            # Windows, causing a config rule like ``docs/*.md`` to behave
            # differently on Windows vs. Linux CI.
            if fnmatch.fnmatchcase(norm, norm_pat):
                return True
    return False


# ─── classify_path ────────────────────────────────────────────────────


def classify_path(path: str, classify_entries: list[ClassifyEntry]) -> str | None:
    """Return the first matching classify type (``'spec'`` or ``'doc'``).

    Per DESIGN.md §6: the classify list is ordered; **first-match-wins**.
    A file that matches no entry returns ``None`` and is excluded from
    indexing (the caller is responsible for surfacing a warning).

    Args:
        path:             Repo-relative file path.
        classify_entries: Ordered classify list from :attr:`Config.classify`.

    Returns:
        ``'spec'``, ``'doc'``, or ``None`` if no entry matches.
    """
    for entry in classify_entries:
        if matches_globs(path, [entry.glob]):
            return entry.type
    return None


# ─── is_safety_excluded ───────────────────────────────────────────────


def is_safety_excluded(path: str, exclude_globs: list[str]) -> bool:
    """Return ``True`` if *path* matches any hard safety-exclude glob.

    Per DESIGN.md §6.1: safety excludes always win.  Even frontmatter
    ``skip: false`` cannot un-exclude a path that matches the global
    ``exclude:`` list (e.g. ``secrets/**``).

    Args:
        path:          Repo-relative file path.
        exclude_globs: Global ``exclude:`` list from :attr:`Config.exclude`.

    Returns:
        ``True`` if *any* exclude glob matches.
    """
    return matches_globs(path, exclude_globs)


# ─── should_index ─────────────────────────────────────────────────────


def should_index(
    path: str,
    frontmatter: Frontmatter | None,
    include_globs: list[str],
    exclude_globs: list[str],
) -> bool:
    """Decide whether a file should be indexed.

    Precedence (highest first), per DESIGN.md §6 and §6.1:

    1. **Hard safety excludes** — ``exclude:`` glob matches always win,
       even when frontmatter has ``skip: false``.
    2. **Frontmatter ``skip: true``** — opts out even when an ``include:``
       glob matches.
    3. **Include glob** — path must match at least one ``include:`` glob.
    4. All conditions satisfied → index the file.

    Args:
        path:          Repo-relative file path.
        frontmatter:   Parsed per-file frontmatter, or ``None`` if absent.
        include_globs: Global ``include:`` list from :attr:`Config.include`.
        exclude_globs: Global ``exclude:`` list from :attr:`Config.exclude`.

    Returns:
        ``True`` only when the file should be indexed.
    """
    # 1. Hard safety exclude — wins over everything including frontmatter.
    if is_safety_excluded(path, exclude_globs):
        return False
    # 2. Frontmatter skip:true — opts out even when include matches.
    if frontmatter is not None and frontmatter.skip:
        return False
    # 3. Must match at least one include glob.
    return matches_globs(path, include_globs)
