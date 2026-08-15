"""Deterministic claim extractor for markdown documentation.

Parses markdown into atomic, typed ``Claim`` objects using
regex/heuristic patterns — no LLM required.  Handles:

- Backticked symbols (function names, class names, variables)
- Numeric values with units (defaults, timeouts, limits)
- Environment variable names (ALL_CAPS_PATTERN)
- API route paths (/api/v1/...)
- File/directory paths (src/..., docs/...)
- Enum/set member lists
- Universal quantifiers (every, all, each, never, always, no)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from scry.claims.model import (
    Claim,
    ClaimType,
    DocSpan,
)

# ───── Patterns ──────────────────────────────────────────────────────

# Backticked code references — potential symbols
_BACKTICK_RE = re.compile(r"`([A-Za-z_]\w*(?:\.\w+)*(?:\(\))?)`")

# Numeric values with optional units
_NUMERIC_RE = re.compile(
    r"\b(\d[\d,]*(?:\.\d+)?)\s*"
    r"(seconds?|secs?|s\b|ms|milliseconds?|minutes?|mins?|hours?|hrs?|"
    r"bytes?|[KMGT]i?B|MiB|GiB|%|states?|members?|tools?|files?|images?|modules?|"
    r"retries?|attempts?|chars?|characters?)?"
)

# Environment variables — at least 3 chars, ALL_CAPS with underscores
_ENVVAR_RE = re.compile(r"\b([A-Z][A-Z0-9_]{2,})\b")

# API route paths
_ROUTE_RE = re.compile(r"(?:GET|POST|PUT|PATCH|DELETE|WS|WSS|HEAD|OPTIONS)?\s*`?(/(?:api|ws)/v\d+/[^\s`'\"]+)`?")

# Also catch routes without method prefix but in backticks
_ROUTE_BACKTICK_RE = re.compile(r"`(/(?:api|ws)/v\d+/[^\s`]+)`")

# File paths — repo-relative
_FILEPATH_RE = re.compile(
    r"`((?:src|docs|tests|scripts|infra|services|agents|harness|migrations|ui|evaluation)"
    r"/[^\s`]+)`"
)

# Universal quantifiers signaling coverage assertions
_UNIVERSAL_RE = re.compile(
    r"\b(every|all|each|always|never|no|none of|must always|must never)\b",
    re.IGNORECASE,
)

# Default/limit value patterns
_DEFAULT_RE = re.compile(
    r"(?:default(?:s to|s?:?\s*(?:is|=|of)?)|fallback(?:s to)?|"
    r"limit(?:s to|:?\s*(?:is)?)?|timeout(?:s to|:?\s*(?:is)?)?|"
    r"max(?:imum)?(?::\s*|\s+(?:is\s+)?)|"
    r"min(?:imum)?(?::\s*|\s+(?:is\s+)?))"
    r"\s*`?(\d[\d,]*(?:\.\d+)?)`?\s*"
    r"(seconds?|secs?|s\b|ms|minutes?|hours?|bytes?|[KMGT]i?B|MiB|GiB|%)?",
    re.IGNORECASE,
)

# Enum/count patterns: "N states", "N members", "N tools"
_COUNT_RE = re.compile(
    r"\b(\d+)\s+(states?|members?|tools?|types?|modes?|roles?|strategies?|"
    r"values?|fields?|methods?|endpoints?|tables?|columns?|modules?|images?|"
    r"providers?|plugins?|handlers?|verifiers?)\b",
    re.IGNORECASE,
)

# Common noise env vars to skip
_ENVVAR_NOISE = frozenset({
    "API", "URL", "GET", "POST", "PUT", "DELETE", "HEAD", "OPTIONS", "PATCH",
    "HTTP", "HTTPS", "TCP", "UDP", "DNS", "SSL", "TLS", "SSH", "SQL",
    "JSON", "YAML", "TOML", "CSV", "HTML", "XML", "JWT", "HMAC",
    "TRUE", "FALSE", "NULL", "NONE", "YES", "AND", "NOT",
    "RLS", "ACL", "ACR", "ACI", "ACA", "IDS", "PID", "WS", "WSS",
    "WAL", "FTS", "UUID", "SHA", "LLM", "MCP", "CLI", "AST",
    "NOTE", "TODO", "FIXME", "HACK", "XXX", "WIP",
    "TABLE", "INDEX", "WHERE", "FROM", "INTO", "SELECT",
    # Common prose/status words that happen to be ALL_CAPS in markdown
    "PULLING", "PUSHING", "RUNNING", "BUILDING", "DEPLOYING",
    "CANCELLED", "FAILED", "COMPLETED", "PENDING", "BLOCKED",
    "ENABLED", "DISABLED", "REQUIRED", "OPTIONAL", "DEFAULT",
    "EXAMPLE", "WARNING", "ERROR", "INFO", "DEBUG", "CRITICAL",
    "IMPORTANT", "RECOMMENDED", "DEPRECATED", "BREAKING",
    "MUST", "SHALL", "SHOULD", "WILL", "WOULD", "COULD", "MAY",
})

# Common noise symbols to skip — builtins, generic words, prose terms
_SYMBOL_NOISE = frozenset({
    # Python builtins/types
    "True", "False", "None", "self", "cls", "str", "int", "float",
    "bool", "list", "dict", "set", "tuple", "bytes", "type",
    "Optional", "Any", "Union", "Literal", "object", "super",
    "print", "len", "range", "enumerate", "isinstance", "hasattr",
    # Generic prose/log words often backticked
    "crash", "error", "warning", "info", "debug", "success", "failure",
    "running", "pending", "done", "cancelled", "failed", "completed",
    "pulling", "pushing", "building", "deploying", "testing",
    "enabled", "disabled", "active", "inactive", "ready", "blocked",
    "default", "custom", "manual", "automatic", "required", "optional",
    "example", "output", "input", "result", "response", "request",
    "config", "settings", "options", "params", "args", "kwargs",
    # File extensions / formats when standalone
    "json", "yaml", "toml", "csv", "html", "xml", "txt", "log", "md",
    # Infrastructure / Kubernetes / Docker terms
    "localhost", "emptyDir", "Dockerfile", "llmproxy",
    "hostPath", "configMap", "secretRef", "nodePort",
    # Common short tokens that are never real symbols
    "id", "ok", "on", "off", "up", "no", "yes",
})

# Bare filenames that are config/docs, not code symbols
_FILENAME_NOISE_RE = re.compile(
    r"^[a-z][\w.-]*\.(ya?ml|json|toml|md|txt|cfg|ini|env|lock|log|sh|bat|ps1)$",
    re.IGNORECASE,
)


def _is_noise_symbol(sym: str, line: str) -> bool:
    """Enhanced noise detection for symbol claims."""
    # Already in static noise set
    if sym in _SYMBOL_NOISE or sym.lower() in {s.lower() for s in _SYMBOL_NOISE}:
        return True
    # All-lowercase single word under 6 chars with no underscores — likely prose
    if len(sym) < 6 and sym.islower() and "_" not in sym and "." not in sym:
        return True
    # Looks like a bare filename (config.yaml, setup.py, etc.)
    if _FILENAME_NOISE_RE.match(sym):
        return True
    # Pure gerund/past-tense verbs (ends in ing/ed) — likely prose
    if sym.islower() and (sym.endswith("ing") or sym.endswith("ed")) and "_" not in sym:
        return True
    return False


# ───── Section parser ────────────────────────────────────────────────


@dataclass
class _Section:
    """A section of markdown with heading context."""

    heading_path: list[str]
    lines: list[str]
    start_line: int  # 1-based


def _parse_sections(text: str) -> list[_Section]:
    """Parse markdown into sections by heading hierarchy."""
    lines = text.splitlines()
    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    current_start = 1

    for i, line in enumerate(lines, 1):
        m = re.match(r"^(#{1,6})\s+(.+?)(?:\s*#+)?\s*$", line)
        if m:
            # Flush previous section
            if current_lines:
                sections.append(_Section(
                    heading_path=[h for _, h in heading_stack],
                    lines=current_lines,
                    start_line=current_start,
                ))
            level = len(m.group(1))
            title = m.group(2).strip()
            # Pop stack to correct level
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            current_lines = [line]
            current_start = i
        else:
            current_lines.append(line)

    # Flush last section
    if current_lines:
        sections.append(_Section(
            heading_path=[h for _, h in heading_stack],
            lines=current_lines,
            start_line=current_start,
        ))

    return sections


# ───── Extraction ────────────────────────────────────────────────────


def extract_claims(doc_path: str, text: str) -> list[Claim]:
    """Extract verifiable claims from markdown text.

    Args:
        doc_path: Repo-relative path to the document.
        text: Full markdown text content.

    Returns:
        List of extracted claims, sorted by document position.
    """
    claims: list[Claim] = []
    sections = _parse_sections(text)
    seen_ids: set[str] = set()

    # Track fenced code blocks for context
    in_fenced_block = False

    for section in sections:
        section_title = " > ".join(section.heading_path) if section.heading_path else ""
        section_text = "\n".join(section.lines)

        for line_offset, line in enumerate(section.lines):
            line_no = section.start_line + line_offset

            # Track fenced code blocks — skip content inside them
            stripped = line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fenced_block = not in_fenced_block
                continue
            if in_fenced_block:
                continue

            # Skip pure headings, empty lines, and HTML comments
            if not stripped or stripped.startswith("#") or stripped.startswith("<!--"):
                continue

            # --- Symbol existence claims ---
            for m in _BACKTICK_RE.finditer(line):
                sym = m.group(1)
                if len(sym) < 3 or sym.isdigit() or sym.isupper():
                    continue
                if _is_noise_symbol(sym, line):
                    continue
                # Context: if preceded by = or : it's likely a value, not a symbol
                pre_ctx = line[:m.start()]
                if pre_ctx.rstrip().endswith(("=", ":", "=>", "set to", "value")):
                    continue
                claim = Claim(
                    doc_path=doc_path,
                    section=section_title,
                    span=DocSpan(start_line=line_no, end_line=line_no),
                    raw_text=line.strip(),
                    claim_text=f"Symbol `{sym}` exists",
                    claim_type=ClaimType.SYMBOL_EXISTS,
                    subject=sym,
                    predicate="exists",
                    evidence_selectors=[sym],
                    confidence=0.7,
                )
                if claim.id not in seen_ids:
                    seen_ids.add(claim.id)
                    claims.append(claim)

            # --- Numeric/default value claims ---
            for m in _DEFAULT_RE.finditer(line):
                value_str = m.group(1).replace(",", "")
                unit = m.group(2) or ""
                try:
                    value = float(value_str) if "." in value_str else int(value_str)
                except ValueError:
                    continue
                claim = Claim(
                    doc_path=doc_path,
                    section=section_title,
                    span=DocSpan(start_line=line_no, end_line=line_no),
                    raw_text=line.strip(),
                    claim_text=f"Default/limit value is {value} {unit}".strip(),
                    claim_type=ClaimType.NUMERIC_VALUE,
                    subject=section_title,
                    predicate="equals",
                    object={"value": value, "unit": unit},
                    confidence=0.85,
                )
                if claim.id not in seen_ids:
                    seen_ids.add(claim.id)
                    claims.append(claim)

            # --- Count claims (N states, N tools, etc.) ---
            for m in _COUNT_RE.finditer(line):
                count = int(m.group(1))
                noun = m.group(2).rstrip("s")
                # Only extract if the count seems meaningful (> 1)
                if count <= 1:
                    continue
                claim = Claim(
                    doc_path=doc_path,
                    section=section_title,
                    span=DocSpan(start_line=line_no, end_line=line_no),
                    raw_text=line.strip(),
                    claim_text=f"There are {count} {noun}(s)",
                    claim_type=ClaimType.ENUM_COUNT,
                    subject=section_title,
                    predicate="count_equals",
                    object={"count": count, "noun": noun},
                    confidence=0.8,
                )
                if claim.id not in seen_ids:
                    seen_ids.add(claim.id)
                    claims.append(claim)

            # --- Environment variable claims ---
            for m in _ENVVAR_RE.finditer(line):
                var = m.group(1)
                if var in _ENVVAR_NOISE or len(var) < 4:
                    continue
                # Must be backticked or in an env-var-looking context
                backticked = f"`{var}`" in line
                env_context = any(kw in line.lower() for kw in ("env", "variable", "inject", "set", "export"))
                if not backticked and not env_context:
                    continue
                claim = Claim(
                    doc_path=doc_path,
                    section=section_title,
                    span=DocSpan(start_line=line_no, end_line=line_no),
                    raw_text=line.strip(),
                    claim_text=f"Environment variable `{var}` is used",
                    claim_type=ClaimType.ENV_VAR_NAME,
                    subject=var,
                    predicate="exists_in_code",
                    evidence_selectors=[var],
                    confidence=0.8,
                )
                if claim.id not in seen_ids:
                    seen_ids.add(claim.id)
                    claims.append(claim)

            # --- Route path claims ---
            for pattern in (_ROUTE_RE, _ROUTE_BACKTICK_RE):
                for m in pattern.finditer(line):
                    route = m.group(1) if pattern == _ROUTE_BACKTICK_RE else m.group(1)
                    # Extract method if present
                    method_match = re.match(r"(GET|POST|PUT|PATCH|DELETE|WS|WSS|HEAD|OPTIONS)\s", m.group(0))
                    method = method_match.group(1) if method_match else None
                    claim = Claim(
                        doc_path=doc_path,
                        section=section_title,
                        span=DocSpan(start_line=line_no, end_line=line_no),
                        raw_text=line.strip(),
                        claim_text=f"Route {method + ' ' if method else ''}{route} exists",
                        claim_type=ClaimType.ROUTE_PATH,
                        subject=route,
                        predicate="route_exists",
                        object={"path": route, "method": method},
                        confidence=0.9,
                    )
                    if claim.id not in seen_ids:
                        seen_ids.add(claim.id)
                        claims.append(claim)

            # --- File path claims ---
            for m in _FILEPATH_RE.finditer(line):
                fpath = m.group(1)
                claim = Claim(
                    doc_path=doc_path,
                    section=section_title,
                    span=DocSpan(start_line=line_no, end_line=line_no),
                    raw_text=line.strip(),
                    claim_text=f"File `{fpath}` exists",
                    claim_type=ClaimType.FILE_PATH,
                    subject=fpath,
                    predicate="file_exists",
                    confidence=0.95,
                )
                if claim.id not in seen_ids:
                    seen_ids.add(claim.id)
                    claims.append(claim)

            # --- Coverage assertion claims ---
            for m in _UNIVERSAL_RE.finditer(line):
                quantifier = m.group(1).lower()
                claim = Claim(
                    doc_path=doc_path,
                    section=section_title,
                    span=DocSpan(start_line=line_no, end_line=line_no),
                    raw_text=line.strip(),
                    claim_text=f"Universal assertion: {line.strip()[:120]}",
                    claim_type=ClaimType.COVERAGE_ASSERTION,
                    subject=section_title,
                    predicate=quantifier,
                    confidence=0.4,
                    needs_llm=True,
                )
                if claim.id not in seen_ids:
                    seen_ids.add(claim.id)
                    claims.append(claim)

    # Sort by document position
    claims.sort(key=lambda c: (c.span.start_line, c.span.end_line))
    return claims


def extract_claims_from_file(doc_path: str, repo_root: Path | None = None) -> list[Claim]:
    """Extract claims from a markdown file on disk.

    Args:
        doc_path: Repo-relative path to the document.
        repo_root: Repository root directory.  If None, uses cwd.

    Returns:
        List of extracted claims.
    """
    root = repo_root or Path.cwd()
    full_path = root / doc_path
    if not full_path.is_file():
        return []
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    return extract_claims(doc_path, text)
