"""Pre-scanned repository index for fast claim verification.

Builds indexes of symbols, string literals, and file paths once,
then verifiers query the index instead of re-scanning per claim.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SymbolDef:
    """A symbol definition found in code."""

    name: str
    kind: str  # "function", "class", "variable", "method"
    file_path: str  # repo-relative
    line: int
    snippet: str = ""


@dataclass
class StringLiteral:
    """A string literal found in code."""

    value: str
    file_path: str
    line: int
    context: str = ""  # the full line


@dataclass
class RouteDef:
    """An API route definition."""

    path: str
    method: str | None
    file_path: str
    line: int
    snippet: str = ""


@dataclass
class EnumDef:
    """An enum class definition with its members."""

    name: str
    file_path: str
    line: int
    members: list[str] = field(default_factory=list)


# ───── Fuzzy matching helpers ────────────────────────────────────────

_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def _normalize_symbol(name: str) -> str:
    """Normalize a symbol to lowercase with underscore separators."""
    # Split camelCase
    parts = _CAMEL_RE.sub("_", name).lower().replace("-", "_").split("_")
    return "_".join(p for p in parts if p)


def _trigrams(s: str) -> set[str]:
    """Generate trigrams from a string."""
    s = s.lower()
    return {s[i : i + 3] for i in range(max(0, len(s) - 2))}


def _symbol_similarity(query: str, query_normalized: str, candidate: str) -> float:
    """Score similarity between query symbol and a candidate.

    Returns 0.0–1.0. Uses multiple strategies:
    - Exact case-insensitive match → 1.0
    - Normalized form match → 0.95
    - Trigram similarity
    """
    cand_lower = candidate.lower()
    query_lower = query.lower()

    # Exact case-insensitive
    if query_lower == cand_lower:
        return 1.0

    # Normalized match (camelCase vs snake_case)
    cand_normalized = _normalize_symbol(candidate)
    if query_normalized == cand_normalized:
        return 0.95

    # One contains the other
    if query_lower in cand_lower or cand_lower in query_lower:
        ratio = min(len(query_lower), len(cand_lower)) / max(len(query_lower), len(cand_lower))
        return 0.6 + ratio * 0.3

    # Trigram similarity
    q_tri = _trigrams(query)
    c_tri = _trigrams(candidate)
    if not q_tri or not c_tri:
        return 0.0
    intersection = len(q_tri & c_tri)
    union = len(q_tri | c_tri)
    return intersection / union if union else 0.0


# ─────────────────────────────────────────────────────────────────────


@dataclass
class RepoIndex:
    """Pre-scanned repository index for fast verification lookups."""

    symbols: dict[str, list[SymbolDef]] = field(default_factory=dict)
    strings: dict[str, list[StringLiteral]] = field(default_factory=dict)
    routes: list[RouteDef] = field(default_factory=list)
    enums: dict[str, EnumDef] = field(default_factory=dict)
    files: set[str] = field(default_factory=set)  # all repo-relative paths

    def lookup_symbol(self, name: str) -> list[SymbolDef]:
        """Find symbol definitions by name (exact or last-component match)."""
        results = self.symbols.get(name, [])
        if not results and "." in name:
            last = name.rsplit(".", 1)[-1]
            results = self.symbols.get(last, [])
        return results

    def lookup_string(self, value: str) -> list[StringLiteral]:
        """Find string literal occurrences."""
        return self.strings.get(value, [])

    def find_similar_symbols(self, name: str, max_results: int = 5) -> list[str]:
        """Find symbols with similar names using fuzzy matching.

        Uses multiple strategies: prefix match, case-normalized match,
        snake/camel transforms, and trigram similarity.
        """
        bare = name.rstrip("()").split(".")[-1]
        if len(bare) < 3:
            return []

        # Normalize for comparison
        normalized = _normalize_symbol(bare)
        candidates: list[tuple[float, str]] = []

        for sym_name in self.symbols:
            if sym_name == bare:
                continue
            score = _symbol_similarity(bare, normalized, sym_name)
            if score > 0.5:
                candidates.append((score, sym_name))

        candidates.sort(key=lambda x: -x[0])
        return [c[1] for c in candidates[:max_results]]

    def find_similar_strings(self, value: str, prefix_len: int = 8) -> list[str]:
        """Find string literals with similar prefix."""
        prefix = value[:prefix_len].upper() if len(value) >= prefix_len else value.upper()
        similar: list[str] = []
        for s in self.strings:
            if s.upper().startswith(prefix) and s != value:
                similar.append(s)
                if len(similar) >= 5:
                    break
        return similar

    def file_exists(self, path: str) -> bool:
        """Check if a repo-relative file path exists."""
        # Normalize separators
        normalized = path.replace("\\", "/")
        return normalized in self.files


# Route decorator pattern
_ROUTE_DEC_RE = re.compile(
    r'@\w*\.(?P<method>get|post|put|patch|delete|head|options|websocket|api_route)\s*\(\s*["\'](?P<path>[^"\']+)["\']',
    re.IGNORECASE,
)


def build_index(
    repo_root: Path,
    *,
    glob_patterns: list[str] | None = None,
    exclude_dirs: frozenset[str] | None = None,
) -> RepoIndex:
    """Scan the repository and build a searchable index.

    Args:
        repo_root: Repository root directory.
        glob_patterns: File patterns to scan. Defaults to ["**/*.py"].
        exclude_dirs: Directory names to skip.

    Returns:
        Pre-built RepoIndex for fast lookups.
    """
    if glob_patterns is None:
        glob_patterns = ["**/*.py"]
    if exclude_dirs is None:
        exclude_dirs = frozenset({
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
            ".scry",
        })

    index = RepoIndex()

    # Collect all file paths (for file_path verifier)
    for fp in repo_root.rglob("*"):
        if any(d in fp.parts for d in exclude_dirs):
            continue
        if fp.is_file():
            index.files.add(str(fp.relative_to(repo_root)).replace("\\", "/"))

    # Scan Python files for symbols, strings, routes, enums
    for pattern in glob_patterns:
        for fp in repo_root.glob(pattern):
            if any(d in fp.parts for d in exclude_dirs):
                continue
            if not fp.is_file():
                continue

            rel_path = str(fp.relative_to(repo_root)).replace("\\", "/")

            try:
                source = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            # AST-based extraction for Python files
            if fp.suffix == ".py":
                _extract_python(source, rel_path, index)

            # Regex-based extraction for routes and strings (all files)
            _extract_strings_and_routes(source, rel_path, index)

    return index


def _extract_python(source: str, rel_path: str, index: RepoIndex) -> None:
    """Extract symbols and enums from Python source via AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return

    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            child.parent = parent  # type: ignore[attr-defined]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if isinstance(getattr(node, "parent", None), ast.ClassDef):
                continue
            _add_symbol(index, node.name, "function", rel_path, node.lineno, source)
        elif isinstance(node, ast.ClassDef):
            _add_symbol(index, node.name, "class", rel_path, node.lineno, source)

            # Check if it's an enum
            is_enum = any(
                (isinstance(b, ast.Name) and b.id in ("Enum", "StrEnum", "IntEnum", "Flag"))
                or (isinstance(b, ast.Attribute) and b.attr in ("Enum", "StrEnum", "IntEnum", "Flag"))
                for b in node.bases
            )
            if is_enum:
                members = []
                for item in node.body:
                    if isinstance(item, ast.Assign):
                        for t in item.targets:
                            if isinstance(t, ast.Name):
                                members.append(t.id)
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        if item.value is not None:
                            members.append(item.target.id)
                if members:
                    index.enums[node.name] = EnumDef(
                        name=node.name,
                        file_path=rel_path,
                        line=node.lineno,
                        members=members,
                    )

            # Extract methods and class attributes (Pydantic fields, etc.)
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef):
                    _add_symbol(index, item.name, "method", rel_path, item.lineno, source)
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    # Annotated class attribute: field_name: Type = ...
                    _add_symbol(index, item.target.id, "attribute", rel_path, item.lineno, source)
                elif isinstance(item, ast.Assign):
                    for t in item.targets:
                        if isinstance(t, ast.Name) and not t.id.startswith("__"):
                            _add_symbol(index, t.id, "attribute", rel_path, item.lineno, source)

        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and not t.id.startswith("_"):
                    _add_symbol(index, t.id, "variable", rel_path, node.lineno, source)


def _add_symbol(
    index: RepoIndex,
    name: str,
    kind: str,
    file_path: str,
    line: int,
    source: str,
) -> None:
    """Add a symbol to the index."""
    lines = source.splitlines()
    snippet = lines[line - 1].strip() if line <= len(lines) else ""
    sym = SymbolDef(name=name, kind=kind, file_path=file_path, line=line, snippet=snippet[:200])
    index.symbols.setdefault(name, []).append(sym)


def _extract_strings_and_routes(source: str, rel_path: str, index: RepoIndex) -> None:
    """Extract string literals and route definitions from source."""
    for i, line in enumerate(source.splitlines(), 1):
        # Route decorators
        for m in _ROUTE_DEC_RE.finditer(line):
            index.routes.append(RouteDef(
                path=m.group("path"),
                method=m.group("method").upper(),
                file_path=rel_path,
                line=i,
                snippet=line.strip()[:200],
            ))

        # String literals — env vars (ALL_CAPS) and identifiers (snake_case, camelCase)
        for m in re.finditer(r'["\']([A-Za-z][A-Za-z0-9_]{2,})["\']', line):
            val = m.group(1)
            lit = StringLiteral(value=val, file_path=rel_path, line=i, context=line.strip()[:200])
            index.strings.setdefault(val, []).append(lit)
