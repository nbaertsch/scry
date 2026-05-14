"""Tree-sitter code symbol extraction for scry.

Implements workstream W1b.  DESIGN.md references: §3.1, §3.2, §5.4, §6, §15.3.

Public API
----------
    extract_code_symbols(path, repo_root, *, language, config) -> list[Anchor]

Supported languages (Wave 1 scope)
------------------------------------
* **python**         - ``function_definition``, ``class_definition``,
                       ``decorated_definition`` (inner name unwrapped, §15.3)
* **typescript/tsx** - ``function_declaration``, ``class_declaration``,
                       ``interface_declaration``, ``type_alias_declaration``;
                       ``function_signature`` used internally for overload
                       detection (§15.3); not exposed in default config list
* **zig**            - ``FnProto``, ``ContainerDecl``

Anchor ID format (§3.2 Layer 1)
---------------------------------
* Top-level symbols  : ``<repo-relative-path>:<symbol_name>``
* Nested methods     : ``<path>:<OuterClass>.<method_name>``
* TS overloads       : ``<path>:<fn_name>@<sig-hash[:6]>``     (§15.3, via
                       :func:`scry.anchor_id.derive_code_id`)
* Same-name collision: first bare, subsequent ``<name>@2``, ``<name>@3`` (§15.3,
                       via :func:`scry.anchor_id.disambiguate_siblings` semantics)

``granularity: "file"`` (DESIGN.md §13, open question #13)
-----------------------------------------------------------
Wave 1 does **not** implement file-level anchors.  If the config sets
``granularity: "file"``, a warning is logged and the extractor falls back to
``"symbol"`` mode.  Wave N will implement ``<path>``-only IDs and whole-file
content hashes when this option is finalised.

``transitive_hash_status``
--------------------------
All Wave 1 code anchors carry ``TransitiveHashStatus.LSP_UNAVAILABLE``.
Wave 3 (W3a-c) will refine this via ``callHierarchy/outgoingCalls``.

Cross-module canonicalization invariant
----------------------------------------
This module **delegates** content canonicalization, hashing, and SimHash
fingerprinting to :mod:`scry.anchor_id` so that hashes computed here agree
byte-for-byte with hashes computed by the markdown extractor (W1a) and any
other consumer that reuses the W1c primitives.  Do **not** reintroduce a
local copy of :func:`canonicalize_content` — see review-w1b BLOCKING finding.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from scry.anchor_id import (
    canonicalize_content,
    content_hash,
    fingerprint_simhash,
    slugify,
)
from scry.extract._test_detection import is_test_path
from scry.models import Anchor, AnchorType, CodeAnchorsConfig, TransitiveHashStatus

# Re-export the W1c canonicalizer under the historical name for callers that
# imported `scry.extract.code.canonicalize`.  This is the SAME function, not
# a duplicate.
canonicalize = canonicalize_content

__all__ = ["canonicalize", "extract_code_symbols"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Language configuration
# ---------------------------------------------------------------------------

# Default symbol_kinds per language (DESIGN.md §6 code_anchors.symbol_kinds).
# The TypeScript entry intentionally omits ``function_signature`` because that
# node kind is handled implicitly for overload detection (§15.3) - it is not a
# user-configurable kind.
_DEFAULT_SYMBOL_KINDS: dict[str, list[str]] = {
    "python": ["function_definition", "class_definition", "decorated_definition"],
    "typescript": [
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        # SR5-2: catch ``export const fn = () => {}`` / ``export const Cls = class {}``
        "lexical_declaration",
        "enum_declaration",
        "internal_module",  # tree-sitter TS spelling for ``namespace N {}``
    ],
    "tsx": [
        "function_declaration",
        "class_declaration",
        "interface_declaration",
        "type_alias_declaration",
        "lexical_declaration",
        "enum_declaration",
        "internal_module",
    ],
    # SR5-3: JS arrow fns / class expressions live inside lexical_declaration
    "javascript": ["function_declaration", "class_declaration", "lexical_declaration"],
    "jsx": ["function_declaration", "class_declaration", "lexical_declaration"],
    "zig": ["FnProto", "ContainerDecl"],
    # SR5-1: Go top-level declarations.  ``method_declaration`` covers
    # receiver methods (``func (s *T) Bar()``); ``type_declaration``
    # wraps ``type_spec`` / ``type_alias`` for struct, interface, alias.
    "go": [
        "function_declaration",
        "method_declaration",
        "type_declaration",
    ],
    # SR5-1: Rust items.  ``impl_item`` so we capture method bodies via
    # the impl walker; ``macro_definition`` for ``macro_rules!``.
    "rust": [
        "function_item",
        "struct_item",
        "trait_item",
        "impl_item",
        "macro_definition",
        "enum_item",
        "type_item",
    ],
}

# Normalise user-supplied language names → tree-sitter-language-pack grammar names.
_GRAMMAR_NAME: dict[str, str] = {
    "python": "python",
    "py": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "tsx": "tsx",
    "javascript": "javascript",
    "js": "javascript",
    "jsx": "jsx",
    "zig": "zig",
    # SR5-1: Go and Rust grammars are bundled with tree_sitter_language_pack
    # and were already routed by index._EXT_TO_LANG; we just need the
    # walkers (defined below).
    "go": "go",
    "golang": "go",
    "rust": "rust",
    "rs": "rust",
}

_DEFAULT_MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB (matches IndexConfig default)


# ---------------------------------------------------------------------------
# §5.4 Canonicalization (delegated to scry.anchor_id — see module docstring)
# ---------------------------------------------------------------------------


def _sig_hash6(text: str) -> str:
    """6-char hex prefix of SHA-256 — used for overload disambiguation.

    Length matches :func:`scry.anchor_id.derive_code_id`'s
    ``signature_hash[:6]`` truncation so that an ID minted here is byte-
    identical to one minted by ``derive_code_id(..., signature_hash=full_hash)``.
    """
    return hashlib.sha256(canonicalize_content(text).encode("utf-8")).hexdigest()[:6]


# ---------------------------------------------------------------------------
# AST helpers (tree-sitter nodes typed as ``Any``; module has no stubs)
# ---------------------------------------------------------------------------


def _node_text(node: Any, src: bytes) -> str:
    return src[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def _child_of_type(node: Any, *types: str) -> Any | None:
    """Return the first direct child whose ``.type`` is in *types*, or None."""
    for child in node.children:
        if child.type in types:
            return child
    return None


def _find_descendant_of_type(node: Any, *types: str) -> Any | None:
    """Depth-first search for the first descendant of one of *types*."""
    for child in node.children:
        if child.type in types:
            return child
        found = _find_descendant_of_type(child, *types)
        if found is not None:
            return found
    return None


# ---------------------------------------------------------------------------
# Collision resolution helpers
# ---------------------------------------------------------------------------


def _apply_collisions(raw: list[str]) -> list[str]:
    """Given a list of raw symbol names (in order), return collision-resolved names.

    First occurrence keeps the bare name; subsequent occurrences receive
    ``@2``, ``@3``, … suffixes per §15.3.
    """
    counts: dict[str, int] = {}
    out: list[str] = []
    for name in raw:
        counts[name] = counts.get(name, 0) + 1
        if counts[name] == 1:
            out.append(name)
        else:
            out.append(f"{name}@{counts[name]}")
    return out


# ---------------------------------------------------------------------------
# Python extraction
# ---------------------------------------------------------------------------


def _py_symbol_name(node: Any, src: bytes) -> str | None:
    """Return the user-visible symbol name for a Python AST node.

    For ``decorated_definition``, unwraps to the inner ``function_definition``
    or ``class_definition`` and returns *that* node's identifier (§15.3).
    Returns ``None`` for anonymous or unrecognised shapes.
    """
    if node.type == "decorated_definition":
        inner = _child_of_type(
            node, "function_definition", "async_function_definition", "class_definition"
        )
        if inner is None:
            return None
        node = inner
    return _py_identifier(node, src)


def _py_identifier(node: Any, src: bytes) -> str | None:
    """Return the text of the first ``identifier`` child of *node*."""
    child = _child_of_type(node, "identifier")
    if child is None:
        return None
    return src[child.start_byte : child.end_byte].decode("utf-8", errors="replace")


def _py_inner_node(node: Any) -> Any:
    """Unwrap a ``decorated_definition`` to its inner def/class node."""
    if node.type != "decorated_definition":
        return node
    inner = _child_of_type(
        node, "function_definition", "async_function_definition", "class_definition"
    )
    return inner if inner is not None else node


def _py_is_class(node: Any) -> bool:
    """Return True when *node* is (or wraps) a class definition."""
    n = _py_inner_node(node)
    return bool(n.type == "class_definition")


def _py_class_body(node: Any) -> Any | None:
    """Return the ``block`` child of a class (possibly decorated) node."""
    n = _py_inner_node(node)
    if n.type != "class_definition":
        return None
    return _child_of_type(n, "block")


# Raw record used during Python tree walk.
_PyRec = tuple[str, str, int, int]  # (qualified_symbol_path, raw_content_text, def_line, def_char)


def _walk_python(
    nodes: list[Any],
    src: bytes,
    kinds: set[str],
    scope_prefix: str,
) -> list[_PyRec]:
    """Recursively walk *nodes* and return (symbol_path, raw_content) pairs.

    *scope_prefix* is the dot-separated class context, e.g. ``"OuterClass"``
    for methods, or ``""`` for top-level symbols.  Collision resolution is
    applied *per scope level* before recursing into class bodies.
    """
    # Collect matching nodes at this scope level.
    raw_names: list[str] = []
    matching: list[Any] = []
    for node in nodes:
        if node.type not in kinds:
            continue
        name = _py_symbol_name(node, src)
        if name is None:
            continue
        raw_names.append(name)
        matching.append(node)

    resolved_names = _apply_collisions(raw_names)

    results: list[_PyRec] = []
    for node, resolved in zip(matching, resolved_names, strict=True):
        symbol_path = f"{scope_prefix}.{resolved}" if scope_prefix else resolved
        content_raw = _node_text(node, src)
        results.append((symbol_path, content_raw, node.start_point[0], node.start_point[1]))

        # Recurse into class bodies to extract methods.
        if _py_is_class(node):
            body = _py_class_body(node)
            if body is not None:
                child_scope = symbol_path
                results.extend(_walk_python(list(body.children), src, kinds, child_scope))

    return results


# ---------------------------------------------------------------------------
# TypeScript / TSX extraction
# ---------------------------------------------------------------------------


def _ts_initializer_is_require_import(declarator: Any, src: bytes) -> bool:
    """Return True if a TS/JS variable_declarator's value is a CommonJS import.

    UAT-24 / review-u7 MEDIUM: handles both ``const x = require('y')`` and
    ``const x = require('y').member`` (member_expression rooted at a bare
    ``require(...)`` call).  ``module.require(...)`` is NOT treated as a
    CommonJS import (the call expression's function is a member_expression,
    not a bare ``require``).  ``require.resolve(...)`` is also NOT treated
    as an import (the function is a member_expression on require).
    """

    def _is_bare_require_call(call_node: Any) -> bool:
        if call_node.type != "call_expression":
            return False
        fn = _child_of_type(call_node, "identifier")
        if fn is None:
            return False
        fn_text = src[fn.start_byte : fn.end_byte].decode("utf-8", errors="replace")
        return fn_text == "require"

    # Direct: ``const x = require('y')``
    direct = _child_of_type(declarator, "call_expression")
    if direct is not None and _is_bare_require_call(direct):
        return True
    # Member: ``const x = require('y').member`` parses as
    # member_expression → object: call_expression → fn: identifier(require)
    member = _child_of_type(declarator, "member_expression")
    if member is not None:
        # The leftmost child of member_expression should be the call.
        for child in member.children:
            if child.type == "call_expression" and _is_bare_require_call(child):
                return True
    return False


def _ts_symbol_name(node: Any, src: bytes) -> str | None:
    """Return the identifier for a TypeScript top-level declaration."""
    # SR5-2: ``lexical_declaration`` (e.g. ``const fn = () => {}``) wraps
    # one or more ``variable_declarator`` children whose first child is
    # the identifier we want.  Pull the FIRST declarator's name.
    if node.type == "lexical_declaration":
        declarator = _child_of_type(node, "variable_declarator")
        if declarator is not None:
            ident = _child_of_type(declarator, "identifier", "property_identifier")
            if ident is not None:
                # UAT-24: skip ``const x = require('y')`` (or any call_expression
                # whose function is a bare ``require`` identifier).  Per
                # review-u7 MEDIUM: also catch ``const x = require('y').member``
                # which parses as a member_expression rooted at a require call.
                # ``module.require(...)`` and ``require.resolve(...)`` are out
                # of scope (latter is a method call ON require, not require()).
                if _ts_initializer_is_require_import(declarator, src):
                    return None
                return src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
        return None
    # Most TS declarations use ``identifier`` or ``type_identifier`` as the
    # second child (after keyword).
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            return src[child.start_byte : child.end_byte].decode("utf-8", errors="replace")
    return None


def _ts_method_name(node: Any, src: bytes) -> str | None:
    """Return the name of a ``method_definition`` node."""
    child = _child_of_type(node, "property_identifier", "identifier")
    if child is None:
        return None
    return src[child.start_byte : child.end_byte].decode("utf-8", errors="replace")


def _ts_class_methods(
    class_node: Any,
    src: bytes,
    class_name: str,
) -> list[_PyRec]:
    """Extract ``method_definition`` anchors from a TypeScript class body."""
    body = _child_of_type(class_node, "class_body")
    if body is None:
        return []

    raw_names: list[str] = []
    method_nodes: list[Any] = []
    for child in body.children:
        if child.type != "method_definition":
            continue
        name = _ts_method_name(child, src)
        if name is None:
            continue
        raw_names.append(name)
        method_nodes.append(child)

    resolved = _apply_collisions(raw_names)
    results: list[_PyRec] = []
    for node, resolved_name in zip(method_nodes, resolved, strict=True):
        symbol_path = f"{class_name}.{resolved_name}"
        results.append(
            (symbol_path, _node_text(node, src), node.start_point[0], node.start_point[1])
        )
    return results


# Groups nodes collected during the TypeScript top-level walk.
_TsGroup = dict[str, list[Any]]  # name -> [node, …]


def _ts_unwrap(node: Any) -> Any:
    """Unwrap an ``export_statement`` (and any nested ``export_statement``)
    to expose the underlying declaration.

    UT5-1 BLOCKING fix: top-level TypeScript symbols are almost always
    written as ``export class Foo {}`` / ``export function bar() {}``.
    The tree-sitter parse for that is::

        export_statement
          decorator? class_declaration | function_declaration | …

    Pre-fix the walker iterated ``root.children`` and only matched
    direct types (``class_declaration``, ``function_declaration``), so
    every exported symbol was silently dropped — a polyglot scry repo
    with TS frontend + Python backend produced ZERO TS anchors.

    Also handles ``export default <decl>`` and tolerates wrappers
    appearing more than once defensively.
    """
    seen = 0
    cur = node
    while cur is not None and getattr(cur, "type", None) == "export_statement" and seen < 4:
        seen += 1
        # The wrapped declaration is the last named child (skipping
        # ``export``, ``default`` keywords and decorator nodes).
        unwrapped: Any = None
        for child in cur.children:
            if getattr(child, "is_named", False) and child.type not in (
                "decorator",
                "export",
                "default",
            ):
                unwrapped = child
        if unwrapped is None:
            return node
        cur = unwrapped
    return cur


def _walk_typescript(
    nodes: list[Any],
    src: bytes,
    kinds: set[str],
) -> list[_PyRec]:
    """Return (symbol_path, raw_content) pairs from TypeScript source nodes.

    Top-level ``export_statement`` wrappers are unwrapped via
    :func:`_ts_unwrap` so that ``export class Foo {}`` is extracted as
    ``Foo`` (UT5-1 BLOCKING fix).

    Overload handling (§15.3):
    * ``function_signature`` nodes at the top level are TypeScript overload
      declarations.  Each is given a ``@<sig-hash[:6]>`` suffix appended to
      the symbol path via :func:`scry.anchor_id.derive_code_id`.
    * ``function_declaration`` nodes with the same name that follow overload
      signatures are treated as the implementation body and extracted with
      the bare name (or ``@N`` collision suffix).

    SR5-5: in addition to the regular declaration kinds, this walker
    also extracts Jest-style test-framework calls
    (``describe()``, ``it()``, ``test()``, hooks, plus
    ``.skip`` / ``.only`` / ``.each`` / ``.concurrent`` variants).
    Without this, TypeScript test files produce zero anchors —
    significant test/prod parity gap.

    Note: ``function_signature`` nodes inside interface bodies are NOT
    extracted here - only top-level ones (direct children of *nodes*).
    """
    results: list[_PyRec] = []

    # Single pass: separate overload signatures from regular declarations.
    decl_names: list[str] = []
    decl_nodes: list[tuple[str, Any]] = []  # (raw_name, node)
    sig_entries: list[tuple[str, Any]] = []  # (raw_name, node) for function_signature

    for raw_node in nodes:
        node = _ts_unwrap(raw_node)
        if node.type == "function_signature":
            name = _ts_symbol_name(node, src)
            if name is not None:
                sig_entries.append((name, node))
        elif node.type in kinds:
            name = _ts_symbol_name(node, src)
            if name is not None:
                decl_names.append(name)
                decl_nodes.append((name, node))

    # Emit overload signatures with @<sig-hash[:6]> suffix.  Match
    # derive_code_id semantics so an ID minted here is byte-identical to
    # one that downstream code might recreate via that helper.
    for sig_name, sig_node in sig_entries:
        sig_text = _node_text(sig_node, src)
        suffix = _sig_hash6(sig_text)
        symbol_path = f"{sig_name}@{suffix}"
        results.append((symbol_path, sig_text, sig_node.start_point[0], sig_node.start_point[1]))

    # Emit regular declarations with collision resolution.
    resolved = _apply_collisions(decl_names)
    for (_raw_name, node), resolved_name in zip(decl_nodes, resolved, strict=True):
        results.append(
            (resolved_name, _node_text(node, src), node.start_point[0], node.start_point[1])
        )

        # Recurse into class body for methods.
        if node.type == "class_declaration":
            results.extend(_ts_class_methods(node, src, resolved_name))

    # SR5-5: Jest-style test-framework anchors.  Recursive walker
    # collects describe/it/test/hooks anywhere in the file (top-level
    # AND nested inside other describes), building hierarchical
    # symbol paths so nested ``it``s carry their parent ``describe``
    # name as a prefix.
    results.extend(_walk_typescript_test_calls(nodes, src, parent_path=()))

    return results


# ---------------------------------------------------------------------------
# SR5-5: Jest / Mocha / Vitest test-framework anchors
# ---------------------------------------------------------------------------

#: Test-framework function names worth anchoring.  Includes the bare
#: forms (``describe``, ``it``, …) plus the well-known ``.skip``,
#: ``.only``, ``.each``, and ``.concurrent`` variants (member
#: expressions resolve to e.g. ``describe.skip``).  Also covers the
#: jest legacy prefix forms ``xdescribe`` / ``fit``.
_TS_TEST_NAMED_FNS: frozenset[str] = frozenset(
    {
        "describe",
        "it",
        "test",
        "context",  # mocha alias
        "suite",  # mocha alias
        "specify",  # mocha alias
        # Jest legacy skip/focus prefixes.
        "xdescribe",
        "xit",
        "xtest",
        "fdescribe",
        "fit",
        "ftest",
    }
)

#: Hooks take ``(callback)`` not ``(name, callback)`` — handle separately.
_TS_TEST_HOOK_FNS: frozenset[str] = frozenset(
    {
        "beforeEach",
        "afterEach",
        "beforeAll",
        "afterAll",
        "before",
        "after",
        "setup",
        "teardown",
    }
)


def _ts_callee_name(call_node: Any, src: bytes) -> str | None:
    """Return the canonical callee name for a call_expression.

    Handles three patterns:
      * ``foo(...)`` → ``"foo"``
      * ``obj.method(...)`` → ``"obj.method"``
      * ``foo.each(...)("name", cb)`` → ``"foo.each"`` (the OUTER call's
        callee is itself a call_expression; we read THAT inner call's
        callee as the canonical name).
    """
    callee = (
        call_node.child_by_field_name("function")
        if hasattr(call_node, "child_by_field_name")
        else None
    )
    if callee is None:
        # Fall back to the first non-syntax child.
        for c in call_node.children:
            if c.is_named:
                callee = c
                break
    if callee is None:
        return None
    if callee.type == "identifier":
        return src[callee.start_byte : callee.end_byte].decode("utf-8", errors="replace")
    if callee.type == "member_expression":
        return src[callee.start_byte : callee.end_byte].decode("utf-8", errors="replace")
    if callee.type == "call_expression":
        # ``describe.each(...)('name', cb)`` — callee is itself a call.
        inner = _ts_callee_name(callee, src)
        return inner
    return None


def _ts_call_first_string_arg(call_node: Any, src: bytes) -> str | None:
    """Return the FIRST argument of a call_expression as a literal string.

    Returns the unquoted/decoded text for ``string`` and
    ``template_string`` literal arguments.  Returns ``None`` when the
    first arg is dynamic (an identifier, expression, etc.).
    """
    args = (
        call_node.child_by_field_name("arguments")
        if hasattr(call_node, "child_by_field_name")
        else None
    )
    if args is None:
        for c in call_node.children:
            if c.type == "arguments":
                args = c
                break
    if args is None:
        return None
    for c in args.children:
        if not c.is_named:
            continue
        if c.type == "string":
            text = src[c.start_byte : c.end_byte].decode("utf-8", errors="replace")
            return text.strip("\"'`")
        if c.type == "template_string":
            text = src[c.start_byte : c.end_byte].decode("utf-8", errors="replace")
            return text.strip("`")
        # First named child wasn't a literal — give up.
        return None
    return None


def _ts_call_test_name(call_node: Any, src: bytes) -> str:
    """Build the suffix for a test-framework anchor name.

    Prefers the literal first-argument string (slugified to keep
    AnchorId chars in the well-known set ``[A-Za-z0-9_./@#%:-]``);
    falls back to ``@<line>:<col>`` so dynamically-named tests still
    get a stable, bounded ID (per code-review: must NOT use full
    call text — it includes the callback body and can blow past
    AnchorId length).
    """
    name = _ts_call_first_string_arg(call_node, src)
    if name is None:
        return f"@{call_node.start_point[0] + 1}:{call_node.start_point[1] + 1}"
    slug = slugify(name, fallback_prefix="test")
    return slug


def _ts_test_hook_name(call_node: Any) -> str:
    """SR5-5: hooks (``beforeEach``, etc.) take a callback as arg 0,
    not a name + callback.  Anchor name uses ``@line:col`` suffix.
    """
    return f"@{call_node.start_point[0] + 1}:{call_node.start_point[1] + 1}"


def _walk_typescript_test_calls(
    nodes: list[Any],
    src: bytes,
    *,
    parent_path: tuple[str, ...],
    sibling_seen: dict[tuple[str, ...], dict[str, int]] | None = None,
) -> list[_PyRec]:
    """SR5-5: recursive walker that emits anchors for Jest-style
    test-framework calls anywhere in the source tree.

    Builds a hierarchical symbol path so nested ``it`` blocks carry
    their parent ``describe`` name as a prefix
    (e.g. ``describe:auth-flow::it:rejects-expired-tokens``).

    review-r6sr5-5: sibling tests with the same name (e.g. two
    ``it('handles foo')`` under one describe) get an ``@2`` / ``@3``
    suffix so AnchorIds remain unique — without this they'd collide
    on the DB primary-key constraint and silently overwrite.
    """
    results: list[_PyRec] = []
    if sibling_seen is None:
        sibling_seen = {}

    def _resolve_collision(symbol: str) -> str:
        bucket = sibling_seen.setdefault(parent_path, {})
        n = bucket.get(symbol, 0) + 1
        bucket[symbol] = n
        return symbol if n == 1 else f"{symbol}@{n}"

    for raw_node in nodes:
        node = _ts_unwrap(raw_node)
        # Recurse into class/function bodies for nested test calls.
        if node.type in (
            "class_body",
            "statement_block",
            "program",
        ):
            results.extend(
                _walk_typescript_test_calls(
                    list(node.children),
                    src,
                    parent_path=parent_path,
                    sibling_seen=sibling_seen,
                )
            )
            continue

        if node.type != "expression_statement":
            # Walk into other compound nodes too — some test frameworks
            # wrap calls in additional layers.
            if hasattr(node, "children") and node.children:
                results.extend(
                    _walk_typescript_test_calls(
                        list(node.children),
                        src,
                        parent_path=parent_path,
                        sibling_seen=sibling_seen,
                    )
                )
            continue

        # expression_statement → call_expression
        call = None
        for c in node.children:
            if c.type == "call_expression":
                call = c
                break
        if call is None:
            continue

        callee_name = _ts_callee_name(call, src)
        if callee_name is None:
            continue
        # Normalize "describe.skip" → base "describe" for the dispatch
        # but keep the FULL string for the anchor's display name.
        base = callee_name.split(".")[0]

        if base in _TS_TEST_NAMED_FNS:
            test_name = _ts_call_test_name(call, src)
            symbol = _resolve_collision(f"{callee_name}:{test_name}")
            full_path = "::".join((*parent_path, symbol))
            results.append(
                (full_path, _node_text(call, src), call.start_point[0], call.start_point[1])
            )
            # Recurse into the callback body to surface nested tests.
            # New nested scope gets a fresh sibling_seen sub-dict.
            results.extend(
                _walk_typescript_test_calls(
                    list(call.children),
                    src,
                    parent_path=(*parent_path, symbol),
                    sibling_seen=sibling_seen,
                )
            )
        elif base in _TS_TEST_HOOK_FNS:
            symbol = _resolve_collision(f"{callee_name}{_ts_test_hook_name(call)}")
            full_path = "::".join((*parent_path, symbol))
            results.append(
                (full_path, _node_text(call, src), call.start_point[0], call.start_point[1])
            )
        # Otherwise: not a test-framework call — keep walking to find
        # nested describes inside e.g. an IIFE.
        elif hasattr(call, "children") and call.children:
            results.extend(
                _walk_typescript_test_calls(
                    list(call.children),
                    src,
                    parent_path=parent_path,
                    sibling_seen=sibling_seen,
                )
            )

    return results


# ---------------------------------------------------------------------------
# Zig extraction
# ---------------------------------------------------------------------------


def _zig_fn_name(fn_proto_node: Any, src: bytes) -> str | None:
    """Return the function name from a Zig ``FnProto`` node."""
    child = _child_of_type(fn_proto_node, "IDENTIFIER")
    if child is None:
        return None
    return src[child.start_byte : child.end_byte].decode("utf-8", errors="replace")


def _zig_var_name(var_decl_node: Any, src: bytes) -> str | None:
    """Return the variable name from a Zig ``VarDecl`` node."""
    child = _child_of_type(var_decl_node, "IDENTIFIER")
    if child is None:
        return None
    return src[child.start_byte : child.end_byte].decode("utf-8", errors="replace")


def _zig_has_container_decl(var_decl_node: Any) -> bool:
    """Return True if *var_decl_node* contains a ``ContainerDecl`` descendant."""
    return _find_descendant_of_type(var_decl_node, "ContainerDecl") is not None


def _walk_zig(
    root: Any,
    src: bytes,
    kinds: set[str],
) -> list[_PyRec]:
    """Return (symbol_path, raw_content) pairs from a Zig source file root.

    The Zig grammar places top-level declarations as ``Decl`` children of
    ``source_file``, with optional ``pub`` siblings immediately preceding them::

        source_file
          [pub]          ← visibility keyword (sibling, not inside Decl)
          [Decl]
            [FnProto]    ← function prototype
            [Block]      ← function body
          [Decl]
            [VarDecl]    ← const declaration (may contain ContainerDecl)

    Content text for a symbol includes the ``pub`` prefix when present.
    """
    children: list[Any] = list(root.children)
    raw_names: list[str] = []
    raw_entries: list[tuple[str, str, int, int]] = []  # (name, content_text, def_line, def_char)

    for i, child in enumerate(children):
        if child.type != "Decl":
            continue

        # Determine whether a `pub` keyword immediately precedes this Decl.
        has_pub = i > 0 and children[i - 1].type == "pub"
        content_start = children[i - 1].start_byte if has_pub else child.start_byte
        content_bytes = src[content_start : child.end_byte]
        content_text = content_bytes.decode("utf-8", errors="replace")

        # Check for FnProto (function).
        fn_proto = _child_of_type(child, "FnProto")
        if fn_proto is not None and "FnProto" in kinds:
            name = _zig_fn_name(fn_proto, src)
            if name is not None:
                raw_names.append(name)
                raw_entries.append(
                    (name, content_text, fn_proto.start_point[0], fn_proto.start_point[1])
                )
            continue  # A Decl is either a fn or a var, not both.

        # Check for VarDecl containing a ContainerDecl (struct/enum/union).
        var_decl = _child_of_type(child, "VarDecl")
        if var_decl is not None and "ContainerDecl" in kinds and _zig_has_container_decl(var_decl):
            name = _zig_var_name(var_decl, src)
            if name is not None:
                raw_names.append(name)
                raw_entries.append(
                    (name, content_text, var_decl.start_point[0], var_decl.start_point[1])
                )

    resolved = _apply_collisions(raw_names)
    results: list[_PyRec] = []
    for (_, content, def_line, def_char), resolved_name in zip(raw_entries, resolved, strict=True):
        results.append((resolved_name, content, def_line, def_char))
    return results


# ---------------------------------------------------------------------------
# Go extraction (SR5-1)
# ---------------------------------------------------------------------------


def _go_method_receiver_type(method_node: Any, src: bytes) -> str | None:
    """Return the (struct) type name from a Go ``method_declaration`` receiver.

    A receiver looks like ``func (s *Service) Bar()`` or
    ``func (s Service) Bar()``.  We pull the unqualified type name from
    the first ``parameter_list`` (the receiver list, which always
    precedes the method name).
    """
    receiver = _child_of_type(method_node, "parameter_list")
    if receiver is None:
        return None
    decl = _child_of_type(receiver, "parameter_declaration")
    if decl is None:
        return None
    # Pointer receiver: pointer_type → type_identifier
    pointer = _child_of_type(decl, "pointer_type")
    if pointer is not None:
        ident = _child_of_type(pointer, "type_identifier")
        if ident is not None:
            return src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
    # Value receiver: type_identifier directly under parameter_declaration
    ident = _child_of_type(decl, "type_identifier")
    if ident is not None:
        return src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
    return None


def _go_symbol_name(node: Any, src: bytes) -> str | None:
    """Return the qualified symbol name for a Go top-level declaration.

    * ``function_declaration`` → ``Foo`` (identifier child)
    * ``method_declaration``   → ``Service.Bar`` (receiver type + field_identifier)
    * ``type_declaration``     → ``Service`` (type_spec/type_alias child's type_identifier)
    """
    if node.type == "function_declaration":
        ident = _child_of_type(node, "identifier")
        if ident is not None:
            return src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
        return None
    if node.type == "method_declaration":
        # field_identifier holds the method name (after the receiver list).
        method_name_node = _child_of_type(node, "field_identifier")
        if method_name_node is None:
            return None
        method_name = src[method_name_node.start_byte : method_name_node.end_byte].decode(
            "utf-8", errors="replace"
        )
        receiver_type = _go_method_receiver_type(node, src)
        if receiver_type is not None:
            return f"{receiver_type}.{method_name}"
        return method_name
    if node.type == "type_declaration":
        # type_declaration wraps either type_spec (struct/interface/alias)
        # or type_alias (the explicit ``type T = U`` form).
        spec = _child_of_type(node, "type_spec", "type_alias")
        if spec is None:
            return None
        ident = _child_of_type(spec, "type_identifier")
        if ident is not None:
            return src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
        return None
    return None


def _walk_go(
    nodes: list[Any],
    src: bytes,
    kinds: set[str],
) -> list[_PyRec]:
    """Return (symbol_path, raw_content) records from Go source nodes.

    Top-level Go declarations are direct children of ``source_file`` so
    we iterate without recursion (tree-sitter's Go grammar already
    surfaces method receivers via ``method_declaration``).
    """
    raw_names: list[str] = []
    raw_entries: list[tuple[str, str, int, int]] = []

    for node in nodes:
        if node.type not in kinds:
            continue
        name = _go_symbol_name(node, src)
        if name is None:
            continue
        raw_names.append(name)
        raw_entries.append((name, _node_text(node, src), node.start_point[0], node.start_point[1]))

    resolved = _apply_collisions(raw_names)
    results: list[_PyRec] = []
    for (_, content, def_line, def_char), resolved_name in zip(raw_entries, resolved, strict=True):
        results.append((resolved_name, content, def_line, def_char))
    return results


# ---------------------------------------------------------------------------
# Rust extraction (SR5-1)
# ---------------------------------------------------------------------------


def _rust_symbol_name(node: Any, src: bytes) -> str | None:
    """Return the symbol name for a Rust top-level item.

    Most Rust items expose a ``type_identifier`` (struct/trait/enum/type)
    or ``identifier`` (fn/macro) child.  ``impl_item`` is special: its
    name is the type being implemented.
    """
    if node.type == "function_item":
        ident = _child_of_type(node, "identifier")
        if ident is not None:
            return src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
    elif node.type in ("struct_item", "trait_item", "enum_item", "type_item"):
        ident = _child_of_type(node, "type_identifier")
        if ident is not None:
            return src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
    elif node.type == "impl_item":
        # Impl block: the implemented type appears as type_identifier.
        # For ``impl Trait for Type`` we'd ideally encode both; for now
        # we use the LAST type_identifier child which is the receiver
        # type.  Distinct impls on the same type get @N collision suffixes.
        type_idents = [c for c in node.children if c.type == "type_identifier"]
        if type_idents:
            ident = type_idents[-1]
            return "impl_" + src[ident.start_byte : ident.end_byte].decode(
                "utf-8", errors="replace"
            )
    elif node.type == "macro_definition":
        ident = _child_of_type(node, "identifier")
        if ident is not None:
            return src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
    return None


def _walk_rust_impl_methods(impl_node: Any, src: bytes, impl_name: str) -> list[_PyRec]:
    """Extract method ``function_item`` anchors from a Rust ``impl_item`` body.

    Rust impl blocks contain method definitions inside a ``declaration_list``
    child.  Each method's symbol path is qualified as ``<impl_name>.<method>``.
    """
    body = _child_of_type(impl_node, "declaration_list")
    if body is None:
        return []
    method_names: list[str] = []
    method_nodes: list[Any] = []
    for child in body.children:
        if child.type == "function_item":
            ident = _child_of_type(child, "identifier")
            if ident is None:
                continue
            method_names.append(
                src[ident.start_byte : ident.end_byte].decode("utf-8", errors="replace")
            )
            method_nodes.append(child)
    resolved = _apply_collisions(method_names)
    results: list[_PyRec] = []
    for method_node, resolved_name in zip(method_nodes, resolved, strict=True):
        results.append(
            (
                f"{impl_name}.{resolved_name}",
                _node_text(method_node, src),
                method_node.start_point[0],
                method_node.start_point[1],
            )
        )
    return results


def _walk_rust(
    nodes: list[Any],
    src: bytes,
    kinds: set[str],
) -> list[_PyRec]:
    """Return (symbol_path, raw_content) records from Rust source nodes."""
    raw_names: list[str] = []
    raw_entries: list[tuple[str, str, int, int]] = []
    impl_blocks: list[tuple[str, Any]] = []

    for node in nodes:
        if node.type not in kinds:
            continue
        name = _rust_symbol_name(node, src)
        if name is None:
            continue
        raw_names.append(name)
        raw_entries.append((name, _node_text(node, src), node.start_point[0], node.start_point[1]))
        if node.type == "impl_item":
            impl_blocks.append((name, node))

    resolved = _apply_collisions(raw_names)
    results: list[_PyRec] = []
    name_remap: dict[int, str] = {}
    for i, ((_, content, def_line, def_char), resolved_name) in enumerate(
        zip(raw_entries, resolved, strict=True)
    ):
        results.append((resolved_name, content, def_line, def_char))
        name_remap[i] = resolved_name

    # Recurse into impl blocks to extract method bodies.  Use the
    # collision-resolved impl name so methods get a stable scope.
    for original_name, impl_node in impl_blocks:
        # Find the resolved name we emitted for this exact impl node.
        resolved_impl = original_name
        for entry, candidate in zip(raw_entries, resolved, strict=True):
            if entry[0] == original_name and entry[1] == _node_text(impl_node, src):
                resolved_impl = candidate
                break
        results.extend(_walk_rust_impl_methods(impl_node, src, resolved_impl))

    return results


# ---------------------------------------------------------------------------
# Anchor construction helper
# ---------------------------------------------------------------------------


def _make_anchor(
    path_str: str,
    symbol_path: str,
    raw_content: str,
    def_line: int = 0,
    def_char: int = 0,
    *,
    is_test: bool = False,
) -> Anchor:
    """Build an ``Anchor`` for a code symbol.

    *symbol_path* is the full qualified path portion of the ID
    (e.g. ``"MyClass.method"`` or ``"fn_name@abc123"``).
    The ``id`` is ``<path_str>:<symbol_path>``.

    Content is canonicalised per §5.4 before hashing.
    ``transitive_hash_status`` is set to ``LSP_UNAVAILABLE`` (Wave 1; Wave 3
    will refine via ``callHierarchy``).
    ``def_line`` and ``def_char`` carry the tree-sitter ``start_point`` for
    this symbol — used by the W3d LSP enrichment pass to call
    ``textDocument/prepareCallHierarchy`` at the right position.
    """
    import re as _re

    canon = canonicalize_content(raw_content)
    chash = content_hash(raw_content)  # content_hash canonicalizes internally
    shash = fingerprint_simhash(raw_content)

    # symbol_name is the leaf name (last component), with both the `@N`
    # collision suffix AND the `@<sig-hash>` overload suffix stripped, to
    # keep it human-readable.  The suffix-stripping regex matches `@`
    # followed by digits (collision) OR by hex chars (sig hash).
    #
    # UAT-25: Rust impl methods are qualified as ``impl_<Type>.<method>``
    # (e.g. ``impl_User.validate``).  Stripping to the leaf collapses
    # ``impl_User.validate`` and ``impl_Order.validate`` both to
    # ``validate`` — a collision hazard.  For ``impl_*.<method>`` paths
    # in Rust source files we retain the full qualified path as
    # ``symbol_name`` so distinct impls of the same method name remain
    # distinguishable.  Per review-u7 LOW: scoping by the path's .rs
    # extension prevents accidental cross-language API drift (e.g. a
    # Python class literally named ``impl_User`` would otherwise also
    # be affected).
    is_rust_impl = (
        path_str.endswith(".rs") and symbol_path.startswith("impl_") and "." in symbol_path
    )
    if is_rust_impl:
        symbol_name = _re.sub(r"@(?:\d+|[0-9a-f]+)$", "", symbol_path)
    else:
        leaf = symbol_path.split(".")[-1]
        symbol_name = _re.sub(r"@(?:\d+|[0-9a-f]+)$", "", leaf)

    return Anchor(
        id=f"{path_str}:{symbol_path}",
        type=AnchorType.CODE,
        path=path_str,
        symbol_name=symbol_name,
        content_text=canon,
        content_hash=chash,
        fingerprint_simhash=shash,
        transitive_hash_status=TransitiveHashStatus.LSP_UNAVAILABLE,
        is_test=is_test,
        def_line=def_line,
        def_char=def_char,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_code_symbols(
    path: Path,
    repo_root: Path,
    *,
    language: str,
    config: CodeAnchorsConfig | None = None,
    max_file_size_bytes: int = _DEFAULT_MAX_FILE_SIZE,
) -> list[Anchor]:
    """Extract ``AnchorType.CODE`` anchors from *path* using tree-sitter.

    Parameters
    ----------
    path:
        Absolute or repo-relative path to the source file.
    repo_root:
        Repository root used to compute the repo-relative path stored in each
        anchor's ``path`` field.
    language:
        Grammar name or alias (``"python"``, ``"typescript"``, ``"ts"``,
        ``"zig"``, …).  Must resolve via the internal alias table.
    config:
        Optional ``CodeAnchorsConfig`` from ``.scry/config.yaml``.  Controls
        ``languages.<lang>`` skip behaviour, ``symbol_kinds``, and
        ``granularity``.  Defaults to ``CodeAnchorsConfig()`` when omitted.
    max_file_size_bytes:
        Per-file byte cap (DESIGN.md §15.4).  Pass
        ``IndexConfig.max_file_size_bytes`` from the loaded config to honor
        the user's override; defaults to ``_DEFAULT_MAX_FILE_SIZE`` (5 MB)
        for callers that don't have an ``IndexConfig`` handy.

    Returns
    -------
    list[Anchor]
        Zero or more ``Anchor`` objects of type ``AnchorType.CODE``.
        Returns an empty list (never raises) for:
        * Unknown / unsupported language
        * ``languages.<lang>: skip`` in config
        * File size exceeds ``max_file_size_bytes`` (default 5 MB)
        * Empty file
        * ``granularity: "file"`` (falls back to symbol mode after warning)
    """
    if config is None:
        config = CodeAnchorsConfig()

    # Warn and fall back when granularity="file" (DESIGN.md §13 #13).
    if config.granularity == "file":
        logger.warning(
            "code_anchors.granularity='file' is not yet implemented (DESIGN.md §13 "
            "open question #13); falling back to 'symbol' mode for %s",
            path,
        )

    # Normalise language name → grammar name.
    lang_key = language.lower()
    grammar_name = _GRAMMAR_NAME.get(lang_key)
    if grammar_name is None:
        logger.warning("extract_code_symbols: unsupported language %r for %s", language, path)
        return []

    # Check per-language skip directive.
    lang_directive = config.languages.get(lang_key) or config.languages.get(grammar_name)
    if lang_directive == "skip":
        logger.debug("Skipping %s (language=%r is configured as 'skip')", path, language)
        return []

    # Resolve repo-relative path (forward-slash form per Anchor.path validator).
    try:
        path_resolved = path.resolve()
        repo_resolved = repo_root.resolve()
        rel = path_resolved.relative_to(repo_resolved)
    except ValueError:
        # path is not under repo_root; use as-is (best-effort)
        rel = path
    path_str = rel.as_posix()

    # File size guard (DESIGN.md §15.4).
    try:
        file_size = path.stat().st_size
    except OSError as exc:
        logger.warning("extract_code_symbols: cannot stat %s: %s", path, exc)
        return []
    if file_size > max_file_size_bytes:
        logger.warning(
            "extract_code_symbols: %s exceeds max_file_size (%d bytes); skipping",
            path,
            max_file_size_bytes,
        )
        return []

    # Read source bytes.
    try:
        src_bytes = path.read_bytes()
    except OSError as exc:
        logger.warning("extract_code_symbols: cannot read %s: %s", path, exc)
        return []

    if not src_bytes:
        return []

    # SR3-4 / §15.4: detect UTF-16 BOM and skip with a clear warning,
    # mirroring extract_markdown's behaviour.  Tree-sitter would
    # otherwise treat the entire byte stream as a single ERROR node
    # and silently produce zero anchors with no diagnostic.
    if src_bytes.startswith(b"\xff\xfe") or src_bytes.startswith(b"\xfe\xff"):
        logger.warning(
            "extract_code_symbols: skipping %s — UTF-16 encoded; transcode to UTF-8 to index",
            path,
        )
        return []

    # Resolve symbol kinds (config override or language default).
    config_kinds = config.symbol_kinds.get(lang_key) or config.symbol_kinds.get(grammar_name)
    if config_kinds is not None:
        kinds: set[str] = set(config_kinds)
    else:
        kinds = set(_DEFAULT_SYMBOL_KINDS.get(grammar_name, []))

    if not kinds:
        return []

    # Parse with tree-sitter.
    try:
        from tree_sitter_language_pack import get_parser

        parser = get_parser(grammar_name)  # type: ignore[arg-type]
        tree = parser.parse(src_bytes)
    except Exception as exc:
        logger.warning("extract_code_symbols: parse error for %s: %s", path, exc)
        return []

    root = tree.root_node

    # SR3-7: tree-sitter recovers from many syntax errors but its
    # ERROR-node insertion can swallow trailing top-level declarations
    # without any signal to the operator.  Emit a single warning when
    # the parse contains errors so users can correlate "missing
    # symbols" with "unparseable file".
    if root.has_error:
        logger.warning(
            "extract_code_symbols: %s has syntax errors; some symbols may be missing",
            path,
        )

    # Dispatch to language-specific walker.
    if grammar_name == "python":
        records = _walk_python(list(root.children), src_bytes, kinds, "")
    elif grammar_name in ("typescript", "tsx", "javascript", "jsx"):
        records = _walk_typescript(list(root.children), src_bytes, kinds)
    elif grammar_name == "zig":
        records = _walk_zig(root, src_bytes, kinds)
    elif grammar_name == "go":
        records = _walk_go(list(root.children), src_bytes, kinds)
    elif grammar_name == "rust":
        records = _walk_rust(list(root.children), src_bytes, kinds)
    else:
        logger.warning(
            "extract_code_symbols: no walker for grammar %r (language=%r)",
            grammar_name,
            language,
        )
        return []

    # Build Anchor objects.
    # SR5-6: file-level test detection.  SR5-5 test-framework anchors
    # always carry is_test=True regardless of filename (a ``describe``
    # call is a test construct even if it lives outside the test
    # filename heuristic — rare but real).  Detection key: a symbol
    # path that contains "::" was emitted by the SR5-5 nested walker,
    # OR the leaf segment matches a known test-fn prefix.
    file_is_test = is_test_path(path_str)
    anchors: list[Anchor] = []
    for symbol_path, raw_content, def_line, def_char in records:
        is_sr5_5_anchor = "::" in symbol_path or any(
            symbol_path.startswith(f"{fn}.") or symbol_path.startswith(f"{fn}:")
            for fn in (*_TS_TEST_NAMED_FNS, *_TS_TEST_HOOK_FNS)
        )
        anchor = _make_anchor(
            path_str,
            symbol_path,
            raw_content,
            def_line,
            def_char,
            is_test=file_is_test or is_sr5_5_anchor,
        )
        anchors.append(anchor)

    return anchors
