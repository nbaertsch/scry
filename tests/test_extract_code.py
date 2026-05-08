"""Tests for the tree-sitter code symbol extractor (W1b).

Covers DESIGN.md §3.2, §5.4, §6, §15.3 edge cases.

Fixture structure
-----------------
All source fixtures are defined as in-module byte strings and written to
``tmp_path`` by the relevant test.  This keeps the test file self-contained
and avoids an external fixture directory for W1b.

Test matrix
-----------
* Python  : function, class with methods, decorated function, nested class,
            same-name collision (@2), async function
* TypeScript : function, class with methods, interface, type alias,
               overloaded function (@<sig-hash[:6]>)
* Zig     : fn (with pub), struct ContainerDecl, enum ContainerDecl
* Config  : granularity="file" falls back to symbol mode (warning logged)
* Config  : languages.python="skip" returns empty list
* Edge    : empty file returns empty list
* Edge    : file too large returns empty list
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from scry.extract.code import canonicalize, extract_code_symbols
from scry.models import AnchorType, CodeAnchorsConfig, TransitiveHashStatus

# ---------------------------------------------------------------------------
# Shared fixture sources
# ---------------------------------------------------------------------------

_PY_SRC = b'''\
def top_func():
    """A top-level function."""
    pass


class MyClass:
    def method(self):
        pass

    def another_method(self) -> int:
        return 42

    class Nested:
        def nested_method(self):
            pass


@decorator
def decorated():
    pass


async def async_func():
    pass


class Config:
    x: int = 0


class Config:
    y: str = ""
'''

_TS_SRC = b"""\
function greet(name: string): void {}

class MyClass {
    method(): void {}
    staticMethod(): string { return ""; }
}

interface IShape {
    area(): number;
}

type StringOrNumber = string | number;

function overloaded(x: string): void;
function overloaded(x: number): void;
function overloaded(x: any): void {}
"""

_ZIG_SRC = (
    b'const std = @import("std");\n\n'
    b"pub fn main() void {\n"
    b"    _ = std;\n"
    b"}\n\n"
    b"const MyStruct = struct {\n"
    b"    x: i32 = 0,\n"
    b"    y: i32 = 0,\n"
    b"};\n\n"
    b"const Direction = enum {\n"
    b"    North,\n"
    b"    South,\n"
    b"};\n\n"
    b"fn helper() i32 {\n"
    b"    return 42;\n"
    b"}\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ids(anchors: list) -> list[str]:  # type: ignore[type-arg]
    return [a.id for a in anchors]


def _names(anchors: list) -> list[str]:  # type: ignore[type-arg]
    return [a.symbol_name for a in anchors]


def _write(tmp_path: Path, name: str, src: bytes) -> tuple[Path, Path]:
    """Write *src* to *tmp_path/name* and return (file_path, repo_root)."""
    p = tmp_path / name
    p.write_bytes(src)
    return p, tmp_path


# ---------------------------------------------------------------------------
# Python extraction tests
# ---------------------------------------------------------------------------


class TestPythonExtraction:
    def test_top_level_function(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        ids = _ids(anchors)
        assert "mod.py:top_func" in ids

    def test_class_anchor(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        ids = _ids(anchors)
        assert "mod.py:MyClass" in ids

    def test_class_methods_extracted(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        ids = _ids(anchors)
        assert "mod.py:MyClass.method" in ids
        assert "mod.py:MyClass.another_method" in ids

    def test_nested_class_and_method(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        ids = _ids(anchors)
        assert "mod.py:MyClass.Nested" in ids
        assert "mod.py:MyClass.Nested.nested_method" in ids

    def test_decorated_function_uses_inner_name(self, tmp_path: Path) -> None:
        """Decorated definition → anchor symbol_name is the inner function (§15.3)."""
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        ids = _ids(anchors)
        assert "mod.py:decorated" in ids
        # The decorated_definition wrapper must not appear as a separate entry.
        assert not any("decorated_definition" in aid for aid in ids)

    def test_async_function(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        ids = _ids(anchors)
        assert "mod.py:async_func" in ids

    def test_same_name_collision(self, tmp_path: Path) -> None:
        """Two ``Config`` classes at module level → bare + @2 (§15.3)."""
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        ids = _ids(anchors)
        assert "mod.py:Config" in ids
        assert "mod.py:Config@2" in ids

    def test_anchor_type_is_code(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        assert all(a.type == AnchorType.CODE for a in anchors)

    def test_transitive_hash_status_lsp_unavailable(self, tmp_path: Path) -> None:
        """Wave 1: all code anchors carry LSP_UNAVAILABLE (§5.3)."""
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        assert all(
            a.transitive_hash_status == TransitiveHashStatus.LSP_UNAVAILABLE for a in anchors
        )

    def test_content_hash_format(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        for a in anchors:
            assert a.content_hash.startswith("sha256:")
            assert len(a.content_hash) == len("sha256:") + 64

    def test_path_is_repo_relative_posix(self, tmp_path: Path) -> None:
        sub = tmp_path / "pkg"
        sub.mkdir()
        p = sub / "mod.py"
        p.write_bytes(_PY_SRC)
        anchors = extract_code_symbols(p, tmp_path, language="python")
        assert all(a.path == "pkg/mod.py" for a in anchors)

    def test_path_no_backslash(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="python")
        assert all("\\" not in a.path for a in anchors)

    def test_language_alias_py(self, tmp_path: Path) -> None:
        """Language alias ``'py'`` resolves to the Python grammar."""
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        anchors = extract_code_symbols(p, root, language="py")
        assert any(a.id.endswith(":top_func") for a in anchors)


# ---------------------------------------------------------------------------
# TypeScript extraction tests
# ---------------------------------------------------------------------------


class TestTypescriptExtraction:
    def test_function_declaration(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="typescript")
        ids = _ids(anchors)
        assert "mod.ts:greet" in ids

    def test_class_anchor(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="typescript")
        ids = _ids(anchors)
        assert "mod.ts:MyClass" in ids

    def test_class_methods_extracted(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="typescript")
        ids = _ids(anchors)
        assert "mod.ts:MyClass.method" in ids
        assert "mod.ts:MyClass.staticMethod" in ids

    def test_interface_declaration(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="typescript")
        ids = _ids(anchors)
        assert "mod.ts:IShape" in ids

    def test_type_alias_declaration(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="typescript")
        ids = _ids(anchors)
        assert "mod.ts:StringOrNumber" in ids

    def test_overloaded_function_sig_hash_suffix(self, tmp_path: Path) -> None:
        """Overload signatures get ``@<sig-hash[:6]>`` suffix (§15.3).

        Format aligns with :func:`scry.anchor_id.derive_code_id` so an ID
        minted here is byte-identical to one minted through that helper.
        """
        import re

        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="typescript")
        # Anchors named overloaded@<6-hex-chars> are the overload signatures
        overload_re = re.compile(r"^mod\.ts:overloaded@[0-9a-f]{6}$")
        overload_ids = [a.id for a in anchors if overload_re.match(a.id)]
        # Two overload signatures → two @... anchors with different hashes.
        assert len(overload_ids) == 2, f"expected 2 overload ids, got {overload_ids}"
        # They must have distinct hashes.
        assert overload_ids[0] != overload_ids[1]

    def test_overload_implementation_extracted(self, tmp_path: Path) -> None:
        """The implementation function_declaration is also extracted."""
        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="typescript")
        ids = _ids(anchors)
        assert "mod.ts:overloaded" in ids

    def test_language_alias_ts(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="ts")
        assert any(a.id.endswith(":greet") for a in anchors)

    def test_symbol_name_for_overload_is_bare_name(self, tmp_path: Path) -> None:
        """symbol_name for overload anchors is the function name (no hash)."""
        import re

        p, root = _write(tmp_path, "mod.ts", _TS_SRC)
        anchors = extract_code_symbols(p, root, language="typescript")
        overload_re = re.compile(r"^mod\.ts:overloaded@[0-9a-f]{6}$")
        overloads = [a for a in anchors if overload_re.match(a.id)]
        for a in overloads:
            assert a.symbol_name == "overloaded"


# ---------------------------------------------------------------------------
# Zig extraction tests
# ---------------------------------------------------------------------------


class TestZigExtraction:
    def test_fn_extracted(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "main.zig", _ZIG_SRC)
        anchors = extract_code_symbols(p, root, language="zig")
        ids = _ids(anchors)
        assert "main.zig:main" in ids

    def test_pub_fn_content_includes_pub(self, tmp_path: Path) -> None:
        """Content text for `pub fn main` must start with 'pub'."""
        p, root = _write(tmp_path, "main.zig", _ZIG_SRC)
        anchors = extract_code_symbols(p, root, language="zig")
        main_anchor = next(a for a in anchors if a.id.endswith(":main"))
        assert main_anchor.content_text.startswith("pub fn main")

    def test_private_fn_extracted(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "main.zig", _ZIG_SRC)
        anchors = extract_code_symbols(p, root, language="zig")
        ids = _ids(anchors)
        assert "main.zig:helper" in ids

    def test_struct_container_decl(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "main.zig", _ZIG_SRC)
        anchors = extract_code_symbols(p, root, language="zig")
        ids = _ids(anchors)
        assert "main.zig:MyStruct" in ids

    def test_enum_container_decl(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "main.zig", _ZIG_SRC)
        anchors = extract_code_symbols(p, root, language="zig")
        ids = _ids(anchors)
        assert "main.zig:Direction" in ids

    def test_no_anchors_for_import_const(self, tmp_path: Path) -> None:
        """``const std = @import(...)`` is a VarDecl without ContainerDecl → skipped."""
        p, root = _write(tmp_path, "main.zig", _ZIG_SRC)
        anchors = extract_code_symbols(p, root, language="zig")
        ids = _ids(anchors)
        assert "main.zig:std" not in ids


# ---------------------------------------------------------------------------
# Config / behaviour edge-case tests
# ---------------------------------------------------------------------------


class TestConfigEdgeCases:
    def test_skip_language_returns_empty(self, tmp_path: Path) -> None:
        """``languages.python: skip`` → empty list returned."""
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        cfg = CodeAnchorsConfig(languages={"python": "skip"})
        anchors = extract_code_symbols(p, root, language="python", config=cfg)
        assert anchors == []

    def test_granularity_file_falls_back_to_symbol(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """``granularity="file"`` is unsupported in Wave 1; falls back to symbol.

        DESIGN.md §13 open question #13: Wave 1 logs a warning and continues
        in symbol extraction mode rather than erroring.
        """
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        cfg = CodeAnchorsConfig(granularity="file")
        with caplog.at_level(logging.WARNING, logger="scry.extract.code"):
            anchors = extract_code_symbols(p, root, language="python", config=cfg)
        # Must still return anchors (symbol mode fallback).
        assert len(anchors) > 0
        # Warning must be logged.
        assert any("granularity" in rec.message for rec in caplog.records)

    def test_custom_symbol_kinds_filter(self, tmp_path: Path) -> None:
        """Configuring only ``class_definition`` excludes functions."""
        p, root = _write(tmp_path, "mod.py", _PY_SRC)
        cfg = CodeAnchorsConfig(symbol_kinds={"python": ["class_definition"]})
        anchors = extract_code_symbols(p, root, language="python", config=cfg)
        ids = _ids(anchors)
        assert "mod.py:MyClass" in ids
        assert "mod.py:top_func" not in ids

    def test_unsupported_language_returns_empty(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "mod.rs", b"fn main() {}")
        anchors = extract_code_symbols(p, root, language="rust")
        assert anchors == []


# ---------------------------------------------------------------------------
# File-level edge cases
# ---------------------------------------------------------------------------


class TestFileEdgeCases:
    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        p, root = _write(tmp_path, "empty.py", b"")
        anchors = extract_code_symbols(p, root, language="python")
        assert anchors == []

    def test_whitespace_only_file(self, tmp_path: Path) -> None:
        """A file with no declarations → empty list."""
        p, root = _write(tmp_path, "blank.py", b"\n\n# just a comment\n")
        anchors = extract_code_symbols(p, root, language="python")
        assert anchors == []

    def test_file_too_large_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Files exceeding max_file_size_bytes are skipped with a warning."""
        p, root = _write(tmp_path, "big.py", _PY_SRC)
        # Monkeypatch stat to report a huge file size without actually creating one.
        import os

        orig_stat = os.stat

        def fake_stat(path: str | bytes | int, **kwargs: object) -> os.stat_result:  # type: ignore[misc]
            result = orig_stat(path, **kwargs)
            # Patch st_size to be > 5 MB.
            return os.stat_result(
                (
                    result.st_mode,
                    result.st_ino,
                    result.st_dev,
                    result.st_nlink,
                    result.st_uid,
                    result.st_gid,
                    6 * 1024 * 1024,  # 6 MB - exceeds default 5 MB limit
                    result.st_atime,
                    result.st_mtime,
                    result.st_ctime,
                )
            )

        monkeypatch.setattr(os, "stat", fake_stat)
        anchors = extract_code_symbols(p, root, language="python")
        assert anchors == []

    def test_crlf_normalization_same_hash(self, tmp_path: Path) -> None:
        """CRLF and LF sources for the same content produce identical content_hash."""
        lf_src = b"def f():\n    pass\n"
        crlf_src = b"def f():\r\n    pass\r\n"
        p_lf, root = _write(tmp_path, "lf.py", lf_src)
        p_crlf, _ = _write(tmp_path, "crlf.py", crlf_src)
        a_lf = extract_code_symbols(p_lf, root, language="python")
        a_crlf = extract_code_symbols(p_crlf, root, language="python")
        assert len(a_lf) == 1
        assert len(a_crlf) == 1
        assert a_lf[0].content_hash == a_crlf[0].content_hash


# ---------------------------------------------------------------------------
# Canonicalize unit tests
# ---------------------------------------------------------------------------


class TestCanonicalize:
    def test_strips_bom(self) -> None:
        assert canonicalize("\ufeffhello\n") == "hello\n"

    def test_normalises_crlf(self) -> None:
        assert canonicalize("a\r\nb\r\n") == "a\nb\n"

    def test_normalises_bare_cr(self) -> None:
        assert canonicalize("a\rb\r") == "a\nb\n"

    def test_trims_trailing_whitespace(self) -> None:
        assert canonicalize("hello   \nworld  \n") == "hello\nworld\n"

    def test_single_trailing_newline(self) -> None:
        assert canonicalize("hello\n\n\n") == "hello\n"

    def test_no_trailing_newline_preserved(self) -> None:
        """W1c semantics: input without trailing newline produces output without one.

        Critical cross-module invariant — `extract.code.canonicalize` MUST
        equal `anchor_id.canonicalize_content` byte-for-byte so that
        hashes computed for tree-sitter slices (which typically have no
        trailing newline) agree with hashes computed by the markdown
        extractor or anchor_id directly.
        """
        assert canonicalize("hello") == "hello"

    def test_canonicalize_is_w1c_canonicalize_content(self) -> None:
        """Cross-module invariant: extract.code.canonicalize IS canonicalize_content.

        Per the W1b module docstring and the BLOCKING fix from review-w1b,
        these are not two functions that happen to agree — they are the
        same function object exported under two names. This test pins
        that invariant so a future refactor can't silently re-introduce
        a divergent local copy.
        """
        from scry.anchor_id import canonicalize_content

        assert canonicalize is canonicalize_content

    @pytest.mark.parametrize(
        "text",
        [
            "",
            "\n",
            "noEOL",
            "with-eol\n",
            "a\r\nb",
            "a\r\nb\r\n",
            "a\rb",
            "a\rb\r",
            "trailing-spaces  \n",
            "tabs\there\n",
            "\ufeffleading-bom",
            "many\n\n\nblanks",
            "def f():\n    pass",
            "def f():\n    pass\n",
        ],
    )
    def test_canonicalize_matches_anchor_id_byte_for_byte(self, text: str) -> None:
        """For every input the W1b and W1c canonicalizers MUST agree exactly."""
        from scry.anchor_id import canonicalize_content

        assert canonicalize(text) == canonicalize_content(text)
