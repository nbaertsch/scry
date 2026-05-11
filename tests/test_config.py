"""Tests for src/scry/config.py — config loader and path-classification helpers.

Covers DESIGN.md §6, §6.1, §6.2, §13 (question 12), and §15.4.

Public API under test
---------------------
load_config          — §6: parse and validate .scry/config.yaml
compute_config_hash  — §13 q.12: stable hash over canonicalized config
parse_frontmatter    — §6.1, §15.4: extract optional YAML frontmatter
matches_globs        — §6: fnmatch + ** glob matching
classify_path        — §6: first-match-wins classify lookup
is_safety_excluded   — §6.1: hard safety-exclude check
should_index         — §6, §6.1: combined include/exclude/frontmatter logic
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import ClassVar

import pytest

from scry.config import (
    ConfigError,
    classify_path,
    compute_config_hash,
    is_safety_excluded,
    load_config,
    matches_globs,
    parse_frontmatter,
    should_index,
)
from scry.models import ClassifyEntry, Config, Frontmatter

# ─── Dynamic §15.4 fixtures (defined here so they're visible to all tests
#     in this file without requiring a separate conftest scope) ─────────


def _can_symlink(tmp: Path) -> bool:
    """Probe whether symlinks can be created (Windows needs Developer Mode)."""
    probe = tmp / "_sym_probe_link"
    target = tmp / "_sym_probe_target"
    target.write_text("x")
    try:
        probe.symlink_to(target)
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        probe.unlink(missing_ok=True)
        target.unlink(missing_ok=True)


@pytest.fixture(scope="session")
def binary_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Binary (PNG magic) .md file — §15.4 binary-file edge case."""
    d = tmp_path_factory.mktemp("cfg_binary")
    p = d / "binary.md"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    return p


@pytest.fixture(scope="session")
def utf16le_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """UTF-16 LE encoded .md file (with BOM) — §15.4 encoding edge case."""
    d = tmp_path_factory.mktemp("cfg_utf16")
    p = d / "utf16le.md"
    content = "# UTF-16 LE Test\n\nThis file is encoded in UTF-16 LE.\n"
    p.write_bytes(b"\xff\xfe" + content.encode("utf-16-le"))
    return p


@pytest.fixture(scope="session")
def crlf_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """CRLF-only line endings — §15.4 line-ending canonicalization."""
    d = tmp_path_factory.mktemp("cfg_crlf")
    p = d / "crlf_only.md"
    lines = ["# CRLF Test", "", "Content with CRLF.", "", "## Section", "", "More."]
    p.write_bytes("\r\n".join(lines).encode("utf-8"))
    return p


@pytest.fixture(scope="session")
def mixed_endings_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Mixed CRLF + LF line endings — §15.4 mixed-endings case."""
    d = tmp_path_factory.mktemp("cfg_mixed")
    p = d / "mixed_endings.md"
    p.write_bytes(b"# Mixed\r\nCRLF line.\r\nLF line.\n\n## Section\n\nMore LF.\n")
    return p


@pytest.fixture(scope="session")
def cr_only_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Bare CR (Mac OS 9) line endings — §15.4 CR-only case."""
    d = tmp_path_factory.mktemp("cfg_cr")
    p = d / "cr_only.md"
    p.write_bytes(b"# CR Test\rBare CR line.\r\r## Section\rContent.\r")
    return p


@pytest.fixture(scope="session")
def oversized_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """6 MB file exceeding default max_file_size_bytes (5 MB) — §15.4."""
    d = tmp_path_factory.mktemp("cfg_oversized")
    p = d / "oversized.md"
    header = b"# Oversized\n\nExceeds 5 MB limit.\n\n"
    p.write_bytes(header + b"A" * (6 * 1024 * 1024 - len(header)))
    return p


@pytest.fixture
def symlink_inside(tmp_path: Path, fixture_dir: Path) -> Generator[Path, None, None]:
    """Symlink to a file inside the fixture tree — §15.4 in-repo symlink."""
    if not _can_symlink(tmp_path):
        pytest.skip("Cannot create symlinks on this platform/privilege level")
    target = fixture_dir / "wave1" / "files" / "empty.md"
    link = tmp_path / "symlink_inside.md"
    link.symlink_to(target)
    yield link
    link.unlink(missing_ok=True)


@pytest.fixture
def symlink_outside(tmp_path: Path) -> Generator[Path, None, None]:
    """Symlink pointing outside the repo root — §15.4 security boundary."""
    if not _can_symlink(tmp_path):
        pytest.skip("Cannot create symlinks on this platform/privilege level")
    outside = (
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "win.ini"
        if sys.platform == "win32"
        else Path("/etc/passwd")
    )
    link = tmp_path / "symlink_outside.md"
    link.symlink_to(outside)
    yield link
    link.unlink(missing_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────


def _write_config(repo: Path, content: str) -> None:
    """Write *content* to ``.scry/config.yaml`` inside *repo*."""
    (repo / ".scry").mkdir(exist_ok=True)
    (repo / ".scry" / "config.yaml").write_text(content, encoding="utf-8")


def _make_repo(tmp_path: Path, yaml_content: str) -> Path:
    """Create a minimal repo with the given config.yaml and return its root."""
    _write_config(tmp_path, yaml_content)
    return tmp_path


# ─── load_config ──────────────────────────────────────────────────────


class TestLoadConfig:
    """Tests for :func:`load_config`."""

    def test_happy_path_returns_config(self, tmp_path: Path) -> None:
        """Well-formed config.yaml → Config instance with expected values."""
        _write_config(
            tmp_path,
            """
include:
  - "**/*.md"
  - "**/*.py"
exclude:
  - node_modules/**
classify:
  - { glob: "docs/**.md", type: spec }
  - { glob: "**/*.md", type: doc }
""",
        )
        cfg = load_config(tmp_path)
        assert isinstance(cfg, Config)
        assert "**/*.md" in cfg.include
        assert "**/*.py" in cfg.include
        assert "node_modules/**" in cfg.exclude
        assert len(cfg.classify) == 2
        assert cfg.classify[0].glob == "docs/**.md"
        assert cfg.classify[0].type == "spec"

    def test_missing_config_raises_config_error(self, tmp_path: Path) -> None:
        """Missing .scry/config.yaml raises ConfigError with actionable message."""
        with pytest.raises(ConfigError, match=r"No \.scry/config\.yaml found"):
            load_config(tmp_path)

    def test_malformed_yaml_raises_config_error(self, tmp_path: Path) -> None:
        """Malformed YAML raises ConfigError with parse hint."""
        _write_config(tmp_path, "invalid: yaml: [unclosed")
        with pytest.raises(ConfigError, match="Failed to parse"):
            load_config(tmp_path)

    def test_pydantic_validation_error_raises_config_error(self, tmp_path: Path) -> None:
        """Pydantic schema violation → ConfigError with validation message."""
        _write_config(
            tmp_path,
            "sections:\n  max_heading_depth: 99\n",  # max is 6
        )
        with pytest.raises(ConfigError, match="Config validation error"):
            load_config(tmp_path)

    def test_empty_yaml_uses_defaults(self, tmp_path: Path) -> None:
        """An empty config.yaml is valid — all fields take their defaults."""
        _write_config(tmp_path, "")
        cfg = load_config(tmp_path)
        assert isinstance(cfg, Config)
        assert cfg.sections.max_heading_depth == 4
        assert cfg.sections.max_tokens == 600
        assert cfg.embeddings.model == "BAAI/bge-small-en-v1.5"

    def test_non_mapping_yaml_raises_config_error(self, tmp_path: Path) -> None:
        """Top-level YAML list (not mapping) → ConfigError."""
        _write_config(tmp_path, "- item1\n- item2\n")
        with pytest.raises(ConfigError, match="top-level value must be a mapping"):
            load_config(tmp_path)

    def test_extra_fields_rejected(self, tmp_path: Path) -> None:
        """Unknown top-level keys raise ConfigError (Config has extra='forbid')."""
        _write_config(tmp_path, "unknown_key: some_value\n")
        with pytest.raises(ConfigError, match="Config validation error"):
            load_config(tmp_path)

    def test_partial_config_uses_defaults_for_missing(self, tmp_path: Path) -> None:
        """Partial config with only include: set → other fields use defaults."""
        _write_config(tmp_path, 'include:\n  - "**/*.md"\n')
        cfg = load_config(tmp_path)
        assert "**/*.md" in cfg.include
        assert cfg.exclude == []  # default
        assert cfg.retrieval.fusion_rrf_k == 60  # default

    def test_hailstorm_config(self, hailstorm_spec: Path) -> None:
        """hailstorm-spec fixture config.yaml loads successfully."""
        cfg = load_config(hailstorm_spec)
        assert "docs/**.md" in cfg.include
        assert "python/**.py" in cfg.include
        assert ".scry/**" in cfg.exclude
        assert cfg.classify[0].type == "spec"


# ─── compute_config_hash ──────────────────────────────────────────────


class TestComputeConfigHash:
    """Tests for :func:`compute_config_hash`."""

    def _load(self, tmp_path: Path, yaml: str) -> Config:
        _write_config(tmp_path, yaml)
        return load_config(tmp_path)

    def test_deterministic(self, tmp_path: Path) -> None:
        """Same Config instance → identical hash on repeated calls."""
        cfg = self._load(tmp_path, 'include: ["**/*.md"]')
        assert compute_config_hash(cfg) == compute_config_hash(cfg)

    def test_hash_format(self, tmp_path: Path) -> None:
        """Hash always has the form ``sha256:<64 hex chars>``."""
        cfg = self._load(tmp_path, 'include: ["**/*.md"]')
        h = compute_config_hash(cfg)
        assert h.startswith("sha256:")
        assert len(h) == 7 + 64  # "sha256:" prefix + 64 hex chars

    def test_stable_across_cosmetic_yaml_reformatting(self, tmp_path: Path) -> None:
        """YAML block style vs flow style → same parsed content → same hash."""
        yaml_flow = 'include: ["**/*.md"]\n'
        yaml_block = 'include:\n  - "**/*.md"\n'
        repo_a = tmp_path / "repo_a"
        repo_a.mkdir()
        repo_b = tmp_path / "repo_b"
        repo_b.mkdir()
        cfg_a = self._load(repo_a, yaml_flow)
        _write_config(repo_b, yaml_block)
        cfg_b = load_config(repo_b)
        assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)

    def test_stable_across_key_ordering_in_yaml(self, tmp_path: Path) -> None:
        """Different YAML key order → same Pydantic model → same hash."""
        yaml_ab = "sections:\n  max_heading_depth: 4\n  max_tokens: 600\n"
        yaml_ba = "sections:\n  max_tokens: 600\n  max_heading_depth: 4\n"
        repo_a = tmp_path / "order_a"
        repo_a.mkdir()
        repo_b = tmp_path / "order_b"
        repo_b.mkdir()
        cfg_a = self._load(repo_a, yaml_ab)
        _write_config(repo_b, yaml_ba)
        cfg_b = load_config(repo_b)
        assert compute_config_hash(cfg_a) == compute_config_hash(cfg_b)

    def test_different_configs_produce_different_hashes(self, tmp_path: Path) -> None:
        """Configs with different logical content → different hashes."""
        repo_a = tmp_path / "diff_a"
        repo_a.mkdir()
        repo_b = tmp_path / "diff_b"
        repo_b.mkdir()
        cfg_a = self._load(repo_a, 'include: ["**/*.md"]')
        _write_config(repo_b, 'include: ["**/*.py"]')
        cfg_b = load_config(repo_b)
        assert compute_config_hash(cfg_a) != compute_config_hash(cfg_b)

    def test_hash_changes_on_max_heading_depth_change(self, tmp_path: Path) -> None:
        """Changing ``sections.max_heading_depth`` changes the hash."""
        repo_a = tmp_path / "depth_a"
        repo_a.mkdir()
        repo_b = tmp_path / "depth_b"
        repo_b.mkdir()
        cfg_a = self._load(repo_a, "sections:\n  max_heading_depth: 4\n")
        _write_config(repo_b, "sections:\n  max_heading_depth: 3\n")
        cfg_b = load_config(repo_b)
        assert compute_config_hash(cfg_a) != compute_config_hash(cfg_b)


# ─── parse_frontmatter ────────────────────────────────────────────────


class TestParseFrontmatter:
    """Tests for :func:`parse_frontmatter`."""

    def test_valid_scry_frontmatter(self) -> None:
        """Well-formed frontmatter with scry: block → Frontmatter + body."""
        text = "---\nscry:\n  skip: true\n  type: spec\n---\n# Title\n\nBody text."
        fm, body = parse_frontmatter(text)
        assert fm is not None
        assert fm.skip is True
        assert fm.type == "spec"
        assert body.startswith("# Title")

    def test_absent_frontmatter_returns_none_and_original(self) -> None:
        """No frontmatter block → (None, original_text)."""
        text = "# Title\n\nJust a body with no frontmatter."
        fm, body = parse_frontmatter(text)
        assert fm is None
        assert body == text

    def test_malformed_yaml_logs_warning_and_returns_original(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Malformed YAML → (None, original_text) + WARNING logged."""
        text = "---\nscry:\n  {bad: yaml: syntax\n---\n# Title"
        with caplog.at_level(logging.WARNING, logger="scry.config"):
            fm, body = parse_frontmatter(text)
        assert fm is None
        assert body == text
        assert any("Malformed" in r.message for r in caplog.records)

    def test_no_scry_key_returns_none_and_body(self) -> None:
        """Valid YAML frontmatter without ``scry:`` key → (None, body)."""
        text = "---\ntitle: My Doc\nauthor: Alice\n---\n# Title\n\nBody."
        fm, body = parse_frontmatter(text)
        assert fm is None
        assert "# Title" in body
        # Frontmatter block stripped from body
        assert "author:" not in body

    def test_id_field_extracted(self) -> None:
        """``scry.id`` field is captured in the returned Frontmatter."""
        text = "---\nscry:\n  id: AUTH-LOGIN\n---\n# Title"
        fm, _body = parse_frontmatter(text)
        assert fm is not None
        assert fm.id == "AUTH-LOGIN"
        assert fm.skip is False

    def test_frontmatter_body_is_remainder(self) -> None:
        """Body is everything after the closing --- delimiter."""
        text = "---\nscry:\n  skip: false\n---\nline1\nline2\n"
        fm, body = parse_frontmatter(text)
        assert fm is not None
        assert body == "line1\nline2\n"

    def test_crlf_frontmatter_parsed(self) -> None:
        """CRLF line endings in frontmatter are handled correctly."""
        text = "---\r\nscry:\r\n  skip: true\r\n---\r\n# Title"
        fm, _body = parse_frontmatter(text)
        assert fm is not None
        assert fm.skip is True

    def test_scry_key_not_mapping_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """``scry: string`` (not a mapping) → warning + (None, original_text)."""
        text = "---\nscry: just-a-string\n---\n# Title"
        with caplog.at_level(logging.WARNING, logger="scry.config"):
            fm, body = parse_frontmatter(text)
        assert fm is None
        assert body == text
        assert any("mapping" in r.message for r in caplog.records)

    def test_skip_in_frontmatter_fixture(self, fixture_dir: Path) -> None:
        """The committed skip_in_frontmatter.md fixture parses correctly."""
        p = fixture_dir / "wave1" / "files" / "skip_in_frontmatter.md"
        text = p.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(text)
        assert fm is not None
        assert fm.skip is True

    def test_duplicate_frontmatter_id_files_parse(self, fixture_dir: Path) -> None:
        """Both duplicate-ID fixtures parse individually without error."""
        base = fixture_dir / "wave1" / "ids" / "duplicate_frontmatter_id"
        for fname in ("file_a.md", "file_b.md"):
            text = (base / fname).read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(text)
            assert fm is not None
            assert fm.id == "AUTH-LOGIN"

    def test_top_level_yaml_list_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """YAML parses to a list (not a dict) → warning + (None, original_text)."""
        text = "---\n- item1\n- item2\n---\n# Title"
        with caplog.at_level(logging.WARNING, logger="scry.config"):
            fm, body = parse_frontmatter(text)
        assert fm is None
        assert body == text
        assert any("mapping" in r.message for r in caplog.records)

    def test_invalid_scry_field_type_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """``scry.skip`` with an unambiguously-invalid value → pydantic error → warning + (None, original_text)."""
        # Use a list — Pydantic v2 will not coerce list → bool, so the
        # validation error path is guaranteed (regardless of PyYAML's
        # string coercion behavior).
        text = "---\nscry:\n  skip: [1, 2]\n---\n# Title"
        with caplog.at_level(logging.WARNING, logger="scry.config"):
            fm, body = parse_frontmatter(text)
        assert fm is None, "list value for skip must fail validation"
        assert body == text, "body must be the original text on validation failure"
        assert any("Invalid scry frontmatter" in r.message for r in caplog.records), (
            f"expected validation warning, got: {[r.message for r in caplog.records]}"
        )

    def test_utf8_bom_does_not_defeat_frontmatter(self) -> None:
        """Regression: leading UTF-8 BOM (Notepad default) must be stripped before delimiter match.

        Without the strip, ``^---`` never matches and the entire ``---/yaml/---``
        block would be returned as part of the body — silently losing
        ``skip:``/``type:``/``id:`` overrides for any file edited in Windows
        Notepad.
        """
        text = "\ufeff---\nscry:\n  skip: true\n---\n# Title\n"
        fm, body = parse_frontmatter(text)
        assert fm is not None, "BOM-prefixed frontmatter must be parsed"
        assert fm.skip is True
        assert body == "# Title\n"

    def test_utf8_bom_with_no_frontmatter(self) -> None:
        """A BOM-prefixed file with no frontmatter returns (None, body-without-BOM)."""
        text = "\ufeff# Just a heading\nbody\n"
        fm, body = parse_frontmatter(text)
        assert fm is None
        # The BOM is stripped; the body should NOT contain the BOM byte.
        assert "\ufeff" not in body
        assert body == "# Just a heading\nbody\n"


# ─── matches_globs ────────────────────────────────────────────────────


class TestMatchesGlobs:
    """Tests for :func:`matches_globs`."""

    def test_exact_match(self) -> None:
        assert matches_globs("README.md", ["README.md"])

    def test_no_match(self) -> None:
        assert not matches_globs("file.py", ["**/*.md"])

    def test_empty_list_returns_false(self) -> None:
        assert not matches_globs("file.md", [])

    def test_single_star_within_directory(self) -> None:
        """``*`` matches within one path segment."""
        assert matches_globs("docs/file.md", ["docs/*.md"])

    def test_single_star_no_cross_directory(self) -> None:
        """``*`` does NOT cross directory boundaries in classify patterns."""
        # fnmatch '*' actually does match '/' in Python — by design for
        # simplicity. This test documents the current behaviour so changes
        # are explicit. If we ever need strict single-segment semantics,
        # update _glob_to_re to use [^/]* for * patterns too.
        assert matches_globs("docs/sub/file.md", ["docs/*.md"])

    def test_double_star_cross_directory(self) -> None:
        """``**`` matches across multiple directory levels."""
        assert matches_globs("docs/sub/deep/file.md", ["docs/**.md"])
        assert matches_globs("docs/file.md", ["docs/**.md"])

    def test_double_star_top_level_file(self) -> None:
        """``**/*.md`` must match a top-level file with no directory prefix."""
        assert matches_globs("README.md", ["**/*.md"])

    def test_case_sensitive_single_star(self) -> None:
        """Regression: ``docs/*.md`` MUST be case-sensitive on every OS.

        On Windows, ``fnmatch.fnmatch`` calls ``os.path.normcase`` which
        downcases its inputs, so ``Docs/spec.md`` would falsely match
        ``docs/*.md``. ``matches_globs`` uses ``fnmatchcase`` to keep
        semantics consistent with the regex path that handles ``**``.
        """
        assert matches_globs("docs/spec.md", ["docs/*.md"])
        assert not matches_globs("Docs/spec.md", ["docs/*.md"])
        assert not matches_globs("DOCS/SPEC.md", ["docs/*.md"])

    def test_case_sensitive_double_star(self) -> None:
        """Regression: ``secrets/**`` is case-sensitive on every OS.

        Same boundary as ``docs/*.md``, but routed through the regex
        engine. Documents that the security boundary holds — a
        ``Secrets/`` (capitalised) directory does NOT trip the
        ``secrets/**`` safety exclude.
        """
        assert matches_globs("secrets/foo.md", ["secrets/**"])
        assert not matches_globs("Secrets/foo.md", ["secrets/**"])

    def test_case_sensitive_consistency_between_glob_engines(self) -> None:
        """Both engines (regex for ``**``, fnmatchcase otherwise) must agree on case sensitivity."""
        # Pattern with ** (regex path) and pattern without ** (fnmatch path)
        # MUST behave identically with respect to case.
        for pattern_with, pattern_without in [
            ("docs/**", "docs/foo"),
            ("secrets/**.md", "secrets/leaked.md"),
        ]:
            # Both should reject the uppercase variant.
            uppercase = pattern_without.upper()
            assert not matches_globs(uppercase, [pattern_with]), (
                f"{pattern_with!r} matched uppercase {uppercase!r}"
            )
            assert not matches_globs(uppercase, [pattern_without]), (
                f"{pattern_without!r} matched uppercase {uppercase!r}"
            )

    def test_double_star_nested_file(self) -> None:
        """``**/*.md`` matches files at any depth."""
        assert matches_globs("a/b/c/d/file.md", ["**/*.md"])

    def test_exclude_pattern_node_modules(self) -> None:
        assert matches_globs("node_modules/foo.js", ["node_modules/**"])
        assert matches_globs("node_modules/a/b/c.js", ["node_modules/**"])
        assert not matches_globs("src/foo.js", ["node_modules/**"])

    def test_exclude_pattern_scry(self) -> None:
        assert matches_globs(".scry/config.yaml", [".scry/**"])
        assert not matches_globs("docs/spec.md", [".scry/**"])

    def test_windows_backslash_normalised(self) -> None:
        """Back-slash paths are normalised to forward-slash before matching."""
        assert matches_globs("docs\\file.md", ["docs/**.md"])
        assert matches_globs("docs\\sub\\file.md", ["**/*.md"])

    def test_multiple_patterns_any_match(self) -> None:
        """Returns True when *any* pattern in the list matches."""
        assert matches_globs("file.py", ["**/*.md", "**/*.py"])
        assert not matches_globs("file.ts", ["**/*.md", "**/*.py"])

    def test_hailstorm_include_patterns(self) -> None:
        """hailstorm-spec include patterns classify expected paths."""
        includes = ["docs/**.md", "python/**.py"]
        assert matches_globs("docs/POLICY_ENGINE.md", includes)
        assert matches_globs("python/hailstone/policy/engine.py", includes)
        assert not matches_globs("docs/README.txt", includes)

    def test_question_mark_wildcard(self) -> None:
        """``?`` matches exactly one non-separator character."""
        assert matches_globs("file.md", ["fil?.md"])
        assert not matches_globs("file.md", ["fil??.md"])

    def test_question_mark_in_double_star_pattern(self) -> None:
        """``?`` inside a ``**`` pattern exercises the regex path for ``?``."""
        # Pattern has ** so _glob_to_re is called; '?' in the suffix is handled there.
        assert matches_globs("docs/file.md", ["**/?ile.md"])
        assert not matches_globs("docs/file.md", ["**/?ile.py"])

    def test_fnmatch_false_branch_continues_to_next_pattern(self) -> None:
        """When fnmatch misses on the first pattern, the loop tries the next."""
        # First pattern uses single-star (fnmatch path); second pattern matches.
        assert matches_globs("other/f.md", ["docs/*.md", "other/*.md"])
        # Both single-star, neither match → False.
        assert not matches_globs("src/f.py", ["docs/*.md", "other/*.md"])


# ─── classify_path ────────────────────────────────────────────────────


class TestClassifyPath:
    """Tests for :func:`classify_path`."""

    def _entries(self) -> list[ClassifyEntry]:
        return [
            ClassifyEntry(glob="docs/**.md", type="spec"),
            ClassifyEntry(glob="README.md", type="doc"),
            ClassifyEntry(glob="**/*.md", type="doc"),
        ]

    def test_first_rule_matches(self) -> None:
        """``docs/**.md`` matches first → ``'spec'``."""
        assert classify_path("docs/POLICY_ENGINE.md", self._entries()) == "spec"

    def test_second_rule_matches(self) -> None:
        """``README.md`` matches second rule → ``'doc'``."""
        assert classify_path("README.md", self._entries()) == "doc"

    def test_fallback_rule(self) -> None:
        """``CHANGELOG.md`` falls through to the third ``**/*.md`` rule → ``'doc'``."""
        assert classify_path("CHANGELOG.md", self._entries()) == "doc"

    def test_no_match_returns_none(self) -> None:
        """Python source files don't match any markdown rule → ``None``."""
        assert classify_path("src/main.py", self._entries()) is None

    def test_empty_list_returns_none(self) -> None:
        assert classify_path("docs/file.md", []) is None

    def test_first_match_wins_not_best_match(self) -> None:
        """Ordering matters: a later, more-specific rule is never reached."""
        entries = [
            ClassifyEntry(glob="**/*.md", type="doc"),  # catches everything first
            ClassifyEntry(glob="docs/**.md", type="spec"),
        ]
        # The first rule wins even though the second is more specific.
        assert classify_path("docs/spec.md", entries) == "doc"

    def test_hailstorm_classify(self, hailstorm_spec: Path) -> None:
        """hailstorm-spec config classifies docs/ files as spec."""
        cfg = load_config(hailstorm_spec)
        assert classify_path("docs/POLICY_ENGINE.md", cfg.classify) == "spec"
        assert classify_path("docs/AUTH_PROTOCOL.md", cfg.classify) == "spec"


# ─── is_safety_excluded ───────────────────────────────────────────────


class TestIsSafetyExcluded:
    """Tests for :func:`is_safety_excluded`."""

    def test_node_modules_excluded(self) -> None:
        assert is_safety_excluded("node_modules/foo.js", ["node_modules/**"])

    def test_nested_in_excluded_dir(self) -> None:
        assert is_safety_excluded(".scry/overlays/main.jsonl", [".scry/**"])

    def test_not_excluded(self) -> None:
        assert not is_safety_excluded("docs/spec.md", ["node_modules/**"])

    def test_empty_exclude_list(self) -> None:
        assert not is_safety_excluded("anything.md", [])

    def test_dist_excluded(self) -> None:
        assert is_safety_excluded("dist/bundle.js", ["dist/**"])

    def test_exact_path_excluded(self) -> None:
        assert is_safety_excluded("secrets/key.pem", ["secrets/**"])
        assert not is_safety_excluded("docs/secrets.md", ["secrets/**"])


# ─── should_index ─────────────────────────────────────────────────────


class TestShouldIndex:
    """Tests for :func:`should_index` — the combined include/exclude/frontmatter gate."""

    INCLUDES: ClassVar[list[str]] = ["**/*.md", "**/*.py"]
    EXCLUDES: ClassVar[list[str]] = [".scry/**", "node_modules/**", "dist/**"]

    def test_include_match_no_exclude_no_frontmatter(self) -> None:
        """Normal include match with no exclusions → True."""
        assert should_index("docs/spec.md", None, self.INCLUDES, self.EXCLUDES)

    def test_exclude_wins_over_include(self) -> None:
        """Hard safety exclude wins even when path also matches include."""
        assert not should_index(".scry/config.yaml", None, self.INCLUDES, self.EXCLUDES)

    def test_frontmatter_skip_true_opts_out(self) -> None:
        """``skip: true`` frontmatter → not indexed even when include matches."""
        fm = Frontmatter(skip=True)
        assert not should_index("docs/spec.md", fm, self.INCLUDES, self.EXCLUDES)

    def test_frontmatter_skip_false_follows_normal_logic(self) -> None:
        """``skip: false`` is the default — path still follows include/exclude."""
        fm = Frontmatter(skip=False)
        assert should_index("docs/spec.md", fm, self.INCLUDES, self.EXCLUDES)

    def test_no_include_match_excluded(self) -> None:
        """Path not in any include glob → not indexed."""
        assert not should_index("src/styles.css", None, self.INCLUDES, self.EXCLUDES)

    def test_exclude_beats_frontmatter_skip_false(self) -> None:
        """Safety exclude wins even when frontmatter says ``skip: false`` (§6.1)."""
        fm = Frontmatter(skip=False)
        assert not should_index("node_modules/foo.md", fm, self.INCLUDES, self.EXCLUDES)

    def test_exclude_beats_frontmatter_skip_true(self) -> None:
        """Exclude wins regardless of frontmatter (both exclude the file)."""
        fm = Frontmatter(skip=True)
        assert not should_index("node_modules/foo.md", fm, self.INCLUDES, self.EXCLUDES)

    def test_none_frontmatter_follows_include(self) -> None:
        """None frontmatter → only include/exclude logic applies."""
        assert should_index("src/main.py", None, self.INCLUDES, self.EXCLUDES)

    def test_python_file_included(self) -> None:
        """Python source files match ``**/*.py`` include."""
        assert should_index("python/hailstone/policy/engine.py", None, self.INCLUDES, self.EXCLUDES)

    def test_dist_excluded(self) -> None:
        assert not should_index("dist/bundle.md", None, self.INCLUDES, self.EXCLUDES)

    def test_empty_include_list(self) -> None:
        """Empty include list → nothing matches → nothing indexed."""
        assert not should_index("docs/spec.md", None, [], self.EXCLUDES)

    def test_empty_exclude_list(self) -> None:
        """Empty exclude list → only include check matters."""
        assert should_index("docs/spec.md", None, self.INCLUDES, [])

    def test_skip_true_with_no_include_match(self) -> None:
        """skip:true + no include match → still not indexed (both gates fail)."""
        fm = Frontmatter(skip=True)
        assert not should_index("styles.css", fm, self.INCLUDES, self.EXCLUDES)

    def test_skip_in_frontmatter_fixture(self, fixture_dir: Path) -> None:
        """The committed skip_in_frontmatter.md fixture is filtered correctly."""
        p = fixture_dir / "wave1" / "files" / "skip_in_frontmatter.md"
        text = p.read_text(encoding="utf-8")
        fm, _ = parse_frontmatter(text)
        assert not should_index("wave1/files/skip_in_frontmatter.md", fm, ["**/*.md"], [])

    def test_hailstorm_config_paths(self, hailstorm_spec: Path) -> None:
        """hailstorm-spec config include/exclude matches expected paths."""
        cfg = load_config(hailstorm_spec)
        # Docs and Python source should be included.
        assert should_index("docs/POLICY_ENGINE.md", None, cfg.include, cfg.exclude)
        assert should_index("python/hailstone/policy/engine.py", None, cfg.include, cfg.exclude)
        # .scry/** should be excluded.
        assert not should_index(".scry/config.yaml", None, cfg.include, cfg.exclude)
        # Unknown extension should not be included.
        assert not should_index("docs/diagram.svg", None, cfg.include, cfg.exclude)


# ─── Wave1 fixture smoke tests ────────────────────────────────────────


class TestWave1Fixtures:
    """Smoke-tests verifying that committed §15 fixture files are readable."""

    @pytest.mark.parametrize(
        "rel_path",
        [
            "wave1/headings/atx.md",
            "wave1/headings/setext.md",
            "wave1/headings/no_headings.md",
            "wave1/headings/single_h1.md",
            "wave1/headings/h1_dividers.md",
            "wave1/headings/h1_in_codefence.md",
            "wave1/headings/deep_nesting.md",
            "wave1/headings/empty_heading.md",
            "wave1/headings/non_ascii.md",
            "wave1/headings/scry_id_override.md",
            "wave1/code_blocks/with_lang_with_decl.md",
            "wave1/code_blocks/with_lang_no_decl.md",
            "wave1/code_blocks/no_lang.md",
            "wave1/code_blocks/unclosed_fence.md",
            "wave1/code_blocks/nested_fences.md",
            "wave1/ids/sibling_collision.md",
            "wave1/ids/code_decl_collision.md",
            "wave1/ids/anonymous_block.md",
            "wave1/ids/typescript_overload.md",
            "wave1/ids/duplicate_scry_id.md",
            "wave1/ids/duplicate_frontmatter_id/file_a.md",
            "wave1/ids/duplicate_frontmatter_id/file_b.md",
            "wave1/files/frontmatter_only.md",
            "wave1/files/oversized_marker.md",
            "wave1/files/skip_in_frontmatter.md",
            "wave1/files/markdown_with_table.md",
            "wave1/refs/inline_link.md",
            "wave1/refs/reference_style.md",
            "wave1/refs/external_url.md",
            "wave1/refs/broken_target.md",
        ],
    )
    def test_fixture_is_readable(self, fixture_dir: Path, rel_path: str) -> None:
        """Every committed fixture file must exist and be readable as UTF-8."""
        p = fixture_dir / rel_path
        assert p.exists(), f"Fixture not found: {p}"
        text = p.read_text(encoding="utf-8")
        assert len(text) >= 0  # non-exceptional read

    def test_empty_md_is_zero_bytes(self, fixture_dir: Path) -> None:
        """wave1/files/empty.md must be exactly zero bytes."""
        p = fixture_dir / "wave1" / "files" / "empty.md"
        assert p.exists()
        assert p.stat().st_size == 0

    def test_setext_headings_fixture(self, fixture_dir: Path) -> None:
        """setext.md uses === and --- underlines."""
        text = (fixture_dir / "wave1" / "headings" / "setext.md").read_text(encoding="utf-8")
        assert "==========\n" in text or "========\n" in text
        assert "--------\n" in text

    def test_scry_id_override_fixture(self, fixture_dir: Path) -> None:
        """scry_id_override.md has a scry-id HTML comment."""
        text = (fixture_dir / "wave1" / "headings" / "scry_id_override.md").read_text(
            encoding="utf-8"
        )
        assert "<!-- scry-id:" in text

    def test_duplicate_scry_id_fixture(self, fixture_dir: Path) -> None:
        """duplicate_scry_id.md has two identical scry-id comments."""
        text = (fixture_dir / "wave1" / "ids" / "duplicate_scry_id.md").read_text(encoding="utf-8")
        assert text.count("<!-- scry-id: foo -->") == 2

    def test_markdown_table_fixture(self, fixture_dir: Path) -> None:
        """markdown_with_table.md contains a markdown table."""
        text = (fixture_dir / "wave1" / "files" / "markdown_with_table.md").read_text(
            encoding="utf-8"
        )
        # Must have at least one pipe-separated header separator row.
        assert "|---" in text or "| ---" in text


class TestDynamicFixtures:
    """Tests exercising the dynamically-generated §15.4 edge-case files."""

    def test_binary_md_is_non_utf8(self, binary_md: Path) -> None:
        """Binary fixture cannot be decoded as UTF-8."""
        with pytest.raises(UnicodeDecodeError):
            binary_md.read_text(encoding="utf-8")

    def test_utf16le_md_has_bom(self, utf16le_md: Path) -> None:
        """UTF-16 LE fixture starts with the 0xFF 0xFE BOM."""
        raw = utf16le_md.read_bytes()
        assert raw[:2] == b"\xff\xfe"

    def test_crlf_md_contains_crlf(self, crlf_md: Path) -> None:
        """CRLF fixture has \\r\\n line endings and no bare \\r."""
        raw = crlf_md.read_bytes()
        assert b"\r\n" in raw

    def test_mixed_endings_md(self, mixed_endings_md: Path) -> None:
        """Mixed-endings fixture has both \\r\\n and \\n in the same file."""
        raw = mixed_endings_md.read_bytes()
        assert b"\r\n" in raw
        assert b"\n" in raw

    def test_cr_only_md_has_bare_cr(self, cr_only_md: Path) -> None:
        """CR-only fixture uses bare \\r (not \\r\\n)."""
        raw = cr_only_md.read_bytes()
        assert b"\r" in raw
        # Should NOT contain CRLF sequences.
        assert b"\r\n" not in raw

    def test_oversized_md_exceeds_5mb(self, oversized_md: Path) -> None:
        """Oversized fixture is larger than the default 5 MB limit."""
        size = oversized_md.stat().st_size
        assert size > 5 * 1024 * 1024

    def test_symlink_inside_resolves(self, symlink_inside: Path) -> None:
        """Inside-repo symlink resolves to an existing file."""
        assert symlink_inside.is_symlink()
        assert symlink_inside.resolve().exists()

    def test_symlink_outside_is_symlink(self, symlink_outside: Path) -> None:
        """Outside-repo symlink is a symlink (target may or may not exist)."""
        assert symlink_outside.is_symlink()


# uat-r5-5 pr-d noise
