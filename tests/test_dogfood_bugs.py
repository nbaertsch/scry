"""Regression tests for dogfooding bugs found while running scry on its own repo.

Each test corresponds to a bug-id (B1-B9) catalogued during the dogfooding
session.  These exercise scenarios the original Wave 1-6 unit tests missed
because they used synthetic fixtures rather than a real-world corpus.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# ─── B5: simhash uint8 overflow on numpy 2.x ──────────────────────────


def test_b5_simhash_repeated_token_no_overflow() -> None:
    """Repeated-token text must not raise OverflowError on numpy 2.x."""
    from scry.anchor_id import fingerprint_simhash

    # 5000 repeats of a single 4-byte token would push the simhash
    # weight above uint8.max.  Pre-fix this raised OverflowError.
    text = ("scry " * 5000).strip()
    h = fingerprint_simhash(text)
    assert isinstance(h, int)
    # 64-bit hash range
    assert 0 <= h < 2**64


# ─── B8: include glob with `**` must match root-level files ──────────


def test_b8_iterate_files_includes_root_level_markdown(tmp_path: Path) -> None:
    """``include: ['**/*.md']`` must capture root-level files like DESIGN.md.

    Pre-fix the indexer used a homemade fnmatch helper that collapsed
    ``**`` to ``*`` — fnmatch's ``*/*.md`` requires at least one
    directory segment, silently dropping every root-level markdown.
    """
    from scry.config import load_config
    from scry.index import Indexer

    # Build a tiny repo: a root-level markdown plus one in a subdir.
    (tmp_path / ".scry").mkdir()
    (tmp_path / ".scry" / "config.yaml").write_text(
        "include: ['**/*.md', '**/*.py']\nexclude: []\nclassify:\n  - {glob: '**/*.md', type: doc}\n",
        encoding="utf-8",
    )
    (tmp_path / "DESIGN.md").write_text("# Root Spec\n", encoding="utf-8")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "guide.md").write_text("# Subdir Doc\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "foo.py").write_text("def f(): pass\n", encoding="utf-8")

    config = load_config(tmp_path)
    idx = Indexer(repo_root=tmp_path, config=config)
    files = idx.discover_files()
    rel = sorted(p.relative_to(tmp_path).as_posix() for p in files)
    assert "DESIGN.md" in rel, f"root-level DESIGN.md missing: {rel}"
    assert "docs/guide.md" in rel
    assert "src/foo.py" in rel


# ─── B6 + B7: indexer must not parse frontmatter for excluded paths ──


def test_b6_no_frontmatter_warning_for_excluded_files(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Markdown files inside an excluded subtree must NOT have their
    frontmatter parsed (pre-fix this generated noisy malformed-YAML
    warnings for huggingface card templates inside ``.venv``).
    """
    import logging

    from scry.config import load_config
    from scry.index import Indexer

    (tmp_path / ".scry").mkdir()
    (tmp_path / ".scry" / "config.yaml").write_text(
        "include: ['**/*.md']\nexclude: ['.venv/**']\n"
        "classify:\n  - {glob: '**/*.md', type: doc}\n",
        encoding="utf-8",
    )
    venv_md = tmp_path / ".venv" / "site-packages" / "card.md"
    venv_md.parent.mkdir(parents=True)
    venv_md.write_text(
        "---\n{{ card_data }}\n---\n# Card\n",  # malformed YAML on purpose
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("# Real doc\n", encoding="utf-8")

    config = load_config(tmp_path)
    idx = Indexer(repo_root=tmp_path, config=config)

    with caplog.at_level(logging.WARNING):
        files = idx.discover_files()

    rel = sorted(p.relative_to(tmp_path).as_posix() for p in files)
    assert ".venv/site-packages/card.md" not in rel
    assert "README.md" in rel
    # No malformed-YAML warnings should be emitted for the excluded file.
    bad = [r for r in caplog.records if "Malformed YAML" in r.getMessage()]
    assert bad == [], f"unexpected frontmatter warnings: {[r.getMessage() for r in bad]}"


# ─── B2: scry validate must respect exclude config ───────────────────


def test_b2_validate_skips_excluded_fixture_dir(tmp_path: Path) -> None:
    """validate must not flag duplicate frontmatter ids from excluded dirs.

    Mirrors the real-world failure: ``tests/fixtures/wave1/ids/duplicate_*``
    intentionally contains duplicate scry.id values for a unit test, but
    a real repo's ``scry validate`` should ignore them when they match an
    exclude glob.
    """
    import scry.cli as cli

    (tmp_path / ".scry").mkdir()
    (tmp_path / ".scry" / "config.yaml").write_text(
        "include: ['**/*.md']\nexclude: ['fixtures/**']\n"
        "classify:\n  - {glob: '**/*.md', type: doc}\n",
        encoding="utf-8",
    )
    fix = tmp_path / "fixtures"
    fix.mkdir()
    for name in ("a.md", "b.md"):
        (fix / name).write_text("---\nscry:\n  id: DUPLICATE-ID\n---\n# x\n", encoding="utf-8")

    # The helper used by the validate walker should consider 'fixtures'
    # excluded so the directory is pruned during os.walk.
    assert cli._path_excluded(fix, tmp_path, ["fixtures/**"])

# uat-r5-5 pr-d noise
