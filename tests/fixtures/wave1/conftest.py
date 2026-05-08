"""Dynamic fixtures for the wave1 §15 edge-case test suite.

This conftest is scoped to ``tests/fixtures/wave1/`` and does NOT touch the
root ``tests/conftest.py``.  It generates files that cannot be safely committed
to git:

* Binary content (PNG magic bytes)
* UTF-16 LE encoded content (BOM + payload)
* CRLF-only, mixed CRLF+LF, and CR-only line endings
* Symlinks (inside-repo and outside-repo)
* An oversized file (> 5 MB) for the size-limit test

All session-scoped fixtures create files in a temporary directory managed by
``tmp_path_factory`` and are automatically cleaned up at session end.
Function-scoped symlink fixtures use ``tmp_path``.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Generator
from pathlib import Path

import pytest

# Resolve the wave1 fixture directory at import time so individual fixtures
# can reference committed files (e.g. the symlink-inside target).
_WAVE1_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def wave1_dir() -> Path:
    """Absolute path to the wave1 fixture directory."""
    return _WAVE1_DIR


# ─── Binary file ──────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def binary_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A file with binary (PNG magic) content and a ``.md`` extension.

    Per §15.4: binary files whose content cannot be decoded as UTF-8 must be
    skipped with a warning; no anchors are created.
    """
    d = tmp_path_factory.mktemp("wave1_binary")
    p = d / "binary.md"
    # PNG magic: \\x89PNG\\r\\n\\x1a\\n followed by null padding
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 200)
    return p


# ─── UTF-16 LE file ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def utf16le_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A ``.md`` file encoded in UTF-16 LE with BOM.

    Per §15.4: detected via \\xFF\\xFE BOM; skipped with a
    "transcode to UTF-8 to index" warning.
    """
    d = tmp_path_factory.mktemp("wave1_utf16le")
    p = d / "utf16le.md"
    content = "# UTF-16 LE Test\n\nThis file is encoded in UTF-16 LE.\n"
    bom = b"\xff\xfe"  # UTF-16 LE BOM
    p.write_bytes(bom + content.encode("utf-16-le"))
    return p


# ─── Line-ending variants ─────────────────────────────────────────────


@pytest.fixture(scope="session")
def crlf_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A markdown file with Windows CRLF (\\r\\n) line endings throughout.

    Per §15.4: canonicalised to LF before hashing; indexed normally.
    """
    d = tmp_path_factory.mktemp("wave1_crlf")
    p = d / "crlf_only.md"
    lines = [
        "# CRLF Test",
        "",
        "This file uses Windows CRLF line endings throughout.",
        "",
        "## Section One",
        "",
        "Content with CRLF endings.",
    ]
    p.write_bytes("\r\n".join(lines).encode("utf-8"))
    return p


@pytest.fixture(scope="session")
def mixed_endings_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A markdown file with mixed CRLF and LF line endings.

    Per §15.4: canonicalise \\r\\n → \\n first, then bare \\r → \\n.
    """
    d = tmp_path_factory.mktemp("wave1_mixed")
    p = d / "mixed_endings.md"
    # Deliberately interleave CRLF and LF endings in a single file.
    content = (
        b"# Mixed Endings Test\r\n"
        b"\r\n"
        b"This line ends with CRLF.\r\n"
        b"This line ends with LF.\n"
        b"\r\n"
        b"## Section\n"
        b"\n"
        b"More content with LF.\n"
    )
    p.write_bytes(content)
    return p


@pytest.fixture(scope="session")
def cr_only_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A markdown file with bare CR (\\r) line endings — Mac OS 9 style.

    Per §15.4: bare \\r → \\n canonicalisation step; indexed normally.
    """
    d = tmp_path_factory.mktemp("wave1_cr")
    p = d / "cr_only.md"
    content = (
        b"# CR-Only Test\r"
        b"This line ends with bare CR.\r"
        b"\r"
        b"## Section\r"
        b"Content with CR-only line endings.\r"
    )
    p.write_bytes(content)
    return p


# ─── Oversized file ───────────────────────────────────────────────────


@pytest.fixture(scope="session")
def oversized_md(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A markdown file exceeding the default 5 MB size limit (6 MB).

    Per §15.4: files exceeding ``index.max_file_size_bytes`` (default 5242880)
    are skipped with a warning; no anchors are created.
    """
    d = tmp_path_factory.mktemp("wave1_oversized")
    p = d / "oversized.md"
    header = b"# Oversized File\n\nThis file exceeds the 5 MB limit.\n\n"
    # Pad to 6 MB
    padding_len = 6 * 1024 * 1024 - len(header)
    p.write_bytes(header + b"A" * padding_len)
    return p


# ─── Symlink fixtures ─────────────────────────────────────────────────


def _can_symlink(tmp: Path) -> bool:
    """Probe whether symlinks can be created in *tmp* (Windows requires elevation)."""
    probe = tmp / "_probe_link"
    probe_target = tmp / "_probe_target"
    probe_target.write_text("x")
    try:
        probe.symlink_to(probe_target)
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        probe.unlink(missing_ok=True)
        probe_target.unlink(missing_ok=True)


@pytest.fixture
def symlink_inside(tmp_path: Path, wave1_dir: Path) -> Generator[Path, None, None]:
    """A ``.md`` symlink inside the fixture directory pointing to ``empty.md``.

    Per §15.4: symlinks to files inside the repo are followed; if the inode
    resolves to an already-indexed path, the canonical (non-symlink) path wins.

    Skipped on platforms / privilege levels where symlinks cannot be created.
    """
    if not _can_symlink(tmp_path):
        pytest.skip("Cannot create symlinks on this platform/privilege level")
    target = wave1_dir / "files" / "empty.md"
    link = tmp_path / "symlink_inside.md"
    link.symlink_to(target)
    yield link
    # tmp_path cleanup is automatic; explicit unlink for robustness.
    link.unlink(missing_ok=True)


@pytest.fixture
def symlink_outside(tmp_path: Path) -> Generator[Path, None, None]:
    """A ``.md`` symlink pointing outside the repo root (security boundary).

    Per §15.4: symlinks whose resolved path is outside the repo root are
    skipped with a warning.

    Skipped on platforms / privilege levels where symlinks cannot be created.
    """
    if not _can_symlink(tmp_path):
        pytest.skip("Cannot create symlinks on this platform/privilege level")
    # Point to a known read-safe file outside any scry repo.
    if sys.platform == "win32":
        outside_target = Path(os.environ.get("WINDIR", r"C:\Windows")) / "win.ini"
    else:
        outside_target = Path("/etc/passwd")
    link = tmp_path / "symlink_outside.md"
    link.symlink_to(outside_target)
    yield link
    link.unlink(missing_ok=True)
