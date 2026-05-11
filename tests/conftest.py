"""Shared pytest fixtures.

Per workstream W0a in `plan.md`. Specific fixtures are added by the
workstream that needs them; this file holds only cross-cutting helpers.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

# Make sure src/scry is importable in tests run via `pytest` from the repo root
# (hatch usually handles this for installed mode; this is for editable dev).
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Generator[Path, None, None]:
    """A temp directory simulating a git repo root, with `.scry/` already created."""
    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir()
    (scry_dir / "overlays").mkdir()
    yield tmp_path


@pytest.fixture
def fixture_dir() -> Path:
    """Path to the tests/fixtures directory."""
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def hailstorm_spec(fixture_dir: Path) -> Path:
    """The synthetic hailstorm-style spec fixture (created in W1d)."""
    p = fixture_dir / "hailstorm-spec"
    if not p.exists():
        pytest.skip("hailstorm-spec fixture not yet created (W1d)")
    return p


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Honor ``@pytest.mark.unix_only`` / ``@pytest.mark.windows_only`` markers.

    Without this hook the markers are inert — they're declared in
    pyproject.toml's ``tool.pytest.ini_options.markers`` but pytest
    has no built-in skip semantics for arbitrary markers.  Tests that
    use the ``unix_only`` / ``windows_only`` fixtures (defined below)
    skip correctly via the fixture's pytest.skip; tests that ONLY use
    the marker (without the fixture) would silently run on the wrong
    OS without this hook.
    """
    skip_unix = pytest.mark.skip(reason="Unix-only test (skipped on Windows)")
    skip_windows = pytest.mark.skip(reason="Windows-only test (skipped on Unix)")
    for item in items:
        if "unix_only" in item.keywords and os.name == "nt":
            item.add_marker(skip_unix)
        elif "windows_only" in item.keywords and os.name != "nt":
            item.add_marker(skip_windows)


@pytest.fixture
def windows_only() -> None:
    """Skip a test unless running on Windows."""
    if os.name != "nt":
        pytest.skip("Windows-only test")


@pytest.fixture
def unix_only() -> None:
    """Skip a test unless running on Linux/macOS."""
    if os.name == "nt":
        pytest.skip("Unix-only test")


@pytest.fixture
def make_temp_file() -> Generator[Path, None, None]:
    """Yield a temp file path that is cleaned up after the test."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    p = Path(path)
    try:
        yield p
    finally:
        if p.exists():
            p.unlink()


@pytest.fixture
def cleanup_dir() -> Generator[Path, None, None]:
    """Yield a temp dir that is fully cleaned up after the test."""
    p = Path(tempfile.mkdtemp())
    try:
        yield p
    finally:
        if p.exists():
            shutil.rmtree(p, ignore_errors=True)

# uat-r5-5 pr-d noise
