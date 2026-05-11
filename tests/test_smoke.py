"""Smoke test verifying the package is importable and version is reachable."""

from __future__ import annotations


def test_import() -> None:
    import scry

    assert scry.__version__


def test_version_format() -> None:
    import re

    import scry

    # SemVer-ish: digits.digits.digits with optional pre-release tag
    assert re.match(r"^\d+\.\d+\.\d+", scry.__version__), scry.__version__

# uat-r5-5 pr-d noise
