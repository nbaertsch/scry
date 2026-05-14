"""SR5-6: filename-based test-file detection.

Heuristic mapping a repo-relative path to ``is_test``.  Used by both
extractors (markdown, code) at construction time so the resulting
anchor carries an ``is_test`` field that callers can filter on.

Recognised patterns:
    Python:   ``test_*.py``, ``*_test.py``, ``*_tests.py``,
              ``conftest.py``, ``tests/**``, ``test/**``
    JS/TS:    ``*.test.{ts,tsx,js,jsx}``, ``*.spec.{ts,tsx,js,jsx}``,
              ``__tests__/**``
    Go:       ``*_test.go``
    Rust:     ``tests/**``, ``*_test.rs``
    Java/Kt:  ``*Test.java``, ``*Tests.java``, ``*Test.kt``, ``*Spec.kt``

Returns False for everything else, including markdown (no test/prod
distinction for docs) and benchmark/fixture files (too high a
false-positive rate; revisit if users ask).
"""

from __future__ import annotations

import re

_TEST_FILENAME_RE = re.compile(
    r"""
    (?:^|/)
    (?:
        # Python
        test_[^/]+\.py
      | [^/]+_tests?\.py
      | conftest\.py
      | tests?/
        # JS / TS  — *.test.* and *.spec.*
      | [^/]+\.(?:test|spec)\.(?:ts|tsx|js|jsx|mjs|cjs)
      | __tests__/
        # Go
      | [^/]+_test\.go
        # Rust
      | [^/]+_test\.rs
        # Java / Kotlin
      | [^/]+(?:Test|Tests|Spec)\.(?:java|kt)
    )
    """,
    re.VERBOSE,
)


def is_test_path(path: str) -> bool:
    """Return True when *path* (repo-relative, forward-slash form)
    looks like a test file by the heuristics above.

    Matches anywhere in the path so e.g. ``src/foo/tests/bar.py``
    is recognized in addition to top-level ``tests/bar.py``.
    """
    if not path:
        return False
    return _TEST_FILENAME_RE.search(path) is not None
