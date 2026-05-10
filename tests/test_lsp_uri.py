"""Tests for scry.lsp.adapters.uri.is_local_file_uri."""

from __future__ import annotations

import pytest

from scry.lsp.adapters.uri import is_local_file_uri


@pytest.mark.parametrize(
    "uri",
    [
        "file:///abs/path.py",
        "file://localhost/abs/path.py",
        "file://Localhost/abs/path.py",  # case-insensitive host
        "file:/abs/path.py",  # opaque form, no authority
        "file:///C:/Users/x.py",  # Windows
        "FILE:///x.py",  # case-insensitive scheme
    ],
)
def test_local_file_uris_accepted(uri: str) -> None:
    assert is_local_file_uri(uri)


@pytest.mark.parametrize(
    "uri",
    [
        "https://example.com/x",
        "http://example.com/x",
        "data:text/plain,abc",
        "scry://repo/x.py",
        "untitled:Untitled-1",
        "ftp://example.com/x",
        "javascript:alert(1)",
        "",
        "/abs/path",  # bare path, no scheme
        "relative/path",  # bare relative path
        "file://remote/x.py",  # non-localhost authority
        "file://192.168.1.1/x.py",
    ],
)
def test_non_local_uris_rejected(uri: str) -> None:
    assert not is_local_file_uri(uri)
