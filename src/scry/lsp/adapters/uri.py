"""URI validation helpers shared across LSP adapters.

Adapters use :func:`is_local_file_uri` to gate their
``language_id_for_uri`` implementations so that virtual / remote URIs
(``http``, ``https``, ``data``, ``scry``, ``untitled``, in-memory
documents) do not get routed into a local LSP session by accident.

This is review-w3c LOW: URI scheme validation.
"""

from __future__ import annotations

from urllib.parse import urlparse

__all__ = ["is_local_file_uri"]


def is_local_file_uri(uri: str) -> bool:
    """Return ``True`` if *uri* is a local ``file://`` URI.

    A local file URI has scheme ``file`` and may optionally have an
    empty or ``localhost`` host (per RFC 8089).  All other schemes
    (``http``, ``https``, ``data``, ``scry``, ``untitled``, etc.) and
    relative paths return ``False``.

    Bare paths without a scheme also return ``False`` — adapters that
    receive raw filesystem paths should convert them via
    ``Path(...).as_uri()`` first.
    """
    try:
        parsed = urlparse(uri)
    except ValueError:
        return False
    if parsed.scheme.lower() != "file":
        return False
    # RFC 8089 permits empty netloc or "localhost"; reject any other host.
    return not (parsed.netloc and parsed.netloc.lower() != "localhost")
