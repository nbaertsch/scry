"""Authentication helpers."""

from __future__ import annotations


def login(user: str, password: str) -> dict | None:
    """Authenticate *user* with *password* and return a session token dict.

    This is a stub implementation.  A real implementation would verify
    credentials against a database or identity provider.

    Args:
        user:     The username (or email) to authenticate.
        password: The plaintext password (hashed before comparison in prod).

    Returns:
        A dict with ``{'token': str, 'expires_in': int}`` on success,
        or ``None`` when credentials are invalid.
    """
    # Stub: always return None (no valid credentials in fixture).
    return None
