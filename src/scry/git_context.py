"""Git context polling with per-process HEAD cache.

Implements DESIGN.md §7.2 v3.1 — git state polling, branch detection,
dirty-worktree detection, and the overlay slug derivation rule from §3.5.0.

Public surface
--------------
GitContext             — frozen snapshot of git state at a point in time
GitContextError        — raised when git is unavailable or repo is not a git tree
GitContextProvider     — cached polling engine
derive_overlay_slug    — §3.5.0: branch / detached-SHA → overlay filename
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from scry.models import IndexConfig

__all__ = [
    "GitContext",
    "GitContextError",
    "GitContextProvider",
    "derive_overlay_slug",
    "get_current_user",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitContext:
    """Snapshot of repo's current git state at a point in time.

    Attributes:
        head_sha:     Full 40-character commit SHA.
        head_short:   First 12 characters of *head_sha*.
        branch:       Branch name from ``git symbolic-ref --short``, or
                      ``None`` for a detached HEAD.
        overlay_slug: Filename for ``.scry/overlays/<slug>.jsonl`` per §3.5.0.
        is_detached:  ``True`` iff *branch* is ``None``.
        dirty_files:  Repo-relative forward-slash paths of uncommitted changes
                      to tracked files (``git status --porcelain=v1 -uno``).
                      Empty tuple when the worktree is clean or
                      ``poll_dirty=False``.
    """

    head_sha: str
    head_short: str
    branch: str | None
    overlay_slug: str
    is_detached: bool
    dirty_files: tuple[str, ...]


class GitContextError(Exception):
    """Raised when git is unavailable or the path is not a git working tree."""


def _run_git(
    args: list[str], repo_root: Path, *, timeout: float = 10.0
) -> subprocess.CompletedProcess[str]:
    """Run a git subcommand inside *repo_root* and return the result without raising.

    Args:
        args:       git args (e.g. ``["rev-parse", "HEAD"]``).
        repo_root:  working directory passed via ``-C``.
        timeout:    seconds before subprocess.run raises ``TimeoutExpired``.
                    Defaults to 10s — matches the worst-case observed for
                    network-mounted repos (NFS/SMB) per DESIGN.md §13 Q10
                    while still preventing an indefinite block under the
                    cache lock that would freeze every other ``get()`` caller.
    """
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        # UT3-1 BLOCKING fix: git inherits the parent's stdin handle by
        # default on Windows (CreateProcess bInheritHandles=True).  When
        # scry runs as an MCP stdio server, sys.stdin is the pipe to the
        # MCP client — git inherits it and never exits, freezing the
        # event loop and breaking ALL write tools.  Always close stdin.
        stdin=subprocess.DEVNULL,
    )


def _run_git_bytes(
    args: list[str], repo_root: Path, *, timeout: float = 10.0
) -> subprocess.CompletedProcess[bytes]:
    """Like :func:`_run_git` but returns raw bytes (for ``-z`` NUL-delimited output)."""
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        check=False,
        timeout=timeout,
        stdin=subprocess.DEVNULL,  # UT3-1 — see _run_git for rationale.
    )


# UAT-10: per-process cached user identity for the ``created_by``
# field on link records.  Checks ``SCRY_USER`` env first (lets users
# set a non-PII handle), then falls back to ``git config user.email``.
_user_identity_cache: dict[str, str | None] = {}


def get_current_user(repo_root: Path) -> str | None:
    """Return the current user identity for link attribution (UAT-10).

    Resolution order:
      1. ``SCRY_USER`` environment variable (non-PII handle preferred).
      2. ``git config user.email`` from *repo_root*'s config.
      3. ``None`` (unavailable — record persists un-attributed).

    Cached per ``repo_root`` per process so repeated calls (e.g.
    during a batch ``scry link`` loop) don't re-spawn git.
    """
    env: str | None = os.environ.get("SCRY_USER")
    if env:
        return env.strip()
    key = str(repo_root)
    cached = _user_identity_cache.get(key)
    if cached is not None:
        return cached
    if key in _user_identity_cache:
        # Explicitly cached as None (no email found).
        return None
    try:
        result = _run_git(["config", "user.email"], repo_root, timeout=5.0)
        email: str | None = result.stdout.strip() if result.returncode == 0 else None
    except Exception:
        email = None
    resolved: str | None = email if email else None
    _user_identity_cache[key] = resolved
    return resolved


def derive_overlay_slug(branch: str | None, head_sha: str) -> str:
    """Derive the overlay filename per DESIGN.md §3.5.0.

    Steps:

    1. If *branch* is not ``None``, use it as input.
       Otherwise use ``"detached-<head_sha[:12]>"``.
    2. ``urllib.parse.quote(input, safe='')`` — URL-encodes every character
       that would be problematic in a flat filename (``/``, ``\\``, ``:``,
       ``*``, ``?``, ``"``, ``<``, ``>``, ``|``, etc.).
    3. If the URL-encoded result exceeds 200 chars, replace with
       ``"long-<sha256(input)[:12]>"``.
    4. Append ``".jsonl"``.

    Examples::

        derive_overlay_slug("main", sha)              →  "main.jsonl"
        derive_overlay_slug("feature/auth-login", sha) →  "feature%2Fauth-login.jsonl"
        derive_overlay_slug(None, "abc123def456...")   →  "detached-abc123def456.jsonl"

    Args:
        branch:   Branch name, or ``None`` for a detached HEAD.
        head_sha: Full 40-character commit SHA.

    Returns:
        Overlay filename (no directory component), e.g. ``"main.jsonl"``,
        ``"feature%2Fauth-login.jsonl"``, or ``"detached-abc123def456.jsonl"``.
    """
    raw = branch if branch is not None else f"detached-{head_sha[:12]}"
    encoded = urllib.parse.quote(raw, safe="")
    if len(encoded) > 200:
        digest = hashlib.sha256(raw.encode()).hexdigest()
        encoded = f"long-{digest[:12]}"
    return f"{encoded}.jsonl"


def _fetch_context(repo_root: Path, poll_dirty: bool) -> GitContext:
    """Fetch a fresh :class:`GitContext` by invoking git subprocesses.

    Args:
        repo_root:  Absolute path to the repository root.
        poll_dirty: When ``True``, include dirty-worktree detection.

    Returns:
        A freshly constructed :class:`GitContext`.

    Raises:
        GitContextError: If ``git rev-parse HEAD`` returns a non-zero exit code,
            meaning git is absent or *repo_root* is not a git working tree.
    """
    # Step 1: Full HEAD SHA.
    rev_result = _run_git(["rev-parse", "HEAD"], repo_root)
    if rev_result.returncode != 0:
        detail = rev_result.stderr.strip()
        msg = f"Not a git repository: {repo_root}"
        if detail:
            msg = f"{msg} ({detail})"
        raise GitContextError(msg)

    head_sha = rev_result.stdout.strip()
    if not head_sha:
        raise GitContextError(f"git rev-parse HEAD returned empty output for {repo_root}")
    head_short = head_sha[:12]

    # Step 2: Branch name.  Non-zero exit = detached HEAD (not an error).
    sym_result = _run_git(["symbolic-ref", "--short", "-q", "HEAD"], repo_root)
    branch: str | None
    if sym_result.returncode == 0:
        raw_branch = sym_result.stdout.strip()
        branch = raw_branch if raw_branch else None
    else:
        branch = None  # Detached HEAD.

    is_detached = branch is None
    overlay_slug = derive_overlay_slug(branch, head_sha)

    # Step 3: Dirty-worktree detection (optional).
    dirty_files: tuple[str, ...]
    if poll_dirty:
        # `-z` flag emits NUL-terminated, RAW (unquoted) entries — required
        # to round-trip non-ASCII paths.  Without it git C-style-quotes
        # paths containing unusual chars (e.g. `"na\303\257ve.txt"`),
        # which the old line-based parser would corrupt.  Verified by
        # review-w2f's HIGH finding.
        status_result = _run_git_bytes(
            ["status", "-z", "--porcelain=v1", "-uno", "--no-renames"],
            repo_root,
        )
        if status_result.returncode != 0:
            logger.warning(
                "git status failed for %s (rc=%d); treating worktree as clean: %s",
                repo_root,
                status_result.returncode,
                status_result.stderr.decode("utf-8", errors="replace").strip(),
            )
            dirty_files = ()
        else:
            paths: list[str] = []
            # `-z` output: each entry is "XY PATH\0" — 2-char status,
            # 1 space, repo-relative path, NUL terminator.  Split on NUL
            # and decode each entry as UTF-8 (git stores paths as raw
            # bytes; `replace` errors-on-invalid keeps us robust).
            for chunk in status_result.stdout.split(b"\x00"):
                if len(chunk) < 4:
                    continue
                # Strip the 3-byte "XY " prefix; everything else is the
                # path verbatim.  No backslash-replace: porcelain v1
                # always emits forward slashes for directory separators
                # (git's wire format is POSIX), so a real backslash in
                # the path would be a legitimate filename character.
                raw_path = chunk[3:]
                try:
                    path = raw_path.decode("utf-8")
                except UnicodeDecodeError:
                    path = raw_path.decode("utf-8", errors="replace")
                path = path.strip()
                if path:
                    paths.append(path)
            dirty_files = tuple(paths)
    else:
        dirty_files = ()

    return GitContext(
        head_sha=head_sha,
        head_short=head_short,
        branch=branch,
        overlay_slug=overlay_slug,
        is_detached=is_detached,
        dirty_files=dirty_files,
    )


class GitContextProvider:
    """Polls git state with a per-process cache.

    Cache rules (DESIGN.md §7.2 v3.1):

    * Read-only ``get()`` calls honor the cache (default 30 s TTL).
    * Write-triggering tool calls should call ``invalidate()`` first so the
      next ``get()`` fetches fresh state and routes writes to the correct
      overlay.
    * ``head_poll_interval_seconds=0`` disables caching entirely — every
      ``get()`` invokes git.

    Args:
        repo_root:                  Absolute path to the repository root.
        head_poll_interval_seconds: Cache TTL in seconds (≥ 0). ``0`` = no cache.
        poll_dirty:                 When ``True``, include dirty-worktree
                                    detection in each poll.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        head_poll_interval_seconds: int = 30,
        poll_dirty: bool = True,
    ) -> None:
        self._repo_root = repo_root
        self._ttl = head_poll_interval_seconds
        self._poll_dirty = poll_dirty
        self._cache: GitContext | None = None
        self._cache_time: float = 0.0
        self._lock = threading.Lock()

    def get(self, *, force_refresh: bool = False) -> GitContext:
        """Return the current GitContext (cached or fresh).

        Honors the TTL cache unless *force_refresh* is ``True`` or the cache
        has expired.  When ``head_poll_interval_seconds=0``, every call
        re-invokes git regardless.

        Args:
            force_refresh: Bypass the TTL cache and fetch a fresh context.

        Returns:
            A :class:`GitContext` snapshot.

        Raises:
            GitContextError: If git is unavailable or *repo_root* is not
                a git working tree.
        """
        with self._lock:
            now = time.monotonic()
            if (
                not force_refresh
                and self._ttl > 0
                and self._cache is not None
                and (now - self._cache_time) < self._ttl
            ):
                return self._cache

            ctx = _fetch_context(self._repo_root, self._poll_dirty)
            if self._ttl > 0:
                self._cache = ctx
                self._cache_time = now
            return ctx

    def invalidate(self) -> None:
        """Drop the cache.

        Call before any state-mutating tool call (``propose_link``,
        ``accept_link``, ``commit_links``, ``reindex``) so the next
        ``get()`` sees the latest git state and routes writes to the
        correct overlay.
        """
        with self._lock:
            self._cache = None
            self._cache_time = 0.0

    @classmethod
    def from_config(cls, repo_root: Path, index_config: IndexConfig) -> GitContextProvider:
        """Convenience factory wired to a loaded :class:`~scry.models.IndexConfig`.

        Args:
            repo_root:    Absolute path to the repository root.
            index_config: Validated config from ``.scry/config.yaml``.

        Returns:
            A :class:`GitContextProvider` configured with the TTL and
            ``poll_dirty`` values from *index_config*.
        """
        return cls(
            repo_root,
            head_poll_interval_seconds=index_config.head_poll_interval_seconds,
            poll_dirty=index_config.poll_dirty,
        )
