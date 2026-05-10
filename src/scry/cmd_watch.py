"""File-watcher hot-reindex daemon (DESIGN.md line 1119, §10 IPC, Wave 6 W6c).

Implements ``scry watch`` — sits on a watchdog file watcher and triggers an
incremental reindex whenever source files change.

Leader / follower coordination
-------------------------------
* **No leader detected** — no other scry process holds the leader lock.
  ``scry watch`` refuses to run and exits with code 2, asking the user to
  start ``scry mcp`` first.  Rationale: allowing a watcher to index directly
  creates a write-path race if a leader starts later; the watcher cannot
  fully take on the leader role without IPC setup.  (W6c BLOCKING #1 fix:
  preferred "refuse-without-leader" approach chosen over acquiring the lock
  because it is simpler and avoids any lock-stealing edge cases.)
* **Follower path** — another process (typically ``scry mcp``) holds the leader
  lock.  This process sends an ``IPCRequest(op="reindex")`` to the leader over
  the Unix socket so the leader owns the single write path.
* **No metadata** — the leader lock is held but its metadata hasn't been
  written yet (cold-start race).  The watch command exits with code 1 and
  asks the user to retry.

Leader failover (reconnect)
----------------------------
If the leader disappears mid-watch, the follower retries IPC with exponential
backoff (2 s initial → 30 s cap) and re-runs
:func:`~scry.process.leader.detect_leader_state` after each failure.  If no
leader is detected within ``reconnect_timeout`` seconds (default 60 s), the
watcher exits with code 2.

Debouncing
----------
File-system events are collected for ``debounce_ms`` milliseconds after the
*last* event in a burst.  N rapid saves → exactly 1 reindex.  Events arriving
while a reindex is in flight are accumulated and trigger one more reindex after
the current one completes.

Ignored paths
-------------
``.git``, ``.scry``, ``node_modules``, ``__pycache__``, ``.venv``, ``venv``,
``dist``, ``build``, ``target``, ``out``, ``coverage``, ``.next``, ``.nuxt``,
``.parcel-cache``, ``.cache``, ``.pytest_cache``, ``.mypy_cache``,
``.ruff_cache``, ``tmp``, ``.tmp`` are always silently dropped.
Note: ``.gitignore``-based filtering is a planned future enhancement.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from watchdog.events import (
    FileSystemEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from scry.models import new_idempotency_token
from scry.process.ipc import IPCClient, parse_endpoint_uri
from scry.process.leader import LeaderState, detect_leader_state

__all__ = [
    "WatchError",
    "WatchHandler",
    "run_watch",
]

logger = logging.getLogger(__name__)

# Directories whose events should always be dropped.
# Note: .gitignore-based filtering is a planned future enhancement.
_IGNORE_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        ".scry",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        # Build outputs and tooling caches (W6c MEDIUM #1)
        "dist",
        "build",
        "target",
        "out",
        "coverage",
        ".next",
        ".nuxt",
        ".parcel-cache",
        ".cache",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        "tmp",
        ".tmp",
    }
)

# Exponential-backoff parameters for IPC reconnect (W6c HIGH #1).
_RECONNECT_BACKOFF_INITIAL: float = 2.0
_RECONNECT_BACKOFF_CAP: float = 30.0


# ─── Exceptions ───────────────────────────────────────────────────────────────


class WatchError(Exception):
    """Raised for fatal watch startup errors (e.g. missing leader metadata)."""


class _WatchReconnectError(Exception):
    """Raised from _follower_reindex when the IPC reconnect timeout expires.

    Inherits from ``Exception`` so the main loop can catch it specifically
    before the generic ``except Exception`` handler that logs and continues.
    """


# ─── Ignore helper ────────────────────────────────────────────────────────────


def _should_ignore(path: Path, repo_root: Path) -> bool:
    """Return ``True`` if *path* should not trigger a reindex.

    Drops paths outside *repo_root* and any path whose ancestors include one of
    the well-known noise directories in :data:`_IGNORE_DIRS`.

    Args:
        path:      Absolute (or relative) filesystem path from a watchdog event.
        repo_root: Absolute repository root.

    Returns:
        ``True`` when the event should be silently discarded.
    """
    try:
        rel = path.resolve().relative_to(repo_root.resolve())
    except ValueError:
        return True  # outside repo — ignore
    return any(part in _IGNORE_DIRS for part in rel.parts)


# ─── Watchdog handler ─────────────────────────────────────────────────────────


class WatchHandler(FileSystemEventHandler):
    """Watchdog callback that posts relevant changes to an asyncio event loop.

    Watchdog invokes callbacks in its own thread.  This handler uses
    ``loop.call_soon_threadsafe`` to safely enqueue notifications into the
    asyncio event loop running in the main thread.

    Args:
        loop:      The running asyncio event loop (from ``asyncio.get_running_loop()``
                   captured before the watcher starts).
        changed:   Mutable set that accumulates changed :class:`~pathlib.Path`
                   objects.  Owned by the event loop; only modified via
                   ``call_soon_threadsafe``.
        trigger:   :class:`asyncio.Event` set whenever *changed* grows.
        repo_root: Repository root used to filter out-of-tree and ignored paths.
    """

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        changed: set[Path],
        trigger: asyncio.Event,
        repo_root: Path,
    ) -> None:
        super().__init__()
        self._loop = loop
        self._changed = changed
        self._trigger = trigger
        self._repo_root = repo_root

    def _post(self, path: Path) -> None:
        """Thread-safe: add *path* to *changed* and wake the debounce loop."""

        def _update() -> None:
            self._changed.add(path)
            self._trigger.set()

        self._loop.call_soon_threadsafe(_update)

    def on_any_event(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        src = Path(str(event.src_path))
        if not _should_ignore(src, self._repo_root):
            self._post(src)


# ─── Main watch loop ──────────────────────────────────────────────────────────


async def run_watch(
    repo: Path,
    *,
    debounce_ms: int = 500,
    once: bool = False,
    reconnect_timeout: int = 60,
    _observer_class: Any | None = None,
) -> int:
    """Main watch loop.

    Detects the leader/follower role, builds the appropriate reindex callable,
    starts the file watcher, and debounces events until shutdown.

    Args:
        repo:               Repository root (must contain ``.scry/``).
        debounce_ms:        Quiet-period in milliseconds after the last event
                            before a reindex is triggered (default 500 ms).
        once:               If ``True``, run exactly one reindex then return
                            without starting the watcher (useful for testing).
        reconnect_timeout:  Seconds to keep retrying IPC before giving up when
                            the leader disappears mid-watch (default 60 s).
        _observer_class:    Inject a custom watchdog observer class (e.g.
                            ``PollingObserver`` in unit tests to avoid inotify).
                            Defaults to the platform-native ``Observer``.

    Returns:
        Exit code: 0 on clean exit or successful ``--once`` run, 1 on
        cold-start metadata error, 2 on no-leader or reconnect-timeout.

    Raises:
        :exc:`WatchError`: For fatal configuration problems detected before
            the watcher loop starts (re-raised by the CLI as exit 1).
    """
    # ── 1. Determine leader / follower role ───────────────────────────────
    state, metadata = detect_leader_state(repo)

    # W6c BLOCKING #1: refuse to run without a running leader.
    # Allowing the watcher to index directly risks write-path collisions if a
    # leader process starts later.  "Refuse" is chosen over acquiring the lock
    # ourselves (see module docstring for rationale).
    if state == LeaderState.LEADER:
        print(
            "error: scry watch requires a running leader. "
            "Start one first with 'scry mcp' (in another terminal) "
            "or 'scry index' will start a one-shot leader for you.",
            file=sys.stderr,
        )
        return 2

    # ── FOLLOWER path only below this point ───────────────────────────
    if metadata is None:
        # Leader lock is held but metadata not yet written — cold-start race.
        logger.error(
            "watch: a leader process is running but its lock-file metadata "
            "is not yet available (still in cold-start). "
            "Wait a moment and retry, or start 'scry mcp' first."
        )
        return 1

    endpoint = parse_endpoint_uri(metadata.endpoint_uri, repo)
    ipc_client: IPCClient | None = IPCClient(endpoint)

    async def _follower_reindex(n_files: int) -> None:
        """Send reindex IPC, retrying with backoff if the leader is unreachable.

        On IPC failure: logs a WARNING, sleeps with exponential backoff
        (``_RECONNECT_BACKOFF_INITIAL`` → ``_RECONNECT_BACKOFF_CAP``),
        re-runs :func:`~scry.process.leader.detect_leader_state`, and
        rebuilds the client.  If the deadline passes, raises
        :exc:`_WatchReconnectError` (exit code 2).
        """
        nonlocal ipc_client
        token = new_idempotency_token()
        logger.info("watch: sending reindex request to leader (%d files changed)", n_files)

        loop = asyncio.get_running_loop()
        deadline = loop.time() + reconnect_timeout
        backoff = _RECONNECT_BACKOFF_INITIAL

        while True:
            try:
                assert ipc_client is not None
                await ipc_client.call("reindex", {}, idempotency_token=token, timeout_seconds=None)
                return  # success
            except (OSError, TimeoutError) as exc:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    logger.error(
                        "watch: IPC to leader failed and reconnect-timeout (%ds) "
                        "exceeded; giving up: %s",
                        reconnect_timeout,
                        exc,
                    )
                    raise _WatchReconnectError from exc

                sleep_for = min(backoff, remaining)
                logger.warning(
                    "watch: leader IPC error: %s — retrying in %.0fs (%.0fs remaining)",
                    exc,
                    sleep_for,
                    remaining,
                )
                await asyncio.sleep(sleep_for)
                backoff = min(backoff * 2, _RECONNECT_BACKOFF_CAP)

                # Re-detect leader and rebuild client with potentially new endpoint.
                new_state, new_metadata = detect_leader_state(repo)
                if new_state != LeaderState.FOLLOWER or new_metadata is None:
                    if loop.time() < deadline:
                        logger.warning("watch: no leader detected; will retry")
                        continue
                    logger.error(
                        "watch: no leader detected after reconnect-timeout (%ds); exiting",
                        reconnect_timeout,
                    )
                    raise _WatchReconnectError from None

                if ipc_client is not None:
                    old_client: IPCClient = ipc_client
                    with contextlib.suppress(Exception):
                        await old_client.close()
                new_endpoint = parse_endpoint_uri(new_metadata.endpoint_uri, repo)
                ipc_client = IPCClient(new_endpoint)
                logger.info("watch: reconnected to leader at %s", new_metadata.endpoint_uri)

    reindex_fn: Callable[[int], Awaitable[None]] = _follower_reindex

    # ── 2. --once: single reindex, no watcher ────────────────────────────
    if once:
        try:
            await reindex_fn(0)
        except _WatchReconnectError:
            return 2
        finally:
            if ipc_client is not None:
                await ipc_client.close()
        return 0

    # ── 3. Start file watcher ─────────────────────────────────────────────
    loop = asyncio.get_running_loop()
    changed: set[Path] = set()
    trigger = asyncio.Event()

    handler = WatchHandler(loop, changed, trigger, repo)
    observer_cls: Any = _observer_class if _observer_class is not None else Observer
    observer = observer_cls()
    observer.schedule(handler, str(repo), recursive=True)
    observer.start()

    logger.info("watch: watching %s (debounce=%dms)", repo, debounce_ms)
    debounce_s = debounce_ms / 1000.0

    _exit_code = 0
    try:
        while True:
            # Wait for the first file event.
            await trigger.wait()
            trigger.clear()

            # Debounce: keep resetting the quiet-period timer whenever a new
            # event arrives.  The inner loop exits when no event arrives within
            # debounce_s seconds.
            while True:
                try:
                    await asyncio.wait_for(trigger.wait(), timeout=debounce_s)
                    trigger.clear()
                except TimeoutError:
                    break  # quiet period elapsed → fire reindex

            n_changed = len(changed)
            changed.clear()

            try:
                await reindex_fn(n_changed)
            except _WatchReconnectError:
                _exit_code = 2
                break  # propagate exit code; finally block cleans up
            except Exception as exc:
                logger.error("watch: reindex error: %s", exc)

    except asyncio.CancelledError:
        logger.info("watch: shutting down")
    finally:
        observer.stop()
        observer.join()
        if ipc_client is not None:
            await ipc_client.close()

    return _exit_code
