"""Leader election and lock-file management for scry.

Implements DESIGN.md §10 (all sub-sections):

    §10.1 — Leader / follower roles
    §10.2 — Leader election ordering (lock-first) and failover
    §10.4 — Cold-start sequence for the single-process case

Key invariants (§10.2 v3.1):

    1.  The OS advisory lock (``.scry/leader.lock``) is acquired **first** —
        it is the authoritative "who is the leader" signal, *not* the PID
        written in the lock file.  This ordering eliminates the cold-start race
        where DB verification and auto-reconcile run before lock acquisition.

    2.  Only after the OS lock is held does the caller bind its IPC endpoint
        and then call :meth:`LeaderLock.write_metadata`.

    3.  Stale-lock detection: ``fcntl.flock`` / ``msvcrt.locking`` are
        released automatically by the kernel when the holding process dies.
        As a defense-in-depth for Windows (where ``uv.exe`` parent kills
        can orphan child fds), :meth:`LeaderLock.try_acquire` checks the
        recorded PID liveness when the OS lock is contended.  If the PID
        is confirmed dead, the lock file is removed and acquisition retried.

    4.  The boot-epoch token (UUIDv4) guards against PID recycling: after a
        crash and restart, a new scry process with the same PID generates a
        different token, letting a future reader distinguish "same scry process"
        from "different process that happened to reuse the PID."

Cross-platform locking
~~~~~~~~~~~~~~~~~~~~~~
- **POSIX** (Linux, macOS): ``fcntl.flock(fd, LOCK_EX | LOCK_NB)``
- **Windows**: ``msvcrt.locking(fd, LK_NBLCK, 1)`` — stdlib only;
  ``pywin32`` is not required for the advisory lock itself.

Lock-file format (JSON written after OS lock is held and IPC is ready)::

    {
        "pid":               12345,
        "endpoint_uri":      "unix:.scry/scry.sock",
        "boot_epoch_token":  "<uuidv4>",
        "scry_version":      "0.0.1"
    }

Public surface
~~~~~~~~~~~~~~
:class:`LeaderMetadata`              — Frozen dataclass snapshot of the lock file
:class:`LeaderState`                 — StrEnum: ``LEADER`` / ``FOLLOWER``
:class:`LockTimeout`                 — Raised by :meth:`LeaderLock.acquire_blocking`
:class:`StaleLockError`              — Signalled on detected stale-lock conditions
:class:`LeaderLock`                  — Advisory lock handle + metadata I/O
:func:`read_leader_metadata_if_present` — Read metadata without acquiring the lock
:func:`detect_leader_state`          — Non-destructive probe for leader/follower status
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

__all__ = [
    "LeaderLock",
    "LeaderMetadata",
    "LeaderState",
    "LockTimeout",
    "StaleLockError",
    "detect_leader_state",
    "read_leader_metadata_if_present",
]

logger = logging.getLogger(__name__)

# Name of the lock file within `.scry/`
_LOCK_FILENAME: str = "leader.lock"

# Polling cadence for acquire_blocking (seconds between non-blocking attempts)
_POLL_INTERVAL: float = 0.05


# ── Platform-specific lock primitives ─────────────────────────────────────────
#
# Both branches define the same three helper signatures so that call-sites
# outside the conditional blocks type-check correctly on every platform.
# Mypy's platform narrowing (via sys.platform checks) evaluates only the
# matching branch, ensuring correct stub resolution.

if sys.platform == "win32":
    import msvcrt as _msvcrt

    # Lock a byte well beyond any realistic metadata payload so that reads of
    # the JSON at offset 0 are not blocked by the mandatory Windows byte-range
    # lock.  msvcrt.locking uses LockFile (exclusive), which would prevent
    # other handles from reading the locked byte if it were at offset 0.
    _LOCK_BYTE: int = 1 << 30  # 1 GiB offset

    def _open_lock_file(path: Path) -> int:
        """Open (or create) the lock file; return a raw CRT file descriptor."""
        return os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_BINARY, 0o600)

    def _try_lock(fd: int) -> bool:
        """Try to acquire an exclusive lock on 1 byte at _LOCK_BYTE offset.

        Uses ``msvcrt.LK_NBLCK`` — fails immediately if the byte is already
        locked by another descriptor.  Returns ``True`` on success.
        The lock position is beyond the metadata region so followers can still
        read bytes 0..N (the JSON payload) without hitting a sharing violation.
        """
        os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
        try:
            _msvcrt.locking(fd, _msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        """Release the byte-range lock at _LOCK_BYTE offset.  Idempotent on error."""
        with contextlib.suppress(OSError):
            os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
            _msvcrt.locking(fd, _msvcrt.LK_UNLCK, 1)

else:
    import fcntl as _fcntl

    def _open_lock_file(path: Path) -> int:
        """Open (or create) the lock file; return a raw POSIX file descriptor."""
        return os.open(str(path), os.O_RDWR | os.O_CREAT, 0o600)

    def _try_lock(fd: int) -> bool:
        """Try a non-blocking exclusive ``flock``.  Returns ``True`` on success."""
        try:
            _fcntl.flock(fd, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
            return True
        except OSError:
            return False

    def _unlock(fd: int) -> None:
        """Release the ``flock``.  Idempotent on error."""
        with contextlib.suppress(OSError):
            _fcntl.flock(fd, _fcntl.LOCK_UN)


# ── Internal JSON helpers ──────────────────────────────────────────────────────


def _read_fd_json(fd: int) -> dict[str, Any] | None:
    """Read all bytes from *fd* (seeking to offset 0 first) and parse as JSON.

    Returns ``None`` on empty content, non-dict JSON, or any parse/IO failure.
    Callers must hold the OS lock before calling this to avoid a partial-read
    race with a concurrent writer.
    """
    try:
        os.lseek(fd, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        if not raw.strip():
            return None
        parsed: Any = json.loads(raw)
        if not isinstance(parsed, dict):
            return None
        return cast(dict[str, Any], parsed)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _parse_metadata(payload: dict[str, Any]) -> LeaderMetadata | None:
    """Convert a decoded JSON dict to a :class:`LeaderMetadata`.

    Returns ``None`` if any required key is absent or has an unexpected type.
    """
    try:
        return LeaderMetadata(
            pid=int(payload["pid"]),
            endpoint_uri=str(payload["endpoint_uri"]),
            boot_epoch_token=str(payload["boot_epoch_token"]),
            scry_version=str(payload["scry_version"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def _read_path_json(path: Path) -> dict[str, Any] | None:
    """Read *path* without acquiring any lock and parse as JSON.

    Returns ``None`` on any error including the file not existing.
    This is intentionally racy — callers treat the result as best-effort.
    """
    try:
        raw = path.read_bytes()
        if not raw.strip():
            return None
        parsed: Any = json.loads(raw)
        if not isinstance(parsed, dict):
            return None
        return cast(dict[str, Any], parsed)
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a running process (best-effort).

    On Windows, avoids ``os.kill(pid, 0)`` which can leak an inheritable
    process handle (CPython opens with PROCESS_ALL_ACCESS).  Instead we use
    ctypes to call OpenProcess with bInheritHandle=FALSE and immediately
    close the handle.
    """
    if sys.platform == "win32":
        import ctypes
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if handle:
            kernel32.CloseHandle(handle)
            return True
        return False
    # POSIX: signal 0 is the standard liveness check.
    try:
        os.kill(pid, 0)
        return True
    except (OSError, SystemError):
        return False


# ── Public types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LeaderMetadata:
    """Frozen snapshot of the lock file's JSON contents (DESIGN.md §10.2)."""

    pid: int
    """PID of the process that currently holds the leader lock."""

    endpoint_uri: str
    """IPC endpoint URI written by the leader after binding its socket/pipe.

    Format: ``unix:.scry/scry.sock`` on POSIX, ``pipe:scry-<hex>`` on Windows.
    See DESIGN.md §10.3 for URI conventions.
    """

    boot_epoch_token: str
    """UUIDv4 generated once per :class:`LeaderLock` instance.

    Guards against PID recycling: a reader can detect "this PID is now a
    different process" by comparing the stored token with a known-good value.
    """

    scry_version: str
    """The ``scry`` package version string (e.g. ``"0.0.1"``)."""


class LeaderState(StrEnum):
    """Election outcome returned by :func:`detect_leader_state`."""

    LEADER = "leader"
    FOLLOWER = "follower"


class LockTimeout(Exception):
    """Raised by :meth:`LeaderLock.acquire_blocking` when the timeout expires."""


class StaleLockError(Exception):
    """Raised when a stale-lock condition is explicitly detected.

    Per DESIGN.md §10.2 step 4 and §13 Q11, the OS lock is the primary
    liveness signal.  This exception is reserved for callers that want to
    surface the anomaly to the user; normal code paths log a warning instead.
    """


# ── LeaderLock ─────────────────────────────────────────────────────────────────


class LeaderLock:
    """OS-level advisory lock on ``.scry/leader.lock`` with metadata I/O.

    Acquisition order per DESIGN.md §10.2 v3.1:

    1. Acquire the OS lock — ``fcntl.flock`` on POSIX, ``msvcrt.locking``
       on Windows.  The lock is the authoritative "who is leader" signal.
    2. Caller binds its IPC endpoint.
    3. Caller calls :meth:`write_metadata` with the bound endpoint URI.
    4. Caller runs DB verification + auto-reconcile under the held lock.

    Usage::

        lock = LeaderLock.try_acquire(repo_root)
        if lock is None:
            # someone else is leader
            ...
        with lock:
            lock.write_metadata(endpoint_uri="unix:.scry/scry.sock",
                                 scry_version="0.0.1")
            # serve MCP tools …
    """

    def __init__(self, repo_root: Path) -> None:
        """Initialise a *not-yet-acquired* lock handle for *repo_root*.

        Do not call directly — use :meth:`try_acquire` or
        :meth:`acquire_blocking` to obtain a held instance.
        """
        self._repo_root: Path = repo_root
        self._lock_path: Path = repo_root / ".scry" / _LOCK_FILENAME
        self._fd: int | None = None
        self._locked: bool = False
        self._boot_epoch_token: str | None = None

    # ── Class-method constructors ──────────────────────────────────────────

    @classmethod
    def try_acquire(cls, repo_root: Path) -> LeaderLock | None:
        """Try to acquire the leader lock non-blockingly.

        Opens (or creates) ``.scry/leader.lock`` and attempts an exclusive
        lock without blocking.

        Returns a held :class:`LeaderLock` instance on success, or ``None``
        if another process already holds the lock.

        If the lock is held but the recorded PID is confirmed dead (process
        was force-killed or orphaned), the stale lock file is removed and
        acquisition is retried once.  This handles the Windows case where
        ``uv.exe`` parent termination orphans the child holding the fd.

        Raises :exc:`OSError` if the lock file cannot be opened (e.g. the
        ``.scry/`` directory does not exist — run ``scry init`` first).
        """
        instance = cls(repo_root)
        fd = _open_lock_file(instance._lock_path)
        if not _try_lock(fd):
            os.close(fd)
            # Check if the holder is dead — if so, force-recover.
            metadata = _read_path_json(instance._lock_path)
            if metadata is not None:
                pid = metadata.get("pid")
                if pid is not None and not _pid_alive(pid):
                    logger.warning(
                        "scry: leader lock held by dead PID %s — forcing recovery",
                        pid,
                    )
                    with contextlib.suppress(OSError):
                        os.unlink(instance._lock_path)
                    # Retry once.
                    fd2 = _open_lock_file(instance._lock_path)
                    if _try_lock(fd2):
                        instance._fd = fd2
                        instance._locked = True
                        return instance
                    os.close(fd2)
            return None
        instance._fd = fd
        instance._locked = True
        return instance

    @classmethod
    def acquire_blocking(
        cls,
        repo_root: Path,
        *,
        timeout_seconds: float = 5.0,
    ) -> LeaderLock:
        """Block up to *timeout_seconds* waiting for the leader lock.

        Polls with :data:`_POLL_INTERVAL` cadence — no busy-loop, but also
        not a pure blocking syscall so the timeout is honoured precisely.

        Raises :exc:`LockTimeout` if the lock cannot be acquired within the
        allotted time.
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            lock = cls.try_acquire(repo_root)
            if lock is not None:
                return lock
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise LockTimeout(
                    f"Could not acquire the scry leader lock within {timeout_seconds}s. "
                    "Another leader process may be starting up."
                )
            time.sleep(min(_POLL_INTERVAL, remaining))

    # ── Metadata I/O ──────────────────────────────────────────────────────

    def write_metadata(
        self,
        *,
        endpoint_uri: str,
        scry_version: str,
    ) -> LeaderMetadata:
        """Write ``{pid, endpoint_uri, boot_epoch_token, scry_version}`` to the lock file.

        **Must** be called only after a successful :meth:`try_acquire` /
        :meth:`acquire_blocking` call **and** only after the IPC endpoint is
        bound and accepting connections (DESIGN.md §10.2 v3.1 step 2b).

        The :attr:`boot_epoch_token` is generated on the first call and reused
        on subsequent calls within the same process instance, so that repeated
        metadata writes (e.g. to update the endpoint URI) do not change the
        token.

        Returns the :class:`LeaderMetadata` that was persisted.
        """
        if not self._locked or self._fd is None:
            raise RuntimeError(
                "write_metadata() requires an active lock — "
                "call try_acquire() or acquire_blocking() first."
            )

        if self._boot_epoch_token is None:
            self._boot_epoch_token = str(uuid.uuid4())

        metadata = LeaderMetadata(
            pid=os.getpid(),
            endpoint_uri=endpoint_uri,
            boot_epoch_token=self._boot_epoch_token,
            scry_version=scry_version,
        )

        payload: bytes = json.dumps(
            {
                "pid": metadata.pid,
                "endpoint_uri": metadata.endpoint_uri,
                "boot_epoch_token": metadata.boot_epoch_token,
                "scry_version": metadata.scry_version,
            }
        ).encode()

        os.lseek(self._fd, 0, os.SEEK_SET)
        os.ftruncate(self._fd, 0)
        os.write(self._fd, payload)

        return metadata

    def read_metadata(self) -> LeaderMetadata | None:
        """Read the lock file's current metadata without releasing the OS lock.

        Useful for inspecting what was written or verifying a round-trip.
        Returns ``None`` if the file is empty, missing, or unparseable.

        Safe to call on a held or not-yet-held instance, though meaningful
        results are only available after :meth:`write_metadata` has been called.
        """
        if self._fd is None:
            return None
        payload = _read_fd_json(self._fd)
        if payload is None:
            return None
        return _parse_metadata(payload)

    # ── Lock lifecycle ─────────────────────────────────────────────────────

    def release(self) -> None:
        """Release the OS lock and remove the lock file.  Idempotent.

        Steps (in order):

        1. Truncate the lock file to zero (clears stale metadata for any
           reader that has the path open but hasn't acquired the lock yet).
        2. On POSIX: unlink the path *while still holding the lock* to
           eliminate the TOCTOU window where a reader opens the old inode
           and then a new file is created at the same path.
        3. Release the OS lock.
        4. Close the file descriptor.
        5. On Windows: attempt to delete the (now-unlocked) file; failures
           are silently ignored because another process may have re-opened
           the path between step 3 and step 5.
        """
        if not self._locked or self._fd is None:
            return

        fd = self._fd

        # 1. Truncate
        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)

        if sys.platform != "win32":
            # 2. POSIX: unlink while lock is held — new openers get a fresh inode
            with contextlib.suppress(OSError):
                os.unlink(self._lock_path)

        # 3. Release OS lock
        _unlock(fd)

        # Mark released before close so idempotent re-calls see a clean state
        self._locked = False
        self._fd = None

        # 4. Close fd
        with contextlib.suppress(OSError):
            os.close(fd)

        if sys.platform == "win32":
            # 5. Windows: best-effort delete after releasing the lock
            with contextlib.suppress(OSError):
                os.unlink(self._lock_path)

    # ── Context manager ────────────────────────────────────────────────────

    def __enter__(self) -> LeaderLock:
        """Enter the context manager — returns ``self`` (already held)."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Release the lock on context exit, even if an exception was raised."""
        self.release()

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def boot_epoch_token(self) -> str:
        """The UUIDv4 boot-epoch token written to the lock file.

        Available only after :meth:`write_metadata` has been called at least
        once on this instance.  Raises :exc:`RuntimeError` otherwise.
        """
        if self._boot_epoch_token is None:
            raise RuntimeError(
                "boot_epoch_token is not available until write_metadata() is called."
            )
        return self._boot_epoch_token


# ── Module-level helpers ───────────────────────────────────────────────────────


def read_leader_metadata_if_present(repo_root: Path) -> LeaderMetadata | None:
    """Read ``.scry/leader.lock`` metadata **without** acquiring the OS lock.

    Intended for followers discovering the leader's IPC endpoint after they
    have determined that the OS lock is already held by another process.

    The read is best-effort and intentionally racy — the file may be partially
    written (leader still in cold-start) or already deleted (leader exiting).
    Returns ``None`` in any of those cases.
    """
    lock_path = repo_root / ".scry" / _LOCK_FILENAME
    payload = _read_path_json(lock_path)
    if payload is None:
        return None
    metadata = _parse_metadata(payload)
    if metadata is None:
        logger.warning(
            "scry leader lock file exists but metadata is unreadable — "
            "the leader may still be in cold-start.  Caller should retry."
        )
    return metadata


def detect_leader_state(
    repo_root: Path,
) -> tuple[LeaderState, LeaderMetadata | None]:
    """Probe whether this process can become the leader (non-destructive).

    Attempts a non-blocking lock acquisition:

    * If the lock is **available**: this process *could* become the leader.
      The probe immediately releases the lock (the caller re-acquires when
      ready) and returns ``(LeaderState.LEADER, None)``.

    * If the lock is **held** by another process: returns
      ``(LeaderState.FOLLOWER, metadata_or_none)`` where ``metadata`` is the
      leader's :class:`LeaderMetadata` (or ``None`` if the leader has not yet
      written it — caller should retry after a short back-off).

    .. important::
        This function does **not** leave the caller holding the lock.  Use
        :meth:`LeaderLock.try_acquire` or :meth:`LeaderLock.acquire_blocking`
        to actually become the leader.
    """
    probe = LeaderLock.try_acquire(repo_root)
    if probe is not None:
        # We won the probe — release immediately so the caller can re-acquire.
        probe.release()
        return LeaderState.LEADER, None

    # Lock is held — read whoever is the current leader.
    metadata = read_leader_metadata_if_present(repo_root)
    return LeaderState.FOLLOWER, metadata
