"""Tests for scry.process.leader — W2g leader election + lock-file management.

Covers:
    - try_acquire happy-path and contended-path
    - acquire_blocking timeout and success-after-release
    - write_metadata / read_metadata round-trip
    - boot_epoch_token stability and cross-acquisition uniqueness
    - release idempotency
    - context-manager exit (normal + exception)
    - read_leader_metadata_if_present (no lock held)
    - detect_leader_state (LEADER / FOLLOWER)
    - process-death lock release (via subprocess)
    - platform-specific: windows_only / unix_only markers

Threading is used for same-process contention tests; subprocess.Popen is
used for the process-death simulation so the target is importable from any
context.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from scry.process.leader import (
    LeaderLock,
    LeaderMetadata,
    LeaderState,
    LockTimeout,
    detect_leader_state,
    read_leader_metadata_if_present,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")

_ENDPOINT = "unix:.scry/scry.sock"
_VERSION = "0.0.1"


# ── Basic acquisition ─────────────────────────────────────────────────────────


def test_try_acquire_succeeds_when_uncontended(tmp_repo: Path) -> None:
    """Happy path: no other holder → try_acquire returns a LeaderLock."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    lock.release()


def test_try_acquire_returns_none_when_lock_held(tmp_repo: Path) -> None:
    """try_acquire must return None while another descriptor holds the lock.

    On POSIX, two separate os.open() calls create two independent open-file
    descriptions, so flock conflict is real even within the same process.
    On Windows, msvcrt.locking on the same byte range fails similarly.
    """
    lock1 = LeaderLock.try_acquire(tmp_repo)
    assert lock1 is not None
    try:
        lock2 = LeaderLock.try_acquire(tmp_repo)
        assert lock2 is None
    finally:
        lock1.release()


# ── acquire_blocking ──────────────────────────────────────────────────────────


def test_acquire_blocking_raises_lock_timeout_when_contended(tmp_repo: Path) -> None:
    """acquire_blocking must raise LockTimeout if the holder doesn't release."""
    acquired = threading.Event()
    hold = threading.Event()

    def holder() -> None:
        lock = LeaderLock.try_acquire(tmp_repo)
        assert lock is not None
        acquired.set()
        hold.wait(timeout=10)
        lock.release()

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    acquired.wait(timeout=5)

    try:
        with pytest.raises(LockTimeout):
            LeaderLock.acquire_blocking(tmp_repo, timeout_seconds=0.2)
    finally:
        hold.set()
        t.join(timeout=5)


def test_acquire_blocking_succeeds_after_holder_releases(tmp_repo: Path) -> None:
    """acquire_blocking must succeed once the previous holder releases."""
    released = threading.Event()

    def holder() -> None:
        lock = LeaderLock.try_acquire(tmp_repo)
        assert lock is not None
        time.sleep(0.1)
        lock.release()
        released.set()

    t = threading.Thread(target=holder, daemon=True)
    t.start()
    released.wait(timeout=5)

    # Should now succeed within a generous window.
    lock = LeaderLock.acquire_blocking(tmp_repo, timeout_seconds=3.0)
    assert lock is not None
    lock.release()
    t.join(timeout=3)


# ── write_metadata / read_metadata ────────────────────────────────────────────


def test_write_metadata_returns_correct_leadermetadata(tmp_repo: Path) -> None:
    """write_metadata must echo back a LeaderMetadata with the right fields."""
    import os

    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        meta = lock.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)
        assert meta.pid == os.getpid()
        assert meta.endpoint_uri == _ENDPOINT
        assert meta.scry_version == _VERSION
        assert len(meta.boot_epoch_token) == 36  # UUID4 canonical form
    finally:
        lock.release()


def test_write_metadata_round_trips_through_read_metadata(tmp_repo: Path) -> None:
    """read_metadata must reproduce exactly what write_metadata persisted."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        written = lock.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)
        read = lock.read_metadata()
        assert read is not None
        assert read.pid == written.pid
        assert read.endpoint_uri == written.endpoint_uri
        assert read.boot_epoch_token == written.boot_epoch_token
        assert read.scry_version == written.scry_version
    finally:
        lock.release()


def test_read_metadata_returns_none_on_empty_file(tmp_repo: Path) -> None:
    """read_metadata must return None when the lock file contains no JSON."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        # No write_metadata call → file is empty / never written.
        assert lock.read_metadata() is None
    finally:
        lock.release()


def test_write_metadata_raises_without_active_lock(tmp_repo: Path) -> None:
    """write_metadata must raise RuntimeError if no lock is held."""
    lock = LeaderLock(tmp_repo)  # not acquired
    with pytest.raises(RuntimeError, match="active lock"):
        lock.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)


# ── boot_epoch_token ──────────────────────────────────────────────────────────


def test_boot_epoch_token_is_stable_within_instance(tmp_repo: Path) -> None:
    """Multiple write_metadata calls on the same instance reuse the token."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        m1 = lock.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)
        m2 = lock.write_metadata(endpoint_uri="unix:.scry/other.sock", scry_version=_VERSION)
        assert m1.boot_epoch_token == m2.boot_epoch_token
        assert lock.boot_epoch_token == m1.boot_epoch_token
    finally:
        lock.release()


def test_boot_epoch_token_differs_across_acquisitions(tmp_repo: Path) -> None:
    """Each new LeaderLock instance must generate a fresh boot-epoch token."""
    lock1 = LeaderLock.try_acquire(tmp_repo)
    assert lock1 is not None
    lock1.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)
    token1 = lock1.boot_epoch_token
    lock1.release()

    lock2 = LeaderLock.try_acquire(tmp_repo)
    assert lock2 is not None
    lock2.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)
    token2 = lock2.boot_epoch_token
    lock2.release()

    assert token1 != token2, "Boot-epoch tokens should differ across instances"


def test_boot_epoch_token_raises_before_write_metadata(tmp_repo: Path) -> None:
    """Accessing boot_epoch_token before write_metadata must raise RuntimeError."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        with pytest.raises(RuntimeError):
            _ = lock.boot_epoch_token
    finally:
        lock.release()


# ── release idempotency ────────────────────────────────────────────────────────


def test_release_is_idempotent(tmp_repo: Path) -> None:
    """Multiple release() calls on the same instance must not raise."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    lock.release()
    lock.release()  # second call — must be a no-op
    lock.release()  # third call — still no-op


# ── Context manager ────────────────────────────────────────────────────────────


def test_context_manager_releases_on_normal_exit(tmp_repo: Path) -> None:
    """The lock must be released when the ``with`` block exits normally."""
    with LeaderLock.acquire_blocking(tmp_repo, timeout_seconds=2.0) as lock:
        lock.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)
        # Lock is held here.

    # After the with-block, a fresh acquisition must succeed.
    lock2 = LeaderLock.try_acquire(tmp_repo)
    assert lock2 is not None
    lock2.release()


def test_context_manager_releases_on_exception(tmp_repo: Path) -> None:
    """The lock must be released even when the ``with`` block raises."""
    try:
        with LeaderLock.acquire_blocking(tmp_repo, timeout_seconds=2.0):
            raise ValueError("deliberate test error")
    except ValueError:
        pass

    # Lock must be free after the exception.
    lock2 = LeaderLock.try_acquire(tmp_repo)
    assert lock2 is not None
    lock2.release()


# ── read_leader_metadata_if_present ───────────────────────────────────────────


def test_read_leader_metadata_if_present_returns_none_when_no_file(
    tmp_repo: Path,
) -> None:
    """Returns None when the lock file does not exist."""
    assert read_leader_metadata_if_present(tmp_repo) is None


def test_read_leader_metadata_if_present_reads_written_metadata(
    tmp_repo: Path,
) -> None:
    """Followers can read the metadata written by the leader without the lock."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        written = lock.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)
        # Read WITHOUT holding the lock — simulates a follower.
        read = read_leader_metadata_if_present(tmp_repo)
        assert read is not None
        assert read.pid == written.pid
        assert read.endpoint_uri == written.endpoint_uri
        assert read.boot_epoch_token == written.boot_epoch_token
        assert read.scry_version == written.scry_version
    finally:
        lock.release()


# ── detect_leader_state ────────────────────────────────────────────────────────


def test_detect_leader_state_returns_leader_when_unowned(tmp_repo: Path) -> None:
    """detect_leader_state returns LEADER for an uncontended lock."""
    state, metadata = detect_leader_state(tmp_repo)
    assert state == LeaderState.LEADER
    assert metadata is None

    # Must have released the probe lock so the caller can re-acquire.
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    lock.release()


def test_detect_leader_state_returns_follower_when_owned_no_metadata(
    tmp_repo: Path,
) -> None:
    """detect_leader_state returns FOLLOWER with None metadata when the leader
    holds the lock but hasn't written metadata yet (cold-start window)."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        state, metadata = detect_leader_state(tmp_repo)
        assert state == LeaderState.FOLLOWER
        # File is empty → no metadata yet.
        assert metadata is None
    finally:
        lock.release()


def test_detect_leader_state_returns_follower_with_metadata_when_owned(
    tmp_repo: Path,
) -> None:
    """detect_leader_state returns FOLLOWER with the leader's metadata after
    the leader has called write_metadata."""
    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        written = lock.write_metadata(endpoint_uri=_ENDPOINT, scry_version=_VERSION)
        state, detected = detect_leader_state(tmp_repo)
        assert state == LeaderState.FOLLOWER
        assert detected is not None
        assert detected.pid == written.pid
        assert detected.endpoint_uri == written.endpoint_uri
        assert detected.boot_epoch_token == written.boot_epoch_token
    finally:
        lock.release()


# ── Process-death simulation ───────────────────────────────────────────────────


def _make_subprocess_script(repo_root: Path) -> str:
    """Return a Python -c script that acquires the lock and blocks until killed."""
    return textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {_SRC_DIR!r})
        from pathlib import Path
        from scry.process.leader import LeaderLock
        lock = LeaderLock.try_acquire(Path({str(repo_root)!r}))
        if lock is None:
            print("failed", flush=True)
            sys.exit(1)
        print("acquired", flush=True)
        time.sleep(3600)
        """
    )


def test_process_death_releases_lock(tmp_repo: Path) -> None:
    """The OS must release the advisory lock when the holding process is killed.

    Uses subprocess.Popen so we exercise a *real* separate process (not just
    a thread sharing the same address space).
    """
    script = _make_subprocess_script(tmp_repo)
    p = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
    )
    assert p.stdout is not None

    try:
        # Wait for the subprocess to signal that it has acquired the lock.
        # readline() blocks until the subprocess flushes a newline.
        deadline = time.monotonic() + 10
        out = b""
        while b"\n" not in out:
            if time.monotonic() > deadline:
                pytest.fail("Subprocess never signalled lock acquisition")
            chunk = p.stdout.read(1)
            out += chunk
        assert b"acquired" in out, f"Unexpected output: {out!r}"

        # Verify we cannot acquire while the subprocess holds the lock.
        assert LeaderLock.try_acquire(tmp_repo) is None

        # Kill the subprocess — OS should release the lock automatically.
        p.kill()
        p.wait(timeout=5)
    finally:
        if p.poll() is None:
            p.kill()
            p.wait()

    # Retry acquisition; on some systems there's a brief kernel delay.
    acquired: LeaderLock | None = None
    deadline2 = time.monotonic() + 3.0
    while time.monotonic() < deadline2:
        acquired = LeaderLock.try_acquire(tmp_repo)
        if acquired is not None:
            break
        time.sleep(0.05)

    assert acquired is not None, "Lock was not released after process death"
    acquired.release()


# ── Platform-specific ──────────────────────────────────────────────────────────


def test_windows_locking_uses_msvcrt(tmp_repo: Path, windows_only: None) -> None:
    """Windows path: verify msvcrt.locking is effective (byte-range contention)."""
    import msvcrt

    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        # A second acquisition attempt must fail — the msvcrt lock is held.
        lock2 = LeaderLock.try_acquire(tmp_repo)
        assert lock2 is None
    finally:
        lock.release()

    # After release the lock file is gone; a fresh acquisition must succeed.
    lock3 = LeaderLock.try_acquire(tmp_repo)
    assert lock3 is not None
    lock3.release()

    # Suppress unused-import warning — import is the point of this test.
    _ = msvcrt


def test_unix_locking_uses_fcntl(tmp_repo: Path, unix_only: None) -> None:
    """Unix path: verify fcntl.flock is effective (separate open-file descriptions)."""
    import fcntl

    lock = LeaderLock.try_acquire(tmp_repo)
    assert lock is not None
    try:
        # Two os.open() calls create independent open-file descriptions;
        # flock conflict must work even within the same process.
        lock2 = LeaderLock.try_acquire(tmp_repo)
        assert lock2 is None
    finally:
        lock.release()

    # After release the lock file is gone; a fresh acquisition must succeed.
    lock3 = LeaderLock.try_acquire(tmp_repo)
    assert lock3 is not None
    lock3.release()

    # Suppress unused-import warning — import is the point of this test.
    _ = fcntl


# ── LeaderMetadata dataclass ───────────────────────────────────────────────────


def test_leader_metadata_is_frozen() -> None:
    """LeaderMetadata must be immutable (frozen dataclass)."""
    meta = LeaderMetadata(
        pid=1,
        endpoint_uri=_ENDPOINT,
        boot_epoch_token="abc",
        scry_version=_VERSION,
    )
    with pytest.raises((AttributeError, TypeError)):
        meta.pid = 2  # type: ignore[misc]

# uat-r5-5 pr-d noise
