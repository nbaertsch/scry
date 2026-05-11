"""Tests for ``scry.store.overlay`` — W2c: branch-aware overlay management.

Covers DESIGN.md §3.5 (link layers), §3.5.0 (overlay slug derivation via
GitContextProvider), §3.5.4 (commit-links atomicity + crash recovery), and
§7.2 v3.1 (cache-invalidation-on-writes contract).

All tests that interact with git use real tiny git repos created via
``tmp_path`` — no mocking of the git layer.

Public API under test
---------------------
OverlayManager.overlay_dir
OverlayManager.overlay_path_for
OverlayManager.current_overlay_path
OverlayManager.replay_active
OverlayManager.list_pending_overlay_records
OverlayManager.list_pending_branches
OverlayManager.append_to_current_branch_overlay
OverlayManager.promote_pending
OverlayManager.recover_pending
PendingPromotion
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scry.git_context import GitContextProvider
from scry.models import AnchorType, LinkRecord, LinkType, new_event_id, new_link_id
from scry.store.links import LinkStore
from scry.store.overlay import OverlayManager, PendingPromotion

# ─── Shared hash constants ────────────────────────────────────────────────────

_H_A = "sha256:" + "a" * 64
_H_B = "sha256:" + "b" * 64

_FROM = "docs/spec.md::intro"
_TO = "src/app.py:main"

# ─── Record builders ──────────────────────────────────────────────────────────


def _upsert(
    link_id: str | None = None,
    *,
    event_id: str | None = None,
    supersedes: str | None = None,
    from_: str = _FROM,
    to: str = _TO,
) -> LinkRecord:
    """Build a minimal valid UPSERT LinkRecord."""
    lid: str = link_id or new_link_id()
    eid: str = event_id or new_event_id()
    return LinkRecord.model_validate(
        {
            "op": "upsert",
            "link_id": lid,
            "event_id": eid,
            "from": from_,
            "from_type": AnchorType.SECTION,
            "to": to,
            "to_type": AnchorType.CODE,
            "type": LinkType.IMPLEMENTS,
            "from_content_hash": _H_A,
            "to_content_hash": _H_B,
            "supersedes": supersedes,
        }
    )


# ─── Git helpers ──────────────────────────────────────────────────────────────


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run git *args* in *cwd*, raising on non-zero exit."""
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def _init_repo(path: Path) -> str:
    """Initialise a git repo, make an initial commit, return the HEAD SHA."""
    _git(["init"], path)
    _git(["config", "user.email", "test@scry.test"], path)
    _git(["config", "user.name", "Scry Test"], path)
    (path / "README.md").write_text("init")
    _git(["add", "README.md"], path)
    _git(["commit", "-m", "init"], path)
    return _current_sha(path)


def _current_sha(path: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _current_branch(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "symbolic-ref", "--short", "-q", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    branch = result.stdout.strip()
    return branch if branch else None


def _make_commit(path: Path, filename: str = "extra.txt", content: str = "data") -> str:
    """Write *content* to *filename*, stage and commit; return the new SHA."""
    (path / filename).write_text(content)
    _git(["add", filename], path)
    _git(["commit", "-m", f"add {filename}"], path)
    return _current_sha(path)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def git_repo(tmp_path: Path) -> tuple[Path, str]:
    """A real git repo with an initial commit; yields (repo_root, head_sha)."""
    sha = _init_repo(tmp_path)
    return tmp_path, sha


@pytest.fixture
def mgr(git_repo: tuple[Path, str]) -> OverlayManager:
    """OverlayManager with TTL=0 (no caching) for deterministic tests."""
    repo_root, _ = git_repo
    return OverlayManager(
        repo_root,
        git_context=GitContextProvider(repo_root, head_poll_interval_seconds=0),
    )


# ─── Tests: structural properties ────────────────────────────────────────────


def test_overlay_dir_property(git_repo: tuple[Path, str]) -> None:
    """overlay_dir returns ``<repo_root>/.scry/overlays/``."""
    repo_root, _ = git_repo
    mgr = OverlayManager(repo_root)
    assert mgr.overlay_dir == repo_root / ".scry" / "overlays"


def test_overlay_path_for(git_repo: tuple[Path, str]) -> None:
    """overlay_path_for returns the absolute path for a given slug."""
    repo_root, _ = git_repo
    mgr = OverlayManager(repo_root)
    p = mgr.overlay_path_for("main.jsonl")
    assert p == repo_root / ".scry" / "overlays" / "main.jsonl"


def test_current_overlay_path_matches_branch(git_repo: tuple[Path, str]) -> None:
    """current_overlay_path reflects the current branch's overlay slug."""
    repo_root, _ = git_repo
    mgr = OverlayManager(
        repo_root,
        git_context=GitContextProvider(repo_root, head_poll_interval_seconds=0),
    )
    ctx = mgr._git_context.get()
    assert mgr.current_overlay_path() == mgr.overlay_dir / ctx.overlay_slug


# ─── Tests: basic append + replay ────────────────────────────────────────────


def test_basic_append_and_replay(mgr: OverlayManager) -> None:
    """Appending a record to the overlay makes it visible in replay_active."""
    record = _upsert()
    mgr.append_to_current_branch_overlay(record)

    result = mgr.replay_active()
    assert record.link_id in result.active_links
    assert result.merge_conflicts == []


def test_replay_active_empty_repo(mgr: OverlayManager) -> None:
    """replay_active on a fresh repo (no baseline, no overlay) returns empty table."""
    result = mgr.replay_active()
    assert result.active_links == {}
    assert result.merge_conflicts == []


def test_replay_merges_baseline_and_overlay(git_repo: tuple[Path, str]) -> None:
    """Records in baseline AND overlay both appear in replay_active."""
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    link_store = LinkStore(repo_root)
    mgr = OverlayManager(repo_root, git_context=git_ctx, link_store=link_store)

    r_baseline = _upsert()
    r_overlay = _upsert()

    # Write directly to baseline (bypassing overlay)
    link_store.append_baseline(r_baseline)
    # Write to overlay via manager
    mgr.append_to_current_branch_overlay(r_overlay)

    result = mgr.replay_active()
    assert r_baseline.link_id in result.active_links
    assert r_overlay.link_id in result.active_links


# ─── Tests: branch switching ─────────────────────────────────────────────────


def test_branch_switch_writes_to_new_overlay(git_repo: tuple[Path, str]) -> None:
    """After a branch switch, writes go to the new branch's overlay file."""
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    mgr = OverlayManager(repo_root, git_context=git_ctx)

    # Write on initial branch
    r1 = _upsert()
    mgr.append_to_current_branch_overlay(r1)
    initial_branch = _current_branch(repo_root) or "main"
    initial_slug = git_ctx.get().overlay_slug

    # Switch to a new branch and write
    _git(["checkout", "-b", "feature-x"], repo_root)
    r2 = _upsert()
    mgr.append_to_current_branch_overlay(r2)

    # feature branch overlay exists and contains r2
    feature_slug = git_ctx.get().overlay_slug
    assert feature_slug != initial_slug
    feature_path = mgr.overlay_dir / feature_slug
    assert feature_path.exists()

    feature_records = mgr._link_store.read_records(feature_path)
    feature_link_ids = {r.link_id for r in feature_records}
    assert r2.link_id in feature_link_ids
    assert r1.link_id not in feature_link_ids

    # initial branch overlay still has r1 and NOT r2
    initial_path = mgr.overlay_dir / initial_slug
    initial_records = mgr._link_store.read_records(initial_path)
    initial_link_ids = {r.link_id for r in initial_records}
    assert r1.link_id in initial_link_ids
    assert r2.link_id not in initial_link_ids

    # Switch back and verify reads from initial overlay
    _git(["checkout", initial_branch], repo_root)
    pending = mgr.list_pending_overlay_records()
    pending_ids = {r.link_id for r in pending}
    assert r1.link_id in pending_ids
    assert r2.link_id not in pending_ids


def test_branch_switch_reads_from_new_overlay(git_repo: tuple[Path, str]) -> None:
    """After a branch switch, list_pending_overlay_records reads the new branch's overlay."""
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    mgr = OverlayManager(repo_root, git_context=git_ctx)

    # Write on initial branch
    r_initial = _upsert()
    mgr.append_to_current_branch_overlay(r_initial)

    # Create + switch to feature branch, write a record there
    _git(["checkout", "-b", "reads-test-branch"], repo_root)
    r_feature = _upsert()
    mgr.append_to_current_branch_overlay(r_feature)

    # While on feature branch, pending records are from feature overlay only
    pending = mgr.list_pending_overlay_records()
    pending_ids = {r.link_id for r in pending}
    assert r_feature.link_id in pending_ids
    assert r_initial.link_id not in pending_ids

    # replay_active likewise only sees the feature overlay (plus shared baseline)
    result = mgr.replay_active()
    assert r_feature.link_id in result.active_links
    assert r_initial.link_id not in result.active_links


# ─── Tests: detached HEAD ─────────────────────────────────────────────────────


def test_detached_head_overlay(git_repo: tuple[Path, str]) -> None:
    """In detached HEAD state, reads/writes use detached-<sha12>.jsonl."""
    repo_root, sha = git_repo
    _git(["checkout", "--detach"], repo_root)

    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    mgr = OverlayManager(repo_root, git_context=git_ctx)

    ctx = git_ctx.get()
    assert ctx.is_detached
    expected_slug = f"detached-{sha[:12]}.jsonl"
    assert ctx.overlay_slug == expected_slug

    record = _upsert()
    mgr.append_to_current_branch_overlay(record)

    overlay_path = mgr.current_overlay_path()
    assert overlay_path.name == expected_slug
    assert overlay_path.exists()

    result = mgr.replay_active()
    assert record.link_id in result.active_links

    pending = mgr.list_pending_overlay_records()
    assert any(r.link_id == record.link_id for r in pending)


# ─── Tests: list_pending_branches ────────────────────────────────────────────


def test_list_pending_branches_empty(git_repo: tuple[Path, str]) -> None:
    """list_pending_branches returns [] when no overlay files exist."""
    repo_root, _ = git_repo
    mgr = OverlayManager(repo_root)
    assert mgr.list_pending_branches() == []


def test_list_pending_branches_no_overlay_dir(tmp_path: Path) -> None:
    """list_pending_branches returns [] when overlay directory doesn't exist."""
    sha = _init_repo(tmp_path)
    # .scry/overlays/ doesn't exist yet
    assert not (tmp_path / ".scry" / "overlays").exists()
    mgr = OverlayManager(
        tmp_path,
        git_context=GitContextProvider(tmp_path, head_poll_interval_seconds=0),
    )
    # Confirm we get [] and no exception
    result = mgr.list_pending_branches()
    assert result == []
    _ = sha  # used to initialize the repo


def test_list_pending_branches_enumerates_all(git_repo: tuple[Path, str]) -> None:
    """list_pending_branches lists slugs from all overlay files (no .jsonl suffix)."""
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    mgr = OverlayManager(repo_root, git_context=git_ctx)

    # Write on initial branch
    mgr.append_to_current_branch_overlay(_upsert())
    initial_slug = git_ctx.get().overlay_slug  # e.g. "main.jsonl"

    # Create + write on two more branches
    _git(["checkout", "-b", "branch-alpha"], repo_root)
    mgr.append_to_current_branch_overlay(_upsert())
    _git(["checkout", "-b", "branch-beta"], repo_root)
    mgr.append_to_current_branch_overlay(_upsert())

    branches = mgr.list_pending_branches()
    # All three overlay stems should be present
    initial_stem = initial_slug[: -len(".jsonl")]  # strip ".jsonl"
    assert initial_stem in branches
    assert "branch-alpha" in branches
    assert "branch-beta" in branches
    assert len(branches) == 3
    assert branches == sorted(branches)  # must be sorted


# ─── Tests: list_pending_overlay_records ────────────────────────────────────


def test_list_pending_overlay_records_current_branch(mgr: OverlayManager) -> None:
    """list_pending_overlay_records returns only the current branch's records."""
    r1 = _upsert()
    r2 = _upsert()
    mgr.append_to_current_branch_overlay(r1)
    mgr.append_to_current_branch_overlay(r2)

    records = mgr.list_pending_overlay_records()
    link_ids = {r.link_id for r in records}
    assert r1.link_id in link_ids
    assert r2.link_id in link_ids
    assert len(records) == 2


def test_list_pending_overlay_records_empty_branch(mgr: OverlayManager) -> None:
    """list_pending_overlay_records returns [] on a branch with no overlay file."""
    assert mgr.list_pending_overlay_records() == []


# ─── Tests: promote_pending ──────────────────────────────────────────────────


def test_promote_pending_all(git_repo: tuple[Path, str]) -> None:
    """promote_pending() with no event_ids promotes everything to baseline."""
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    link_store = LinkStore(repo_root)
    mgr = OverlayManager(repo_root, git_context=git_ctx, link_store=link_store)

    r1 = _upsert()
    r2 = _upsert()
    mgr.append_to_current_branch_overlay(r1)
    mgr.append_to_current_branch_overlay(r2)

    promoted = mgr.promote_pending()

    assert set(promoted) == {r1.event_id, r2.event_id}
    # Overlay is now empty
    assert mgr.list_pending_overlay_records() == []
    # Baseline has both records
    baseline_records = link_store.read_records(link_store.baseline_path)
    baseline_link_ids = {r.link_id for r in baseline_records}
    assert r1.link_id in baseline_link_ids
    assert r2.link_id in baseline_link_ids


def test_promote_pending_subset(git_repo: tuple[Path, str]) -> None:
    """promote_pending(event_ids=[...]) promotes only the specified records."""
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    link_store = LinkStore(repo_root)
    mgr = OverlayManager(repo_root, git_context=git_ctx, link_store=link_store)

    r1 = _upsert()
    r2 = _upsert()
    r3 = _upsert()
    mgr.append_to_current_branch_overlay(r1)
    mgr.append_to_current_branch_overlay(r2)
    mgr.append_to_current_branch_overlay(r3)

    promoted = mgr.promote_pending(event_ids=[r1.event_id, r2.event_id])

    assert set(promoted) == {r1.event_id, r2.event_id}
    # r3 still in overlay
    remaining = mgr.list_pending_overlay_records()
    remaining_ids = {r.link_id for r in remaining}
    assert r3.link_id in remaining_ids
    assert r1.link_id not in remaining_ids
    assert r2.link_id not in remaining_ids
    # r1, r2 in baseline
    baseline_records = link_store.read_records(link_store.baseline_path)
    baseline_link_ids = {r.link_id for r in baseline_records}
    assert r1.link_id in baseline_link_ids
    assert r2.link_id in baseline_link_ids
    assert r3.link_id not in baseline_link_ids


def test_promote_pending_returns_empty_on_no_records(mgr: OverlayManager) -> None:
    """promote_pending() on an empty overlay returns [] without error."""
    result = mgr.promote_pending()
    assert result == []


# ─── Tests: recover_pending ──────────────────────────────────────────────────


def test_recover_pending_no_markers(mgr: OverlayManager) -> None:
    """recover_pending returns [] when no marker files are present."""
    result = mgr.recover_pending()
    assert result == []


def test_recover_pending_after_crash(git_repo: tuple[Path, str]) -> None:
    """recover_pending completes step 2+3 after a simulated step-1-only crash.

    Scenario: step 1 completed (records appended to baseline + marker written)
    but the process crashed before step 2 (overlay rewrite) or step 3 (marker
    delete).  recover_pending should strip the promoted records from the
    overlay and delete the marker.
    """
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    link_store = LinkStore(repo_root)
    mgr = OverlayManager(repo_root, git_context=git_ctx, link_store=link_store)

    # Append two records to the overlay
    r1 = _upsert()
    r2 = _upsert()
    mgr.append_to_current_branch_overlay(r1)
    mgr.append_to_current_branch_overlay(r2)

    overlay_slug = git_ctx.get().overlay_slug
    overlay_path = mgr.current_overlay_path()

    # Simulate step 1: append the records to baseline (already "promoted")
    link_store.append_baseline(r1)
    link_store.append_baseline(r2)

    # Simulate step 1: write a marker file (crash occurred before step 2)
    txn_id = "crash-sim-txn-001"
    marker_data = {
        "txn_id": txn_id,
        "overlay_path": f"overlays/{overlay_slug}",
        "promoted_event_ids": [r1.event_id, r2.event_id],
        "ts": datetime.now(UTC).isoformat(),
    }
    scry_dir = repo_root / ".scry"
    marker_path = scry_dir / f"commit-links.{txn_id}.marker"
    marker_path.write_text(json.dumps(marker_data), encoding="utf-8")

    # Verify pre-recovery state: overlay still has both records
    assert len(link_store.read_records(overlay_path)) == 2
    assert marker_path.exists()

    # Recover
    recovered = mgr.recover_pending()

    # One PendingPromotion returned (one marker)
    assert len(recovered) == 1
    assert recovered[0].txn_id == txn_id
    assert set(recovered[0].promoted_event_ids) == {r1.event_id, r2.event_id}
    assert recovered[0].overlay_path == overlay_path

    # Step 2 completed: overlay no longer contains the promoted records
    assert link_store.read_records(overlay_path) == []

    # Step 3 completed: marker deleted
    assert not marker_path.exists()

    # Baseline still intact
    baseline_records = link_store.read_records(link_store.baseline_path)
    baseline_ids = {r.link_id for r in baseline_records}
    assert r1.link_id in baseline_ids
    assert r2.link_id in baseline_ids


def test_recover_pending_ts_parsed(git_repo: tuple[Path, str]) -> None:
    """PendingPromotion.ts is correctly parsed from the marker file."""
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    link_store = LinkStore(repo_root)
    mgr = OverlayManager(repo_root, git_context=git_ctx, link_store=link_store)

    # Append one record and build a marker manually
    r1 = _upsert()
    mgr.append_to_current_branch_overlay(r1)
    overlay_slug = git_ctx.get().overlay_slug
    link_store.append_baseline(r1)

    ts_str = "2026-01-15T10:30:00+00:00"
    txn_id = "ts-test-txn"
    scry_dir = repo_root / ".scry"
    marker_data = {
        "txn_id": txn_id,
        "overlay_path": f"overlays/{overlay_slug}",
        "promoted_event_ids": [r1.event_id],
        "ts": ts_str,
    }
    (scry_dir / f"commit-links.{txn_id}.marker").write_text(
        json.dumps(marker_data), encoding="utf-8"
    )

    recovered = mgr.recover_pending()
    assert len(recovered) == 1
    assert recovered[0].ts == datetime.fromisoformat(ts_str)


def test_recover_pending_missing_ts_uses_fallback(git_repo: tuple[Path, str]) -> None:
    """A marker without a 'ts' key uses datetime.min (UTC) as fallback."""
    repo_root, _ = git_repo
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=0)
    link_store = LinkStore(repo_root)
    mgr = OverlayManager(repo_root, git_context=git_ctx, link_store=link_store)

    r1 = _upsert()
    mgr.append_to_current_branch_overlay(r1)
    overlay_slug = git_ctx.get().overlay_slug
    link_store.append_baseline(r1)

    txn_id = "no-ts-txn"
    scry_dir = repo_root / ".scry"
    marker_data = {
        "txn_id": txn_id,
        "overlay_path": f"overlays/{overlay_slug}",
        "promoted_event_ids": [r1.event_id],
        # no "ts" key
    }
    (scry_dir / f"commit-links.{txn_id}.marker").write_text(
        json.dumps(marker_data), encoding="utf-8"
    )

    recovered = mgr.recover_pending()
    assert len(recovered) == 1
    assert recovered[0].ts == datetime.min.replace(tzinfo=UTC)


# ─── Tests: cross-process visibility ─────────────────────────────────────────


def test_cross_process_two_managers_see_each_others_writes(
    git_repo: tuple[Path, str],
) -> None:
    """Two independent OverlayManager instances share the same JSONL files.

    This simulates two processes (or two objects in one process) both
    operating on the same repo root.  After mgr1 writes, mgr2 can read the
    record without any explicit sync or notification.
    """
    repo_root, _ = git_repo
    mgr1 = OverlayManager(
        repo_root,
        git_context=GitContextProvider(repo_root, head_poll_interval_seconds=0),
    )
    mgr2 = OverlayManager(
        repo_root,
        git_context=GitContextProvider(repo_root, head_poll_interval_seconds=0),
    )

    record = _upsert()
    mgr1.append_to_current_branch_overlay(record)

    # mgr2 reads from disk — no in-memory caching of file contents
    result = mgr2.replay_active()
    assert record.link_id in result.active_links

    pending = mgr2.list_pending_overlay_records()
    assert any(r.link_id == record.link_id for r in pending)


# ─── Tests: cache invalidation on write ──────────────────────────────────────


def test_cache_invalidation_on_append(git_repo: tuple[Path, str]) -> None:
    """append_to_current_branch_overlay invalidates the cache before writing.

    Without invalidation, a write after a branch switch would use the stale
    cached overlay slug (from the previous branch) and route to the wrong
    file.  This test verifies the opposite: the write goes to the FRESH
    branch's overlay even when the cache is warm with the old branch's data.
    """
    repo_root, _ = git_repo

    # Long TTL — cache won't expire naturally during the test
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=3600)
    mgr = OverlayManager(repo_root, git_context=git_ctx)

    # Warm the cache with the initial branch's overlay slug
    ctx_initial = git_ctx.get()
    initial_slug = ctx_initial.overlay_slug  # e.g. "main.jsonl"

    # Switch to a new branch (cache is now stale)
    _git(["checkout", "-b", "cache-inv-test"], repo_root)

    # The cache still reports the initial branch (TTL not expired)
    cached = git_ctx.get()
    assert cached.overlay_slug == initial_slug, "cache should still be warm with initial slug"

    # Write via manager — must invalidate first, then route to fresh branch
    record = _upsert()
    mgr.append_to_current_branch_overlay(record)

    # New overlay ("cache-inv-test.jsonl") must exist with the record
    new_overlay_path = mgr.overlay_dir / "cache-inv-test.jsonl"
    assert new_overlay_path.exists(), "write should go to the fresh branch overlay"

    # Old overlay ("main.jsonl" etc.) must NOT have been touched
    old_overlay_path = mgr.overlay_dir / initial_slug
    assert not old_overlay_path.exists(), "stale overlay must not be written"


def test_cache_invalidation_on_promote(git_repo: tuple[Path, str]) -> None:
    """promote_pending invalidates the cache before resolving the overlay path.

    Setup: write a record to the initial branch's overlay via ``link_store``
    directly (bypasses OverlayManager's cache invalidation so the cache can
    be controlled manually).  Warm the cache to a *different* branch slug.
    Switch back to the initial branch.  At this point the cache is stale.
    Call ``promote_pending()`` — it MUST invalidate first, then see the
    initial branch's overlay and promote the record.
    """
    repo_root, _ = git_repo

    # Long TTL so the cache never expires naturally
    git_ctx = GitContextProvider(repo_root, head_poll_interval_seconds=3600)
    link_store = LinkStore(repo_root)
    mgr = OverlayManager(repo_root, git_context=git_ctx, link_store=link_store)

    # Get the initial branch slug (force refresh, no side-effects on mgr)
    ctx_initial = git_ctx.get(force_refresh=True)
    initial_slug = ctx_initial.overlay_slug  # e.g. "main.jsonl"
    initial_branch = initial_slug[: -len(".jsonl")]  # e.g. "main"

    # Write a record directly via link_store so the git cache is untouched
    record = _upsert()
    overlay_path_initial = mgr.overlay_path_for(initial_slug)
    link_store.append_overlay(record, overlay_path_initial)
    # Cache is still "initial_slug" (link_store doesn't touch git_ctx)
    assert git_ctx.get().overlay_slug == initial_slug

    # Switch to a new branch and force-refresh the cache to the new slug
    _git(["checkout", "-b", "promote-stale-test"], repo_root)
    new_ctx = git_ctx.get(force_refresh=True)
    new_slug = new_ctx.overlay_slug  # "promote-stale-test.jsonl"
    assert new_slug != initial_slug

    # Switch BACK to the initial branch — cache is stale (still says new_slug)
    _git(["checkout", initial_branch], repo_root)
    stale = git_ctx.get()
    assert stale.overlay_slug == new_slug, "cache must be stale with new-branch slug"

    # promote_pending() must invalidate → see initial branch → promote the record.
    # Without invalidation it would look at the empty "promote-stale-test.jsonl"
    # and return [].  With invalidation it reads "initial_slug" and returns the event.
    promoted = mgr.promote_pending()
    assert set(promoted) == {record.event_id}, (
        "promote_pending must invalidate the cache and promote from the real current overlay"
    )

    # Baseline has the promoted record
    baseline_records = link_store.read_records(link_store.baseline_path)
    assert any(r.link_id == record.link_id for r in baseline_records)


# ─── Tests: PendingPromotion dataclass ───────────────────────────────────────


def test_pending_promotion_is_frozen() -> None:
    """PendingPromotion is a frozen dataclass (immutable)."""
    pp = PendingPromotion(
        txn_id="abc",
        overlay_path=Path("/tmp/x.jsonl"),
        promoted_event_ids=[],
        ts=datetime.now(UTC),
    )
    with pytest.raises((AttributeError, TypeError)):
        pp.txn_id = "other"  # type: ignore[misc]


def test_pending_promotion_fields() -> None:
    """PendingPromotion carries all four required fields."""
    now = datetime.now(UTC)
    eids: list[str] = ["evt_abc123", "evt_def456"]
    pp = PendingPromotion(
        txn_id="txn-001",
        overlay_path=Path("/repo/.scry/overlays/main.jsonl"),
        promoted_event_ids=eids,
        ts=now,
    )
    assert pp.txn_id == "txn-001"
    assert pp.overlay_path == Path("/repo/.scry/overlays/main.jsonl")
    assert pp.promoted_event_ids == eids
    assert pp.ts == now


# uat-r5-5 pr-d noise
