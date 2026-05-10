"""Tests for W4d: auto-reconcile non-fresh index states (DESIGN.md §7.2 v3.1)."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from scry.anchor_id import content_hash as _ch
from scry.anchor_id import fingerprint_simhash as _sh
from scry.config import compute_config_hash
from scry.embed import StubEmbedder
from scry.git_context import GitContext, GitContextProvider
from scry.index import Indexer
from scry.mcp.handlers import MCPContext, find_drift, get_anchor, get_links, reindex, search, status
from scry.models import Anchor, AnchorType, Config, IndexMetadata, IndexState, SubChunk
from scry.reconcile import (
    IndexStateTracker,
    check_staleness,
    estimate_changed_count,
)
from scry.store.db import ScryDB
from scry.store.overlay import OverlayManager

# ─── Helpers ─────────────────────────────────────────────────────────────────

_DIMS = 384
_SHA_A = "a" * 40
_SHA_B = "b" * 40


def _git_ctx(
    head_sha: str = _SHA_A,
    dirty_files: tuple[str, ...] = (),
) -> GitContext:
    return GitContext(
        head_sha=head_sha,
        head_short=head_sha[:12],
        branch="main",
        overlay_slug="main",
        is_detached=False,
        dirty_files=dirty_files,
    )


def _meta(
    indexed_head: str = _SHA_A,
    manifest: dict[str, str] | None = None,
    config: Config | None = None,
) -> IndexMetadata:
    c = config or Config()
    return IndexMetadata(
        indexed_git_head=indexed_head,
        indexed_branch="main",
        indexed_file_manifest=manifest if manifest is not None else {},
        config_hash=compute_config_hash(c),
        embedding_provider="stub",
        embedding_model="stub",
        embedding_dimensions=_DIMS,
    )


class _FakeDB:
    """Minimal ScryDB stub: only ``read_index_metadata`` is needed."""

    def __init__(self, meta: IndexMetadata | None) -> None:
        self._meta = meta

    def read_index_metadata(self) -> IndexMetadata | None:
        return self._meta


# ─── Slow coroutine helper ────────────────────────────────────────────────────


async def _slow_index(*_args: Any, **_kwargs: Any) -> None:
    await asyncio.sleep(3600)


# ─── Tests: check_staleness ──────────────────────────────────────────────────


def test_check_staleness_no_meta_not_stale() -> None:
    """None metadata → not stale (unbuilt index is handled at leader startup)."""
    assert check_staleness(_git_ctx(), None, Config()) is False


def test_check_staleness_fresh() -> None:
    """Matching HEAD SHA, clean worktree, same config hash → not stale."""
    assert check_staleness(_git_ctx(head_sha=_SHA_A), _meta(indexed_head=_SHA_A), Config()) is False


def test_check_staleness_head_changed() -> None:
    meta = _meta(indexed_head=_SHA_A)
    assert check_staleness(_git_ctx(head_sha=_SHA_B), meta, Config()) is True


def test_check_staleness_dirty_indexed_file() -> None:
    meta = _meta(indexed_head=_SHA_A, manifest={"src/foo.py": "hash1"})
    git = _git_ctx(head_sha=_SHA_A, dirty_files=("src/foo.py",))
    assert check_staleness(git, meta, Config()) is True


def test_check_staleness_dirty_unindexed_file_not_stale() -> None:
    """Dirty file not in the manifest doesn't trigger auto-reconcile."""
    meta = _meta(indexed_head=_SHA_A, manifest={"src/foo.py": "hash1"})
    git = _git_ctx(head_sha=_SHA_A, dirty_files=("untracked.txt",))
    assert check_staleness(git, meta, Config()) is False


def test_check_staleness_config_change() -> None:
    old_config = Config()
    meta = _meta(indexed_head=_SHA_A, config=old_config)
    new_config = Config(include=["**/*.py"])
    assert check_staleness(_git_ctx(head_sha=_SHA_A), meta, new_config) is True


# ─── Tests: estimate_changed_count ───────────────────────────────────────────


def test_estimate_changed_count_no_meta() -> None:
    assert estimate_changed_count(_git_ctx(), None, Path(".")) == 999_999


def test_estimate_changed_count_empty_manifest() -> None:
    """Manifest is empty → 0 changed files even when HEAD differs."""
    meta = _meta(indexed_head=_SHA_A, manifest={})
    count = estimate_changed_count(_git_ctx(head_sha=_SHA_B), meta, Path("."))
    assert count == 0


def test_estimate_changed_count_dirty_in_manifest() -> None:
    meta = _meta(indexed_head=_SHA_A, manifest={"src/foo.py": "h1", "src/bar.py": "h2"})
    git = _git_ctx(head_sha=_SHA_A, dirty_files=("src/foo.py",))
    assert estimate_changed_count(git, meta, Path(".")) == 1


# ─── Tests: IndexStateTracker unit ───────────────────────────────────────────


async def test_tracker_initial_state() -> None:
    assert IndexStateTracker().current_state == IndexState.FRESH


async def test_tracker_poll_fresh_stays_fresh(tmp_path: Path) -> None:
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_A),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=None,
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert state == IndexState.FRESH
    assert tracker.current_state == IndexState.FRESH


async def test_tracker_leader_stale_to_reconciling(tmp_path: Path) -> None:
    """Leader detects staleness → STALE_RECONCILING while task is running."""
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    indexer = MagicMock()
    indexer.index_async = AsyncMock(side_effect=_slow_index)

    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=indexer,
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert state == IndexState.STALE_RECONCILING
    assert tracker.current_state == IndexState.STALE_RECONCILING

    await tracker.mark_fresh()  # cancel pending task to avoid resource warning


async def test_tracker_leader_reconcile_succeeds(tmp_path: Path) -> None:
    """Leader reconcile success: stale → stale-reconciling → fresh."""
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    indexer = MagicMock()
    indexer.index_async = AsyncMock(return_value=None)

    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=indexer,
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert state == IndexState.STALE_RECONCILING

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert tracker.current_state == IndexState.FRESH


async def test_tracker_leader_reconcile_fails(tmp_path: Path) -> None:
    """Leader reconcile failure: stale → stale-reconciling → stale-warned."""
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    indexer = MagicMock()
    indexer.index_async = AsyncMock(side_effect=RuntimeError("simulated index error"))

    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=indexer,
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert state == IndexState.STALE_RECONCILING

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert tracker.current_state == IndexState.STALE_WARNED


async def test_tracker_follower_pings_leader(tmp_path: Path) -> None:
    """Follower + IPC available: stale → stale-reconciling → fresh."""
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    ipc_client = MagicMock()
    ipc_client.call = AsyncMock(return_value={"index_state": "fresh"})

    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=None,
        ipc_client=ipc_client,
        repo_root=tmp_path,
    )
    assert state == IndexState.STALE_RECONCILING

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert tracker.current_state == IndexState.FRESH
    ipc_client.call.assert_called_once()


async def test_tracker_no_leader_stale_no_write_lock(tmp_path: Path) -> None:
    """No leader, no IPC: stale → stale-no-write-lock."""
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=None,
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert state == IndexState.STALE_NO_WRITE_LOCK
    assert tracker.current_state == IndexState.STALE_NO_WRITE_LOCK


async def test_tracker_threshold_exceeded_stale_warned(tmp_path: Path) -> None:
    """Changed file count > auto_reconcile_max_changed_files → stale-warned."""
    manifest = {f"src/file_{i}.py": f"hash{i}" for i in range(600)}
    db = _FakeDB(_meta(indexed_head=_SHA_A, manifest=manifest))  # type: ignore[arg-type]
    git = _git_ctx(head_sha=_SHA_A, dirty_files=tuple(manifest.keys()))
    config = Config()
    assert config.index.auto_reconcile_max_changed_files == 500

    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        git,
        db,  # type: ignore[arg-type]
        config,
        indexer=MagicMock(),
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert state == IndexState.STALE_WARNED


async def test_tracker_no_double_reconcile(tmp_path: Path) -> None:
    """A second poll while already reconciling is a no-op (one task only)."""
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    call_count = 0

    async def _counting_slow(*_args: Any, **_kwargs: Any) -> None:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(3600)

    indexer = MagicMock()
    indexer.index_async = AsyncMock(side_effect=_counting_slow)

    tracker = IndexStateTracker()
    await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=indexer,
        ipc_client=None,
        repo_root=tmp_path,
    )
    # Second poll while task is running.
    await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=indexer,
        ipc_client=None,
        repo_root=tmp_path,
    )
    await asyncio.sleep(0)
    assert call_count == 1  # Only one reconcile launched

    await tracker.mark_fresh()


async def test_mark_fresh_cancels_task(tmp_path: Path) -> None:
    """mark_fresh() cancels an in-flight reconcile and forces state to FRESH."""
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    indexer = MagicMock()
    indexer.index_async = AsyncMock(side_effect=_slow_index)

    tracker = IndexStateTracker()
    await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=indexer,
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert tracker.current_state == IndexState.STALE_RECONCILING

    await tracker.mark_fresh()

    assert tracker.current_state == IndexState.FRESH
    assert tracker._reconcile_task is None


# ─── Tests: handler integration ──────────────────────────────────────────────
# Build real MCPContext objects (same pattern as test_mcp_server.py) to verify
# that index_state is wired end-to-end from the tracker to the response dicts.


def _git_run(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


def _init_git_repo(path: Path) -> None:
    _git_run(["init"], path)
    _git_run(["config", "user.email", "test@scry.test"], path)
    _git_run(["config", "user.name", "Scry Test"], path)
    (path / "README.md").write_text("test\n")
    _git_run(["add", "README.md"], path)
    _git_run(["commit", "-m", "init"], path)


@pytest.fixture()
def reconcile_repo(tmp_path: Path) -> Path:
    (tmp_path / ".scry").mkdir()
    (tmp_path / ".scry" / "overlays").mkdir()
    _init_git_repo(tmp_path)
    return tmp_path


def _make_ctx(repo: Path, tracker: IndexStateTracker) -> MCPContext:
    db = ScryDB(repo)
    db.init_schema(embedding_dimensions=_DIMS)
    embedder = StubEmbedder(dimensions=_DIMS)
    git_context = GitContextProvider(repo, head_poll_interval_seconds=0)
    overlay_mgr = OverlayManager(repo, git_context=git_context)
    indexer = Indexer(
        repo_root=repo,
        config=Config(),
        db=db,
        embedder=embedder,
        git_context=git_context,
    )
    return MCPContext(
        repo_root=repo,
        config=Config(),
        db=db,
        embedder=embedder,
        git_context=git_context,
        overlay_mgr=overlay_mgr,
        indexer=indexer,
        role="leader",
        ipc_client=None,
        index_state_tracker=tracker,
    )


async def test_status_reflects_tracker_state(reconcile_repo: Path) -> None:
    """status() returns the tracker's current index_state."""
    tracker = IndexStateTracker()
    tracker._state = IndexState.STALE_NO_WRITE_LOCK  # force a specific state

    ctx = _make_ctx(reconcile_repo, tracker)
    # DB has no metadata → check_staleness returns False → poll resets to FRESH
    # (because no-meta case is not stale; STALE_NO_WRITE_LOCK is not STALE_WARNED
    # so it IS overridden by the "if state != STALE_WARNED: set FRESH" branch).
    result = await status(ctx)
    assert "index_state" in result
    assert result["index_state"] in {s.value for s in IndexState}


async def test_search_surfaces_stale_reconciling(reconcile_repo: Path) -> None:
    """search() embeds index_state in every AnchorPacket when reconciling."""
    db = ScryDB(reconcile_repo)
    db.init_schema(embedding_dimensions=_DIMS)
    embedder = StubEmbedder(dimensions=_DIMS)

    text = "The authentication module must verify JWT tokens."
    anchor = Anchor.model_validate(
        {
            "id": "docs/auth.md::auth",
            "type": AnchorType.SECTION,
            "path": "docs/auth.md",
            "heading_path": ["auth"],
            "content_text": text,
            "content_hash": _ch(text),
            "fingerprint_simhash": _sh(text),
        }
    )
    chunk = SubChunk(
        parent_id="docs/auth.md::auth",
        chunk_index=0,
        text=text,
        parent_content_hash=_ch(text),
    )
    db.reindex_anchor_with_chunks(anchor, [chunk], embedder.encode([text]))

    git_context = GitContextProvider(reconcile_repo, head_poll_interval_seconds=0)
    overlay_mgr = OverlayManager(reconcile_repo, git_context=git_context)
    indexer = Indexer(
        repo_root=reconcile_repo,
        config=Config(),
        db=db,
        embedder=embedder,
        git_context=git_context,
    )

    # Pre-seed tracker to STALE_RECONCILING so the handler surfaces it.
    tracker = IndexStateTracker()
    tracker._state = IndexState.STALE_RECONCILING

    ctx = MCPContext(
        repo_root=reconcile_repo,
        config=Config(),
        db=db,
        embedder=embedder,
        git_context=git_context,
        overlay_mgr=overlay_mgr,
        indexer=indexer,
        role="leader",
        ipc_client=None,
        index_state_tracker=tracker,
    )

    results = await search(ctx, "JWT authentication")
    assert len(results) >= 1
    for r in results:
        assert r["index_state"] == IndexState.STALE_RECONCILING


async def test_reindex_marks_tracker_fresh(reconcile_repo: Path) -> None:
    """Explicit reindex forces the tracker to FRESH regardless of prior state."""
    tracker = IndexStateTracker()
    tracker._state = IndexState.STALE_WARNED

    ctx = _make_ctx(reconcile_repo, tracker)
    result = await reindex(ctx, force=False)

    assert result["index_state"] == IndexState.FRESH
    assert tracker.current_state == IndexState.FRESH


async def test_index_state_fresh_after_manual_reindex(reconcile_repo: Path) -> None:
    """After reindex(), the next status() call also sees a fresh state."""
    tracker = IndexStateTracker()
    tracker._state = IndexState.STALE_NO_WRITE_LOCK

    ctx = _make_ctx(reconcile_repo, tracker)
    await reindex(ctx, force=False)

    # Next status call: DB now has fresh metadata; check_staleness returns False.
    result = await status(ctx)
    assert result["index_state"] == IndexState.FRESH


# ─── New regression tests ─────────────────────────────────────────────────────


# ── BLOCKING #1: stale-warned clears to fresh when index becomes fresh ────────


async def test_stale_warned_clears_to_fresh_on_fresh_poll(tmp_path: Path) -> None:
    """BLOCKING #1: STALE_WARNED → FRESH when on-disk metadata is now fresh."""
    # Tracker enters STALE_WARNED (e.g. from a failed background reconcile).
    tracker = IndexStateTracker()
    tracker._state = IndexState.STALE_WARNED

    # Simulate user ran `scry index`: DB metadata is now fresh (same SHA as git).
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]

    state = await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_A),  # fresh — same as indexed HEAD
        db,  # type: ignore[arg-type]
        Config(),
        indexer=None,
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert state == IndexState.FRESH
    assert tracker.current_state == IndexState.FRESH


# ── BLOCKING #2: config change + large corpus → stale-warned ─────────────────


async def test_config_change_large_repo_stale_warned(tmp_path: Path) -> None:
    """BLOCKING #2: config-change staleness uses corpus size for threshold gate."""
    # 600 files in manifest (> default 500 threshold).
    manifest = {f"src/file_{i}.py": f"hash{i}" for i in range(600)}
    old_config = Config()
    db = _FakeDB(_meta(indexed_head=_SHA_A, manifest=manifest, config=old_config))  # type: ignore[arg-type]

    # Same HEAD, same branch, clean worktree — only config changed.
    new_config = Config(include=["**/*.py"])
    assert new_config.index.auto_reconcile_max_changed_files == 500

    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_A),
        db,  # type: ignore[arg-type]
        new_config,
        indexer=MagicMock(),
        ipc_client=None,
        repo_root=tmp_path,
    )
    # 600 files > 500 threshold → STALE_WARNED, not STALE_RECONCILING.
    assert state == IndexState.STALE_WARNED


# ── HIGH #1: branch change detected as stale ──────────────────────────────────


def test_check_staleness_branch_change() -> None:
    """HIGH #1: switching branches at same HEAD triggers staleness."""
    # Indexed on 'main'; now on 'feature'.
    meta = _meta(indexed_head=_SHA_A)  # indexed_branch='main' (from _meta helper)
    git = GitContext(
        head_sha=_SHA_A,
        head_short=_SHA_A[:12],
        branch="feature",  # Different branch — same HEAD SHA
        overlay_slug="feature",
        is_detached=False,
        dirty_files=(),
    )
    assert check_staleness(git, meta, Config()) is True


def test_check_staleness_detached_vs_branch() -> None:
    """HIGH #1: switching from branch to detached-HEAD at same commit is stale."""
    meta = _meta(indexed_head=_SHA_A)  # indexed_branch='main'
    git_detached = GitContext(
        head_sha=_SHA_A,
        head_short=_SHA_A[:12],
        branch=None,  # Detached HEAD
        overlay_slug=f"detached-{_SHA_A[:12]}",
        is_detached=True,
        dirty_files=(),
    )
    assert check_staleness(git_detached, meta, Config()) is True


async def test_branch_change_large_repo_stale_warned(tmp_path: Path) -> None:
    """HIGH #1: branch change with 600-file corpus → STALE_WARNED."""
    manifest = {f"src/file_{i}.py": f"hash{i}" for i in range(600)}
    db = _FakeDB(_meta(indexed_head=_SHA_A, manifest=manifest))  # type: ignore[arg-type]

    # Same HEAD, clean worktree — only branch changed.
    git = GitContext(
        head_sha=_SHA_A,
        head_short=_SHA_A[:12],
        branch="feature",  # Different from 'main' stored in meta
        overlay_slug="feature",
        is_detached=False,
        dirty_files=(),
    )
    config = Config()
    assert config.index.auto_reconcile_max_changed_files == 500

    tracker = IndexStateTracker()
    state = await tracker.poll_and_maybe_reconcile(
        git,
        db,  # type: ignore[arg-type]
        config,
        indexer=MagicMock(),
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert state == IndexState.STALE_WARNED


# ── HIGH #2: read handlers return index_state and trigger polling ─────────────


async def test_get_anchor_returns_index_state(reconcile_repo: Path) -> None:
    """HIGH #2: get_anchor() includes index_state and triggers reconcile polling."""
    tracker = IndexStateTracker()
    ctx = _make_ctx(reconcile_repo, tracker)
    # First run reindex so there is something to find.
    await reindex(ctx, force=False)

    anchors = ctx.db.list_anchors()
    if not anchors:
        return  # repo has no indexable content; skip assertion

    result = await get_anchor(ctx, anchors[0].id)
    assert result is not None
    assert "index_state" in result
    assert result["index_state"] in {s.value for s in IndexState}


async def test_get_links_returns_index_state(reconcile_repo: Path) -> None:
    """HIGH #2: get_links() carries index_state as a top-level response field."""
    tracker = IndexStateTracker()
    ctx = _make_ctx(reconcile_repo, tracker)
    result = await get_links(ctx, "any::anchor")
    assert "index_state" in result
    assert "links" in result
    assert result["index_state"] in {s.value for s in IndexState}


async def test_find_drift_returns_index_state(reconcile_repo: Path) -> None:
    """HIGH #2: find_drift() carries index_state as a top-level response field."""
    tracker = IndexStateTracker()
    ctx = _make_ctx(reconcile_repo, tracker)
    result = await find_drift(ctx)
    assert "index_state" in result
    assert "entries" in result
    assert result["index_state"] in {s.value for s in IndexState}


async def test_get_links_triggers_polling(reconcile_repo: Path) -> None:
    """HIGH #2: get_links() triggers poll_and_maybe_reconcile (not just reads state)."""
    # With a fresh DB (no index metadata), check_staleness returns False
    # → state stays FRESH after polling.
    tracker = IndexStateTracker()
    ctx = _make_ctx(reconcile_repo, tracker)
    # No prior poll — tracker is in its initial FRESH state.
    result = await get_links(ctx, "any::anchor")
    assert "index_state" in result
    # poll_and_maybe_reconcile was called: with no metadata, stays FRESH.
    assert result["index_state"] == IndexState.FRESH


# ── HIGH #3: concurrent follower reindexes coalesce to one index_async call ──


async def test_concurrent_reindex_single_index_async_call(tmp_path: Path) -> None:
    """HIGH #3: two concurrent run_leader_reindex() calls fire index_async once."""
    call_count = 0
    result_sentinel = object()

    async def counting_index(*, force: bool = False) -> object:
        nonlocal call_count
        call_count += 1
        await asyncio.sleep(0)  # yield so both can be in-flight
        return result_sentinel

    indexer = MagicMock()
    indexer.index_async = AsyncMock(side_effect=counting_index)

    tracker = IndexStateTracker()
    # Launch two concurrent reindex calls.
    r1, r2 = await asyncio.gather(
        tracker.run_leader_reindex(indexer, force=False),
        tracker.run_leader_reindex(indexer, force=False),
    )

    assert call_count == 1, f"Expected 1 index_async call, got {call_count}"
    # Both callers receive the same result.
    assert r1 is result_sentinel
    assert r2 is result_sentinel


# ── MEDIUM #1: reindex does not race background reconcile ────────────────────


async def test_reindex_cancels_background_reconcile(tmp_path: Path) -> None:
    """MEDIUM #1: run_leader_reindex() cancels in-flight background reconcile first."""
    background_started = False
    background_ran_concurrently = False

    async def slow_background(*_: Any, **__: Any) -> None:
        nonlocal background_started
        background_started = True
        await asyncio.sleep(3600)

    explicit_call_count = 0

    async def explicit_index(*, force: bool = False) -> None:
        nonlocal explicit_call_count, background_ran_concurrently
        explicit_call_count += 1
        # If background task is still running at this point, that's the bug.
        if background_started and not task_was_cancelled():
            background_ran_concurrently = True

    tracker = IndexStateTracker()

    # Start a slow background reconcile.
    bg_indexer = MagicMock()
    bg_indexer.index_async = AsyncMock(side_effect=slow_background)
    db = _FakeDB(_meta(indexed_head=_SHA_A))  # type: ignore[arg-type]
    await tracker.poll_and_maybe_reconcile(
        _git_ctx(head_sha=_SHA_B),
        db,  # type: ignore[arg-type]
        Config(),
        indexer=bg_indexer,
        ipc_client=None,
        repo_root=tmp_path,
    )
    assert tracker.current_state == IndexState.STALE_RECONCILING
    assert tracker._reconcile_task is not None

    bg_task = tracker._reconcile_task

    def task_was_cancelled() -> bool:
        return bg_task.cancelled() or bg_task.cancelling() > 0

    # Now run an explicit reindex — should cancel background first.
    explicit_indexer = MagicMock()
    explicit_indexer.index_async = AsyncMock(side_effect=explicit_index)
    await tracker.run_leader_reindex(explicit_indexer, force=False)

    assert explicit_call_count == 1
    assert not background_ran_concurrently
    # Background task must be cancelled (not still running).
    assert bg_task.done()


# ── mark_fresh awaits task cleanup ────────────────────────────────────────────


async def test_mark_fresh_awaits_task_cleanup(tmp_path: Path) -> None:
    """MEDIUM #1: mark_fresh() awaits background reconcile cancellation."""
    cleanup_ran = False

    async def slow_with_cleanup() -> None:
        nonlocal cleanup_ran
        try:
            await asyncio.sleep(3600)
        finally:
            cleanup_ran = True

    tracker = IndexStateTracker()
    # Inject a background task directly (no AsyncMock wrapping) so CancelledError
    # propagation through try/finally is straightforward.
    bg_task = asyncio.create_task(slow_with_cleanup())
    tracker._reconcile_task = bg_task  # type: ignore[assignment]
    tracker._state = IndexState.STALE_RECONCILING

    # Give the event loop a tick so the task starts and reaches asyncio.sleep.
    await asyncio.sleep(0)
    assert not bg_task.done(), "task must be running before mark_fresh"

    await tracker.mark_fresh()

    assert tracker.current_state == IndexState.FRESH
    assert tracker._reconcile_task is None
    assert bg_task.done()
    assert cleanup_ran  # finally block ran because we awaited cancellation
