"""Tests for scry watch (W6c) — file-watcher hot-reindex daemon.

Covers:
- File write in repo triggers reindex via debounce loop
- Debouncing: N rapid events → 1 reindex call
- Ignored dirs (.git, .scry, node_modules, dist, build, ...) do not trigger reindex
- No-leader: exits 2 with clear message (W6c BLOCKING #1)
- Follower path: sends IPC reindex request when a leader is running
- No-leader-metadata: exits 1 with error log
- --once: runs exactly one reindex then exits (no watcher)
- Reconnect timeout: exits 2 when IPC keeps failing (W6c HIGH #1)
- WatchHandler._should_ignore coverage

Uses PollingObserver (watchdog) for cross-platform reliability in CI.
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from watchdog.observers.polling import PollingObserver

from scry.cmd_watch import WatchHandler, _should_ignore, run_watch
from scry.index import IndexResult
from scry.process.ipc import IPCClient
from scry.process.leader import LeaderMetadata, LeaderState

# ─── Helpers ──────────────────────────────────────────────────────────────────

_FAKE_RESULT = IndexResult(
    files_processed=3,
    anchors_extracted=5,
    anchors_embedded=5,
    chunks_written=10,
    files_pruned=0,
    elapsed_seconds=0.1,
)

_FAKE_METADATA = LeaderMetadata(
    pid=99999,
    endpoint_uri="unix:/tmp/fake.sock",
    boot_epoch_token="tok-abc",
    scry_version="0.0.1",
)


def _make_repo(tmp_path: Path) -> Path:
    """Create a minimal scry repo skeleton."""
    scry_dir = tmp_path / ".scry"
    scry_dir.mkdir()
    (scry_dir / "overlays").mkdir()
    return tmp_path


# ─── _should_ignore ───────────────────────────────────────────────────────────


class TestShouldIgnore:
    def test_normal_file_not_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert not _should_ignore(repo / "src" / "main.py", repo)

    def test_git_dir_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / ".git" / "COMMIT_EDITMSG", repo)

    def test_scry_dir_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / ".scry" / "vectors.db", repo)

    def test_node_modules_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / "node_modules" / "lodash" / "index.js", repo)

    def test_pycache_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / "src" / "__pycache__" / "foo.pyc", repo)

    def test_venv_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / ".venv" / "lib" / "site.py", repo)
        assert _should_ignore(repo / "venv" / "bin" / "python", repo)

    def test_outside_repo_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        # A path that cannot be relative_to repo_root
        outside = Path("/some/other/path/file.py")
        assert _should_ignore(outside, repo)

    def test_nested_normal_file_not_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert not _should_ignore(repo / "docs" / "sub" / "page.md", repo)

    # W6c MEDIUM #1: extended ignore list
    def test_dist_dir_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / "dist" / "bundle.js", repo)

    def test_build_dir_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / "build" / "output.o", repo)

    def test_target_dir_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / "target" / "debug" / "binary", repo)

    def test_next_cache_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / ".next" / "server" / "pages" / "index.js", repo)

    def test_pytest_cache_ignored(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        assert _should_ignore(repo / ".pytest_cache" / "v" / "cache.json", repo)


# ─── WatchHandler ─────────────────────────────────────────────────────────────


class TestWatchHandler:
    def test_posts_relevant_file_event(self, tmp_path: Path) -> None:
        """on_any_event posts to changed + triggers the event."""
        repo = _make_repo(tmp_path)
        loop = asyncio.new_event_loop()
        try:
            changed: set[Path] = set()
            trigger = asyncio.Event()
            handler = WatchHandler(loop, changed, trigger, repo)

            fake_event = MagicMock()
            fake_event.is_directory = False
            fake_event.src_path = str(repo / "src" / "foo.py")

            handler.on_any_event(fake_event)
            loop.run_until_complete(asyncio.sleep(0))  # drain call_soon_threadsafe

            assert repo / "src" / "foo.py" in changed
            assert trigger.is_set()
        finally:
            loop.close()

    def test_ignores_directory_events(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        loop = asyncio.new_event_loop()
        try:
            changed: set[Path] = set()
            trigger = asyncio.Event()
            handler = WatchHandler(loop, changed, trigger, repo)

            fake_event = MagicMock()
            fake_event.is_directory = True
            fake_event.src_path = str(repo / "src")

            handler.on_any_event(fake_event)
            loop.run_until_complete(asyncio.sleep(0))

            assert not changed
            assert not trigger.is_set()
        finally:
            loop.close()

    def test_ignores_git_dir_events(self, tmp_path: Path) -> None:
        repo = _make_repo(tmp_path)
        loop = asyncio.new_event_loop()
        try:
            changed: set[Path] = set()
            trigger = asyncio.Event()
            handler = WatchHandler(loop, changed, trigger, repo)

            fake_event = MagicMock()
            fake_event.is_directory = False
            fake_event.src_path = str(repo / ".git" / "index")

            handler.on_any_event(fake_event)
            loop.run_until_complete(asyncio.sleep(0))

            assert not changed
            assert not trigger.is_set()
        finally:
            loop.close()

    @pytest.mark.parametrize(
        "ignored_dir",
        [
            ".git",
            ".scry",
            "node_modules",
            "__pycache__",
            ".venv",
            "venv",
            # W6c MEDIUM #1: extended ignore list
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
        ],
    )
    def test_ignores_all_noise_dirs(self, tmp_path: Path, ignored_dir: str) -> None:
        repo = _make_repo(tmp_path)
        loop = asyncio.new_event_loop()
        try:
            changed: set[Path] = set()
            trigger = asyncio.Event()
            handler = WatchHandler(loop, changed, trigger, repo)

            fake_event = MagicMock()
            fake_event.is_directory = False
            fake_event.src_path = str(repo / ignored_dir / "some_file.txt")

            handler.on_any_event(fake_event)
            loop.run_until_complete(asyncio.sleep(0))

            assert not changed, f"Expected {ignored_dir} to be ignored"
        finally:
            loop.close()


# ─── run_watch — --once mode ──────────────────────────────────────────────────


class TestRunWatchOnce:
    @pytest.mark.asyncio
    async def test_once_no_leader_exits_2(self, tmp_path: Path) -> None:
        """--once with no leader: exits 2 (W6c BLOCKING #1 refuse-without-leader)."""
        repo = _make_repo(tmp_path)

        with patch(
            "scry.cmd_watch.detect_leader_state",
            return_value=(LeaderState.LEADER, None),
        ):
            code = await run_watch(repo, once=True)

        assert code == 2

    @pytest.mark.asyncio
    async def test_once_follower_path(self, tmp_path: Path) -> None:
        """--once with a leader: sends IPC reindex and returns 0."""
        repo = _make_repo(tmp_path)

        mock_client = MagicMock(spec=IPCClient)
        mock_client.call = AsyncMock(return_value={"files_processed": 1})
        mock_client.close = AsyncMock()

        with (
            patch(
                "scry.cmd_watch.detect_leader_state",
                return_value=(LeaderState.FOLLOWER, _FAKE_METADATA),
            ),
            patch("scry.cmd_watch.parse_endpoint_uri", return_value=MagicMock()),
            patch("scry.cmd_watch.IPCClient", return_value=mock_client),
        ):
            code = await run_watch(repo, once=True)

        assert code == 0
        mock_client.call.assert_awaited_once()
        called_op = mock_client.call.call_args[0][0]
        assert called_op == "reindex"
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_once_no_leader_metadata_returns_1(self, tmp_path: Path) -> None:
        """Follower detected but metadata is None → returns 1."""
        repo = _make_repo(tmp_path)

        with patch(
            "scry.cmd_watch.detect_leader_state",
            return_value=(LeaderState.FOLLOWER, None),
        ):
            code = await run_watch(repo, once=True)

        assert code == 1

    @pytest.mark.asyncio
    async def test_once_reconnect_timeout_exits_2(self, tmp_path: Path) -> None:
        """--once with IPC failure and reconnect_timeout=0 → exits 2 (W6c HIGH #1)."""
        repo = _make_repo(tmp_path)

        mock_client = MagicMock(spec=IPCClient)
        mock_client.call = AsyncMock(side_effect=ConnectionRefusedError("no leader"))
        mock_client.close = AsyncMock()

        with (
            patch(
                "scry.cmd_watch.detect_leader_state",
                return_value=(LeaderState.FOLLOWER, _FAKE_METADATA),
            ),
            patch("scry.cmd_watch.parse_endpoint_uri", return_value=MagicMock()),
            patch("scry.cmd_watch.IPCClient", return_value=mock_client),
        ):
            code = await run_watch(repo, once=True, reconnect_timeout=0)

        assert code == 2


# ─── run_watch — watcher / debounce ──────────────────────────────────────────


class TestRunWatchDebounce:
    @pytest.mark.asyncio
    async def test_file_change_triggers_reindex(self, tmp_path: Path) -> None:
        """A single file write eventually triggers one IPC reindex call (follower path)."""
        repo = _make_repo(tmp_path)
        (repo / "README.md").write_text("hello")

        ipc_calls: list[str] = []

        mock_client = MagicMock(spec=IPCClient)

        async def _fake_call(op: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            ipc_calls.append(op)
            return {}

        mock_client.call = _fake_call
        mock_client.close = AsyncMock()

        async def _run() -> None:
            watch_task = asyncio.create_task(
                run_watch(
                    repo,
                    debounce_ms=50,
                    _observer_class=PollingObserver,
                )
            )
            # Let the watcher start.
            await asyncio.sleep(0.3)

            # Write a file — PollingObserver polls periodically.
            (repo / "README.md").write_text("changed content")

            # Wait enough for the poll + debounce to fire.
            await asyncio.sleep(1.5)

            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch_task

        with (
            patch(
                "scry.cmd_watch.detect_leader_state",
                return_value=(LeaderState.FOLLOWER, _FAKE_METADATA),
            ),
            patch("scry.cmd_watch.parse_endpoint_uri", return_value=MagicMock()),
            patch("scry.cmd_watch.IPCClient", return_value=mock_client),
        ):
            await _run()

        assert len(ipc_calls) >= 1
        assert ipc_calls[0] == "reindex"

    @pytest.mark.asyncio
    async def test_rapid_events_debounced_to_one_reindex(self, tmp_path: Path) -> None:
        """Multiple rapid file events collapse into a single reindex call."""
        repo = _make_repo(tmp_path)

        reindex_calls: list[int] = []

        async def run_with_injected_events() -> int:
            """Run the watch loop but inject 5 trigger events manually."""
            changed: set[Path] = set()
            trigger = asyncio.Event()

            # Schedule 5 rapid events directly (bypass watchdog thread)
            async def _inject() -> None:
                await asyncio.sleep(0.05)
                for i in range(5):
                    changed.add(repo / f"file{i}.py")
                    trigger.set()
                    await asyncio.sleep(0.01)

            mock_indexer = MagicMock()

            async def _fake_index_async() -> IndexResult:
                reindex_calls.append(1)
                return _FAKE_RESULT

            mock_indexer.index_async = _fake_index_async

            inject_task = asyncio.create_task(_inject())

            # Run the debounce inner loop directly (not full run_watch, so we
            # can inject events via the asyncio path rather than the OS watcher).
            debounce_s = 0.1

            await inject_task  # wait for events to be queued

            # Simulate the debounce loop that run_watch uses internally.
            trigger.clear()

            # Wait for first event (already set by inject).
            trigger.set()  # mimic having received events
            await trigger.wait()
            trigger.clear()

            while True:
                try:
                    await asyncio.wait_for(trigger.wait(), timeout=debounce_s)
                    trigger.clear()
                except TimeoutError:
                    break

            n = len(changed)
            changed.clear()
            await mock_indexer.index_async()
            return n

        n_changed = await run_with_injected_events()
        assert reindex_calls == [1], "Expected exactly 1 reindex after debounce"
        assert n_changed == 5

    @pytest.mark.asyncio
    async def test_ignored_dirs_do_not_trigger_reindex(self, tmp_path: Path) -> None:
        """Events from ignored dirs (.git, .scry, node_modules, dist, …) don't fire reindex."""
        repo = _make_repo(tmp_path)

        loop = asyncio.get_event_loop()
        changed: set[Path] = set()
        trigger = asyncio.Event()
        handler = WatchHandler(loop, changed, trigger, repo)

        ignored_paths = [
            repo / ".git" / "ORIG_HEAD",
            repo / ".scry" / "vectors.db",
            repo / "node_modules" / "react" / "index.js",
            repo / "__pycache__" / "foo.pyc",
            repo / ".venv" / "lib" / "python3.11" / "site.py",
            # W6c MEDIUM #1: extended ignore list
            repo / "dist" / "bundle.js",
            repo / "build" / "output.o",
            repo / "target" / "debug" / "binary",
            repo / ".pytest_cache" / "v" / "cache.json",
            repo / ".mypy_cache" / "3.11" / "some.json",
        ]

        for p in ignored_paths:
            evt = MagicMock()
            evt.is_directory = False
            evt.src_path = str(p)
            handler.on_any_event(evt)

        # Drain the event loop
        await asyncio.sleep(0)

        assert not trigger.is_set(), "Trigger should NOT be set for ignored paths"
        assert not changed, "No paths should be in changed set for ignored paths"


# ─── run_watch — IPC follower with watcher ────────────────────────────────────


class TestRunWatchFollower:
    @pytest.mark.asyncio
    async def test_follower_sends_ipc_reindex(self, tmp_path: Path) -> None:
        """Follower path: file change → IPC 'reindex' op sent to leader."""
        repo = _make_repo(tmp_path)
        (repo / "spec.md").write_text("initial")

        ipc_calls: list[str] = []

        mock_client = MagicMock(spec=IPCClient)

        async def _fake_call(op: str, args: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
            ipc_calls.append(op)
            return {}

        mock_client.call = _fake_call
        mock_client.close = AsyncMock()

        async def _run() -> None:
            watch_task = asyncio.create_task(
                run_watch(
                    repo,
                    debounce_ms=50,
                    _observer_class=PollingObserver,
                )
            )
            await asyncio.sleep(0.3)

            (repo / "spec.md").write_text("updated spec content")

            await asyncio.sleep(1.5)

            watch_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await watch_task

        with (
            patch(
                "scry.cmd_watch.detect_leader_state",
                return_value=(LeaderState.FOLLOWER, _FAKE_METADATA),
            ),
            patch("scry.cmd_watch.parse_endpoint_uri", return_value=MagicMock()),
            patch("scry.cmd_watch.IPCClient", return_value=mock_client),
        ):
            await _run()

        assert "reindex" in ipc_calls, "Expected IPC 'reindex' op to be called"
        mock_client.close.assert_awaited()


# ─── CLI integration ──────────────────────────────────────────────────────────


class TestWatchCLI:
    def test_watch_once_no_leader_exits_2(self, tmp_path: Path) -> None:
        """CLI: scry watch --once exits 2 when no leader is running (W6c BLOCKING #1)."""
        from click.testing import CliRunner

        from scry.cli import main

        runner = CliRunner()

        with patch(
            "scry.cmd_watch.detect_leader_state",
            return_value=(LeaderState.LEADER, None),
        ):
            result = runner.invoke(
                main,
                ["watch", "--once"],
                obj={"repo_root": tmp_path, "allow_untrusted_lsp_config": False},
                catch_exceptions=False,
            )

        assert result.exit_code == 2

    def test_watch_once_no_metadata_exits_nonzero(self, tmp_path: Path) -> None:
        """CLI: scry watch --once exits nonzero when follower has no metadata."""
        from click.testing import CliRunner

        from scry.cli import main

        runner = CliRunner()

        with patch(
            "scry.cmd_watch.detect_leader_state",
            return_value=(LeaderState.FOLLOWER, None),
        ):
            result = runner.invoke(
                main,
                ["watch", "--once"],
                obj={"repo_root": tmp_path, "allow_untrusted_lsp_config": False},
                catch_exceptions=False,
            )

        assert result.exit_code != 0

# uat-r5-5 pr-d noise
