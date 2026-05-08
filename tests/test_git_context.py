"""Tests for ``scry.git_context`` — W2f: git polling and overlay slug derivation.

Covers DESIGN.md §7.2 v3.1 (polling, cache, dirty detection) and §3.5.0 (slug
derivation).  All git-interaction tests create real tiny git repos via
``subprocess`` rather than mocking so the behaviour is verified end-to-end.

Public API under test
---------------------
derive_overlay_slug  — §3.5.0: branch/detached-SHA → overlay filename
GitContext           — frozen snapshot of git state
GitContextProvider   — cached polling engine
GitContextError      — raised for non-git directories
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
import urllib.parse
from pathlib import Path

import pytest

from scry.git_context import (
    GitContext,
    GitContextError,
    GitContextProvider,
    derive_overlay_slug,
)
from scry.models import IndexConfig

# ─── repo helpers ─────────────────────────────────────────────────────


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
    """Initialise a git repo, make an initial commit, and return the HEAD SHA."""
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
    return result.stdout.strip() or None if result.returncode == 0 else None


def _make_commit(path: Path, filename: str = "extra.txt", content: str = "data") -> str:
    """Write *content* to *filename*, stage and commit it; return the new SHA."""
    (path / filename).write_text(content)
    _git(["add", filename], path)
    _git(["commit", "-m", f"add {filename}"], path)
    return _current_sha(path)


# ─── derive_overlay_slug (§3.5.0) ─────────────────────────────────────


class TestDeriveOverlaySlug:
    """Pure-function tests for §3.5.0 overlay slug derivation."""

    def test_main_branch(self) -> None:
        """Simple branch name → unchanged with .jsonl suffix."""
        assert derive_overlay_slug("main", "a" * 40) == "main.jsonl"

    def test_master_branch(self) -> None:
        assert derive_overlay_slug("master", "b" * 40) == "master.jsonl"

    def test_branch_with_slash(self) -> None:
        """Slash in branch name → URL-encoded."""
        assert derive_overlay_slug("feature/auth-login", "a" * 40) == "feature%2Fauth-login.jsonl"

    def test_dependabot_multi_slash(self) -> None:
        """Multiple slashes → each slash URL-encoded."""
        assert derive_overlay_slug("dependabot/npm/foo", "a" * 40) == "dependabot%2Fnpm%2Ffoo.jsonl"

    def test_detached_head(self) -> None:
        """Detached HEAD → detached-<sha12>.jsonl."""
        sha = "abc123def456" + "0" * 28
        assert derive_overlay_slug(None, sha) == "detached-abc123def456.jsonl"

    def test_detached_head_uses_first_12_chars(self) -> None:
        """Only the first 12 chars of the SHA appear in the detached slug."""
        sha = "fedcba987654" + "f" * 28
        slug = derive_overlay_slug(None, sha)
        assert slug == "detached-fedcba987654.jsonl"

    def test_long_branch_name_fallback(self) -> None:
        """Branch whose URL-encoded form exceeds 200 chars → long-<hash12>.jsonl."""
        # "a" is unreserved → no encoding; 201 'a's → 201-char encoded string > 200.
        long_name = "a" * 201
        encoded = urllib.parse.quote(long_name, safe="")
        assert len(encoded) > 200, "pre-condition: encoding must exceed 200 chars"
        digest = hashlib.sha256(long_name.encode()).hexdigest()
        expected = f"long-{digest[:12]}.jsonl"
        assert derive_overlay_slug(long_name, "a" * 40) == expected

    def test_slash_encoded_long_branch_fallback(self) -> None:
        """Branch with slashes whose encoded form exceeds 200 chars → long slug."""
        # 51 'x' nodes joined by '/' → 50 slashes; each '/' → '%2F' (3 chars).
        # Encoded length = 51 + 50*3 = 201 > 200.
        long_name = "/".join(["x"] * 51)
        encoded = urllib.parse.quote(long_name, safe="")
        assert len(encoded) > 200, "pre-condition"
        digest = hashlib.sha256(long_name.encode()).hexdigest()
        assert derive_overlay_slug(long_name, "b" * 40) == f"long-{digest[:12]}.jsonl"

    def test_special_chars_encoded(self) -> None:
        """Characters like : and * are URL-encoded."""
        slug = derive_overlay_slug("feat:something", "a" * 40)
        assert "%3A" in slug
        assert slug.endswith(".jsonl")

    def test_exactly_200_chars_not_truncated(self) -> None:
        """Encoded length == 200 is NOT truncated (strictly > 200 triggers fallback)."""
        # Build a name whose URL-encoding is exactly 200 chars.
        # 200 lowercase alphanumerics → 200 unreserved chars.
        name = "a" * 200
        slug = derive_overlay_slug(name, "a" * 40)
        assert slug == f"{'a' * 200}.jsonl"

    def test_empty_branch_returns_jsonl(self) -> None:
        """An empty string branch name yields just '.jsonl'."""
        slug = derive_overlay_slug("", "a" * 40)
        # urllib.parse.quote("", safe="") == "" → ".jsonl"
        assert slug == ".jsonl"


# ─── GitContextProvider: happy path ───────────────────────────────────


class TestGitContextProviderBasic:
    """Happy-path tests for GitContextProvider.get()."""

    def test_get_returns_correct_sha(self, tmp_path: Path) -> None:
        """get() returns the expected full 40-char SHA."""
        sha = _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert isinstance(ctx, GitContext)
        assert ctx.head_sha == sha

    def test_get_returns_correct_branch(self, tmp_path: Path) -> None:
        """get() returns the branch name matching git symbolic-ref."""
        _init_repo(tmp_path)
        branch = _current_branch(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert ctx.branch == branch
        assert not ctx.is_detached

    def test_head_short_is_12_chars(self, tmp_path: Path) -> None:
        """head_short is always exactly 12 characters."""
        _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert len(ctx.head_short) == 12
        assert ctx.head_sha.startswith(ctx.head_short)

    def test_overlay_slug_consistent_with_derive(self, tmp_path: Path) -> None:
        """overlay_slug matches derive_overlay_slug(branch, head_sha)."""
        _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert ctx.overlay_slug == derive_overlay_slug(ctx.branch, ctx.head_sha)

    def test_is_detached_false_on_branch(self, tmp_path: Path) -> None:
        """is_detached is False when on a normal branch."""
        _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert ctx.is_detached is False


# ─── GitContextProvider: cache behaviour ──────────────────────────────


class TestGitContextProviderCache:
    """Tests for the TTL cache and invalidation logic."""

    def test_cache_honored_within_ttl(self, tmp_path: Path) -> None:
        """Second get() returns the same object within TTL (cache hit)."""
        sha1 = _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=3600)
        ctx1 = provider.get()
        assert ctx1.head_sha == sha1

        # New commit — cache should still serve the old SHA.
        _make_commit(tmp_path)
        ctx2 = provider.get()
        assert ctx2.head_sha == sha1  # Cache hit.
        assert ctx2 is ctx1  # Same object returned from cache.

    def test_ttl_zero_never_caches(self, tmp_path: Path) -> None:
        """head_poll_interval_seconds=0 → every call re-invokes git."""
        sha1 = _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx1 = provider.get()
        assert ctx1.head_sha == sha1

        sha2 = _make_commit(tmp_path)
        ctx2 = provider.get()
        assert ctx2.head_sha == sha2  # Fresh git call — new SHA.
        assert sha1 != sha2

    def test_invalidate_forces_refresh(self, tmp_path: Path) -> None:
        """invalidate() causes the next get() to fetch fresh state."""
        sha1 = _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=3600)
        ctx1 = provider.get()
        assert ctx1.head_sha == sha1

        sha2 = _make_commit(tmp_path)
        provider.invalidate()
        ctx2 = provider.get()
        assert ctx2.head_sha == sha2  # Fresh after invalidate.
        assert sha1 != sha2

    def test_force_refresh_bypasses_cache(self, tmp_path: Path) -> None:
        """force_refresh=True bypasses the TTL even when cache is fresh."""
        sha1 = _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=3600)
        provider.get()  # Populate cache with sha1.

        sha2 = _make_commit(tmp_path)
        ctx = provider.get(force_refresh=True)
        assert ctx.head_sha == sha2
        assert sha1 != sha2

    def test_cache_repopulates_after_invalidate(self, tmp_path: Path) -> None:
        """After invalidate + get(), subsequent gets() are served from the new cache."""
        _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=3600)
        provider.invalidate()
        ctx1 = provider.get()  # Cache was empty; now populated.

        _make_commit(tmp_path)
        ctx2 = provider.get()  # Should be cached; no new git call.
        assert ctx2 is ctx1


# ─── GitContextProvider: detached HEAD ────────────────────────────────


class TestGitContextProviderDetached:
    """Tests for detached-HEAD handling."""

    def test_detached_head_branch_is_none(self, tmp_path: Path) -> None:
        """Detached HEAD → branch=None, is_detached=True."""
        sha = _init_repo(tmp_path)
        _git(["checkout", "--detach", "HEAD"], tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert ctx.branch is None
        assert ctx.is_detached is True
        assert ctx.head_sha == sha

    def test_detached_head_slug_uses_short_sha(self, tmp_path: Path) -> None:
        """Detached HEAD → overlay_slug is detached-<sha[:12]>.jsonl."""
        sha = _init_repo(tmp_path)
        _git(["checkout", "--detach", "HEAD"], tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert ctx.overlay_slug == f"detached-{sha[:12]}.jsonl"


# ─── GitContextProvider: branch with slash → URL-encoded slug ─────────


class TestGitContextProviderBranchSlug:
    """Tests that branch names are correctly reflected in overlay_slug."""

    def test_slash_in_branch_url_encoded(self, tmp_path: Path) -> None:
        """Branch 'feature/auth' → overlay_slug 'feature%2Fauth.jsonl'."""
        _init_repo(tmp_path)
        _git(["checkout", "-b", "feature/auth"], tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert ctx.branch == "feature/auth"
        assert ctx.overlay_slug == "feature%2Fauth.jsonl"

    def test_multi_slash_branch_url_encoded(self, tmp_path: Path) -> None:
        """Branch 'dependabot/npm/foo' → each slash URL-encoded."""
        _init_repo(tmp_path)
        _git(["checkout", "-b", "dependabot/npm/foo"], tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        ctx = provider.get()
        assert ctx.branch == "dependabot/npm/foo"
        assert ctx.overlay_slug == "dependabot%2Fnpm%2Ffoo.jsonl"


# ─── GitContextProvider: dirty-file detection ─────────────────────────


class TestGitContextProviderDirty:
    """Tests for dirty-worktree detection (§7.2 --porcelain=v1 -uno)."""

    def test_clean_worktree_empty_dirty_files(self, tmp_path: Path) -> None:
        """Clean worktree → dirty_files is an empty tuple."""
        _init_repo(tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0, poll_dirty=True)
        ctx = provider.get()
        assert ctx.dirty_files == ()

    def test_modified_tracked_file_in_dirty_files(self, tmp_path: Path) -> None:
        """Modifying a tracked file without staging → appears in dirty_files."""
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("modified content")
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0, poll_dirty=True)
        ctx = provider.get()
        assert "README.md" in ctx.dirty_files

    def test_dirty_files_uses_forward_slashes(self, tmp_path: Path) -> None:
        """Paths in dirty_files are normalised to forward-slash form."""
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("changed")
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0, poll_dirty=True)
        ctx = provider.get()
        for p in ctx.dirty_files:
            assert "\\" not in p

    def test_poll_dirty_false_always_empty(self, tmp_path: Path) -> None:
        """poll_dirty=False → dirty_files is always empty, even when repo is dirty."""
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("modified")
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0, poll_dirty=False)
        ctx = provider.get()
        assert ctx.dirty_files == ()

    def test_untracked_file_not_in_dirty_files(self, tmp_path: Path) -> None:
        """Untracked files are NOT reported (--uno suppresses ?? lines)."""
        _init_repo(tmp_path)
        (tmp_path / "new_untracked.py").write_text("print('hello')")
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0, poll_dirty=True)
        ctx = provider.get()
        assert "new_untracked.py" not in ctx.dirty_files

    def test_staged_file_in_dirty_files(self, tmp_path: Path) -> None:
        """Staged (but not committed) changes appear in dirty_files."""
        _init_repo(tmp_path)
        (tmp_path / "README.md").write_text("staged change")
        _git(["add", "README.md"], tmp_path)
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0, poll_dirty=True)
        ctx = provider.get()
        assert "README.md" in ctx.dirty_files

    def test_dirty_files_subdirectory_path(self, tmp_path: Path) -> None:
        """A modified file in a subdirectory is reported with full repo-relative path."""
        _init_repo(tmp_path)
        sub = tmp_path / "src" / "lib"
        sub.mkdir(parents=True)
        f = sub / "mod.py"
        f.write_text("print('orig')\n")
        _git(["add", "src/lib/mod.py"], tmp_path)
        _git(["commit", "-m", "add subdir file"], tmp_path)
        f.write_text("print('modified')\n")
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0, poll_dirty=True)
        ctx = provider.get()
        assert "src/lib/mod.py" in ctx.dirty_files

    def test_dirty_files_non_ascii_path_round_trips(self, tmp_path: Path) -> None:
        """Regression (review-w2f HIGH): non-ASCII filenames must NOT be C-escape-corrupted.

        Old parser used line-based ``git status`` output which C-style-quotes
        unusual paths (``"na\\303\\257ve.txt"`` for ``naïve.txt``).  Combined
        with the legacy ``\\\\`` → ``/`` replace, this turned ``\\303\\257``
        into ``/303/257`` — silently mangling real filenames.  Fixed by
        switching to ``-z`` (NUL-terminated, raw bytes).
        """
        _init_repo(tmp_path)
        f = tmp_path / "naïve.txt"
        try:
            f.write_text("hello\n", encoding="utf-8")
        except (OSError, UnicodeEncodeError):
            pytest.skip("filesystem does not support non-ASCII filenames")
        _git(["add", "--", "naïve.txt"], tmp_path)
        _git(["commit", "-m", "add naïve.txt"], tmp_path)
        f.write_text("modified\n", encoding="utf-8")
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0, poll_dirty=True)
        ctx = provider.get()
        # The path must round-trip exactly (no C-escapes, no quotes).
        assert "naïve.txt" in ctx.dirty_files, f"got: {ctx.dirty_files}"


# ─── GitContextProvider: error conditions ─────────────────────────────


class TestGitContextProviderErrors:
    """Tests for error handling and edge cases."""

    def test_non_git_directory_raises(self, tmp_path: Path) -> None:
        """A directory that is not a git repo raises GitContextError."""
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
        with pytest.raises(GitContextError, match="Not a git repository"):
            provider.get()

    def test_error_not_cached(self, tmp_path: Path) -> None:
        """A GitContextError does not poison the cache; retry succeeds after init."""
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=3600)
        with pytest.raises(GitContextError):
            provider.get()
        # Now initialise the repo — the provider must not serve a cached error.
        _init_repo(tmp_path)
        ctx = provider.get()
        assert isinstance(ctx, GitContext)

    def test_non_git_directory_with_cache_enabled(self, tmp_path: Path) -> None:
        """TTL-enabled provider still raises GitContextError for non-git paths."""
        provider = GitContextProvider(tmp_path, head_poll_interval_seconds=30)
        with pytest.raises(GitContextError):
            provider.get()


# ─── from_config factory ──────────────────────────────────────────────


class TestFromConfig:
    """Tests for the from_config convenience factory."""

    def test_from_config_wires_ttl(self, tmp_path: Path) -> None:
        """from_config reads head_poll_interval_seconds from IndexConfig."""
        _init_repo(tmp_path)
        config = IndexConfig(head_poll_interval_seconds=99, poll_dirty=True)
        provider = GitContextProvider.from_config(tmp_path, config)
        assert provider._ttl == 99
        assert provider._poll_dirty is True

    def test_from_config_poll_dirty_false(self, tmp_path: Path) -> None:
        """from_config propagates poll_dirty=False."""
        _init_repo(tmp_path)
        config = IndexConfig(poll_dirty=False)
        provider = GitContextProvider.from_config(tmp_path, config)
        assert provider._poll_dirty is False

    def test_from_config_ttl_zero(self, tmp_path: Path) -> None:
        """from_config with interval=0 → caching disabled."""
        _init_repo(tmp_path)
        config = IndexConfig(head_poll_interval_seconds=0)
        provider = GitContextProvider.from_config(tmp_path, config)
        assert provider._ttl == 0

    def test_from_config_get_returns_valid_context(self, tmp_path: Path) -> None:
        """Provider created via from_config returns a valid GitContext."""
        _init_repo(tmp_path)
        config = IndexConfig(head_poll_interval_seconds=0, poll_dirty=True)
        provider = GitContextProvider.from_config(tmp_path, config)
        ctx = provider.get()
        assert isinstance(ctx, GitContext)
        assert len(ctx.head_sha) == 40
        assert len(ctx.head_short) == 12
        assert ctx.overlay_slug.endswith(".jsonl")

    def test_from_config_default_values(self, tmp_path: Path) -> None:
        """Default IndexConfig gives TTL=30, poll_dirty=True."""
        _init_repo(tmp_path)
        config = IndexConfig()
        provider = GitContextProvider.from_config(tmp_path, config)
        assert provider._ttl == 30
        assert provider._poll_dirty is True


# ─── Windows/Unix platform markers ────────────────────────────────────


@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific path check")
def test_windows_path_str_conversion(tmp_path: Path) -> None:
    """repo_root is passed as str() to git -C so Windows Path objects work."""
    _init_repo(tmp_path)
    # If Path were passed raw without str(), git may fail on some Windows builds.
    provider = GitContextProvider(tmp_path, head_poll_interval_seconds=0)
    ctx = provider.get()
    assert ctx.head_sha  # Non-empty means git invocation succeeded.
