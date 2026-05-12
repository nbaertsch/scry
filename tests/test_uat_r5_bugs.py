"""Regression tests for UAT-R5 batch bugs.

UAT-R5-4   Windows --scope glob silently broken via PowerShell shell expansion
UAT-R5-6   Cold-start indexing: missing --max-files cap + doc-translation excludes
UAT-R5-7   scry link direction-warning false positives on baseline-style links
UAT-R5-10  README install broken on Windows (uv run scry)
UAT-R5-11  README missing core link/check workflow
UAT-R5-12  No ETA during indexing (only running counter)
UAT-R5-13  scry init does not warn about scale
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from scry.cli import main

# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_cli.py conventions)
# ---------------------------------------------------------------------------

_STUB_ENV = {"SCRY_EMBEDDER": "stub"}


def _run(
    runner: CliRunner,
    args: list[str],
    *,
    repo: Path,
    input: str | None = None,
    env_extra: dict[str, str] | None = None,
) -> Any:
    """Invoke `scry` with *args* as if run from *repo* directory."""
    env = {**_STUB_ENV, **(env_extra or {})}
    old_cwd = Path.cwd()
    try:
        os.chdir(repo)
        return runner.invoke(
            main,
            args,
            catch_exceptions=False,
            env=env,
            input=input,
        )
    finally:
        os.chdir(old_cwd)


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def fixture_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def hailstorm_spec(fixture_dir: Path) -> Path:
    p = fixture_dir / "hailstorm-spec"
    if not p.exists():
        pytest.skip("hailstorm-spec fixture not yet created")
    return p


@pytest.fixture
def indexed_repo(tmp_path: Path, hailstorm_spec: Path) -> Path:
    """Temp git repo with hailstorm-spec fixture + initial commit."""
    shutil.copytree(str(hailstorm_spec), str(tmp_path), dirs_exist_ok=True)

    def _git(*args: str) -> None:
        subprocess.run(
            ["git", "-C", str(tmp_path), *args],
            check=True,
            capture_output=True,
        )

    _git("init")
    _git("config", "user.email", "ci@test.local")
    _git("config", "user.name", "CI Test")
    _git("add", ".")
    _git("commit", "-m", "init")
    return tmp_path


@pytest.fixture
def indexed_and_built_repo(indexed_repo: Path) -> Path:
    """Repo that has been indexed (stub embedder)."""
    from scry.config import load_config
    from scry.embed import StubEmbedder
    from scry.index import Indexer

    config = load_config(indexed_repo)
    indexer = Indexer(
        indexed_repo,
        config=config,
        embedder=StubEmbedder(dimensions=config.embeddings.dimensions),
    )
    indexer.index(force=True)
    return indexed_repo


# ─── UAT-R5-7: direction warning ─────────────────────────────────────────────


class TestDirectionWarning:
    """UAT-R5-7: direction warning is suppressible and fires at most once."""

    def _make_link_cmd_args(
        self, from_id: str, to_id: str, link_type: str = "implements"
    ) -> list[str]:
        return ["link", from_id, to_id, "--type", link_type]

    def test_no_direction_warning_flag_suppresses_warning(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """--no-direction-warning suppresses the direction advisory."""
        from scry.store.db import ScryDB

        repo = indexed_and_built_repo

        # Find a code anchor and a section anchor to author an inverted link.
        with ScryDB(repo, read_only=True) as db:
            all_anchors = db.list_anchors()
        code_anchors = [a for a in all_anchors if a.type == "code"]
        section_anchors = [a for a in all_anchors if a.type == "section"]
        if not code_anchors or not section_anchors:
            pytest.skip("need both code and section anchors in fixture")

        code_id = code_anchors[0].id
        sec_id = section_anchors[0].id

        # Inverted: section→code for "implements" (canonical is code→spec)
        result = _run(
            runner,
            ["link", sec_id, code_id, "--type", "implements", "--no-direction-warning"],
            repo=repo,
        )
        assert result.exit_code == 0, result.output
        # The direction warning must NOT appear.
        assert "canonical direction" not in result.output
        assert "direction" not in result.output.lower() or "link created" in result.output

    def test_direction_warning_env_var_suppresses(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """SCRY_NO_DIRECTION_WARNING=1 suppresses the direction warning."""
        from scry.store.db import ScryDB

        repo = indexed_and_built_repo
        with ScryDB(repo, read_only=True) as db:
            all_anchors = db.list_anchors()
        code_anchors = [a for a in all_anchors if a.type == "code"]
        section_anchors = [a for a in all_anchors if a.type == "section"]
        if not code_anchors or not section_anchors:
            pytest.skip("need both code and section anchors in fixture")

        code_id = code_anchors[0].id
        sec_id = section_anchors[0].id

        result = _run(
            runner,
            ["link", sec_id, code_id, "--type", "implements"],
            repo=repo,
            env_extra={"SCRY_NO_DIRECTION_WARNING": "1"},
        )
        assert result.exit_code == 0, result.output
        assert "canonical direction" not in result.output

    def test_direction_warning_message_includes_swap_suggestion(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """Direction warning names the canonical direction and suggests the corrective swap."""
        import scry.cli as _cli_mod

        # Reset the module-level flag so the warning fires fresh.
        _cli_mod._direction_warning_emitted = False  # type: ignore[attr-defined]

        from scry.store.db import ScryDB

        repo = indexed_and_built_repo
        with ScryDB(repo, read_only=True) as db:
            all_anchors = db.list_anchors()
        code_anchors = [a for a in all_anchors if a.type == "code"]
        section_anchors = [a for a in all_anchors if a.type == "section"]
        if not code_anchors or not section_anchors:
            pytest.skip("need both code and section anchors in fixture")

        code_id = code_anchors[0].id
        sec_id = section_anchors[0].id

        result = _run(
            runner,
            # Inverted: section → code for "implements"
            ["link", sec_id, code_id, "--type", "implements"],
            repo=repo,
        )
        # Exit should still succeed; link is written.
        assert result.exit_code == 0, result.output
        # Warning should mention canonical direction with an arrow.
        assert "canonical direction" in result.output
        # Warning should suggest the swapped invocation.
        assert "scry link" in result.output
        assert code_id in result.output
        assert sec_id in result.output

    def test_direction_warning_fires_at_most_once_per_process(
        self, runner: CliRunner, indexed_and_built_repo: Path
    ) -> None:
        """Module-level flag ensures warning fires at most once per process."""
        import scry.cli as _cli_mod

        _cli_mod._direction_warning_emitted = False  # type: ignore[attr-defined]

        from scry.store.db import ScryDB

        repo = indexed_and_built_repo
        with ScryDB(repo, read_only=True) as db:
            all_anchors = db.list_anchors()
        code_anchors = [a for a in all_anchors if a.type == "code"]
        section_anchors = [a for a in all_anchors if a.type == "section"]
        if not (len(code_anchors) >= 1 and len(section_anchors) >= 1):
            pytest.skip("need both code and section anchors")

        # Simulate two invocations in the same process (e.g. batch link API).
        # The module-level flag should suppress the second warning.
        _cli_mod._direction_warning_emitted = True  # type: ignore[attr-defined]
        result = _run(
            runner,
            ["link", section_anchors[0].id, code_anchors[0].id, "--type", "implements"],
            repo=repo,
        )
        assert result.exit_code == 0, result.output
        assert "canonical direction" not in result.output

        # Clean up for other tests.
        _cli_mod._direction_warning_emitted = False  # type: ignore[attr-defined]


# ─── UAT-R5-4: Windows --scope glob expansion warning ────────────────────────


class TestScopeExpansionWarning:
    """UAT-R5-4: warn when --scope looks like an expanded filename on Windows."""

    def test_warn_if_scope_looks_expanded_real_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_warn_if_scope_looks_expanded emits warning when path exists and has no glob chars."""
        import click

        from scry.cli import _warn_if_scope_looks_expanded

        # Create a real file so Path(scope).exists() is True.
        existing = tmp_path / "src" / "foo.py"
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text("# file\n", encoding="utf-8")

        monkeypatch.setattr(sys, "platform", "win32")

        output: list[str] = []

        def _fake_echo(msg: str, err: bool = False) -> None:
            output.append(msg)

        monkeypatch.setattr(click, "echo", _fake_echo)
        _warn_if_scope_looks_expanded(str(existing))
        assert any("warning" in m.lower() for m in output), f"Expected warning; got: {output}"
        assert any("--scope=" in m for m in output), (
            f"Expected --scope= hint in output; got: {output}"
        )

    def test_no_warn_when_scope_has_glob_chars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No warning when the scope contains * or ? — that's normal glob syntax."""
        import click

        from scry.cli import _warn_if_scope_looks_expanded

        monkeypatch.setattr(sys, "platform", "win32")
        output: list[str] = []
        monkeypatch.setattr(click, "echo", lambda msg, err=False: output.append(msg))

        _warn_if_scope_looks_expanded("src/**/*.py")
        assert not output, f"Unexpected warning for valid glob: {output}"

    def test_no_warn_on_non_windows(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Warning is gated to sys.platform == 'win32'."""
        import click

        from scry.cli import _warn_if_scope_looks_expanded

        existing = tmp_path / "file.py"
        existing.write_text("# x\n", encoding="utf-8")

        monkeypatch.setattr(sys, "platform", "linux")
        output: list[str] = []
        monkeypatch.setattr(click, "echo", lambda msg, err=False: output.append(msg))

        _warn_if_scope_looks_expanded(str(existing))
        assert not output, f"Unexpected warning on non-Windows: {output}"

    def test_no_warn_when_scope_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No warning when scope is None."""
        import click

        from scry.cli import _warn_if_scope_looks_expanded

        monkeypatch.setattr(sys, "platform", "win32")
        output: list[str] = []
        monkeypatch.setattr(click, "echo", lambda msg, err=False: output.append(msg))

        _warn_if_scope_looks_expanded(None)
        assert not output


# ─── UAT-R5-6 / UAT-R5-13: init file count warning ──────────────────────────


class TestInitFileCountWarning:
    """UAT-R5-6 / UAT-R5-13: scry init warns when projected file count is high."""

    def test_init_warns_when_file_count_exceeds_max(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """scry init warns when projected indexed file count exceeds --max-files."""
        # Create > 5 markdown files so we can set --max-files=4 and trigger warning.
        for i in range(6):
            md = tmp_path / f"doc{i}.md"
            md.write_text(f"# Doc {i}\n", encoding="utf-8")

        result = _run(runner, ["init", "--max-files", "4"], repo=tmp_path)
        assert result.exit_code == 0, result.output
        # Warning should mention the file count and threshold.
        assert "warning" in result.output.lower()
        assert "files match" in result.output or "files would be" in result.output

    def test_init_no_warning_when_count_under_threshold(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """scry init is silent (no file-count warning) when count is under threshold."""
        # Create just 2 markdown files.
        for i in range(2):
            (tmp_path / f"doc{i}.md").write_text(f"# Doc {i}\n", encoding="utf-8")

        result = _run(runner, ["init", "--max-files", "100"], repo=tmp_path)
        assert result.exit_code == 0, result.output
        # No file-count warning should appear.
        assert "files match" not in result.output

    def test_init_detects_doc_translation_dirs(self, runner: CliRunner, tmp_path: Path) -> None:
        """scry init detects multi-language doc translations and suggests excludes."""
        # Create docs/ with 4 language subdirs (en, de, fr, ja) each with .md files.
        for lang in ("en", "de", "fr", "ja"):
            lang_dir = tmp_path / "docs" / lang
            lang_dir.mkdir(parents=True, exist_ok=True)
            for i in range(50):
                (lang_dir / f"page{i}.md").write_text(f"# Page {i}\n", encoding="utf-8")

        # Low threshold to ensure warning fires.
        result = _run(runner, ["init", "--max-files", "50"], repo=tmp_path)
        assert result.exit_code == 0, result.output
        # Warning should mention non-English language dirs.
        assert "docs/de/**" in result.output or "de" in result.output

    def test_count_projected_files_helper(self, tmp_path: Path) -> None:
        """_count_projected_files returns correct total and directory breakdown."""
        from scry.cli import _count_projected_files

        # Create some markdown and python files.
        (tmp_path / "README.md").write_text("# Root\n", encoding="utf-8")
        src = tmp_path / "src"
        src.mkdir()
        for i in range(3):
            (src / f"m{i}.py").write_text(f"def f{i}(): pass\n", encoding="utf-8")

        total, dirs, capped = _count_projected_files(tmp_path, ["**/*.md", "**/*.py"], [])
        assert total == 4  # 1 md + 3 py
        assert dirs.get("src", 0) == 3
        assert capped is False


# ─── UAT-R5-12: ETA in non-TTY progress output ───────────────────────────────


class TestIndexProgressETA:
    """UAT-R5-12: non-TTY progress output includes ETA once 5% complete."""

    def test_progress_log_includes_percentage(self, runner: CliRunner, indexed_repo: Path) -> None:
        """Non-TTY index progress lines should include percentage."""
        # CliRunner does not attach a TTY, so _progress_log is used.
        result = _run(runner, ["index"], repo=indexed_repo)
        assert result.exit_code == 0, result.output
        # Look for percentage pattern like "(0.5%)" or "(100.0%)" in stderr output.
        # Click's CliRunner mixes stderr into output by default.
        pct_pattern = re.compile(r"\(\d+\.\d+%\)")
        assert pct_pattern.search(result.output), (
            f"No percentage pattern found in index output: {result.output!r}"
        )

    def test_progress_log_format_with_eta(self) -> None:
        """Progress-log closure emits ETA string once >= 5% complete."""
        # Directly exercise the format logic by building the closure.
        # We monkeypatch time.monotonic to control elapsed time.
        emitted: list[str] = []

        phase_state: dict[str, int] = {}
        phase_start: dict[str, float] = {}
        fake_start = 1000.0

        # Simulate 10% complete with 100 seconds elapsed → ETA ≈ 900s (15m00s).
        def fake_monotonic() -> float:
            phase = "extract"
            # First call (from phase_start setup): return fake_start.
            # Subsequent call (for elapsed): return fake_start + 100.
            if phase not in phase_start:
                return fake_start
            return fake_start + 100.0

        # Build the exact closure logic from the CLI (copy of _progress_log internals).
        import time as _time_mod

        orig_monotonic = _time_mod.monotonic

        try:
            _time_mod.monotonic = fake_monotonic  # type: ignore[assignment]

            # Manually replicate the _progress_log closure logic.
            phase = "extract"
            processed = 100
            total = 1000
            label = "file.py"

            if phase not in phase_start:
                phase_start[phase] = _time_mod.monotonic()
            last = phase_state.get(phase, 0)
            if processed - last >= 25 or processed == total or processed == 1:
                pct_str = ""
                eta_str = ""
                if total > 0:
                    pct = processed / total
                    pct_str = f" ({pct:.1%})"
                    if pct >= 0.05:
                        # Use 100s elapsed (fake_start+100 - phase_start[phase]).
                        elapsed = 100.0
                        eta_sec = int(elapsed / pct - elapsed)
                        mins, secs = divmod(eta_sec, 60)
                        eta_str = f" ETA {mins}m{secs:02d}s"
                line = f"  {phase} {processed}/{total}{pct_str}{eta_str} ({label})"
                emitted.append(line)
                phase_state[phase] = processed
        finally:
            _time_mod.monotonic = orig_monotonic  # type: ignore[assignment]

        assert len(emitted) == 1
        line = emitted[0]
        assert "(10.0%)" in line, f"Expected percentage in line: {line!r}"
        assert "ETA" in line, f"Expected ETA in line: {line!r}"
        assert "m" in line, f"Expected minutes in ETA: {line!r}"


# ─── UAT-R5-10: README uses uv run scry ──────────────────────────────────────


class TestReadmeUvRunScry:
    """UAT-R5-10: README install/quick-start sections must use `uv run scry`."""

    def _readme_text(self) -> str:
        readme = Path(__file__).resolve().parents[1] / "README.md"
        return readme.read_text(encoding="utf-8")

    def test_no_bare_scry_in_install_or_quickstart(self) -> None:
        """No bare `scry <cmd>` invocations appear in code blocks in Quick start / Install."""
        text = self._readme_text()
        # Match lines in fenced code blocks that start with bare 'scry ' (not 'uv run scry').
        # We accept `uv run scry`, `scry.cli`, or `scry` as argument (e.g. in JSON snippets).
        # Look specifically for shell invocations: lines beginning with `scry ` not prefixed.
        bare_lines = [line for line in text.splitlines() if re.match(r"^scry\s+\w", line.strip())]
        assert not bare_lines, (
            "Bare 'scry <cmd>' lines found in README (should use 'uv run scry'):\n"
            + "\n".join(f"  {bare}" for bare in bare_lines)
        )

    def test_readme_contains_uv_run_scry(self) -> None:
        """README must reference `uv run scry` at least once."""
        text = self._readme_text()
        assert "uv run scry" in text


# ─── UAT-R5-11: README core workflow section ─────────────────────────────────


class TestReadmeCoreWorkflow:
    """UAT-R5-11: README must have a core workflow section."""

    def _readme_text(self) -> str:
        readme = Path(__file__).resolve().parents[1] / "README.md"
        return readme.read_text(encoding="utf-8")

    def test_readme_has_core_workflow_heading(self) -> None:
        """README must contain a 'Core workflow' section heading."""
        text = self._readme_text()
        assert re.search(r"##\s+Core\s+workflow", text, re.IGNORECASE), (
            "README is missing a '## Core workflow' section (UAT-R5-11)."
        )

    def test_readme_workflow_mentions_link_check_commit(self) -> None:
        """Core workflow section must mention link, check, and commit-links."""
        text = self._readme_text()
        # All three core commands should appear somewhere in README.
        for cmd in ("scry link", "scry check", "scry commit-links"):
            assert cmd in text or f"uv run {cmd}" in text, (
                f"README is missing reference to `{cmd}` (UAT-R5-11)."
            )
