"""AI-assisted patch proposals for drifted links (DESIGN.md §5.5, Wave 5c).

Command: scry reconcile <link_id> [--all] [--apply] [--yes] [--json]

When a link is in code-changed, spec-changed, or both-changed drift state:
  1. Fetches both endpoint anchors from the DB
  2. Fetches git diffs for the changed side(s) via ``git show <commit>:<path>``
  3. Sends context to the LLM (json_mode=True) to propose a patch
  4. Outputs a unified diff for human review, or applies it directly

This module intentionally does NOT modify ``scry.reconcile`` (W4d's
auto-reconcile state machine — a separate, unrelated concept).
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn

import click

from scry.config import ConfigError, load_config
from scry.drift import DriftEvaluation, evaluate_all_drift, evaluate_link_drift
from scry.git_context import GitContextError, GitContextProvider
from scry.llm import LLMError, LLMProvider, LLMRequest, make_provider
from scry.models import (
    Anchor,
    AnchorType,
    Config,
    DriftStatus,
    Link,
    LinkOp,
    LinkRecord,
    new_event_id,
)
from scry.store.db import LockTimeout, ScryDB
from scry.store.links import LinkStore
from scry.store.overlay import OverlayManager

__all__ = [
    "ReconcileError",
    "ReconcileResult",
    "propose_patch",
    "reconcile_all_drifted",
    "run_reconcile_cmd",
]

logger = logging.getLogger(__name__)

# Drift statuses the AI reconciler can act on.
_ACTIONABLE_STATUSES: frozenset[DriftStatus] = frozenset(
    {DriftStatus.CODE_CHANGED, DriftStatus.SPEC_CHANGED, DriftStatus.BOTH_CHANGED}
)

# Context truncation limits (MEDIUM #1) — prevent unbounded LLM context windows.
_MAX_ANCHOR_TEXT_CHARS: int = 4000
_MAX_DIFF_CHARS: int = 8000


def _truncate(text: str, max_chars: int, label: str) -> str:
    """Return *text* truncated to *max_chars*; log a WARNING when truncation fires."""
    if len(text) > max_chars:
        logger.warning(
            "Truncating %s from %d to %d chars for LLM context.",
            label,
            len(text),
            max_chars,
        )
        return text[:max_chars] + "\n... [truncated]"
    return text


_SYSTEM_PROMPT = (
    "You are a documentation maintenance assistant. "
    "Given a drifted link (a code or spec endpoint changed since the link was "
    "created), propose a patch to either the spec, the code, or the link itself "
    "to restore alignment.\n\n"
    "Respond with a single JSON object containing exactly these three fields:\n"
    '  "target":      "spec" | "code" | "link"\n'
    '  "diff":        unified diff string (empty string when target == "link")\n'
    '  "explanation": one-sentence summary of the proposed change'
)


# ─── Result / Error ───────────────────────────────────────────────────────────


class ReconcileError(Exception):
    """Raised for unrecoverable reconciliation failures."""


@dataclass
class ReconcileResult:
    """The AI-proposed patch for one drifted link."""

    link_id: str
    drift_status: DriftStatus
    target: str  # "spec" | "code" | "link"
    diff: str
    explanation: str


# ─── Git helpers ──────────────────────────────────────────────────────────────


def _git_show(repo_root: Path, commit_sha: str, path: str) -> str | None:
    """Return file content at *commit_sha:path*, or ``None`` if unreachable.

    Returns ``None`` on a non-zero git exit code, timeout, or missing git.
    """
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "show", f"{commit_sha}:{path}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    return result.stdout if result.returncode == 0 else None


def _read_current(repo_root: Path, path: str) -> str | None:
    """Return the current on-disk content of *path* (repo-relative), or ``None``."""
    try:
        return (repo_root / path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def _compute_diff(path: str, old: str, new: str) -> str:
    """Return a unified diff string between *old* and *new*; empty when identical."""
    lines = list(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )
    return "".join(lines)


def _validate_diff_paths(
    diff_text: str,
    repo_root: Path,
    allowed_paths: set[str],
    link_id: str,
) -> None:
    """Parse unified diff headers and reject unsafe paths (BLOCKING #3).

    Raises :class:`ReconcileError` when any path in the diff is:

    * **Absolute** — starts with ``/`` or has a Windows drive letter
    * **Traversal** — contains ``..`` segments
    * **Out-of-repo** — resolves outside *repo_root*
    * **Out-of-scope** — is not one of the two link endpoint anchor paths

    Args:
        diff_text:     The unified diff string returned by the LLM.
        repo_root:     Absolute path to the git repository root.
        allowed_paths: Repo-relative paths the LLM is allowed to modify
                       (the ``from`` and ``to`` anchor paths of the link).
        link_id:       Used verbatim in error messages for traceability.
    """
    # Match ``--- …`` and ``+++ …`` header lines; capture the path token
    # (everything after the mandatory space, up to an optional tab-delimited
    # timestamp suffix that POSIX diff appends).
    header_re = re.compile(r"^(?:---|\+\+\+)\s+(\S[^\t\n]*)(?:\t.*)?$", re.MULTILINE)
    repo_resolved = repo_root.resolve()
    found_paths: set[str] = set()

    for m in header_re.finditer(diff_text):
        raw = m.group(1).strip()
        # ``/dev/null`` (and its Windows equivalent) are sentinel values meaning
        # "file did not exist" — they are not real paths and must be skipped.
        if raw in {"/dev/null", "nul", "NUL"}:
            continue
        # Strip the ``a/`` / ``b/`` prefix that git diff always adds.
        for prefix in ("a/", "b/"):
            if raw.startswith(prefix):
                raw = raw[len(prefix) :]
                break
        found_paths.add(raw)

    for path_str in found_paths:
        # ── Absolute-path guard ───────────────────────────────────────────────
        # PurePosixPath covers Unix-style ``/…`` absolute paths.
        # The drive-letter check (``C:\…``) handles Windows paths that could
        # appear in LLM output even when the repo is on POSIX.
        if PurePosixPath(path_str).is_absolute() or (
            len(path_str) >= 2 and path_str[1] == ":" and path_str[0].isalpha()
        ):
            raise ReconcileError(
                f"LLM diff for link {link_id!r} contains absolute path "
                f"{path_str!r} — rejected for security."
            )

        # ── Path-traversal guard ─────────────────────────────────────────────
        if ".." in PurePosixPath(path_str).parts:
            raise ReconcileError(
                f"LLM diff for link {link_id!r} contains path-traversal "
                f"segment in {path_str!r} — rejected for security."
            )

        # ── Out-of-repo guard ─────────────────────────────────────────────────
        try:
            resolved = (repo_root / path_str).resolve()
        except (OSError, ValueError) as exc:
            raise ReconcileError(
                f"LLM diff for link {link_id!r} contains unresolvable path "
                f"{path_str!r}: {exc} — rejected."
            ) from exc
        if not resolved.is_relative_to(repo_resolved):
            raise ReconcileError(
                f"LLM diff for link {link_id!r} modifies path outside repo: "
                f"{path_str!r} — rejected for security."
            )

        # ── Endpoint-scope guard ─────────────────────────────────────────────
        if path_str not in allowed_paths:
            raise ReconcileError(
                f"LLM diff for link {link_id!r} modifies {path_str!r} which "
                f"is not a link endpoint (allowed: {sorted(allowed_paths)!r}) "
                "— rejected for security."
            )


def _apply_git_patch(repo_root: Path, diff_text: str) -> None:
    """Dry-check then apply *diff_text* (unified diff) to the working tree.

    Runs ``git apply --check`` first to catch syntax errors without mutating
    the tree, then runs ``git apply`` to perform the actual modification.

    Raises :class:`ReconcileError` when either invocation fails.
    """
    for extra_args in (["--check"], []):
        try:
            result = subprocess.run(
                ["git", "-C", str(repo_root), "apply", *extra_args],
                input=diff_text,
                capture_output=True,
                text=True,
                check=False,
                timeout=30.0,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            raise ReconcileError(f"git apply unavailable: {exc}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise ReconcileError(f"git apply failed: {detail!r}")


def _write_post_apply_overlay(
    *,
    repo_root: Path,
    link: Link,
    from_anchor: Anchor,
    to_anchor: Anchor,
) -> None:
    """Write a superseding overlay upsert after a successful ``--apply`` (BLOCKING #1).

    The new record carries the *current DB anchor hashes* as the link's
    content/closure hashes, and the *pre-apply link hashes* as
    ``prior_from/to_content_hash`` (§3.3 prior-hash override semantics).
    This ensures the next :func:`~scry.drift.evaluate_link_drift` call returns
    :attr:`~scry.models.DriftStatus.FRESH` because the link's stored hashes now
    equal the DB anchor hashes.

    Args:
        repo_root:   Repository root used to resolve the overlay path.
        link:        The link as it existed *before* the patch was applied.
                     Supplies stable IDs, type, evidence, and ``last_event_id``
                     (needed for ``supersedes``).
        from_anchor: Current DB state of the *from* anchor (refreshed hashes).
        to_anchor:   Current DB state of the *to* anchor (refreshed hashes).

    Raises:
        :class:`ReconcileError`: When git context is unavailable.
    """
    git_ctx_prov = GitContextProvider(repo_root)
    try:
        git_ctx = git_ctx_prov.get()
        commit_sha: str | None = git_ctx.head_sha
        worktree_dirty = bool(git_ctx.dirty_files)
    except GitContextError as exc:
        raise ReconcileError(f"Cannot write post-apply overlay — git unavailable: {exc}") from exc

    record = LinkRecord.model_validate(
        {
            "op": LinkOp.UPSERT,
            "link_id": link.link_id,
            "event_id": new_event_id(),
            "from": link.from_id,
            "from_type": link.from_type,
            "to": link.to_id,
            "to_type": link.to_type,
            "type": link.type,
            # Set hashes to match the current DB anchor state so that the next
            # evaluate_link_drift() returns FRESH.  Omit prior_*_content_hash
            # here: those fields are for rename/rebase (§3.3) — setting them
            # would cause the drift evaluator to compare against the OLD hash
            # (from_ref = prior_from_content_hash) which breaks idempotency.
            "from_content_hash": from_anchor.content_hash,
            "to_content_hash": to_anchor.content_hash,
            "from_closure_hash": from_anchor.closure_hash,
            "to_closure_hash": to_anchor.closure_hash,
            "commit_sha": commit_sha,
            "worktree_dirty": worktree_dirty,
            "evidence": link.evidence,
            # Required by §3.5.2 rule 5 whenever link_id already exists.
            "supersedes": link.last_event_id,
        }
    )
    overlay_mgr = OverlayManager(repo_root, git_context=git_ctx_prov)
    overlay_mgr.append_to_current_branch_overlay(record)


# ─── Prompt builder ───────────────────────────────────────────────────────────


def _build_user_message(
    *,
    link_id: str,
    link_type: str,
    drift_status: DriftStatus,
    from_anchor: Anchor,
    to_anchor: Anchor,
    commit_sha: str | None,
    worktree_dirty: bool,
    evidence: str | None,
    spec_diff: str | None,
    code_diff: str | None,
    unreachable_note: str | None,
) -> str:
    parts: list[str] = [
        f"## Link `{link_id}` (type: {link_type})",
        f"**Drift status**: `{drift_status}`",
    ]
    if commit_sha:
        parts.append(f"**Baseline commit**: `{commit_sha[:12]}`")
    if worktree_dirty:
        parts.append("**Note**: worktree was dirty at link-creation time.")
    if unreachable_note:
        parts.append(f"**Warning**: {unreachable_note}")

    # MEDIUM #1: cap each anchor's content excerpt to avoid blowing the LLM window.
    from_text = _truncate(
        from_anchor.content_text,
        _MAX_ANCHOR_TEXT_CHARS,
        f"from_anchor({from_anchor.id!r}) content",
    )
    to_text = _truncate(
        to_anchor.content_text,
        _MAX_ANCHOR_TEXT_CHARS,
        f"to_anchor({to_anchor.id!r}) content",
    )

    parts += [
        "",
        "### Source anchor (from)",
        f"- ID: `{from_anchor.id}`",
        f"- Type: `{from_anchor.type}`",
        f"- File: `{from_anchor.path}`",
        "- Content at baseline:",
        "```",
        from_text,
        "```",
        "",
        "### Target anchor (to)",
        f"- ID: `{to_anchor.id}`",
        f"- Type: `{to_anchor.type}`",
        f"- File: `{to_anchor.path}`",
        "- Content at baseline:",
        "```",
        to_text,
        "```",
    ]

    if evidence:
        parts += ["", f"### Evidence\n{evidence}"]

    # MEDIUM #1: cap diff excerpts as well.
    if spec_diff:
        spec_diff = _truncate(spec_diff, _MAX_DIFF_CHARS, "spec_diff")
        parts += [
            "",
            "### Spec-side diff (since baseline)",
            "```diff",
            spec_diff,
            "```",
        ]

    if code_diff:
        code_diff = _truncate(code_diff, _MAX_DIFF_CHARS, "code_diff")
        parts += [
            "",
            "### Code-side diff (since baseline)",
            "```diff",
            code_diff,
            "```",
        ]

    parts += [
        "",
        'Respond with JSON: {"target": "spec"|"code"|"link", "diff": "...", "explanation": "..."}',
    ]
    return "\n".join(parts)


# ─── Core async logic ─────────────────────────────────────────────────────────


async def propose_patch(
    evaluation: DriftEvaluation,
    *,
    repo_root: Path,
    from_anchor: Anchor,
    to_anchor: Anchor,
    provider: LLMProvider,
) -> ReconcileResult:
    """Call the LLM to propose a patch for a single drifted link.

    Fetches git diffs for the changed side(s) when ``link.commit_sha`` is set
    (DESIGN.md §5.5 step 1).  Falls back to a note about unreachable commits
    (e.g. after a rebase) per the spec.

    Raises :class:`ReconcileError` on LLM failure or an invalid JSON response.
    """
    link = evaluation.link
    drift_status = evaluation.drift_status

    spec_diff: str | None = None
    code_diff: str | None = None
    unreachable_note: str | None = None

    if link.commit_sha:
        for anchor in (from_anchor, to_anchor):
            old = _git_show(repo_root, link.commit_sha, anchor.path)
            if old is None:
                if unreachable_note is None:
                    unreachable_note = (
                        f"Cannot reconstruct {anchor.path!r} at commit "
                        f"`{link.commit_sha[:12]}` (may have been rebased away). "
                        "Showing current state vs content-hash mismatch only."
                    )
                continue
            current = _read_current(repo_root, anchor.path)
            if current is None:
                continue
            d = _compute_diff(anchor.path, old, current)
            if not d:
                # File unchanged at this anchor; skip adding an empty diff.
                continue
            if AnchorType(anchor.type) in {AnchorType.SECTION, AnchorType.CODE_IN_DOC}:
                spec_diff = d
            else:
                code_diff = d

    user_msg = _build_user_message(
        link_id=link.link_id,
        link_type=str(link.type),
        drift_status=drift_status,
        from_anchor=from_anchor,
        to_anchor=to_anchor,
        commit_sha=link.commit_sha,
        worktree_dirty=link.worktree_dirty,
        evidence=link.evidence,
        spec_diff=spec_diff,
        code_diff=code_diff,
        unreachable_note=unreachable_note,
    )

    req = LLMRequest(
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
        temperature=0.0,
        json_mode=True,
    )
    try:
        response = await provider.complete(req)
    except LLMError as exc:
        raise ReconcileError(f"LLM error for link {link.link_id}: {exc}") from exc

    try:
        data: dict[str, Any] = json.loads(response.text)
    except (ValueError, TypeError) as exc:
        raise ReconcileError(f"LLM returned invalid JSON for link {link.link_id}: {exc}") from exc

    target = str(data.get("target", ""))
    diff = str(data.get("diff", ""))
    explanation = str(data.get("explanation", ""))

    if target not in {"spec", "code", "link"}:
        raise ReconcileError(
            f"LLM returned unexpected target {target!r} for link {link.link_id}; "
            "expected 'spec', 'code', or 'link'."
        )

    return ReconcileResult(
        link_id=link.link_id,
        drift_status=drift_status,
        target=target,
        diff=diff,
        explanation=explanation,
    )


async def reconcile_all_drifted(
    *,
    repo_root: Path,
    db: ScryDB,
    link_store: LinkStore,
    provider: LLMProvider,
) -> list[ReconcileResult]:
    """Propose patches for every code-changed / spec-changed / both-changed link.

    Silently skips links with broken or missing anchors (they cannot be patched).
    """
    evaluations = evaluate_all_drift(db=db, link_store=link_store)
    results: list[ReconcileResult] = []
    for ev in evaluations:
        if ev.drift_status not in _ACTIONABLE_STATUSES:
            continue
        from_anchor = db.get_anchor(ev.link.from_id)
        to_anchor = db.get_anchor(ev.link.to_id)
        if from_anchor is None or to_anchor is None:
            continue
        result = await propose_patch(
            ev,
            repo_root=repo_root,
            from_anchor=from_anchor,
            to_anchor=to_anchor,
            provider=provider,
        )
        results.append(result)
    return results


# ─── Output helpers ───────────────────────────────────────────────────────────


def _display_results(results: list[ReconcileResult], *, json_output: bool) -> None:
    """Print reconcile results to stdout (human-readable or JSON)."""
    if json_output:
        payload = [
            {
                "link_id": r.link_id,
                "drift_status": r.drift_status,
                "target": r.target,
                "diff": r.diff,
                "explanation": r.explanation,
            }
            for r in results
        ]
        click.echo(json.dumps(payload, indent=2))
        return

    for r in results:
        click.echo(f"\n─── Link: {r.link_id} ───")
        click.echo(f"Drift:       {r.drift_status}")
        click.echo(f"Target:      {r.target}")
        click.echo(f"Explanation: {r.explanation}")
        if r.diff:
            click.echo("\nProposed diff:")
            click.echo(r.diff)
        else:
            click.echo("(no diff — link-level change only)")


# ─── CLI entry point ──────────────────────────────────────────────────────────


def _emit_error(message: str, *, json_output: bool, exit_code: int) -> NoReturn:
    """Print an error in JSON or plain-text form, then SystemExit.

    UT2-2 fix: when ``--json`` is set, error responses must be parseable
    JSON so machine-readable pipelines can route on them.  Plain-text
    error messages on stderr break ``scry reconcile <id> --json | jq``.
    """
    if json_output:
        click.echo(json.dumps({"error": message, "exit_code": exit_code}), err=True)
    else:
        click.echo(f"error: {message}", err=True)
    raise SystemExit(exit_code) from None


def run_reconcile_cmd(
    *,
    repo_root: Path,
    link_id: str | None,
    all_links: bool,
    apply_patch: bool,
    yes: bool,
    json_output: bool,
) -> None:
    """Synchronous entry point for ``scry reconcile``.

    Called from :func:`scry.cli.reconcile`.  All error paths use
    :func:`_emit_error` so ``--json`` callers always get parseable JSON
    on both success and failure (UT2-2 fix).

    Exit codes (UT2-4 refinement):

    * **0** — success / no drift.
    * **1** — LLM unavailable / unconfigured (configuration issue).
    * **2** — operational failure (missing DB, malformed input, git error,
      patch validation rejection, IO error).
    """
    if not all_links and link_id is None:
        _emit_error("provide a LINK_ID or --all.", json_output=json_output, exit_code=2)

    # Load config; fall back to defaults when config.yaml is absent.
    try:
        config = load_config(repo_root)
    except ConfigError:
        config = Config()

    # Build the LLM provider from config.
    # UT2-4: LLM unavailability is a configuration problem (exit 1), not
    # an infrastructure failure (exit 2).
    try:
        provider = make_provider(config.llm)
    except LLMError as exc:
        _emit_error(f"LLM provider unavailable: {exc}", json_output=json_output, exit_code=1)

    db_path = repo_root / ".scry" / "vectors.db"
    if not db_path.exists():
        _emit_error(
            "vectors.db not found. Run `scry index` first.",
            json_output=json_output,
            exit_code=2,
        )

    # ── BLOCKING #2: build an overlay-aware LinkStore ─────────────────────────
    # Mirror the _BranchLinkStore pattern used by ``scry check`` and
    # ``scry suggest-links`` so that links created on the current branch
    # overlay are visible to drift evaluation and reconciliation.
    base_link_store = LinkStore(repo_root)
    try:
        _git_ctx_prov = GitContextProvider(repo_root)
        _overlay_mgr = OverlayManager(
            repo_root, git_context=_git_ctx_prov, link_store=base_link_store
        )
        _current_overlay = _overlay_mgr.current_overlay_path()
    except GitContextError as exc:
        _emit_error(f"git context unavailable: {exc}", json_output=json_output, exit_code=2)

    _ov_path = _current_overlay

    class _BranchLinkStore(LinkStore):
        """LinkStore that always replays baseline ⊕ current-branch overlay."""

        def replay(self, *, overlay_path: Path | None = None) -> Any:
            return super().replay(overlay_path=overlay_path or _ov_path)

    link_store = _BranchLinkStore(repo_root)

    # apply_context holds (link, from_anchor, to_anchor) for every result so
    # that --apply can write the overlay upsert (BLOCKING #1) without reopening
    # the DB.
    apply_context: dict[str, tuple[Link, Anchor, Anchor]] = {}
    results: list[ReconcileResult] = []

    try:
        with ScryDB(repo_root, read_only=True) as db:
            if all_links:
                results = asyncio.run(
                    reconcile_all_drifted(
                        repo_root=repo_root,
                        db=db,
                        link_store=link_store,  # overlay-aware (BLOCKING #2)
                        provider=provider,
                    )
                )
                # Collect apply context for every result while DB is still open.
                if apply_patch and results:
                    fresh_replay = link_store.replay()
                    for r in results:
                        lnk = fresh_replay.active_links.get(r.link_id)
                        if lnk is not None:
                            fa = db.get_anchor(lnk.from_id)
                            ta = db.get_anchor(lnk.to_id)
                            if fa is not None and ta is not None:
                                apply_context[r.link_id] = (lnk, fa, ta)
            else:
                # Single-link path.
                assert link_id is not None  # guaranteed by flag-check above

                # BLOCKING #2: replay with current overlay so branch-only links
                # are visible.
                replay = link_store.replay()
                link = replay.active_links.get(link_id)
                if link is None:
                    _emit_error(
                        f"link not found: {link_id!r}",
                        json_output=json_output,
                        exit_code=2,
                    )

                ev = evaluate_link_drift(
                    link,
                    db=db,
                    merge_conflicts=set(replay.merge_conflicts),
                    config=config.drift,
                )

                if ev.drift_status == DriftStatus.FRESH:
                    click.echo(f"Link {link_id} is fresh — no drift detected.")
                    return

                if ev.drift_status not in _ACTIONABLE_STATUSES:
                    _emit_error(
                        f"Link {link_id} has status {ev.drift_status!r} which is not "
                        "actionable by the AI reconciler "
                        "(expected code-changed, spec-changed, or both-changed).",
                        json_output=json_output,
                        exit_code=2,
                    )

                from_anchor = db.get_anchor(link.from_id)
                to_anchor = db.get_anchor(link.to_id)
                if from_anchor is None or to_anchor is None:
                    _emit_error(
                        f"one or both anchors missing for link {link_id!r}",
                        json_output=json_output,
                        exit_code=2,
                    )

                result = asyncio.run(
                    propose_patch(
                        ev,
                        repo_root=repo_root,
                        from_anchor=from_anchor,
                        to_anchor=to_anchor,
                        provider=provider,
                    )
                )
                results = [result]

                if apply_patch:
                    apply_context[link_id] = (link, from_anchor, to_anchor)

            if not results:
                click.echo("No drifted links to reconcile.")
                return

            _display_results(results, json_output=json_output)

            if apply_patch:
                # ── HIGH #1: collect failures; continue on error; exit 1 at end
                apply_failures: list[str] = []

                for r in results:
                    if not r.diff:
                        click.echo(
                            f"Skipping {r.link_id}: no diff to apply (link-level change only)."
                        )
                        continue
                    if not yes:
                        confirmed = click.confirm(
                            f"\nApply patch for link {r.link_id}?", default=False
                        )
                        if not confirmed:
                            click.echo("Skipped.")
                            continue

                    ctx = apply_context.get(r.link_id)
                    if ctx is None:
                        click.echo(
                            f"error: missing apply context for link {r.link_id!r}",
                            err=True,
                        )
                        apply_failures.append(r.link_id)
                        continue

                    lnk, fa, ta = ctx
                    allowed_paths = {fa.path, ta.path}

                    try:
                        # BLOCKING #3: path validation before touching the tree.
                        _validate_diff_paths(r.diff, repo_root, allowed_paths, r.link_id)
                        # Run --check then actual apply.
                        _apply_git_patch(repo_root, r.diff)
                        click.echo(f"Applied patch for link {r.link_id}.")

                        # BLOCKING #1: write superseding overlay upsert so the
                        # next reconcile reports FRESH (idempotency).
                        _write_post_apply_overlay(
                            repo_root=repo_root,
                            link=lnk,
                            from_anchor=fa,
                            to_anchor=ta,
                        )

                        # Verify the overlay write yielded a FRESH status.
                        new_replay = link_store.replay()
                        new_lnk = new_replay.active_links.get(r.link_id)
                        if new_lnk is not None:
                            new_ev = evaluate_link_drift(
                                new_lnk,
                                db=db,
                                merge_conflicts=set(new_replay.merge_conflicts),
                                config=config.drift,
                            )
                            if new_ev.drift_status != DriftStatus.FRESH:
                                logger.warning(
                                    "link %s still %s after apply — patch may be insufficient",
                                    r.link_id,
                                    new_ev.drift_status,
                                )
                                click.echo(
                                    f"warning: link {r.link_id!r} still "
                                    f"{new_ev.drift_status!r} after patch — "
                                    "the patch may be insufficient.",
                                    err=True,
                                )

                    except ReconcileError as exc:
                        click.echo(
                            f"error applying patch for link {r.link_id!r}: {exc}",
                            err=True,
                        )
                        apply_failures.append(r.link_id)

                # HIGH #1: exit 1 if any apply failed; summarise for --all.
                if apply_failures:
                    if len(results) > 1:
                        succeeded = len(results) - len(apply_failures)
                        click.echo(
                            f"\nApply summary: {succeeded}/{len(results)} succeeded.",
                            err=True,
                        )
                        click.echo(
                            f"Failed links: {', '.join(apply_failures)}",
                            err=True,
                        )
                    raise SystemExit(1) from None

    except ReconcileError as exc:
        _emit_error(str(exc), json_output=json_output, exit_code=2)
    except LLMError as exc:
        _emit_error(f"LLM error: {exc}", json_output=json_output, exit_code=1)
    except (LockTimeout, OSError) as exc:
        _emit_error(str(exc), json_output=json_output, exit_code=2)
