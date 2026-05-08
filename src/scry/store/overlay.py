"""Branch-aware overlay management for scry link records.

Implements DESIGN.md §3.5 (link layer architecture), §3.5.0 (overlay slug
derivation — delegated to GitContextProvider), and §3.5.4 (commit-links
two-phase atomicity protocol — delegated to LinkStore).

``OverlayManager`` is the higher-level API consumed by W2i (MCP server) and
W2j (CLI).  It composes :class:`~scry.git_context.GitContextProvider` with
:class:`~scry.store.links.LinkStore` to provide:

* Branch-aware reads (replay baseline ⊕ current-branch overlay).
* Branch-aware writes with **cache-invalidation-on-write** (§7.2 v3.1).
* Recovery of interrupted ``commit-links`` transactions (§3.5.4).

Public surface
--------------
PendingPromotion  — recovery metadata for an interrupted commit-links txn
OverlayManager    — branch-aware orchestration of LinkStore + GitContextProvider
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from scry.git_context import GitContextProvider
from scry.models import EventId, LinkRecord
from scry.store.links import LinkStore, ReplayResult

__all__ = [
    "OverlayManager",
    "PendingPromotion",
]


@dataclass(frozen=True)
class PendingPromotion:
    """Recovery information about an in-progress commit-links transaction.

    Returned by :meth:`OverlayManager.recover_pending` — one entry per
    ``.scry/commit-links.<txn-id>.marker`` file found during startup recovery.

    Attributes:
        txn_id:              UUID string extracted from the marker filename.
        overlay_path:        Absolute path to the overlay file involved.
        promoted_event_ids:  The ``event_id``s that were (or are being) promoted.
        ts:                  Timestamp recorded in the marker file, or
                             ``datetime.min`` (UTC) if absent or unparseable.
    """

    txn_id: str
    overlay_path: Path
    promoted_event_ids: list[EventId]
    ts: datetime


class OverlayManager:
    """Branch-aware orchestration of LinkStore + GitContextProvider.

    Lifecycle:
        Constructed once per repo.  Reads git context on every public call
        via the cached :class:`~scry.git_context.GitContextProvider` (§7.2
        polling).  Writes go through :class:`~scry.store.links.LinkStore`
        which provides cross-process append safety.

    On startup, callers should invoke :meth:`recover_pending` once to finish
    any ``commit-links`` transaction that crashed mid-flight (§3.5.4).

    See DESIGN.md §3.5, §3.5.0, §3.5.4 v3.1.

    Args:
        repo_root:    Absolute path to the git repository root.
        git_context:  Optional pre-constructed
                      :class:`~scry.git_context.GitContextProvider`.
                      When ``None``, a default provider is created with a
                      30-second TTL and dirty-worktree polling enabled.
        link_store:   Optional pre-constructed
                      :class:`~scry.store.links.LinkStore`.
                      When ``None``, a default store is created for
                      *repo_root*.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        git_context: GitContextProvider | None = None,
        link_store: LinkStore | None = None,
    ) -> None:
        self._repo_root = repo_root
        self._git_context = git_context or GitContextProvider(repo_root)
        self._link_store = link_store or LinkStore(repo_root)

    @property
    def overlay_dir(self) -> Path:
        """``<repo_root>/.scry/overlays/``"""
        return self._repo_root / ".scry" / "overlays"

    def overlay_path_for(self, overlay_slug: str) -> Path:
        """Return the absolute path to ``.scry/overlays/<slug>``.

        Args:
            overlay_slug: Overlay filename derived via §3.5.0
                          (e.g. ``"main.jsonl"``).

        Returns:
            Absolute :class:`~pathlib.Path` under :attr:`overlay_dir`.
            The file may or may not exist yet.
        """
        return self.overlay_dir / overlay_slug

    def current_overlay_path(self) -> Path:
        """Resolve the overlay path for the CURRENT git branch/HEAD.

        Uses :attr:`~scry.git_context.GitContext.overlay_slug` (which
        already encodes the §3.5.0 slug derivation, including detached HEAD
        handling).  Returns the absolute path; the file may or may not exist
        yet.

        Returns:
            Absolute :class:`~pathlib.Path` to the current branch's overlay.

        Raises:
            :class:`~scry.git_context.GitContextError`: If git is unavailable.
        """
        slug = self._git_context.get().overlay_slug
        return self.overlay_path_for(slug)

    # ── Read side ─────────────────────────────────────────────────────────────

    def replay_active(self) -> ReplayResult:
        """Replay baseline ⊕ current-branch overlay.

        Equivalent to
        ``LinkStore.replay(overlay_path=self.current_overlay_path())``.
        Returns the active link table and any merge-conflicts.

        Returns:
            :class:`~scry.store.links.ReplayResult` with the merged active
            link table and a list of conflicting ``link_id``s.

        Raises:
            :class:`~scry.git_context.GitContextError`: If git is unavailable.
        """
        return self._link_store.replay(overlay_path=self.current_overlay_path())

    def list_pending_overlay_records(self) -> list[LinkRecord]:
        """Return all records currently in the active branch's overlay file.

        Useful for ``scry status`` to show pending (uncommitted) link
        mutations on the current branch.

        Returns:
            :class:`~scry.models.LinkRecord` list in document order.
            Empty if the overlay file does not exist.

        Raises:
            :class:`~scry.git_context.GitContextError`: If git is unavailable.
        """
        return self._link_store.read_records(self.current_overlay_path())

    def list_pending_branches(self) -> list[str]:
        """Enumerate every overlay slug present in ``.scry/overlays/``.

        Each entry is a ``.jsonl`` filename stem (i.e. the overlay slug
        *without* the ``.jsonl`` suffix).  Callers can use the list to
        render ``scry status`` cross-branch summaries.  ``scry vacuum``
        (Wave 6) uses it to detect dead branches whose overlays should be
        GC'd.

        Returns:
            Sorted list of overlay slugs (without ``.jsonl`` extension).
            Returns ``[]`` when the overlays directory does not exist or is
            empty.
        """
        if not self.overlay_dir.exists():
            return []
        return sorted(p.stem for p in self.overlay_dir.iterdir() if p.suffix == ".jsonl")

    # ── Write side ────────────────────────────────────────────────────────────

    def append_to_current_branch_overlay(self, record: LinkRecord) -> None:
        """Append *record* to the current branch's overlay file.

        Delegates to
        :meth:`~scry.store.links.LinkStore.append_overlay` which validates
        per §3.5.2 and cross-file revival rules.

        The current branch's overlay path is recomputed on every call (HEAD
        may have moved per the §7.2 polling model).  The
        :class:`~scry.git_context.GitContextProvider` cache is invalidated
        **before** resolving the overlay path, ensuring writes never go to a
        stale branch overlay (§7.2 v3.1 cache-invalidation-on-writes
        contract).

        Args:
            record: The :class:`~scry.models.LinkRecord` to append.

        Raises:
            :class:`~scry.store.links.LinkValidationError`: On §3.5.2
                violations.
            :class:`~scry.git_context.GitContextError`: If git is unavailable.
        """
        # §7.2 v3.1: invalidate the cache before every write so the overlay
        # path is always resolved from a fresh git context.
        self._git_context.invalidate()
        overlay_path = self.current_overlay_path()
        self._link_store.append_overlay(record, overlay_path)

    def promote_pending(
        self,
        *,
        event_ids: Iterable[EventId] | None = None,
    ) -> list[EventId]:
        """Promote the current branch's overlay records to baseline.

        Delegates to
        :meth:`~scry.store.links.LinkStore.promote_overlay_to_baseline` with
        the current overlay path.  ``event_ids=None`` promotes every record in
        the overlay; otherwise only the listed ``event_id``s are promoted.

        The cache is invalidated before resolving the overlay path (same
        rationale as :meth:`append_to_current_branch_overlay`).

        Returns:
            List of promoted ``event_id``s in the order they were appended to
            the baseline.

        Raises:
            :class:`~scry.store.links.LinkValidationError`: If any requested
                ``event_id`` is absent from the overlay.
            :class:`~scry.git_context.GitContextError`: If git is unavailable.
        """
        # §7.2 v3.1: invalidate cache before every write.
        self._git_context.invalidate()
        overlay_path = self.current_overlay_path()
        return self._link_store.promote_overlay_to_baseline(overlay_path, event_ids=event_ids)

    # ── Recovery ─────────────────────────────────────────────────────────────

    def recover_pending(self) -> list[PendingPromotion]:
        """At process startup: finish any ``commit-links`` transaction that
        crashed mid-flight (§3.5.4 v3.1 atomicity protocol).

        Scans for ``.scry/commit-links.<txn-id>.marker`` files, collects
        their metadata, then delegates the actual step-completion to
        :meth:`~scry.store.links.LinkStore.recover_pending_promotions`.

        Returns metadata about each marker that was found — useful for
        logging or ``scry doctor`` output.  Returns ``[]`` when no markers
        are present.

        Returns:
            List of :class:`PendingPromotion` objects — one per marker file
            found.  Empty when no markers exist.
        """
        scry_dir = self._repo_root / ".scry"
        markers = sorted(scry_dir.glob("commit-links.*.marker"))
        results: list[PendingPromotion] = []

        for marker_path in markers:
            try:
                raw: object = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(raw, dict):
                continue

            raw_txn = raw.get("txn_id")
            txn_id = raw_txn if isinstance(raw_txn, str) else ""

            raw_overlay = raw.get("overlay_path")
            if not isinstance(raw_overlay, str):
                continue
            overlay_path = scry_dir / raw_overlay

            raw_promoted = raw.get("promoted_event_ids")
            promoted_event_ids: list[EventId]
            if isinstance(raw_promoted, list):
                promoted_event_ids = [e for e in raw_promoted if isinstance(e, str)]
            else:
                promoted_event_ids = []

            raw_ts = raw.get("ts")
            ts: datetime
            if isinstance(raw_ts, str):
                try:
                    ts = datetime.fromisoformat(raw_ts)
                except ValueError:
                    ts = datetime.min.replace(tzinfo=UTC)
            else:
                ts = datetime.min.replace(tzinfo=UTC)

            results.append(
                PendingPromotion(
                    txn_id=txn_id,
                    overlay_path=overlay_path,
                    promoted_event_ids=promoted_event_ids,
                    ts=ts,
                )
            )

        # Delegate actual step-completion + marker deletion to LinkStore.
        self._link_store.recover_pending_promotions()

        return results
