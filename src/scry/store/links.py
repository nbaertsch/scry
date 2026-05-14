"""JSONL reader/writer and replay engine for scry link records.

Implements the baseline + per-branch overlay link store as described in
DESIGN.md §3.5, §3.5.0 (overlay slug derivation), §3.5.1 (event record
schema), §3.5.2 (replay rules), §3.5.3 (merge driver), §3.5.4
(commit-links promotion).

The active link table is computed by ``replay(baseline) ⊕ replay(overlay)``
where overlay records layer on top of baseline per §3.5.2.

Rule summary (§3.5.2):
    1. File order = ordering within a file; last record for a link_id wins.
    2. Replay order: baseline first, then overlay; overlay records win.
    3. Tombstones are absorbing within the same file — upsert after delete
       in the same file is a write-time validation error.
    4. Cross-file revival (baseline delete + overlay upsert) is allowed;
       the revival upsert MUST carry ``supersedes`` pointing at the
       baseline tombstone event_id.
    5. ``supersedes`` is required on every upsert whose link_id already
       exists in the file being written OR in baseline (overlay writes).
    6. Post-union-merge: if two upserts share the same supersedes event_id,
       or a supersedes references an unknown event_id → merge-conflict.
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import time
import uuid
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from scry.models import EventId, Link, LinkId, LinkOp, LinkRecord

# ─── Exceptions ───────────────────────────────────────────────────────────────


class LinkValidationError(Exception):
    """Raised when an attempted upsert/delete violates §3.5.2 replay rules at write time."""


class MergeConflictError(Exception):
    """Raised at replay time when the supersedes chain has multiple latest upserts
    or references an unknown event_id (§3.5.2 rule 6 v3.1)."""


# ─── Result type ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ReplayResult:
    """Output of replay().

    ``active_links`` is the merged active link table (baseline ⊕ overlay per
    §3.5.2).  ``merge_conflicts`` lists link_ids whose ``supersedes`` chain is
    broken post-merge (§3.5.2 rule 6).
    """

    active_links: dict[LinkId, Link]
    """The active link table — replay(baseline) ⊕ replay(overlay)."""
    merge_conflicts: list[LinkId]
    """link_ids whose supersedes chain is broken (post-merge)."""


# ─── Internal helpers ─────────────────────────────────────────────────────────


def _link_from_record(record: LinkRecord) -> Link:
    """Convert an UPSERT ``LinkRecord`` to an active ``Link`` (§3.5).

    Pre-condition: ``record.op == LinkOp.UPSERT`` and all required upsert
    fields are non-None (enforced by ``LinkRecord._shape_check``).
    """
    assert record.from_ is not None, "upsert record must have from_"
    assert record.from_type is not None, "upsert record must have from_type"
    assert record.to is not None, "upsert record must have to"
    assert record.to_type is not None, "upsert record must have to_type"
    assert record.type is not None, "upsert record must have type"
    assert record.from_content_hash is not None, "upsert record must have from_content_hash"
    assert record.to_content_hash is not None, "upsert record must have to_content_hash"
    return Link(
        link_id=record.link_id,
        from_id=record.from_,
        from_type=record.from_type,
        to_id=record.to,
        to_type=record.to_type,
        type=record.type,
        from_content_hash=record.from_content_hash,
        to_content_hash=record.to_content_hash,
        from_closure_hash=record.from_closure_hash,
        to_closure_hash=record.to_closure_hash,
        prior_from_content_hash=record.prior_from_content_hash,
        prior_to_content_hash=record.prior_to_content_hash,
        commit_sha=record.commit_sha,
        worktree_dirty=record.worktree_dirty,
        evidence=record.evidence,
        last_event_id=record.event_id,
        created_by=record.created_by,
    )


def _fsync_file(path: Path) -> None:
    """Open ``path`` with write-compatible access and fsync for durability.

    Opening with ``"r+b"`` avoids truncation while satisfying FlushFileBuffers
    on Windows (which requires write access).
    """
    with path.open("r+b") as fh:
        os.fsync(fh.fileno())


def _fsync_dir(dir_path: Path) -> None:
    """fsync ``dir_path`` on Unix; no-op on Windows (os.replace is atomic there)."""
    if os.name == "nt":
        return
    try:
        fd = os.open(str(dir_path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass  # best-effort; directory fsync is an optimisation, not a correctness gate


def _atomic_rewrite(path: Path, records: list[LinkRecord]) -> None:
    """Atomically overwrite ``path`` with ``records`` via tempfile + rename.

    Uses ``os.replace`` which is atomic on both POSIX and Windows (NTFS
    ReplaceFile).  The temp file is fsynced before rename to ensure data is
    durable on crash.
    """
    dir_path = path.parent
    fd, tmp_name = tempfile.mkstemp(dir=str(dir_path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            for record in records:
                fh.write(record.model_dump_json(by_alias=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_name)
        raise
    os.replace(tmp_name, path)


# ─── Main class ───────────────────────────────────────────────────────────────


class LinkStore:
    """Reader/writer for links.jsonl + per-branch overlay files.

    Owns:
        - Atomic append to baseline ``links.jsonl``
        - Atomic append to per-branch overlay
        - Validation at write time per §3.5.2 rules
        - Replay across baseline + overlay producing the active ``Link`` table
        - Two-phase atomic overlay→baseline promotion (§3.5.4)
        - Crash recovery via ``.scry/commit-links.*.marker`` files (§3.5.4)

    Delegates overlay slug derivation to the caller (W2c owns §3.5.0 slug
    computation; you just read/write whatever path the caller provides).
    """

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self._scry_dir = repo_root / ".scry"
        self._scry_dir.mkdir(parents=True, exist_ok=True)

    @property
    def baseline_path(self) -> Path:
        """Always ``<repo_root>/.scry/links.jsonl``."""
        return self._scry_dir / "links.jsonl"

    # ── Read side ─────────────────────────────────────────────────────────────

    def read_records(self, path: Path) -> list[LinkRecord]:
        """Read all event records from a JSONL file in document order.

        Returns ``[]`` if the file does not exist.

        Raises ``LinkValidationError`` on a malformed line (with line number).
        """
        if not path.exists():
            return []
        records: list[LinkRecord] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = LinkRecord.model_validate_json(line)
                except Exception as exc:
                    raise LinkValidationError(f"Malformed record on line {lineno}: {exc}") from exc
                records.append(record)
        return records

    def replay(self, *, overlay_path: Path | None = None) -> ReplayResult:
        """Replay baseline + optional overlay per §3.5.2 rules.

        - §3.5.2 rule 1: file order = ordering within a file (last wins).
        - §3.5.2 rule 2: baseline first, then overlay; overlay records win.
        - §3.5.2 rule 6: merge conflicts detected across the union.

        Returns ``ReplayResult`` with the active link table and any
        merge-conflict link_ids.
        """
        baseline_records = self.read_records(self.baseline_path)
        overlay_records = self.read_records(overlay_path) if overlay_path else []
        all_records = baseline_records + overlay_records

        # Rules 1 + 2: process in document order; last record for each
        # link_id wins (baseline first, overlay second → overlay wins ties).
        # review-r6abc-2 follow-up: pop+reassign on existing keys so that
        # ``link_last`` iteration order tracks LAST-EVENT order.  Python's
        # default ``dict[k] = v`` on an existing key preserves the original
        # insertion position, which would silently break consumers that
        # iterate ``active_links.values()`` expecting the most-recently
        # touched link to come last (e.g. ``get_links`` MCP dedup).
        link_last: dict[LinkId, LinkRecord] = {}
        for record in all_records:
            if record.link_id in link_last:
                del link_last[record.link_id]
            link_last[record.link_id] = record

        # Build the active link table from the winning record per link_id.
        # ``active_links`` inherits ``link_last``'s last-event iteration
        # order (DELETEs are skipped but their absence preserves order).
        active_links: dict[LinkId, Link] = {}
        for link_id, last_record in link_last.items():
            if last_record.op == LinkOp.UPSERT:
                active_links[link_id] = _link_from_record(last_record)
            # DELETE → link intentionally absent from the active table.

        # Rule 6: detect broken supersedes chains across the union.
        merge_conflicts = self._find_merge_conflicts(all_records)

        return ReplayResult(active_links=active_links, merge_conflicts=merge_conflicts)

    # ── Write side ────────────────────────────────────────────────────────────

    def append_baseline(self, record: LinkRecord) -> None:
        """Validate per §3.5.2 against the current baseline, then atomically append.

        Raises ``LinkValidationError`` for:
        - Rule 3: upsert after delete within the same (baseline) file.
        - Rule 5: duplicate link_id without a supersedes pointer.
        - Deletes with missing supersedes or referencing an unknown event_id.
        """
        same_file = self.read_records(self.baseline_path)
        self._validate_record(record, same_file, cross_file=None)
        self._do_append(self.baseline_path, record)

    def append_overlay(self, record: LinkRecord, overlay_path: Path) -> None:
        """Validate and atomically append ``record`` to ``overlay_path``.

        Validates against baseline + existing overlay per §3.5.2:
        - Rule 3: upsert after delete within the overlay (same file).
        - Rule 4: cross-file revival allowed — overlay upsert after baseline
          delete MUST carry ``supersedes: <baseline-tombstone-event-id>``.
        - Rule 5: ``supersedes`` required if link_id exists in overlay or
          baseline.

        Raises ``LinkValidationError`` on violations.
        """
        same_file = self.read_records(overlay_path)
        cross_file = self.read_records(self.baseline_path)
        self._validate_record(record, same_file, cross_file=cross_file)
        self._do_append(overlay_path, record)

    # ── Promotion ─────────────────────────────────────────────────────────────

    def promote_overlay_to_baseline(
        self,
        overlay_path: Path,
        *,
        event_ids: Iterable[EventId] | None = None,
    ) -> list[EventId]:
        """Promote selected (or all) overlay records to the baseline.

        Uses the §3.5.4 v3.1 two-phase atomicity protocol:
        1. Append promoted records to ``links.jsonl`` + write a
           ``.scry/commit-links.<txn-id>.marker`` file.  fsync both.
        2. Atomic rewrite of overlay minus promoted records.  fsync directory.
        3. Delete the marker file.  fsync directory.

        Raises ``LinkValidationError`` if any selected event_id is missing
        from the overlay.

        Returns the promoted event_ids in the order they were appended to
        baseline.
        """
        overlay_records = self.read_records(overlay_path)

        # Select which records to promote.
        if event_ids is not None:
            wanted: set[EventId] = set(event_ids)
            to_promote = [r for r in overlay_records if r.event_id in wanted]
            found = {r.event_id for r in to_promote}
            missing = wanted - found
            if missing:
                raise LinkValidationError(f"event_ids not found in overlay: {sorted(missing)}")
        else:
            to_promote = list(overlay_records)

        if not to_promote:
            return []

        promoted_event_ids = [r.event_id for r in to_promote]
        promoted_set: set[EventId] = set(promoted_event_ids)
        txn_id = str(uuid.uuid4())
        marker_path = self._scry_dir / f"commit-links.{txn_id}.marker"

        try:
            overlay_relative: Path = overlay_path.relative_to(self._scry_dir)
        except ValueError:
            overlay_relative = Path(overlay_path.name)

        # ── Step 1a: write marker FIRST, before touching baseline ─────────────
        #
        # Reversed order from the naive design (review-w2b MEDIUM/BLOCKING
        # fix combined with the merge-conflict dedupe): if the process
        # crashes between marker-write and baseline-append, the marker
        # exists with no baseline change — recover_pending_promotions
        # detects this and re-applies. If we instead appended to baseline
        # first, a crash before marker-write would leave duplicate records
        # (overlay still has them + baseline has them) with no marker for
        # recovery to find — exactly the false-positive merge-conflict
        # state that bit review-w2b. Marker-first means baseline is only
        # mutated when there's already a record of the intent.
        marker_data = {
            "txn_id": txn_id,
            "overlay_path": overlay_relative.as_posix(),
            "promoted_event_ids": promoted_event_ids,
            "ts": datetime.now(UTC).isoformat(),
        }
        marker_path.write_text(json.dumps(marker_data), encoding="utf-8")
        _fsync_file(marker_path)
        _fsync_dir(self._scry_dir)

        # ── Step 1b: append the records to baseline (each via _do_append
        # so the cross-process append lock is held — review-w2b MEDIUM
        # fix; previously this path bypassed the lock).
        for record in to_promote:
            self._do_append(self.baseline_path, record)

        # ── Step 2: atomic overlay rewrite, fsync dir ──────────────────────────
        remaining = [r for r in overlay_records if r.event_id not in promoted_set]
        _atomic_rewrite(overlay_path, remaining)
        _fsync_dir(overlay_path.parent)

        # ── Step 3: delete marker, fsync dir ──────────────────────────────────
        marker_path.unlink()
        _fsync_dir(self._scry_dir)
        return promoted_event_ids

    def recover_pending_promotions(self) -> list[EventId]:
        """Finish any promotions interrupted by a crash (§3.5.4 recovery).

        With the marker-first protocol, a crashed promotion can leave
        any of these states:

        1. Marker exists; baseline missing the records; overlay still
           has them — replay step 1b (re-append to baseline), then
           step 2 (rewrite overlay), then step 3 (unlink marker).
        2. Marker exists; baseline already has the records (from a
           prior partial completion); overlay still has them — skip
           step 1b (don't double-append), do steps 2 and 3.
        3. Marker exists; baseline has records; overlay already
           rewritten — only step 3 left.
        4. Marker missing — nothing to recover.

        We detect "baseline already has the records" by checking
        whether the marker's promoted_event_ids are present in the
        baseline records, deduping by event_id.

        Returns the list of promoted event_ids that were finalized
        (across all recovered markers).  Idempotent.
        """
        markers = sorted(self._scry_dir.glob("commit-links.*.marker"))
        completed: list[EventId] = []

        for marker_path in markers:
            try:
                raw: object = json.loads(marker_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue

            if not isinstance(raw, dict):
                continue

            promoted_raw = raw.get("promoted_event_ids")
            if not isinstance(promoted_raw, list):
                continue
            promoted_event_ids: list[EventId] = [e for e in promoted_raw if isinstance(e, str)]

            overlay_rel = raw.get("overlay_path")
            if not isinstance(overlay_rel, str):
                continue
            overlay_path = self._scry_dir / overlay_rel

            # ── Step 1b recovery: re-append any promoted records to baseline
            # that aren't already there.  Look up by event_id in the
            # current baseline + overlay.
            baseline_records = self.read_records(self.baseline_path)
            baseline_event_ids = {r.event_id for r in baseline_records}
            missing_event_ids = [evt for evt in promoted_event_ids if evt not in baseline_event_ids]
            if missing_event_ids and overlay_path.exists():
                overlay_records = self.read_records(overlay_path)
                overlay_by_evt = {r.event_id: r for r in overlay_records}
                for evt in missing_event_ids:
                    rec = overlay_by_evt.get(evt)
                    if rec is not None:
                        self._do_append(self.baseline_path, rec)

            # Step 2: remove promoted records from overlay (if still present).
            if overlay_path.exists():
                overlay_records = self.read_records(overlay_path)
                promoted_set: set[str] = set(promoted_event_ids)
                remaining = [r for r in overlay_records if r.event_id not in promoted_set]
                if len(remaining) != len(overlay_records):
                    _atomic_rewrite(overlay_path, remaining)
                    _fsync_dir(overlay_path.parent)

            # Step 3: delete marker.
            marker_path.unlink(missing_ok=True)
            _fsync_dir(self._scry_dir)
            completed.extend(promoted_event_ids)

        return completed

    # ── Internal ──────────────────────────────────────────────────────────────

    def _validate_record(
        self,
        record: LinkRecord,
        same_file: list[LinkRecord],
        *,
        cross_file: list[LinkRecord] | None,
    ) -> None:
        """Validate ``record`` against §3.5.2 rules.

        ``same_file``:  existing records in the file being appended to.
        ``cross_file``: baseline records when appending to overlay; ``None``
                        when appending to baseline.
        """
        # Last record per link_id within each source.
        file_last: dict[LinkId, LinkRecord] = {}
        for r in same_file:
            file_last[r.link_id] = r

        cross_last: dict[LinkId, LinkRecord] = {}
        if cross_file:
            for r in cross_file:
                cross_last[r.link_id] = r

        # Universe of event_ids available for supersedes references.
        known_evt: set[EventId] = {r.event_id for r in same_file}
        if cross_file:
            known_evt |= {r.event_id for r in cross_file}

        link_id = record.link_id

        if record.op == LinkOp.UPSERT:
            if link_id in file_last:
                last = file_last[link_id]
                if last.op == LinkOp.DELETE:
                    # Rule 3: tombstones are absorbing within the same file.
                    raise LinkValidationError(
                        f"Rule 3: upsert for link_id {link_id!r} follows a delete"
                        " in the same file — tombstones are absorbing (§3.5.2)."
                    )
                # last.op == UPSERT → Rule 5: supersedes required.
                if record.supersedes is None:
                    raise LinkValidationError(
                        f"Rule 5: upsert for existing link_id {link_id!r}"
                        " requires supersedes (§3.5.2)."
                    )
            elif link_id in cross_last:
                # Link exists in the other file (baseline) but not in this one.
                # Cross-file revival of a delete is allowed (rule 4); in all
                # cross-file cases supersedes is required (rule 5).
                if record.supersedes is None:
                    raise LinkValidationError(
                        f"Rule 5: upsert for link_id {link_id!r} (exists in"
                        " baseline) requires supersedes (§3.5.2)."
                    )
            # If supersedes is set, the target event_id must be known.
            if record.supersedes is not None and record.supersedes not in known_evt:
                raise LinkValidationError(
                    f"supersedes references unknown event_id {record.supersedes!r} (§3.5.2 rule 5)."
                )

        elif record.op == LinkOp.DELETE:
            # supersedes is required on every delete (§3.5.1).
            if record.supersedes is None:
                raise LinkValidationError(
                    f"Delete for link_id {link_id!r} requires supersedes (§3.5.1)."
                )
            # link_id must have at least one existing record in the combined context.
            if link_id not in file_last and link_id not in cross_last:
                raise LinkValidationError(
                    f"Delete for link_id {link_id!r} which has no existing record."
                )
            # The referenced event_id must be known.
            if record.supersedes not in known_evt:
                raise LinkValidationError(
                    f"Delete supersedes references unknown event_id"
                    f" {record.supersedes!r} (§3.5.2 rule 5)."
                )
            # UAT-9 review-u14a HIGH: compare-and-swap — supersedes must
            # reference the CURRENT latest event for this link_id, AND
            # the current latest must NOT itself be a DELETE.  Without
            # this, two concurrent CLI/MCP writers can both tombstone
            # the same link with stale supersedes pointers, or one
            # process can tombstone a link that another process has
            # since refreshed (silent re-deletion of the new state).
            current_latest = file_last.get(link_id) or cross_last.get(link_id)
            if current_latest is not None:
                if current_latest.op == LinkOp.DELETE:
                    raise LinkValidationError(
                        f"Delete for link_id {link_id!r} but link is "
                        f"already tombstoned (current event_id="
                        f"{current_latest.event_id!r})."
                    )
                if record.supersedes != current_latest.event_id:
                    raise LinkValidationError(
                        f"Delete for link_id {link_id!r} has stale "
                        f"supersedes={record.supersedes!r}; current latest "
                        f"event is {current_latest.event_id!r}.  "
                        f"Re-read the active table and retry."
                    )

    def _find_merge_conflicts(self, all_records: list[LinkRecord]) -> list[LinkId]:
        """Detect broken supersedes chains per §3.5.2 rule 6.

        Returns a sorted list of link_ids with broken chains:
        - **Condition 1**: two upserts with DISTINCT event_ids share the
          same supersedes target (forked chain — typically the result
          of a git union merge).
        - **Condition 2**: any record whose supersedes references an
          event_id not present in baseline ⊕ overlay.

        Records are deduplicated by ``event_id`` before condition 1
        is evaluated (review-w2b HIGH fix). Without this dedupe, the
        same logical record appearing in both baseline and overlay
        (the post-state of a §3.5.4 step-1-partial-failure crash, or
        a concurrent-promotion edge case) would be counted twice and
        spuriously flagged as a fork.
        """
        known_evt: set[EventId] = {r.event_id for r in all_records}
        conflict_ids: set[LinkId] = set()

        # Condition 1: count how many DISTINCT upserts share the same
        # supersedes target.  Dedupe by event_id (a record present in
        # both baseline and overlay is the SAME logical record, not a
        # fork).
        seen_evt: set[EventId] = set()
        supersedes_to_links: dict[EventId, list[LinkId]] = defaultdict(list)
        for r in all_records:
            if r.op == LinkOp.UPSERT and r.supersedes is not None:
                if r.event_id in seen_evt:
                    continue
                seen_evt.add(r.event_id)
                supersedes_to_links[r.supersedes].append(r.link_id)

        for link_ids_for_evt in supersedes_to_links.values():
            if len(link_ids_for_evt) >= 2:
                conflict_ids.update(link_ids_for_evt)

        # Condition 2: supersedes references an unknown event_id.
        for r in all_records:
            if r.supersedes is not None and r.supersedes not in known_evt:
                conflict_ids.add(r.link_id)

        return sorted(conflict_ids)

    def _do_append(self, path: Path, record: LinkRecord) -> None:
        """Append ``record`` as a JSONL line to ``path``, fsync, with cross-process lock.

        Per DESIGN.md §10, only the leader writes to baseline/overlay in
        production.  This method nevertheless takes an OS-level exclusive
        lock around the append+flush+fsync window so that direct CLI calls
        from multiple processes (or test scenarios that exercise
        concurrent appends) can never interleave bytes within a single
        record OR overwrite each other's writes.

        Locking strategy (sidecar lock file — review-w2b HIGH fix):
            We acquire a lock on a SEPARATE companion file ``<path>.lock``
            at byte 0.  The data file's EOF is irrelevant to the lock.
            * Linux/macOS: ``fcntl.flock`` is whole-file advisory; the
              sidecar approach also works directly on the data file but
              the sidecar gives us a uniform cross-platform contract.
            * Windows: ``msvcrt.locking`` is mandatory byte-range; using
              a sidecar fixed at byte 0 means the lock byte never moves
              with EOF, the locked range never overlaps user writes, and
              no PermissionError is raised on flush under contention.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with _exclusive_file_lock(lock_path), path.open("a", encoding="utf-8") as fh:
            fh.write(record.model_dump_json(by_alias=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())


# ─── Cross-process file lock helper ───────────────────────────────────


@contextlib.contextmanager
def _exclusive_file_lock(lock_path: Path, *, retry_seconds: float = 5.0) -> Iterator[None]:
    """Acquire an exclusive OS-level lock on ``<lock_path>`` for the block.

    Sidecar-file pattern: the lock is placed on a SEPARATE companion
    file, not the data file being written. This decouples the lock
    region from the data file's EOF — important on Windows where
    ``msvcrt.locking`` is mandatory and locking-at-EOF causes
    PermissionError on subsequent flushes if another process's
    write extends past the locked byte.

    * Linux/macOS: ``fcntl.flock(LOCK_EX)`` (whole-file advisory).
      Released automatically on file close + on process death.
    * Windows: ``msvcrt.locking(LK_NBLCK, 1)`` at byte 0 of the lock
      file with retry-with-backoff (default 5s budget). Released by
      explicit ``LK_UNLCK`` at the same byte 0; lock file is closed
      after release. No write ever extends the lock file beyond byte
      0, so the mandatory lock never collides with user data writes.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import msvcrt

        # Open the lock file in binary read+write+create; we never
        # write to it after creating it, but we need a valid byte at
        # offset 0 for msvcrt.locking to lock.
        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            # Ensure at least one byte exists so we can lock byte 0.
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)

            deadline = time.monotonic() + retry_seconds
            while True:
                try:
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise
                    time.sleep(0.01)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)
    else:
        import fcntl

        fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                yield
            finally:
                with contextlib.suppress(OSError):
                    fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            with contextlib.suppress(OSError):
                os.close(fd)
