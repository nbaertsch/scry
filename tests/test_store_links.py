"""Tests for scry.store.links — JSONL reader/writer and replay engine.

Covers all §3.5.2 replay rules, §3.5.4 promotion protocol, crash-recovery,
JSONL serialisation round-trips, and concurrent baseline appends.
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from scry.models import (
    AnchorType,
    LinkId,
    LinkOp,
    LinkRecord,
    LinkType,
    new_event_id,
    new_link_id,
)
from scry.store.links import LinkStore, LinkValidationError, ReplayResult

# ─── Shared hash constants ────────────────────────────────────────────────────

_H_A = "sha256:" + "a" * 64
_H_B = "sha256:" + "b" * 64
_H_C = "sha256:" + "c" * 64

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
    """Build a minimal valid UPSERT LinkRecord (uses model_validate to handle 'from' alias)."""
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


def _delete(
    link_id: str,
    *,
    event_id: str | None = None,
    supersedes: str,
) -> LinkRecord:
    """Build a minimal valid DELETE LinkRecord."""
    eid: str = event_id or new_event_id()
    return LinkRecord.model_validate(
        {
            "op": "delete",
            "link_id": link_id,
            "event_id": eid,
            "supersedes": supersedes,
            "reason": "test-deletion",
        }
    )


# ─── Tests: read_records ──────────────────────────────────────────────────────


class TestReadRecords:
    def test_nonexistent_file_returns_empty(self, tmp_repo: Path) -> None:
        """read_records on a missing file returns []."""
        store = LinkStore(tmp_repo)
        assert store.read_records(store.baseline_path) == []

    def test_reads_records_in_document_order(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_order_a")
        r2 = _upsert("lnk_order_b")
        store.append_baseline(r1)
        store.append_baseline(r2)
        records = store.read_records(store.baseline_path)
        assert len(records) == 2
        assert records[0].link_id == "lnk_order_a"
        assert records[1].link_id == "lnk_order_b"

    def test_malformed_line_raises_with_line_number(self, tmp_repo: Path) -> None:
        """Malformed JSONL raises LinkValidationError with the 1-based line number."""
        store = LinkStore(tmp_repo)
        r = _upsert("lnk_ok")
        store.baseline_path.write_text(
            r.model_dump_json(by_alias=True) + "\nnot valid json\n",
            encoding="utf-8",
        )
        with pytest.raises(LinkValidationError, match="line 2"):
            store.read_records(store.baseline_path)

    def test_empty_lines_are_skipped(self, tmp_repo: Path) -> None:
        """Blank lines in the JSONL file are silently ignored."""
        store = LinkStore(tmp_repo)
        r = _upsert("lnk_skip")
        store.baseline_path.write_text(
            "\n" + r.model_dump_json(by_alias=True) + "\n\n",
            encoding="utf-8",
        )
        records = store.read_records(store.baseline_path)
        assert len(records) == 1

    def test_read_from_overlay_path(self, tmp_repo: Path) -> None:
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        store = LinkStore(tmp_repo)
        r = _upsert("lnk_ov_read")
        store.append_overlay(r, overlay)
        records = store.read_records(overlay)
        assert len(records) == 1
        assert records[0].event_id == r.event_id


# ─── Tests: append_baseline ───────────────────────────────────────────────────


class TestAppendBaseline:
    def test_happy_path_single_upsert(self, tmp_repo: Path) -> None:
        """Single upsert appended then read back correctly."""
        store = LinkStore(tmp_repo)
        r = _upsert("lnk_happy")
        store.append_baseline(r)
        records = store.read_records(store.baseline_path)
        assert len(records) == 1
        assert records[0].link_id == "lnk_happy"
        assert records[0].event_id == r.event_id

    def test_rejects_duplicate_link_id_without_supersedes(self, tmp_repo: Path) -> None:
        """Rule 5: duplicate link_id without supersedes → LinkValidationError."""
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_dup5")
        store.append_baseline(r1)
        r2 = _upsert("lnk_dup5")  # same link_id, no supersedes
        with pytest.raises(LinkValidationError, match="Rule 5"):
            store.append_baseline(r2)

    def test_accepts_update_with_correct_supersedes(self, tmp_repo: Path) -> None:
        """Rule 5 satisfied: supersedes present on update → OK."""
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_upd")
        store.append_baseline(r1)
        r2 = _upsert("lnk_upd", supersedes=r1.event_id)
        store.append_baseline(r2)
        records = store.read_records(store.baseline_path)
        assert len(records) == 2
        assert records[1].supersedes == r1.event_id

    def test_rejects_upsert_after_delete_rule3(self, tmp_repo: Path) -> None:
        """Rule 3: once a delete appears in a file, a subsequent upsert for the
        same link_id in the SAME file is a write-time validation error."""
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_r3")
        store.append_baseline(r1)
        d1 = _delete("lnk_r3", supersedes=r1.event_id)
        store.append_baseline(d1)
        # Attempt to revive within baseline (same file) — must fail.
        r2 = _upsert("lnk_r3", supersedes=d1.event_id)
        with pytest.raises(LinkValidationError, match="Rule 3"):
            store.append_baseline(r2)

    def test_rejects_delete_without_supersedes(self, tmp_repo: Path) -> None:
        """Deletes always require supersedes per §3.5.1."""
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_del_nosup")
        store.append_baseline(r1)
        # Build a delete without supersedes (model-level validation allows it;
        # store-level validation must reject it).
        bad = LinkRecord.model_validate(
            {"op": "delete", "link_id": "lnk_del_nosup", "reason": "test"}
        )
        with pytest.raises(LinkValidationError, match="supersedes"):
            store.append_baseline(bad)

    def test_rejects_upsert_with_unknown_supersedes_event_id(self, tmp_repo: Path) -> None:
        """supersedes must reference a known event_id (§3.5.2 rule 5)."""
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_unk_sup")
        store.append_baseline(r1)
        r2 = _upsert("lnk_unk_sup", supersedes="evt_" + "0" * 32)
        with pytest.raises(LinkValidationError, match="unknown event_id"):
            store.append_baseline(r2)

    def test_rejects_delete_nonexistent_link(self, tmp_repo: Path) -> None:
        """Deleting a link_id that has no existing record is an error."""
        store = LinkStore(tmp_repo)
        r_other = _upsert("lnk_other_del")
        store.append_baseline(r_other)
        bad = _delete("lnk_ghost_del", supersedes=r_other.event_id)
        with pytest.raises(LinkValidationError, match="no existing record"):
            store.append_baseline(bad)


# ─── Tests: append_overlay ────────────────────────────────────────────────────


class TestAppendOverlay:
    def test_overlay_new_link_happy_path(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r = _upsert("lnk_ov_new")
        store.append_overlay(r, overlay)
        records = store.read_records(overlay)
        assert len(records) == 1
        assert records[0].link_id == "lnk_ov_new"

    def test_cross_file_revival_with_supersedes_accepted(self, tmp_repo: Path) -> None:
        """Rule 4: overlay upsert after baseline delete with correct supersedes → OK."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_rv")
        store.append_baseline(r1)
        d1 = _delete("lnk_rv", supersedes=r1.event_id)
        store.append_baseline(d1)
        # Revival in overlay pointing at the baseline tombstone.
        revival = _upsert("lnk_rv", supersedes=d1.event_id)
        store.append_overlay(revival, overlay)
        records = store.read_records(overlay)
        assert len(records) == 1
        assert records[0].supersedes == d1.event_id

    def test_cross_file_revival_without_supersedes_rejected(self, tmp_repo: Path) -> None:
        """Rule 4 + 5: revival MUST carry supersedes; missing → error."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_norv")
        store.append_baseline(r1)
        d1 = _delete("lnk_norv", supersedes=r1.event_id)
        store.append_baseline(d1)
        bad_revival = _upsert("lnk_norv")  # no supersedes
        with pytest.raises(LinkValidationError, match="Rule 5"):
            store.append_overlay(bad_revival, overlay)

    def test_rule3_applies_within_overlay(self, tmp_repo: Path) -> None:
        """Rule 3 is not file-type-specific: upsert after delete in overlay → error."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_ov_r3")
        store.append_overlay(r1, overlay)
        d1 = _delete("lnk_ov_r3", supersedes=r1.event_id)
        store.append_overlay(d1, overlay)
        r2 = _upsert("lnk_ov_r3", supersedes=d1.event_id)
        with pytest.raises(LinkValidationError, match="Rule 3"):
            store.append_overlay(r2, overlay)

    def test_overlay_update_for_baseline_link_requires_supersedes(self, tmp_repo: Path) -> None:
        """Rule 5: overlay upsert for a link_id that lives in baseline needs supersedes."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_bl_nosup")
        store.append_baseline(r1)
        r2 = _upsert("lnk_bl_nosup")  # no supersedes
        with pytest.raises(LinkValidationError, match="Rule 5"):
            store.append_overlay(r2, overlay)


# ─── Tests: replay ────────────────────────────────────────────────────────────


class TestReplay:
    def test_empty_stores_give_empty_result(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        result = store.replay()
        assert isinstance(result, ReplayResult)
        assert result.active_links == {}
        assert result.merge_conflicts == []

    def test_baseline_only_single_upsert(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        r = _upsert("lnk_base_single")
        store.append_baseline(r)
        result = store.replay()
        assert "lnk_base_single" in result.active_links
        link = result.active_links["lnk_base_single"]
        assert link.last_event_id == r.event_id

    def test_overlay_wins_over_baseline(self, tmp_repo: Path) -> None:
        """Rule 2: overlay record for same link_id supersedes baseline record."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_ov_wins")
        store.append_baseline(r1)
        r2 = _upsert("lnk_ov_wins", supersedes=r1.event_id, to="src/other.py:func")
        store.append_overlay(r2, overlay)
        result = store.replay(overlay_path=overlay)
        assert "lnk_ov_wins" in result.active_links
        assert result.active_links["lnk_ov_wins"].last_event_id == r2.event_id

    def test_tombstone_replay_link_absent(self, tmp_repo: Path) -> None:
        """Baseline upsert + overlay delete → link absent from active_links."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_tomb")
        store.append_baseline(r1)
        d1 = _delete("lnk_tomb", supersedes=r1.event_id)
        store.append_overlay(d1, overlay)
        result = store.replay(overlay_path=overlay)
        assert "lnk_tomb" not in result.active_links

    def test_cross_file_revival_link_present(self, tmp_repo: Path) -> None:
        """Baseline delete + overlay upsert (with supersedes) → link present."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_rev_rep")
        store.append_baseline(r1)
        d1 = _delete("lnk_rev_rep", supersedes=r1.event_id)
        store.append_baseline(d1)
        revival = _upsert("lnk_rev_rep", supersedes=d1.event_id)
        store.append_overlay(revival, overlay)
        result = store.replay(overlay_path=overlay)
        assert "lnk_rev_rep" in result.active_links
        assert result.active_links["lnk_rev_rep"].last_event_id == revival.event_id

    def test_merge_conflict_duplicate_supersedes(self, tmp_repo: Path) -> None:
        """Rule 6 condition 1: two upserts share the same supersedes → conflict.

        Simulates a post-git-merge state by force-writing two records that
        both reference the same prior event_id (what would happen after
        `git merge` with the union driver).
        """
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_fork")
        store.append_baseline(r1)
        # Two branches each created an independent update; after union merge,
        # both appear in baseline with the same supersedes → fork.
        r2 = _upsert("lnk_fork", supersedes=r1.event_id)
        r3 = _upsert("lnk_fork", supersedes=r1.event_id)
        with store.baseline_path.open("a", encoding="utf-8") as fh:
            fh.write(r2.model_dump_json(by_alias=True) + "\n")
            fh.write(r3.model_dump_json(by_alias=True) + "\n")
        result = store.replay()
        assert "lnk_fork" in result.merge_conflicts

    def test_merge_conflict_unknown_supersedes_event_id(self, tmp_repo: Path) -> None:
        """Rule 6 condition 2: supersedes references unknown event_id → conflict."""
        store = LinkStore(tmp_repo)
        ghost_evt = "evt_" + "f" * 32
        r = _upsert("lnk_ghost", supersedes=ghost_evt)
        # Force-write to bypass write-time validation.
        with store.baseline_path.open("a", encoding="utf-8") as fh:
            fh.write(r.model_dump_json(by_alias=True) + "\n")
        result = store.replay()
        assert "lnk_ghost" in result.merge_conflicts

    def test_clean_supersedes_chain_has_no_conflicts(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_chain_ok")
        store.append_baseline(r1)
        r2 = _upsert("lnk_chain_ok", supersedes=r1.event_id)
        store.append_baseline(r2)
        result = store.replay()
        assert "lnk_chain_ok" not in result.merge_conflicts
        assert "lnk_chain_ok" in result.active_links

    def test_baseline_delete_within_baseline_only_replay(self, tmp_repo: Path) -> None:
        """Baseline upsert then baseline delete → absent even with no overlay."""
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_bl_del")
        store.append_baseline(r1)
        d1 = _delete("lnk_bl_del", supersedes=r1.event_id)
        store.append_baseline(d1)
        result = store.replay()
        assert "lnk_bl_del" not in result.active_links


# ─── Tests: JSONL serialisation round-trip ────────────────────────────────────


class TestJsonlRoundTrip:
    def test_upsert_round_trips_through_pydantic(self, tmp_repo: Path) -> None:
        """model_dump_json(by_alias=True) → model_validate_json preserves all fields."""
        store = LinkStore(tmp_repo)
        original = _upsert("lnk_rt_up", from_="docs/api.md::auth", to="src/auth.py:login")
        store.append_baseline(original)
        records = store.read_records(store.baseline_path)
        assert len(records) == 1
        rt = records[0]
        assert rt.link_id == original.link_id
        assert rt.event_id == original.event_id
        assert rt.from_ == original.from_
        assert rt.to == original.to
        assert rt.ts == original.ts

    def test_delete_round_trips_through_pydantic(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_rt_del")
        store.append_baseline(r1)
        d = _delete("lnk_rt_del", supersedes=r1.event_id)
        store.append_baseline(d)
        records = store.read_records(store.baseline_path)
        assert records[1].op == LinkOp.DELETE
        assert records[1].supersedes == r1.event_id
        assert records[1].reason == "test-deletion"

    def test_from_alias_serialised_as_from_not_from_underscore(self, tmp_repo: Path) -> None:
        """The 'from' alias (not 'from_') must appear in the serialised JSONL."""
        store = LinkStore(tmp_repo)
        r = _upsert("lnk_alias_chk")
        store.append_baseline(r)
        raw = store.baseline_path.read_text(encoding="utf-8").strip()
        data = json.loads(raw)
        assert "from" in data
        assert "from_" not in data


# ─── Tests: promote_overlay_to_baseline ───────────────────────────────────────


class TestPromoteOverlayToBaseline:
    def test_promote_all_records(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_prom1")
        r2 = _upsert("lnk_prom2")
        store.append_overlay(r1, overlay)
        store.append_overlay(r2, overlay)
        promoted = store.promote_overlay_to_baseline(overlay)
        assert set(promoted) == {r1.event_id, r2.event_id}
        # Both records now in baseline.
        baseline_ids = {r.event_id for r in store.read_records(store.baseline_path)}
        assert r1.event_id in baseline_ids
        assert r2.event_id in baseline_ids
        # Overlay is now empty.
        assert store.read_records(overlay) == []

    def test_promote_subset_by_event_ids(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_sub1")
        r2 = _upsert("lnk_sub2")
        store.append_overlay(r1, overlay)
        store.append_overlay(r2, overlay)
        promoted = store.promote_overlay_to_baseline(overlay, event_ids=[r1.event_id])
        assert promoted == [r1.event_id]
        baseline_ids = {r.event_id for r in store.read_records(store.baseline_path)}
        assert r1.event_id in baseline_ids
        assert r2.event_id not in baseline_ids
        # r2 remains in overlay.
        overlay_ids = {r.event_id for r in store.read_records(overlay)}
        assert r2.event_id in overlay_ids

    def test_promote_missing_event_id_raises(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_miss_prom")
        store.append_overlay(r1, overlay)
        ghost = "evt_" + "9" * 32
        with pytest.raises(LinkValidationError, match="not found in overlay"):
            store.promote_overlay_to_baseline(overlay, event_ids=[ghost])

    def test_promote_empty_overlay_returns_empty_list(self, tmp_repo: Path) -> None:
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        overlay.touch()
        assert store.promote_overlay_to_baseline(overlay) == []

    def test_no_marker_file_after_successful_promotion(self, tmp_repo: Path) -> None:
        """The marker is deleted at step 3; no stale markers left after success."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        store.append_overlay(_upsert("lnk_no_marker"), overlay)
        store.promote_overlay_to_baseline(overlay)
        markers = list((tmp_repo / ".scry").glob("commit-links.*.marker"))
        assert markers == []

    def test_promote_is_reflected_in_replay(self, tmp_repo: Path) -> None:
        """Promoted records appear in replay() without needing an overlay path."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r = _upsert("lnk_prom_replay")
        store.append_overlay(r, overlay)
        store.promote_overlay_to_baseline(overlay)
        result = store.replay()
        assert "lnk_prom_replay" in result.active_links


# ─── Tests: crash recovery ────────────────────────────────────────────────────


class TestCrashRecovery:
    def test_no_markers_returns_empty(self, tmp_repo: Path) -> None:
        """Idempotent: no markers → returns []."""
        store = LinkStore(tmp_repo)
        assert store.recover_pending_promotions() == []

    def test_recover_completes_step2_after_crash_between_step1_and_step2(
        self, tmp_repo: Path
    ) -> None:
        """Crash between step 1 (baseline + marker written) and step 2 (overlay
        rewrite): recover_pending_promotions finishes step 2 + 3."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_crash_rec")
        store.append_overlay(r1, overlay)

        # Simulate step 1: append r1 to baseline.
        with store.baseline_path.open("a", encoding="utf-8") as fh:
            fh.write(r1.model_dump_json(by_alias=True) + "\n")

        # Write the marker (as promote_overlay_to_baseline would after step 1).
        txn_id = str(uuid.uuid4())
        marker_path = tmp_repo / ".scry" / f"commit-links.{txn_id}.marker"
        overlay_rel = overlay.relative_to(tmp_repo / ".scry")
        marker_data = {
            "txn_id": txn_id,
            "overlay_path": overlay_rel.as_posix(),
            "promoted_event_ids": [r1.event_id],
            "ts": "2026-01-01T00:00:00+00:00",
        }
        marker_path.write_text(json.dumps(marker_data), encoding="utf-8")

        # Crash happened here — overlay still has r1, marker still exists.
        completed = store.recover_pending_promotions()

        assert r1.event_id in completed
        assert store.read_records(overlay) == []  # step 2 done
        assert not marker_path.exists()  # step 3 done

    def test_recover_idempotent_no_markers(self, tmp_repo: Path) -> None:
        """Running recover twice in a row with no markers returns [] both times."""
        store = LinkStore(tmp_repo)
        assert store.recover_pending_promotions() == []
        assert store.recover_pending_promotions() == []

    def test_recover_with_already_removed_overlay_records(self, tmp_repo: Path) -> None:
        """If overlay was already rewritten (step 2 done) the marker is still
        cleaned up without error (idempotent step 3)."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "feat.jsonl"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay.touch()  # empty overlay — step 2 already completed

        txn_id = str(uuid.uuid4())
        marker_path = tmp_repo / ".scry" / f"commit-links.{txn_id}.marker"
        overlay_rel = overlay.relative_to(tmp_repo / ".scry")
        marker_data = {
            "txn_id": txn_id,
            "overlay_path": overlay_rel.as_posix(),
            "promoted_event_ids": ["evt_" + "a" * 32],
            "ts": "2026-01-01T00:00:00+00:00",
        }
        marker_path.write_text(json.dumps(marker_data), encoding="utf-8")

        store.recover_pending_promotions()
        assert not marker_path.exists()

    def test_recover_then_replay_is_consistent(self, tmp_repo: Path) -> None:
        """After crash-recovery, replay reflects the promoted records correctly."""
        store = LinkStore(tmp_repo)
        overlay = tmp_repo / ".scry" / "overlays" / "main.jsonl"
        r1 = _upsert("lnk_rec_replay")
        store.append_overlay(r1, overlay)

        # Simulate crash after step 1.
        with store.baseline_path.open("a", encoding="utf-8") as fh:
            fh.write(r1.model_dump_json(by_alias=True) + "\n")
        txn_id = str(uuid.uuid4())
        marker_path = tmp_repo / ".scry" / f"commit-links.{txn_id}.marker"
        overlay_rel = overlay.relative_to(tmp_repo / ".scry")
        marker_path.write_text(
            json.dumps(
                {
                    "txn_id": txn_id,
                    "overlay_path": overlay_rel.as_posix(),
                    "promoted_event_ids": [r1.event_id],
                    "ts": "2026-01-01T00:00:00+00:00",
                }
            ),
            encoding="utf-8",
        )

        store.recover_pending_promotions()
        result = store.replay()
        assert "lnk_rec_replay" in result.active_links


# ─── Tests: concurrent appends ────────────────────────────────────────────────


class TestConcurrentAppends:
    def test_two_processes_produce_valid_union(self, tmp_repo: Path) -> None:
        """Two subprocesses append different records concurrently.
        Both records must be present in the baseline after joining,
        verifying OS-level append atomicity (§3.5.3 union merge driver rationale).
        """
        store = LinkStore(tmp_repo)
        r1 = _upsert("lnk_mp_proc1")
        r2 = _upsert("lnk_mp_proc2")

        # One-liner script run in each subprocess; scry is installed in the venv.
        script = (
            "from pathlib import Path;"
            "from scry.models import LinkRecord;"
            "from scry.store.links import LinkStore;"
            "import sys;"
            "store = LinkStore(Path(sys.argv[1]));"
            "store.append_baseline(LinkRecord.model_validate_json(sys.argv[2]))"
        )

        p1 = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_repo), r1.model_dump_json(by_alias=True)]
        )
        p2 = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_repo), r2.model_dump_json(by_alias=True)]
        )
        rc1 = p1.wait(timeout=60)
        rc2 = p2.wait(timeout=60)

        assert rc1 == 0, f"Subprocess 1 exited with code {rc1}"
        assert rc2 == 0, f"Subprocess 2 exited with code {rc2}"

        records = store.read_records(store.baseline_path)
        link_ids: set[LinkId] = {r.link_id for r in records}
        assert "lnk_mp_proc1" in link_ids
        assert "lnk_mp_proc2" in link_ids


# uat-r5-5 pr-d noise
