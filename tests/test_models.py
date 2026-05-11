"""Unit tests for `scry.models` — the W0b Pydantic schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from scry.models import (
    DRIFT_PRECEDENCE,
    Anchor,
    AnchorLinkProjection,
    AnchorPacket,
    AnchorType,
    Config,
    DriftCounts,
    DriftStatus,
    DriftSummary,
    Frontmatter,
    IndexState,
    Link,
    LinkOp,
    LinkRecord,
    LinkType,
    SubChunk,
    TransitiveHashStatus,
    drift_winner,
    is_well_formed_anchor_id,
    new_event_id,
    new_idempotency_token,
    new_link_id,
)

# Valid sha256 prefix used everywhere.
H = "sha256:" + "a" * 64
H2 = "sha256:" + "b" * 64
H3 = "sha256:" + "c" * 64
H4 = "sha256:" + "d" * 64


# ───── ID generators ─────────────────────────────────────────────────


def test_new_link_id_format() -> None:
    lid = new_link_id()
    assert lid.startswith("lnk_")
    assert len(lid) > len("lnk_")


def test_new_event_id_format() -> None:
    eid = new_event_id()
    assert eid.startswith("evt_")


def test_new_idempotency_token_format() -> None:
    tok = new_idempotency_token()
    assert tok.startswith("tok_")


def test_anchor_id_well_formed() -> None:
    assert is_well_formed_anchor_id("docs/POLICY_ENGINE.md::policy-engine::rule-structure")
    assert is_well_formed_anchor_id("python/hailstone/policy/engine.py:PolicyRule")
    assert not is_well_formed_anchor_id("with spaces")


# ───── Anchor model ──────────────────────────────────────────────────


def _section_anchor(**overrides: object) -> Anchor:
    base = {
        "id": "docs/spec.md::heading::sub",
        "type": AnchorType.SECTION,
        "path": "docs/spec.md",
        "heading_path": ["Heading", "Sub"],
        "content_text": "Hello world",
        "content_hash": H,
        "fingerprint_simhash": 12345,
    }
    base.update(overrides)  # type: ignore[arg-type]
    return Anchor(**base)  # type: ignore[arg-type]


def test_anchor_section_happy_path() -> None:
    a = _section_anchor()
    assert a.type == AnchorType.SECTION.value
    assert a.transitive_hash_status is None


def test_anchor_path_must_be_relative() -> None:
    with pytest.raises(ValidationError, match="must be repo-relative"):
        _section_anchor(path="C:/abs/path.md")
    with pytest.raises(ValidationError, match="must be repo-relative"):
        _section_anchor(path="/abs/unix/path.md")


def test_anchor_path_normalizes_backslashes() -> None:
    a = _section_anchor(path="docs\\spec.md")
    assert a.path == "docs/spec.md"


def test_anchor_transitive_hash_status_only_on_code() -> None:
    with pytest.raises(ValidationError, match="only be set on CODE"):
        _section_anchor(transitive_hash_status=TransitiveHashStatus.COMPLETE)
    code = Anchor(
        id="src/foo.py:Bar",
        type=AnchorType.CODE,
        path="src/foo.py",
        symbol_name="Bar",
        content_text="class Bar: ...",
        content_hash=H,
        fingerprint_simhash=42,
        transitive_hash_status=TransitiveHashStatus.COMPLETE,
    )
    assert code.transitive_hash_status == TransitiveHashStatus.COMPLETE.value


def test_content_hash_format_enforced() -> None:
    with pytest.raises(ValidationError):
        _section_anchor(content_hash="not-a-hash")
    with pytest.raises(ValidationError):
        _section_anchor(content_hash="sha256:short")


# ───── SubChunk ──────────────────────────────────────────────────────


def test_subchunk_happy_path() -> None:
    sc = SubChunk(
        parent_id="docs/x.md::a",
        chunk_index=0,
        text="hello",
        parent_content_hash=H,
    )
    assert sc.overlap_with_prev == 0


def test_subchunk_negative_index_rejected() -> None:
    with pytest.raises(ValidationError):
        SubChunk(parent_id="x", chunk_index=-1, text="t", parent_content_hash=H)


# ───── LinkRecord — upsert shape ─────────────────────────────────────


def _upsert(**overrides: object) -> LinkRecord:
    base: dict[str, object] = {
        "op": LinkOp.UPSERT,
        "link_id": "lnk_test",
    }
    base.update(
        {
            "from": "src/foo.py:Bar",
            "from_type": AnchorType.CODE,
            "to": "docs/spec.md::a",
            "to_type": AnchorType.SECTION,
            "type": LinkType.IMPLEMENTS,
            "from_content_hash": H,
            "to_content_hash": H2,
        }
    )
    base.update(overrides)
    return LinkRecord.model_validate(base)


def test_link_record_upsert_happy_path() -> None:
    r = _upsert()
    assert r.op == LinkOp.UPSERT.value
    assert r.event_id.startswith("evt_")
    assert r.from_ == "src/foo.py:Bar"


def test_link_record_upsert_missing_required_fields() -> None:
    with pytest.raises(ValidationError, match="missing required"):
        LinkRecord.model_validate({"op": "upsert", "link_id": "lnk_x"})


def test_link_record_upsert_with_reason_rejected() -> None:
    with pytest.raises(ValidationError, match="may not carry"):
        _upsert(reason="oops")


def test_link_record_with_supersedes() -> None:
    prior = new_event_id()
    r = _upsert(supersedes=prior)
    assert r.supersedes == prior


def test_link_record_with_prior_hashes() -> None:
    r = _upsert(
        prior_from_content_hash=H3,
        prior_to_content_hash=H4,
    )
    assert r.prior_from_content_hash == H3
    assert r.prior_to_content_hash == H4


# ───── LinkRecord — delete shape ─────────────────────────────────────


def test_link_record_delete_happy_path() -> None:
    r = LinkRecord(op=LinkOp.DELETE, link_id="lnk_x", reason="manual")
    assert r.op == LinkOp.DELETE.value


def test_link_record_delete_with_endpoints_rejected() -> None:
    with pytest.raises(ValidationError, match="must not carry"):
        LinkRecord.model_validate(
            {
                "op": "delete",
                "link_id": "lnk_x",
                "from": "a",
                "from_type": "code",
                "reason": "x",
            }
        )


# ───── LinkRecord — extra=forbid ─────────────────────────────────────


def test_link_record_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        LinkRecord.model_validate(
            {
                "op": "upsert",
                "link_id": "lnk_x",
                "from": "a",
                "from_type": "code",
                "to": "b",
                "to_type": "section",
                "type": "implements",
                "from_content_hash": H,
                "to_content_hash": H2,
                "bogus_field": True,
            }
        )


# ───── Link (active shape) ───────────────────────────────────────────


def test_link_active_shape() -> None:
    link = Link(
        link_id="lnk_y",
        from_id="src/foo.py:Bar",
        from_type=AnchorType.CODE,
        to_id="docs/spec.md::a",
        to_type=AnchorType.SECTION,
        type=LinkType.IMPLEMENTS,
        from_content_hash=H,
        to_content_hash=H2,
        last_event_id=new_event_id(),
    )
    assert link.type == LinkType.IMPLEMENTS.value


# ───── DriftStatus + precedence ──────────────────────────────────────


def test_drift_precedence_ordering() -> None:
    # merge-conflict tops the ladder per v3.1.
    assert DRIFT_PRECEDENCE[0] == DriftStatus.MERGE_CONFLICT
    # Fresh always last.
    assert DRIFT_PRECEDENCE[-1] == DriftStatus.FRESH


def test_drift_winner_picks_highest_precedence() -> None:
    assert drift_winner(DriftStatus.FRESH, DriftStatus.SPEC_CHANGED) == DriftStatus.SPEC_CHANGED
    assert (
        drift_winner(DriftStatus.BROKEN_SOURCE, DriftStatus.MERGE_CONFLICT)
        == DriftStatus.MERGE_CONFLICT
    )
    assert (
        drift_winner(DriftStatus.CODE_CHANGED, DriftStatus.DRIFT_UNKNOWN)
        == DriftStatus.CODE_CHANGED
    )


def test_drift_winner_empty_returns_fresh() -> None:
    assert drift_winner() == DriftStatus.FRESH


# ───── AnchorLinkProjection (transitive_hash_status omission) ─────────


def test_link_projection_omits_status_for_non_code_target() -> None:
    proj = AnchorLinkProjection(
        to="docs/spec.md::a",
        to_type=AnchorType.SECTION,
        type=LinkType.REFERENCES,
        drift_status=DriftStatus.FRESH,
        # NOT set:
        transitive_hash_status=None,
    )
    serialized = proj.serialize()
    assert "transitive_hash_status" not in serialized


def test_link_projection_keeps_status_for_code_target() -> None:
    proj = AnchorLinkProjection(
        to="src/foo.py:Bar",
        to_type=AnchorType.CODE,
        type=LinkType.IMPLEMENTS,
        drift_status=DriftStatus.FRESH,
        transitive_hash_status=TransitiveHashStatus.COMPLETE,
    )
    serialized = proj.serialize()
    assert serialized["transitive_hash_status"] == "complete"


def test_link_projection_semantic_drift_optional() -> None:
    proj = AnchorLinkProjection(
        to="src/foo.py:Bar",
        to_type=AnchorType.CODE,
        type=LinkType.MIRRORS,
        drift_status=DriftStatus.CODE_CHANGED,
        semantic_drift=True,
        transitive_hash_status=TransitiveHashStatus.COMPLETE,
    )
    assert proj.semantic_drift is True


# ───── AnchorPacket ──────────────────────────────────────────────────


def test_anchor_packet_defaults() -> None:
    a = _section_anchor()
    pkt = AnchorPacket(anchor=a, score=0.5)
    assert pkt.index_state == IndexState.FRESH.value
    assert pkt.content_truncated is False
    assert pkt.links == []


# ───── DriftSummary ──────────────────────────────────────────────────


def test_drift_summary_null_scores_for_empty_repo() -> None:
    s = DriftSummary(
        drift_score=None,
        coverage_score=None,
        counts=DriftCounts(),
        drift_coverage="section-only",
    )
    assert s.drift_score is None
    assert s.counts.total == 0


# ───── Config ────────────────────────────────────────────────────────


def test_config_defaults_round_trip() -> None:
    c = Config()
    assert c.embeddings.model == "BAAI/bge-small-en-v1.5"
    assert c.retrieval.fusion_rrf_k == 60
    assert c.drift.scoring.merge_conflict == 1.0
    # v3.1 invariants:
    assert c.code_anchors_extra.transitive_max_depth == 32
    assert c.index.poll_dirty is True
    assert c.retrieval.bm25.index_table_cells is True
    assert c.ipc.timeouts.short == 5.0
    assert c.ipc.idempotency_cache_size == 10_000


def test_config_rejects_unknown_top_level_keys() -> None:
    with pytest.raises(ValidationError):
        Config.model_validate({"unknown_top": 1})


def test_classify_entry_validation() -> None:
    c = Config(
        classify=[
            {"glob": "docs/**.md", "type": "spec"},
            {"glob": "**/*.md", "type": "doc"},
        ]  # type: ignore[arg-type]
    )
    assert len(c.classify) == 2


# ───── Frontmatter ───────────────────────────────────────────────────


def test_frontmatter_uses_skip_not_exclude() -> None:
    fm = Frontmatter(skip=True)
    assert fm.skip is True
    # Unknown fields silently ignored (not extra=forbid for frontmatter).
    fm2 = Frontmatter.model_validate({"skip": False, "extra_user_field": "ok"})
    assert fm2.skip is False

# uat-r5-5 pr-d noise
