"""Tests for scry.audit — semantic documentation audit."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scry.audit import (
    AuditConfig,
    AuditFinding,
    AuditResult,
    _parse_audit_batch,
    build_audit_payload,
    parse_agent_audit,
    run_audit,
    select_audit_pairs,
)
from scry.models import Anchor, AnchorType


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_anchor(
    *,
    id: str,
    type: str = AnchorType.SECTION.value,
    path: str = "docs/test.md",
    content: str = "test content",
    heading_path: list[str] | None = None,
    symbol_name: str | None = None,
) -> Anchor:
    return Anchor(
        id=id,
        type=type,
        path=path,
        content_text=content,
        content_hash="sha256:" + "a" * 64,
        fingerprint_simhash=0,
        heading_path=heading_path,
        symbol_name=symbol_name,
    )


# ─── Tests: _parse_audit_batch ────────────────────────────────────────────────


class TestParseAuditBatch:
    def test_valid_findings(self) -> None:
        raw = json.dumps({
            "findings": [
                {
                    "pair_id": "a_0",
                    "is_drifted": True,
                    "severity": "HIGH",
                    "doc_claim": "uses in-memory queue",
                    "code_reality": "uses Redis queue",
                    "suggestion": "update doc to say Redis",
                },
                {
                    "pair_id": "a_1",
                    "is_drifted": False,
                },
            ]
        })
        result = _parse_audit_batch(raw, ["a_0", "a_1"])
        assert len(result) == 1
        assert result[0]["pair_id"] == "a_0"
        assert result[0]["severity"] == "HIGH"

    def test_invalid_json(self) -> None:
        assert _parse_audit_batch("not json", ["a_0"]) == []

    def test_unknown_pair_id(self) -> None:
        raw = json.dumps({
            "findings": [
                {
                    "pair_id": "unknown",
                    "is_drifted": True,
                    "severity": "HIGH",
                    "doc_claim": "x",
                    "code_reality": "y",
                    "suggestion": "z",
                }
            ]
        })
        assert _parse_audit_batch(raw, ["a_0"]) == []

    def test_invalid_severity(self) -> None:
        raw = json.dumps({
            "findings": [
                {
                    "pair_id": "a_0",
                    "is_drifted": True,
                    "severity": "CRITICAL",
                    "doc_claim": "x",
                    "code_reality": "y",
                    "suggestion": "z",
                }
            ]
        })
        assert _parse_audit_batch(raw, ["a_0"]) == []


# ─── Tests: build_audit_payload ───────────────────────────────────────────────


class TestBuildAuditPayload:
    def test_produces_valid_payload(self) -> None:
        doc = _make_anchor(id="doc::test", path="docs/arch.md", content="doc content",
                           heading_path=["Architecture", "Queue"])
        code = _make_anchor(id="src/queue.py::QueueClient", type=AnchorType.CODE.value,
                            path="src/queue.py", content="class QueueClient: ...",
                            symbol_name="QueueClient")
        pairs = [(doc, [code])]
        config = AuditConfig()
        payload = build_audit_payload(pairs, config)

        assert "system_prompt" in payload
        assert "schema" in payload
        assert "pairs" in payload
        assert len(payload["pairs"]) == 1
        assert payload["pairs"][0]["pair_id"] == "a_0"
        assert payload["pairs"][0]["doc"]["path"] == "docs/arch.md"
        assert payload["pairs"][0]["code"][0]["path"] == "src/queue.py"

    def test_empty_pairs(self) -> None:
        payload = build_audit_payload([], AuditConfig())
        assert payload["_count"] == 0
        assert payload["pairs"] == []


# ─── Tests: parse_agent_audit ─────────────────────────────────────────────────


class TestParseAgentAudit:
    def test_parses_valid_response(self) -> None:
        doc = _make_anchor(id="doc::overview", path="docs/overview.md",
                           heading_path=["Overview"])
        code = _make_anchor(id="src/main.py::main", type=AnchorType.CODE.value,
                            path="src/main.py", symbol_name="main")
        pairs = [(doc, [code])]

        response = {
            "findings": [
                {
                    "pair_id": "a_0",
                    "is_drifted": True,
                    "severity": "HIGH",
                    "doc_claim": "uses SQLite",
                    "code_reality": "uses PostgreSQL",
                    "suggestion": "update to PostgreSQL",
                }
            ]
        }
        findings = parse_agent_audit(response, pairs=pairs)
        assert len(findings) == 1
        assert findings[0].severity == "HIGH"
        assert findings[0].doc_path == "docs/overview.md"
        assert findings[0].code_path == "src/main.py"

    def test_filters_non_drifted(self) -> None:
        doc = _make_anchor(id="doc::test", path="docs/test.md")
        code = _make_anchor(id="src/t.py::f", type=AnchorType.CODE.value, path="src/t.py")
        pairs = [(doc, [code])]

        response = {
            "findings": [
                {"pair_id": "a_0", "is_drifted": False}
            ]
        }
        findings = parse_agent_audit(response, pairs=pairs)
        assert len(findings) == 0

    def test_sorts_by_severity(self) -> None:
        pairs = []
        findings_data = {"findings": []}
        for i, sev in enumerate(["LOW", "HIGH", "MEDIUM"]):
            doc = _make_anchor(id=f"doc::{i}", path=f"docs/{i}.md")
            code = _make_anchor(id=f"src/{i}.py::f", type=AnchorType.CODE.value,
                                path=f"src/{i}.py")
            pairs.append((doc, [code]))
            findings_data["findings"].append({
                "pair_id": f"a_{i}",
                "is_drifted": True,
                "severity": sev,
                "doc_claim": "x",
                "code_reality": "y",
                "suggestion": "z",
            })

        findings = parse_agent_audit(findings_data, pairs=pairs)
        assert [f.severity for f in findings] == ["HIGH", "MEDIUM", "LOW"]
