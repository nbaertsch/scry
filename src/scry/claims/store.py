"""Persistence layer for claims, dependencies, and cached verdicts.

Uses the existing scry SQLite database (vectors.db) with additional
tables for claim verification data.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scry.claims.model import (
    Claim,
    ClaimDependency,
    ClaimType,
    DependencyKind,
    DocSpan,
    Evidence,
    Severity,
    Verdict,
    VerificationResult,
)

_CLAIMS_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    id              TEXT PRIMARY KEY,
    doc_path        TEXT NOT NULL,
    section         TEXT NOT NULL DEFAULT '',
    span_start      INTEGER NOT NULL,
    span_end        INTEGER NOT NULL,
    raw_text        TEXT NOT NULL,
    claim_text      TEXT NOT NULL,
    claim_type      TEXT NOT NULL,
    subject         TEXT NOT NULL DEFAULT '',
    predicate       TEXT NOT NULL DEFAULT '',
    object_json     TEXT,
    evidence_selectors TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 0.5,
    needs_llm       INTEGER NOT NULL DEFAULT 0,
    extracted_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claims_doc ON claims(doc_path);
CREATE INDEX IF NOT EXISTS idx_claims_type ON claims(claim_type);

CREATE TABLE IF NOT EXISTS claim_deps (
    claim_id    TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    file_path   TEXT NOT NULL,
    symbol      TEXT NOT NULL DEFAULT '',
    kind        TEXT NOT NULL DEFAULT 'file',
    PRIMARY KEY (claim_id, file_path, symbol)
);

CREATE INDEX IF NOT EXISTS idx_claim_deps_file ON claim_deps(file_path);

CREATE TABLE IF NOT EXISTS claim_verdicts (
    claim_id         TEXT NOT NULL REFERENCES claims(id) ON DELETE CASCADE,
    verdict          TEXT NOT NULL,
    confidence       REAL NOT NULL DEFAULT 1.0,
    severity         TEXT NOT NULL DEFAULT 'medium',
    reason           TEXT NOT NULL DEFAULT '',
    expected_json    TEXT,
    observed_json    TEXT,
    evidence_json    TEXT NOT NULL DEFAULT '[]',
    verifier         TEXT NOT NULL DEFAULT '',
    code_fingerprint TEXT NOT NULL DEFAULT '',
    verified_at      TEXT NOT NULL,
    PRIMARY KEY (claim_id, code_fingerprint)
);
"""


class ClaimStore:
    """SQLite-backed store for claims and verification results."""

    def __init__(self, db_path: Path | str) -> None:
        self._db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.row_factory = sqlite3.Row
            self._ensure_schema()
        return self._conn

    def _ensure_schema(self) -> None:
        conn = self._conn
        assert conn is not None
        conn.executescript(_CLAIMS_SCHEMA)

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ─── Claims CRUD ──────────────────────────────────────────────

    def upsert_claims(self, claims: list[Claim]) -> int:
        """Insert or replace claims. Returns count of upserted rows."""
        conn = self._connect()
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                c.id,
                c.doc_path,
                c.section,
                c.span.start_line,
                c.span.end_line,
                c.raw_text,
                c.claim_text,
                c.claim_type.value,
                c.subject,
                c.predicate,
                json.dumps(c.object) if c.object is not None else None,
                json.dumps(c.evidence_selectors),
                c.confidence,
                1 if c.needs_llm else 0,
                now,
            )
            for c in claims
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO claims
               (id, doc_path, section, span_start, span_end, raw_text, claim_text,
                claim_type, subject, predicate, object_json, evidence_selectors,
                confidence, needs_llm, extracted_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            rows,
        )
        conn.commit()
        return len(rows)

    def get_claims_for_doc(self, doc_path: str) -> list[Claim]:
        """Get all claims for a document."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM claims WHERE doc_path = ? ORDER BY span_start",
            (doc_path,),
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def get_claims_by_type(self, claim_type: ClaimType) -> list[Claim]:
        """Get all claims of a given type."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM claims WHERE claim_type = ? ORDER BY doc_path, span_start",
            (claim_type.value,),
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def get_all_claims(self) -> list[Claim]:
        """Get all claims."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT * FROM claims ORDER BY doc_path, span_start"
        ).fetchall()
        return [self._row_to_claim(r) for r in rows]

    def delete_claims_for_doc(self, doc_path: str) -> int:
        """Delete all claims for a document. Returns count deleted."""
        conn = self._connect()
        cursor = conn.execute("DELETE FROM claims WHERE doc_path = ?", (doc_path,))
        conn.commit()
        return cursor.rowcount

    # ─── Dependencies ─────────────────────────────────────────────

    def upsert_dependencies(self, deps: list[ClaimDependency]) -> int:
        """Insert or replace claim dependencies."""
        conn = self._connect()
        rows = [
            (d.claim_id, d.file_path, d.symbol or "", d.kind.value)
            for d in deps
        ]
        conn.executemany(
            """INSERT OR REPLACE INTO claim_deps (claim_id, file_path, symbol, kind)
               VALUES (?,?,?,?)""",
            rows,
        )
        conn.commit()
        return len(rows)

    def get_claims_for_file(self, file_path: str) -> list[str]:
        """Get claim IDs that depend on a given file."""
        conn = self._connect()
        rows = conn.execute(
            "SELECT DISTINCT claim_id FROM claim_deps WHERE file_path = ?",
            (file_path,),
        ).fetchall()
        return [r["claim_id"] for r in rows]

    # ─── Verdicts ─────────────────────────────────────────────────

    def save_verdict(self, result: VerificationResult) -> None:
        """Save a verification result."""
        conn = self._connect()
        conn.execute(
            """INSERT OR REPLACE INTO claim_verdicts
               (claim_id, verdict, confidence, severity, reason,
                expected_json, observed_json, evidence_json,
                verifier, code_fingerprint, verified_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                result.claim_id,
                result.verdict.value,
                result.confidence,
                result.severity.value,
                result.reason,
                json.dumps(result.expected) if result.expected is not None else None,
                json.dumps(result.observed) if result.observed is not None else None,
                json.dumps([e.model_dump() for e in result.evidence]),
                result.verifier,
                result.code_fingerprint,
                result.verified_at.isoformat(),
            ),
        )
        conn.commit()

    def get_latest_verdict(self, claim_id: str) -> VerificationResult | None:
        """Get the most recent verdict for a claim."""
        conn = self._connect()
        row = conn.execute(
            """SELECT * FROM claim_verdicts
               WHERE claim_id = ?
               ORDER BY verified_at DESC LIMIT 1""",
            (claim_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_verdict(row)

    def get_failed_verdicts(self) -> list[VerificationResult]:
        """Get all verdicts with failed status."""
        conn = self._connect()
        rows = conn.execute(
            """SELECT cv.* FROM claim_verdicts cv
               INNER JOIN (
                   SELECT claim_id, MAX(verified_at) as max_at
                   FROM claim_verdicts GROUP BY claim_id
               ) latest ON cv.claim_id = latest.claim_id
                       AND cv.verified_at = latest.max_at
               WHERE cv.verdict IN ('contradicted', 'incomplete', 'stale_target')
               ORDER BY cv.claim_id""",
        ).fetchall()
        return [self._row_to_verdict(r) for r in rows]

    # ─── Stats ────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        conn = self._connect()
        total = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
        by_type = dict(
            conn.execute(
                "SELECT claim_type, COUNT(*) FROM claims GROUP BY claim_type"
            ).fetchall()
        )
        by_doc = dict(
            conn.execute(
                "SELECT doc_path, COUNT(*) FROM claims GROUP BY doc_path ORDER BY COUNT(*) DESC"
            ).fetchall()
        )
        verdicts = dict(
            conn.execute(
                """SELECT verdict, COUNT(*) FROM claim_verdicts cv
                   INNER JOIN (
                       SELECT claim_id, MAX(verified_at) as max_at
                       FROM claim_verdicts GROUP BY claim_id
                   ) latest ON cv.claim_id = latest.claim_id
                           AND cv.verified_at = latest.max_at
                   GROUP BY verdict"""
            ).fetchall()
        )
        return {
            "total_claims": total,
            "by_type": by_type,
            "by_doc": by_doc,
            "latest_verdicts": verdicts,
        }

    # ─── Helpers ──────────────────────────────────────────────────

    @staticmethod
    def _row_to_claim(row: sqlite3.Row) -> Claim:
        obj = json.loads(row["object_json"]) if row["object_json"] else None
        selectors = json.loads(row["evidence_selectors"]) if row["evidence_selectors"] else []
        return Claim(
            id=row["id"],
            doc_path=row["doc_path"],
            section=row["section"],
            span=DocSpan(start_line=row["span_start"], end_line=row["span_end"]),
            raw_text=row["raw_text"],
            claim_text=row["claim_text"],
            claim_type=ClaimType(row["claim_type"]),
            subject=row["subject"],
            predicate=row["predicate"],
            object=obj,
            evidence_selectors=selectors,
            confidence=row["confidence"],
            needs_llm=bool(row["needs_llm"]),
        )

    @staticmethod
    def _row_to_verdict(row: sqlite3.Row) -> VerificationResult:
        evidence_raw = json.loads(row["evidence_json"]) if row["evidence_json"] else []
        evidence = [Evidence(**e) for e in evidence_raw]
        return VerificationResult(
            claim_id=row["claim_id"],
            verdict=Verdict(row["verdict"]),
            confidence=row["confidence"],
            severity=Severity(row["severity"]),
            reason=row["reason"],
            expected=json.loads(row["expected_json"]) if row["expected_json"] else None,
            observed=json.loads(row["observed_json"]) if row["observed_json"] else None,
            evidence=evidence,
            verifier=row["verifier"],
            code_fingerprint=row["code_fingerprint"],
            verified_at=datetime.fromisoformat(row["verified_at"]),
        )
