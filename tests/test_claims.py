from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from textwrap import dedent

from scry.claims.extractor import extract_claims
from scry.claims.model import (
    Claim,
    ClaimType,
    DocSpan,
    Verdict,
    VerificationReport,
    VerificationResult,
)
from scry.claims.repo_index import RepoIndex, build_index
from scry.claims.store import ClaimStore
from scry.claims.verifiers import verify_claim
import scry.claims.verifiers.symbol  # noqa: F401
import scry.claims.verifiers.numeric  # noqa: F401
import scry.claims.verifiers.enum  # noqa: F401
import scry.claims.verifiers.paths  # noqa: F401
import scry.claims.verifiers.envvar  # noqa: F401


def _make_claim(
    claim_type: ClaimType,
    subject: str,
    *,
    raw_text: str | None = None,
    section: str = "",
    object_: object = None,
) -> Claim:
    return Claim(
        doc_path="docs/spec.md",
        section=section,
        span=DocSpan(start_line=1, end_line=1),
        raw_text=raw_text or subject,
        claim_text=f"{claim_type.value}:{subject}",
        claim_type=claim_type,
        subject=subject,
        predicate="test",
        object=object_,
    )


def _write(tmp_path: Path, rel_path: str, content: str) -> Path:
    path = tmp_path / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip(), encoding="utf-8")
    return path


def test_extract_claims_extracts_expected_claim_types() -> None:
    text = dedent(
        """
        # Overview

        ## Runtime
        Use `build_index` to scan the repo.
        Default timeout is 30 seconds.
        Set `APP_TIMEOUT` in the environment.
        Call `GET /api/v1/runs`.
        See `src/scry/claims/model.py`.
        RunState has 4 states.
        Every run emits artifacts.
        """
    ).lstrip()

    claims = extract_claims("docs/spec.md", text)

    by_type = {claim.claim_type: claim for claim in claims}

    assert by_type[ClaimType.SYMBOL_EXISTS].subject == "build_index"
    assert by_type[ClaimType.SYMBOL_EXISTS].section == "Overview > Runtime"
    assert by_type[ClaimType.NUMERIC_VALUE].object == {"value": 30, "unit": "seconds"}
    assert by_type[ClaimType.ENV_VAR_NAME].subject == "APP_TIMEOUT"
    assert by_type[ClaimType.ROUTE_PATH].subject == "/api/v1/runs"
    assert by_type[ClaimType.FILE_PATH].subject == "src/scry/claims/model.py"
    assert by_type[ClaimType.ENUM_COUNT].object == {"count": 4, "noun": "state"}
    assert by_type[ClaimType.COVERAGE_ASSERTION].predicate == "every"
    assert by_type[ClaimType.COVERAGE_ASSERTION].needs_llm is True


def test_extract_claims_filters_noise_symbols_and_env_vars() -> None:
    text = dedent(
        """
        # Noise

        Ignore `str`, `Any`, and `True`.
        Set `API`, `URL`, and `GET` in the env.
        Keep `REAL_TIMEOUT` in the environment.
        """
    ).lstrip()

    claims = extract_claims("docs/noise.md", text)
    subjects = {claim.subject for claim in claims}

    assert "REAL_TIMEOUT" in subjects
    assert {"str", "Any", "True", "API", "URL", "GET"}.isdisjoint(subjects)


def test_extract_claims_tracks_nested_section_headings() -> None:
    text = dedent(
        """
        # Root

        ## Child
        Use `verify_claim`.
        """
    ).lstrip()

    [claim] = extract_claims("docs/sections.md", text)

    assert claim.section == "Root > Child"
    assert claim.span.start_line == 4


def test_claim_id_generation_is_stable() -> None:
    claim_a = _make_claim(ClaimType.SYMBOL_EXISTS, "build_index", raw_text="Use `build_index`.")
    claim_b = _make_claim(ClaimType.SYMBOL_EXISTS, "build_index", raw_text="Use `build_index`.")
    claim_c = Claim(
        doc_path="docs/spec.md",
        section="",
        span=DocSpan(start_line=2, end_line=2),
        raw_text="Use `build_index`.",
        claim_text="symbol_exists:build_index",
        claim_type=ClaimType.SYMBOL_EXISTS,
        subject="build_index",
        predicate="test",
    )

    assert claim_a.id == claim_b.id
    assert claim_a.id != claim_c.id


def test_verification_report_counters() -> None:
    report = VerificationReport()

    for verdict in (
        Verdict.CONFIRMED,
        Verdict.CONTRADICTED,
        Verdict.INCOMPLETE,
        Verdict.UNVERIFIABLE,
        Verdict.STALE_TARGET,
        Verdict.ERROR,
    ):
        report.add(VerificationResult(claim_id=f"claim-{verdict}", verdict=verdict))

    assert report.total_claims == 6
    assert report.verified == 4
    assert report.confirmed == 1
    assert report.contradicted == 1
    assert report.incomplete == 1
    assert report.unverifiable == 1
    assert report.stale_target == 1
    assert report.errors == 1
    assert len(report.failed_results) == 3
    assert report.pass_rate == 0.25


def test_verdict_enum_values() -> None:
    assert Verdict.CONFIRMED.value == "confirmed"
    assert Verdict.CONTRADICTED.value == "contradicted"
    assert Verdict.INCOMPLETE.value == "incomplete"
    assert Verdict.UNVERIFIABLE.value == "unverifiable"
    assert Verdict.STALE_TARGET.value == "stale_target"
    assert Verdict.ERROR.value == "error"


def test_verify_claim_symbol_exists_confirms_with_python_function(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/module.py",
        """
        def do_work() -> None:
            pass
        """,
    )
    index = build_index(tmp_path)
    claim = _make_claim(ClaimType.SYMBOL_EXISTS, "do_work()", raw_text="Use `do_work()`.")

    result = verify_claim(claim, tmp_path, index)

    assert result.verdict == Verdict.CONFIRMED
    assert result.evidence[0].file_path == "pkg/module.py"


def test_verify_claim_symbol_exists_returns_stale_target_for_missing_symbol(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/module.py",
        """
        def present_symbol() -> None:
            pass
        """,
    )
    index = build_index(tmp_path)
    claim = _make_claim(ClaimType.SYMBOL_EXISTS, "missing_symbol", raw_text="Use `missing_symbol`.")

    result = verify_claim(claim, tmp_path, index)

    assert result.verdict == Verdict.STALE_TARGET


def test_verify_claim_file_path_checks_existing_and_missing_paths(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/existing.py", "VALUE = 1\n")
    index = build_index(tmp_path)

    confirmed = verify_claim(
        _make_claim(ClaimType.FILE_PATH, "pkg/existing.py", raw_text="See `pkg/existing.py`."),
        tmp_path,
        index,
    )
    contradicted = verify_claim(
        _make_claim(ClaimType.FILE_PATH, "pkg/missing.py", raw_text="See `pkg/missing.py`."),
        tmp_path,
        index,
    )

    assert confirmed.verdict == Verdict.CONFIRMED
    assert contradicted.verdict == Verdict.CONTRADICTED


def test_verify_claim_env_var_confirms_when_string_literal_exists(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/settings.py",
        """
        import os

        VALUE = os.getenv("APP_TOKEN")
        """,
    )
    index = build_index(tmp_path)
    claim = _make_claim(ClaimType.ENV_VAR_NAME, "APP_TOKEN", raw_text="Set `APP_TOKEN`.")

    result = verify_claim(claim, tmp_path, index)

    assert result.verdict == Verdict.CONFIRMED
    assert result.evidence[0].file_path == "pkg/settings.py"


def test_verify_claim_enum_count_confirms_matching_strenum_member_count(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/enums.py",
        """
        from enum import StrEnum

        class RunState(StrEnum):
            QUEUED = "queued"
            RUNNING = "running"
            DONE = "done"
        """,
    )
    index = build_index(tmp_path)
    claim = _make_claim(
        ClaimType.ENUM_COUNT,
        "RunState",
        raw_text="RunState has 3 states.",
        section="RunState",
        object_={"count": 3, "noun": "state"},
    )

    result = verify_claim(claim, tmp_path, index)

    assert result.verdict == Verdict.CONFIRMED
    assert result.observed == 3


def test_verify_claim_route_path_confirms_matching_router_decorator(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/routes.py",
        """
        @router.get("/api/v1/runs")
        def list_runs() -> None:
            pass
        """,
    )
    index = build_index(tmp_path)
    claim = _make_claim(
        ClaimType.ROUTE_PATH,
        "/api/v1/runs",
        raw_text="GET `/api/v1/runs` exists.",
        object_={"path": "/api/v1/runs", "method": "GET"},
    )

    result = verify_claim(claim, tmp_path, index)

    assert result.verdict == Verdict.CONFIRMED
    assert result.evidence[0].file_path == "pkg/routes.py"


def test_build_index_scans_python_files_correctly(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/module.py",
        """
        from enum import StrEnum

        VALUE = 7

        class Worker:
            def run(self) -> None:
                pass

        def helper() -> None:
            pass

        class Mode(StrEnum):
            FAST = "fast"
            SLOW = "slow"
        """,
    )

    index = build_index(tmp_path)

    assert "helper" in index.symbols
    assert "Worker" in index.symbols
    assert "run" in index.symbols
    assert "VALUE" in index.symbols
    assert index.enums["Mode"].members == ["FAST", "SLOW"]


def test_repo_index_symbol_lookup_supports_dotted_names(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/module.py",
        """
        class Worker:
            def run(self) -> None:
                pass
        """,
    )
    index = build_index(tmp_path)

    hits = index.lookup_symbol("Worker.run")

    assert len(hits) == 1
    assert hits[0].name == "run"


def test_repo_index_string_literal_lookup_finds_uppercase_literals(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "pkg/settings.py",
        """
        VALUE = "APP_TOKEN"
        """,
    )
    index = build_index(tmp_path)

    hits = index.lookup_string("APP_TOKEN")

    assert len(hits) == 1
    assert hits[0].file_path == "pkg/settings.py"


def test_repo_index_file_exists_checks_repo_relative_paths(tmp_path: Path) -> None:
    _write(tmp_path, "pkg/module.py", "VALUE = 1\n")
    _write(tmp_path, "README.md", "# test\n")
    index = build_index(tmp_path)

    assert index.file_exists("pkg/module.py") is True
    assert index.file_exists(r"pkg\module.py") is True
    assert index.file_exists("pkg/missing.py") is False


def test_upsert_claims_and_get_claims_for_doc(tmp_path: Path) -> None:
    store = ClaimStore(tmp_path / "claims.db")
    claims = [
        _make_claim(
            ClaimType.SYMBOL_EXISTS,
            "build_index",
            raw_text="Use `build_index`.",
        ),
        Claim(
            doc_path="docs/spec.md",
            section="",
            span=DocSpan(start_line=2, end_line=2),
            raw_text="Use `verify_claim`.",
            claim_text="symbol_exists:verify_claim",
            claim_type=ClaimType.SYMBOL_EXISTS,
            subject="verify_claim",
            predicate="test",
        ),
    ]

    try:
        assert store.upsert_claims(claims) == 2
        stored = store.get_claims_for_doc("docs/spec.md")
    finally:
        store.close()

    assert [claim.subject for claim in stored] == ["build_index", "verify_claim"]


def test_save_verdict_and_get_latest_verdict(tmp_path: Path) -> None:
    store = ClaimStore(tmp_path / "claims.db")
    claim = _make_claim(ClaimType.SYMBOL_EXISTS, "build_index", raw_text="Use `build_index`.")
    older_at = datetime(2024, 1, 1, tzinfo=UTC)
    newer_at = older_at + timedelta(days=1)

    older = VerificationResult(
        claim_id=claim.id,
        verdict=Verdict.CONTRADICTED,
        verifier="test",
        code_fingerprint="old",
        verified_at=older_at,
    )
    newer = VerificationResult(
        claim_id=claim.id,
        verdict=Verdict.CONFIRMED,
        verifier="test",
        code_fingerprint="new",
        verified_at=newer_at,
    )

    try:
        store.upsert_claims([claim])
        store.save_verdict(older)
        store.save_verdict(newer)
        latest = store.get_latest_verdict(claim.id)
    finally:
        store.close()

    assert latest is not None
    assert latest.verdict == Verdict.CONFIRMED
    assert latest.code_fingerprint == "new"


def test_get_failed_verdicts_returns_latest_failures(tmp_path: Path) -> None:
    store = ClaimStore(tmp_path / "claims.db")
    claim_ok = _make_claim(ClaimType.FILE_PATH, "pkg/ok.py", raw_text="See `pkg/ok.py`.")
    claim_fail = Claim(
        doc_path="docs/spec.md",
        section="",
        span=DocSpan(start_line=2, end_line=2),
        raw_text="See `pkg/missing.py`.",
        claim_text="file_path:pkg/missing.py",
        claim_type=ClaimType.FILE_PATH,
        subject="pkg/missing.py",
        predicate="test",
    )
    base_at = datetime(2024, 1, 1, tzinfo=UTC)

    try:
        store.upsert_claims([claim_ok, claim_fail])
        store.save_verdict(
            VerificationResult(
                claim_id=claim_ok.id,
                verdict=Verdict.STALE_TARGET,
                verifier="test",
                code_fingerprint="ok-1",
                verified_at=base_at,
            )
        )
        store.save_verdict(
            VerificationResult(
                claim_id=claim_ok.id,
                verdict=Verdict.CONFIRMED,
                verifier="test",
                code_fingerprint="ok-2",
                verified_at=base_at + timedelta(minutes=1),
            )
        )
        store.save_verdict(
            VerificationResult(
                claim_id=claim_fail.id,
                verdict=Verdict.CONTRADICTED,
                verifier="test",
                code_fingerprint="fail-1",
                verified_at=base_at + timedelta(minutes=2),
            )
        )
        failed = store.get_failed_verdicts()
    finally:
        store.close()

    assert [result.claim_id for result in failed] == [claim_fail.id]
    assert failed[0].verdict == Verdict.CONTRADICTED
