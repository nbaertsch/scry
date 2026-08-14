"""Orchestrator for claim extraction and verification.

Wires together the extractor, verifiers, and store into a single
``verify_docs`` flow usable by both the CLI and MCP tools.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from scry.claims.extractor import extract_claims, extract_claims_from_file
from scry.claims.model import (
    Claim,
    ClaimDependency,
    DependencyKind,
    VerificationReport,
    VerificationResult,
    Verdict,
)
from scry.claims.repo_index import RepoIndex, build_index
from scry.claims.store import ClaimStore
from scry.claims.verifiers import verify_claim

# Force-import verifier modules so they self-register
import scry.claims.verifiers.symbol  # noqa: F401
import scry.claims.verifiers.numeric  # noqa: F401
import scry.claims.verifiers.enum  # noqa: F401
import scry.claims.verifiers.paths  # noqa: F401
import scry.claims.verifiers.envvar  # noqa: F401


def _evidence_changed(cached: VerificationResult, changed_files: set[str]) -> bool:
    """Check if any evidence files are in the changed file set."""
    if not changed_files:
        return False
    return any(e.file_path in changed_files for e in cached.evidence if e.file_path)


def _find_doc_files(repo_root: Path, paths: list[str] | None = None) -> list[str]:
    """Find markdown doc files to verify.

    Args:
        repo_root: Repository root.
        paths: Optional list of specific paths/globs.  If None, searches docs/.

    Returns:
        List of repo-relative doc paths.
    """
    if paths:
        result: list[str] = []
        for p in paths:
            target = repo_root / p
            if target.is_file() and target.suffix in (".md", ".mdx"):
                result.append(str(target.relative_to(repo_root)))
            elif target.is_dir():
                for fp in sorted(target.rglob("*.md")):
                    if ".git" not in fp.parts:
                        result.append(str(fp.relative_to(repo_root)))
        return result

    # Default: search common doc directories
    doc_dirs = ["docs", "doc", "documentation"]
    result = []
    for d in doc_dirs:
        dd = repo_root / d
        if dd.is_dir():
            for fp in sorted(dd.rglob("*.md")):
                if ".git" not in fp.parts:
                    result.append(str(fp.relative_to(repo_root)))
    return result


def _get_changed_files(repo_root: Path) -> list[str] | None:
    """Get files changed vs the default branch using git diff."""
    import subprocess

    # Try to get merge-base with common default branches
    for base in ("origin/main", "origin/dev", "origin/master", "HEAD~1"):
        try:
            result = subprocess.run(
                ["git", "diff", "--name-only", base],
                capture_output=True,
                text=True,
                cwd=str(repo_root),
                timeout=10,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().splitlines()
        except Exception:
            continue
    return None


def verify_docs(
    repo_root: Path,
    *,
    paths: list[str] | None = None,
    changed_only: bool = False,
    store: ClaimStore | None = None,
    skip_llm: bool = True,
    use_cache: bool = True,
) -> VerificationReport:
    """Extract claims from docs and verify them against code.

    Args:
        repo_root: Repository root directory.
        paths: Specific doc paths to verify.  None = all docs.
        changed_only: Only verify claims impacted by changed files.
        store: Optional ClaimStore for persistence.
        skip_llm: Skip claims that need LLM verification.
        use_cache: Use cached verdicts when code hasn't changed.

    Returns:
        VerificationReport with all results.
    """
    report = VerificationReport()

    # Find doc files
    doc_files = _find_doc_files(repo_root, paths)
    if not doc_files:
        return report

    # If --changed, filter to only impacted docs/claims
    changed_files: set[str] | None = None
    if changed_only:
        raw = _get_changed_files(repo_root)
        if raw is not None:
            changed_files = set(raw)
            # Include docs that changed
            changed_docs = {f for f in changed_files if f.endswith(".md")}
            # Also include docs linked to changed code files
            if store:
                for cf in changed_files:
                    if not cf.endswith(".md"):
                        claim_ids = store.get_claims_for_file(cf)
                        if claim_ids:
                            # Get the doc paths for these claims
                            for claim in store.get_all_claims():
                                if claim.id in claim_ids:
                                    changed_docs.add(claim.doc_path)
            if changed_docs:
                doc_files = [f for f in doc_files if f in changed_docs]
            else:
                # No changed docs found
                return report

    # Build repo index once
    index = build_index(repo_root)

    # Extract and verify
    all_claims: list[Claim] = []
    for doc_path in doc_files:
        claims = extract_claims_from_file(doc_path, repo_root)
        all_claims.extend(claims)

    # Persist extracted claims
    if store and all_claims:
        store.upsert_claims(all_claims)

    # Verify each claim
    changed_file_set = changed_files or set()
    for claim in all_claims:
        if skip_llm and claim.needs_llm:
            result = VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.UNVERIFIABLE,
                reason="Skipped: requires LLM verification",
                verifier="skip_llm",
            )
        else:
            # Check cache: skip if verdict exists and evidence files unchanged
            cached = store.get_latest_verdict(claim.id) if (store and use_cache) else None
            if cached and cached.code_fingerprint and not _evidence_changed(cached, changed_file_set):
                result = cached
            else:
                result = verify_claim(claim, repo_root, index)
                # Set code fingerprint from evidence files
                if result.evidence:
                    fp_parts = sorted({e.file_path for e in result.evidence if e.file_path})
                    fingerprint = hashlib.sha256("|".join(fp_parts).encode()).hexdigest()[:16]
                    object.__setattr__(result, "code_fingerprint", fingerprint)

        report.add(result)

        # Persist verdict and auto-generate dependencies
        if store:
            store.save_verdict(result)
            # Generate dependencies from evidence
            if result.evidence:
                deps = [
                    ClaimDependency(
                        claim_id=claim.id,
                        file_path=e.file_path,
                        symbol=e.symbol,
                        kind=DependencyKind.SYMBOL if e.symbol else DependencyKind.FILE,
                    )
                    for e in result.evidence
                    if e.file_path
                ]
                if deps:
                    store.upsert_dependencies(deps)

    return report


def format_report(
    report: VerificationReport,
    claims: list[Claim] | None = None,
    *,
    format: str = "text",
    show_confirmed: bool = False,
) -> str:
    """Format a verification report for display.

    Args:
        report: The verification report.
        claims: Optional claims list for enriching output.
        format: Output format — "text" or "json".
        show_confirmed: Whether to include confirmed claims.

    Returns:
        Formatted report string.
    """
    if format == "json":
        import json

        return json.dumps(report.model_dump(), indent=2, default=str)

    # Text format
    lines: list[str] = []
    lines.append("═" * 60)
    lines.append("  SCRY CLAIM VERIFICATION REPORT")
    lines.append("═" * 60)
    lines.append("")
    lines.append(f"  Total claims:   {report.total_claims}")
    lines.append(f"  Verified:       {report.verified}")
    lines.append(f"  ✅ Confirmed:   {report.confirmed}")
    lines.append(f"  ❌ Contradicted: {report.contradicted}")
    lines.append(f"  ⚠️  Incomplete:  {report.incomplete}")
    lines.append(f"  🔍 Stale target: {report.stale_target}")
    lines.append(f"  ❓ Unverifiable: {report.unverifiable}")
    lines.append(f"  💥 Errors:      {report.errors}")
    if report.verified > 0:
        lines.append(f"  Pass rate:      {report.pass_rate:.0%}")
    lines.append("")

    # Build claim lookup
    claim_map: dict[str, Claim] = {}
    if claims:
        claim_map = {c.id: c for c in claims}

    # Group failures by severity
    failures = report.failed_results
    if failures:
        lines.append("─" * 60)
        lines.append("  FAILURES")
        lines.append("─" * 60)
        for result in sorted(failures, key=lambda r: r.severity.value):
            claim = claim_map.get(result.claim_id)
            doc = claim.doc_path if claim else "?"
            line_no = claim.span.start_line if claim else "?"
            icon = "❌" if result.verdict == Verdict.CONTRADICTED else "⚠️" if result.verdict == Verdict.INCOMPLETE else "🔗"
            lines.append("")
            lines.append(f"  {icon} [{result.severity.value.upper()}] {doc}:{line_no}")
            if claim:
                lines.append(f"     Claim: {claim.claim_text}")
            lines.append(f"     {result.reason}")
            if result.expected is not None:
                lines.append(f"     Expected: {result.expected}")
            if result.observed is not None:
                lines.append(f"     Observed: {result.observed}")
            if result.evidence:
                ev = result.evidence[0]
                lines.append(f"     Evidence: {ev.file_path}:{ev.line or '?'}")

    if show_confirmed:
        confirmed = [r for r in report.results if r.verdict == Verdict.CONFIRMED]
        if confirmed:
            lines.append("")
            lines.append("─" * 60)
            lines.append("  CONFIRMED")
            lines.append("─" * 60)
            for result in confirmed:
                claim = claim_map.get(result.claim_id)
                lines.append(f"  ✅ {claim.claim_text if claim else result.claim_id}")

    lines.append("")
    return "\n".join(lines)
