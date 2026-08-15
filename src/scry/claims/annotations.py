"""GitHub PR annotation reporter for claim verification.

Generates GitHub Actions workflow commands for inline annotations
on pull request diffs. Works with both `::error` / `::warning` commands
and the Checks API markdown summary.
"""

from __future__ import annotations

from scry.claims.model import (
    Claim,
    Verdict,
    VerificationReport,
    VerificationResult,
)


def generate_annotations(
    report: VerificationReport,
    claims: list[Claim],
) -> str:
    """Generate GitHub Actions annotation commands for failed claims.

    Output format: `::error file=path,line=N::message`
    These create inline annotations on PR diffs.
    """
    claim_map = {c.id: c for c in claims}
    lines: list[str] = []

    for r in report.results:
        if r.verdict == Verdict.CONFIRMED:
            continue
        claim = claim_map.get(r.claim_id)
        if not claim:
            continue

        level = "error" if r.verdict == Verdict.CONTRADICTED else "warning"
        file_path = claim.doc_path.replace("\\", "/")
        msg = r.reason.replace("\n", " ")

        lines.append(f"::{level} file={file_path},line={claim.span.start_line}::{msg}")

    return "\n".join(lines)


def generate_summary(report: VerificationReport, claims: list[Claim]) -> str:
    """Generate a markdown summary for the GitHub Checks API or job summary.

    Suitable for writing to $GITHUB_STEP_SUMMARY.
    """
    claim_map = {c.id: c for c in claims}
    lines: list[str] = []

    lines.append("## Scry Claim Verification Report")
    lines.append("")
    lines.append(f"| Metric | Count |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total claims | {report.total_claims} |")
    lines.append(f"| Confirmed | {report.confirmed} |")
    lines.append(f"| Contradicted | {report.contradicted} |")
    lines.append(f"| Stale target | {report.stale_target} |")
    lines.append(f"| Unverifiable | {report.unverifiable} |")
    lines.append("")

    if report.contradicted > 0:
        lines.append("### Contradicted Claims")
        lines.append("")
        for r in report.results:
            if r.verdict != Verdict.CONTRADICTED:
                continue
            claim = claim_map.get(r.claim_id)
            if not claim:
                continue
            file_path = claim.doc_path.replace("\\", "/")
            lines.append(f"- **{file_path}:{claim.span.start_line}** — {r.reason}")
        lines.append("")

    pass_rate = f"{report.pass_rate * 100:.1f}%" if report.verified > 0 else "N/A"
    lines.append(f"**Pass rate: {pass_rate}**")

    return "\n".join(lines)
