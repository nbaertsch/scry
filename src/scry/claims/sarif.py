"""SARIF 2.1.0 report generator for claim verification results.

Produces Static Analysis Results Interchange Format output
compatible with GitHub Code Scanning (upload-sarif action).
"""

from __future__ import annotations

import json
from typing import Any

from scry.claims.model import (
    Claim,
    Verdict,
    VerificationReport,
    VerificationResult,
)

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA = "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/main/sarif-2.1/schema/sarif-schema-2.1.0.json"

_VERDICT_TO_LEVEL = {
    Verdict.CONTRADICTED: "error",
    Verdict.INCOMPLETE: "warning",
    Verdict.STALE_TARGET: "note",
    Verdict.ERROR: "error",
    Verdict.UNVERIFIABLE: "note",
}


def _make_result(claim: Claim, result: VerificationResult) -> dict[str, Any]:
    """Convert a single verification result to a SARIF result object."""
    level = _VERDICT_TO_LEVEL.get(result.verdict, "note")
    message = result.reason or f"Claim: {claim.claim_text}"

    region: dict[str, Any] = {"startLine": claim.span.start_line}
    if claim.span.end_line != claim.span.start_line:
        region["endLine"] = claim.span.end_line

    return {
        "ruleId": f"scry/{claim.claim_type.value}",
        "level": level,
        "message": {"text": message},
        "locations": [
            {
                "physicalLocation": {
                    "artifactLocation": {
                        "uri": claim.doc_path.replace("\\", "/"),
                        "uriBaseId": "%SRCROOT%",
                    },
                    "region": region,
                }
            }
        ],
        "properties": {
            "claim_text": claim.claim_text,
            "subject": claim.subject,
            "verdict": result.verdict.value,
            "severity": result.severity.value,
        },
    }


def generate_sarif(
    report: VerificationReport,
    results: list[VerificationResult],
    claims: list[Claim],
) -> str:
    """Generate SARIF 2.1.0 JSON from verification results.

    Only includes non-confirmed results (contradicted, incomplete,
    stale_target, error).
    """
    # Build claim lookup by ID
    claim_map = {c.id: c for c in claims}

    sarif_results: list[dict[str, Any]] = []
    for r in results:
        if r.verdict == Verdict.CONFIRMED:
            continue
        claim = claim_map.get(r.claim_id)
        if not claim:
            continue
        sarif_results.append(_make_result(claim, r))

    # Collect unique rule IDs
    rule_ids = sorted({r["ruleId"] for r in sarif_results})
    rules = [
        {
            "id": rid,
            "shortDescription": {"text": f"Doc claim verification: {rid.split('/')[-1]}"},
            "helpUri": "https://github.com/nbaertsch/scry#claim-verification",
        }
        for rid in rule_ids
    ]

    sarif = {
        "$schema": SARIF_SCHEMA,
        "version": SARIF_VERSION,
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "scry",
                        "informationUri": "https://github.com/nbaertsch/scry",
                        "version": "0.1.0",
                        "rules": rules,
                    }
                },
                "results": sarif_results,
            }
        ],
    }

    return json.dumps(sarif, indent=2)
