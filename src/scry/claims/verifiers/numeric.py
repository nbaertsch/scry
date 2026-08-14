"""Numeric value verifier.

Checks whether numeric values claimed in documentation (defaults,
timeouts, limits) match what the code actually defines.
"""

from __future__ import annotations

import re
from pathlib import Path

from scry.claims.model import (
    Claim,
    ClaimType,
    Evidence,
    Severity,
    Verdict,
    VerificationResult,
)
from scry.claims.repo_index import RepoIndex
from scry.claims.verifiers import BaseVerifier, register


class NumericValueVerifier(BaseVerifier):
    """Verify numeric claims (defaults, limits, timeouts) against code."""

    @property
    def name(self) -> str:
        return "numeric_value"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.NUMERIC_VALUE})

    def verify(self, claim: Claim, repo_root: Path, index: RepoIndex) -> VerificationResult:
        obj = claim.object
        if not isinstance(obj, dict) or "value" not in obj:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.ERROR,
                reason="Claim object missing 'value' field", verifier=self.name,
            )

        expected = obj["value"]
        str_val = str(int(expected)) if isinstance(expected, float) and expected == int(expected) else str(expected)

        # Search string literals in the index for the value
        hits = index.lookup_string(str_val)
        if hits:
            evidence = [
                Evidence(file_path=h.file_path, line=h.line, snippet=h.context)
                for h in hits[:5]
            ]
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=0.7,
                reason=f"Value {expected} found in code",
                expected=expected, evidence=evidence, verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id, verdict=Verdict.UNVERIFIABLE, confidence=0.3,
            severity=Severity.LOW,
            reason=f"Could not locate value {expected} in code; manual review needed",
            expected=expected, verifier=self.name,
        )


register(NumericValueVerifier())
