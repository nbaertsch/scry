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
from scry.claims.verifiers import BaseVerifier, register


class NumericValueVerifier(BaseVerifier):
    """Verify numeric claims (defaults, limits, timeouts) against code."""

    @property
    def name(self) -> str:
        return "numeric_value"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.NUMERIC_VALUE})

    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult:
        obj = claim.object
        if not isinstance(obj, dict) or "value" not in obj:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.ERROR,
                reason="Claim object missing 'value' field",
                verifier=self.name,
            )

        expected = obj["value"]

        # Look for the expected value in code near related symbols
        # Search for the number in assignment/default contexts
        str_val = str(int(expected)) if isinstance(expected, float) and expected == int(expected) else str(expected)

        patterns = [
            rf"=\s*{re.escape(str_val)}\b",
            rf"default\s*=\s*{re.escape(str_val)}\b",
            rf":\s*{re.escape(str_val)}\b",
            rf"\b{re.escape(str_val)}\b",
        ]

        # If we have section context, try to find related symbols
        context_terms = self._extract_context_terms(claim)

        all_evidence: list[Evidence] = []
        code_values: set[str] = set()

        for pat in patterns[:2]:  # Focus on assignment patterns
            hits = self._search_files(repo_root, pat)
            for fp, line_no, line_text in hits:
                # Check if hit is in a relevant context
                rel_path = str(fp.relative_to(repo_root))
                if self._is_relevant(line_text, context_terms):
                    all_evidence.append(Evidence(
                        file_path=rel_path,
                        line=line_no,
                        snippet=line_text[:200],
                    ))

        if all_evidence:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.CONFIRMED,
                confidence=0.7,
                reason=f"Value {expected} found in code",
                expected=expected,
                evidence=all_evidence[:5],
                verifier=self.name,
            )

        # Value not found — check if a different value exists in similar context
        # This is a weaker signal; mark as unverifiable rather than contradicted
        return VerificationResult(
            claim_id=claim.id,
            verdict=Verdict.UNVERIFIABLE,
            confidence=0.3,
            severity=Severity.LOW,
            reason=f"Could not locate value {expected} in code; manual review needed",
            expected=expected,
            verifier=self.name,
        )

    def _extract_context_terms(self, claim: Claim) -> list[str]:
        """Extract relevant terms from claim context for filtering."""
        terms: list[str] = []
        # From section heading
        if claim.section:
            for part in claim.section.split(" > "):
                for word in re.findall(r"\w{3,}", part.lower()):
                    terms.append(word)
        # From raw text
        for word in re.findall(r"\w{4,}", claim.raw_text.lower()):
            if word not in ("default", "value", "seconds", "timeout", "limit"):
                terms.append(word)
        return terms[:10]

    def _is_relevant(self, line: str, context_terms: list[str]) -> bool:
        """Check if a code line is relevant to the claim context."""
        if not context_terms:
            return True
        line_lower = line.lower()
        return any(term in line_lower for term in context_terms)


register(NumericValueVerifier())
