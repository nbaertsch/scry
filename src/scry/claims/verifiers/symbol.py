"""Symbol existence verifier.

Checks whether backticked symbol names from documentation actually exist
in the codebase using the pre-built RepoIndex.
"""

from __future__ import annotations

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


class SymbolExistsVerifier(BaseVerifier):
    """Verify that a symbol referenced in docs exists in the codebase."""

    @property
    def name(self) -> str:
        return "symbol_exists"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.SYMBOL_EXISTS})

    def verify(self, claim: Claim, repo_root: Path, index: RepoIndex) -> VerificationResult:
        symbol = claim.subject
        if not symbol:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.ERROR,
                reason="No symbol subject in claim", verifier=self.name,
            )

        bare = symbol.rstrip("()")
        hits = index.lookup_symbol(bare)

        if hits:
            evidence = [
                Evidence(file_path=h.file_path, line=h.line, snippet=h.snippet, symbol=h.name)
                for h in hits[:5]
            ]
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=0.95,
                reason=f"Symbol `{symbol}` found in {len(hits)} location(s)",
                evidence=evidence, verifier=self.name,
            )

        similar = index.find_similar_symbols(bare)
        if similar:
            # High-confidence rename detected
            top = similar[0]
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.STALE_TARGET, confidence=0.85,
                severity=Severity.MEDIUM,
                reason=f"Symbol `{symbol}` not found; possible rename to `{top}`",
                expected=symbol, observed=similar[:3], verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id, verdict=Verdict.CONTRADICTED, confidence=0.7,
            severity=Severity.MEDIUM,
            reason=f"Symbol `{symbol}` not found in codebase",
            expected=symbol, verifier=self.name,
        )


register(SymbolExistsVerifier())
