"""Symbol existence verifier.

Checks whether backticked symbol names from documentation actually exist
in the codebase.  Uses grep-based search across Python files.
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


class SymbolExistsVerifier(BaseVerifier):
    """Verify that a symbol referenced in docs exists in the codebase."""

    @property
    def name(self) -> str:
        return "symbol_exists"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.SYMBOL_EXISTS})

    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult:
        symbol = claim.subject
        if not symbol:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.ERROR,
                reason="No symbol subject in claim",
                verifier=self.name,
            )

        # Strip trailing () for function calls
        bare = symbol.rstrip("()")

        # Build search patterns — look for definitions
        patterns = [
            rf"\bdef\s+{re.escape(bare)}\b",
            rf"\bclass\s+{re.escape(bare)}\b",
            rf"\b{re.escape(bare)}\s*=",
            rf"\b{re.escape(bare)}\s*:",
        ]
        # For dotted names like Class.method, search for the last part
        if "." in bare:
            parts = bare.split(".")
            last = parts[-1]
            patterns.extend([
                rf"\bdef\s+{re.escape(last)}\b",
                rf"\b{re.escape(last)}\s*=",
            ])

        all_hits: list[tuple[Path, int, str]] = []
        for pat in patterns:
            hits = self._search_files(repo_root, pat)
            all_hits.extend(hits)

        if all_hits:
            # Deduplicate
            seen: set[tuple[str, int]] = set()
            evidence: list[Evidence] = []
            for fp, line_no, line_text in all_hits:
                key = (str(fp.relative_to(repo_root)), line_no)
                if key not in seen:
                    seen.add(key)
                    evidence.append(Evidence(
                        file_path=str(fp.relative_to(repo_root)),
                        line=line_no,
                        snippet=line_text[:200],
                    ))
                    if len(evidence) >= 5:
                        break

            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.CONFIRMED,
                confidence=0.95,
                reason=f"Symbol `{symbol}` found in {len(seen)} location(s)",
                evidence=evidence,
                verifier=self.name,
            )

        # Check for similar symbols (possible rename)
        similar = self._find_similar(repo_root, bare)
        if similar:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.STALE_TARGET,
                confidence=0.8,
                severity=Severity.HIGH,
                reason=f"Symbol `{symbol}` not found; similar: {', '.join(similar[:3])}",
                expected=symbol,
                observed=similar[:3],
                verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id,
            verdict=Verdict.STALE_TARGET,
            confidence=0.7,
            severity=Severity.MEDIUM,
            reason=f"Symbol `{symbol}` not found in codebase",
            expected=symbol,
            verifier=self.name,
        )

    def _find_similar(self, repo_root: Path, symbol: str) -> list[str]:
        """Find symbols with similar names (simple prefix/suffix match)."""
        # Extract the core name (last part if dotted)
        core = symbol.split(".")[-1] if "." in symbol else symbol
        if len(core) < 4:
            return []

        # Search for definitions containing the core name
        hits = self._search_files(
            repo_root,
            rf"\b(?:def|class)\s+\w*{re.escape(core[:4])}\w*\b",
        )
        names: list[str] = []
        seen: set[str] = set()
        for _, _, line in hits:
            m = re.search(r"\b(?:def|class)\s+(\w+)", line)
            if m:
                name = m.group(1)
                if name != symbol and name not in seen:
                    seen.add(name)
                    names.append(name)
        return names[:5]


# Auto-register
register(SymbolExistsVerifier())
