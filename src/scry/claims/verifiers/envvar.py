"""Environment variable name verifier.

Checks whether environment variable names referenced in documentation
actually appear in the codebase (as string literals, os.environ access,
or env-related patterns).
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


class EnvVarVerifier(BaseVerifier):
    """Verify that an environment variable name exists in the codebase."""

    @property
    def name(self) -> str:
        return "env_var"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.ENV_VAR_NAME})

    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult:
        var_name = claim.subject
        if not var_name:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.ERROR,
                reason="No env var name in claim subject",
                verifier=self.name,
            )

        # Search for the env var in code
        patterns = [
            rf'["\']{re.escape(var_name)}["\']',  # String literal
            rf"\b{re.escape(var_name)}\b",  # Direct reference
        ]

        evidence: list[Evidence] = []
        for pat in patterns:
            hits = self._search_files(repo_root, pat)
            for fp, line_no, line_text in hits:
                rel = str(fp.relative_to(repo_root))
                # Skip doc files — we want code references
                if rel.startswith("docs/") or rel.endswith(".md"):
                    continue
                evidence.append(Evidence(
                    file_path=rel,
                    line=line_no,
                    snippet=line_text[:200],
                ))
                if len(evidence) >= 5:
                    break
            if evidence:
                break

        if evidence:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.CONFIRMED,
                confidence=0.9,
                reason=f"Env var `{var_name}` found in code",
                evidence=evidence,
                verifier=self.name,
            )

        # Check for similar env var names (possible rename)
        prefix = var_name.split("_")[0] + "_" if "_" in var_name else var_name[:4]
        similar_hits = self._search_files(
            repo_root,
            rf'["\']({re.escape(prefix)}\w+)["\']',
        )
        similar_names: list[str] = []
        seen: set[str] = set()
        for _, _, line in similar_hits:
            for m in re.finditer(rf'["\']({re.escape(prefix)}\w+)["\']', line):
                name = m.group(1)
                if name != var_name and name not in seen:
                    seen.add(name)
                    similar_names.append(name)

        if similar_names:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.STALE_TARGET,
                confidence=0.7,
                severity=Severity.HIGH,
                reason=f"Env var `{var_name}` not found; similar: {', '.join(similar_names[:5])}",
                expected=var_name,
                observed=similar_names[:5],
                verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id,
            verdict=Verdict.CONTRADICTED,
            confidence=0.6,
            severity=Severity.MEDIUM,
            reason=f"Env var `{var_name}` not found in codebase",
            expected=var_name,
            verifier=self.name,
        )


register(EnvVarVerifier())
