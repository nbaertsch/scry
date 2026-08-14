"""Route path and file path verifiers using the pre-built RepoIndex."""

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


class RoutePathVerifier(BaseVerifier):
    """Verify that an API route path exists in the codebase."""

    @property
    def name(self) -> str:
        return "route_path"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.ROUTE_PATH})

    def verify(self, claim: Claim, repo_root: Path, index: RepoIndex) -> VerificationResult:
        route = claim.subject
        if not route:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.ERROR,
                reason="No route path in claim subject", verifier=self.name,
            )

        normalized = route.rstrip("/")
        # Build param pattern for matching
        param_pattern = re.sub(r"\{[^}]+\}", r"\\{[^}]+\\}", re.escape(normalized))

        evidence: list[Evidence] = []
        for rd in index.routes:
            if rd.path == normalized or re.fullmatch(param_pattern, rd.path):
                evidence.append(Evidence(
                    file_path=rd.file_path, line=rd.line, snippet=rd.snippet,
                ))
            elif normalized.endswith(rd.path) or rd.path.endswith(normalized.split("/")[-1]):
                evidence.append(Evidence(
                    file_path=rd.file_path, line=rd.line, snippet=rd.snippet,
                ))

        if evidence:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=0.85,
                reason=f"Route `{route}` found in {len(evidence)} location(s)",
                evidence=evidence[:5], verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id, verdict=Verdict.CONTRADICTED, confidence=0.7,
            severity=Severity.HIGH,
            reason=f"Route `{route}` not found in codebase",
            expected=route, verifier=self.name,
        )


class FilePathVerifier(BaseVerifier):
    """Verify that a file or directory path exists."""

    @property
    def name(self) -> str:
        return "file_path"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.FILE_PATH})

    def verify(self, claim: Claim, repo_root: Path, index: RepoIndex) -> VerificationResult:
        file_path = claim.subject
        if not file_path:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.ERROR,
                reason="No file path in claim subject", verifier=self.name,
            )

        # Check index (files set) — also check if it's a directory prefix
        normalized = file_path.replace("\\", "/")
        if index.file_exists(normalized):
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=1.0,
                reason=f"Path `{file_path}` exists",
                evidence=[Evidence(file_path=file_path)], verifier=self.name,
            )

        # Check if it's a directory (any file starts with this path)
        dir_prefix = normalized.rstrip("/") + "/"
        if any(f.startswith(dir_prefix) for f in index.files):
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=1.0,
                reason=f"Directory `{file_path}` exists",
                evidence=[Evidence(file_path=file_path)], verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id, verdict=Verdict.CONTRADICTED, confidence=0.95,
            severity=Severity.HIGH,
            reason=f"Path `{file_path}` does not exist",
            expected=file_path, verifier=self.name,
        )


register(RoutePathVerifier())
register(FilePathVerifier())
