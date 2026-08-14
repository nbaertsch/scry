"""Route path and file path verifiers.

- ``RoutePathVerifier``: checks API route paths against FastAPI/Flask decorators
- ``FilePathVerifier``: checks file/directory existence on disk
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

# Route decorator patterns for common frameworks
_ROUTE_DECORATOR_RE = re.compile(
    r"@\w*\.(?:get|post|put|patch|delete|head|options|websocket|api_route|route)"
    r'\(\s*["\']([^"\']+)["\']',
    re.IGNORECASE,
)

# app.add_api_route / app.mount patterns
_MOUNT_RE = re.compile(
    r"\.(?:add_api_route|mount|add_route|include_router)"
    r'\(\s*["\']([^"\']+)["\']',
)


class RoutePathVerifier(BaseVerifier):
    """Verify that an API route path exists in the codebase."""

    @property
    def name(self) -> str:
        return "route_path"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.ROUTE_PATH})

    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult:
        route = claim.subject
        if not route:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.ERROR,
                reason="No route path in claim subject",
                verifier=self.name,
            )

        # Normalize: strip trailing slash, collapse path params
        normalized = route.rstrip("/")
        # Create a pattern that matches parameterized segments
        param_pattern = re.sub(r"\{[^}]+\}", r"\\{[^}]+\\}", re.escape(normalized))

        # Search for route in decorators and mounts
        evidence: list[Evidence] = []

        for fp in repo_root.rglob("*.py"):
            if ".git" in fp.parts or "__pycache__" in fp.parts:
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            for i, line in enumerate(text.splitlines(), 1):
                for pattern in (_ROUTE_DECORATOR_RE, _MOUNT_RE):
                    for m in pattern.finditer(line):
                        found_path = m.group(1)
                        # Check exact match or parameterized match
                        if found_path == normalized or re.fullmatch(param_pattern, found_path):
                            evidence.append(Evidence(
                                file_path=str(fp.relative_to(repo_root)),
                                line=i,
                                snippet=line.strip()[:200],
                            ))
                        # Also check if the claimed path is a suffix of a composed route
                        elif normalized.endswith(found_path) or found_path.endswith(normalized.split("/")[-1]):
                            evidence.append(Evidence(
                                file_path=str(fp.relative_to(repo_root)),
                                line=i,
                                snippet=line.strip()[:200],
                            ))

        if evidence:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.CONFIRMED,
                confidence=0.85,
                reason=f"Route `{route}` found in {len(evidence)} location(s)",
                evidence=evidence[:5],
                verifier=self.name,
            )

        # Try a broader search for the path string
        escaped = re.escape(normalized.split("/")[-2] + "/" + normalized.split("/")[-1]) if "/" in normalized else re.escape(normalized)
        broad_hits = self._search_files(repo_root, escaped)
        if broad_hits:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.INCOMPLETE,
                confidence=0.5,
                severity=Severity.MEDIUM,
                reason=f"Route `{route}` not found in decorators but path fragment exists in code",
                expected=route,
                evidence=[Evidence(
                    file_path=str(h[0].relative_to(repo_root)),
                    line=h[1],
                    snippet=h[2][:200],
                ) for h in broad_hits[:3]],
                verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id,
            verdict=Verdict.CONTRADICTED,
            confidence=0.7,
            severity=Severity.HIGH,
            reason=f"Route `{route}` not found in codebase",
            expected=route,
            verifier=self.name,
        )


class FilePathVerifier(BaseVerifier):
    """Verify that a file or directory path exists."""

    @property
    def name(self) -> str:
        return "file_path"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.FILE_PATH})

    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult:
        file_path = claim.subject
        if not file_path:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.ERROR,
                reason="No file path in claim subject",
                verifier=self.name,
            )

        target = repo_root / file_path

        if target.exists():
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.CONFIRMED,
                confidence=1.0,
                reason=f"Path `{file_path}` exists",
                evidence=[Evidence(file_path=file_path)],
                verifier=self.name,
            )

        # Check for similar paths (possible rename)
        parent = target.parent
        if parent.exists():
            siblings = [p.name for p in parent.iterdir() if not p.name.startswith(".")]
            target_name = target.name
            similar = [s for s in siblings if _similar(s, target_name)]
            if similar:
                return VerificationResult(
                    claim_id=claim.id,
                    verdict=Verdict.STALE_TARGET,
                    confidence=0.8,
                    severity=Severity.MEDIUM,
                    reason=f"Path `{file_path}` not found; similar: {', '.join(similar[:3])}",
                    expected=file_path,
                    observed=similar[:3],
                    verifier=self.name,
                )

        return VerificationResult(
            claim_id=claim.id,
            verdict=Verdict.CONTRADICTED,
            confidence=0.95,
            severity=Severity.HIGH,
            reason=f"Path `{file_path}` does not exist",
            expected=file_path,
            verifier=self.name,
        )


def _similar(a: str, b: str) -> bool:
    """Check if two filenames are similar (share a long common substring)."""
    a_lower, b_lower = a.lower(), b.lower()
    if a_lower == b_lower:
        return False
    # Share at least 60% of chars
    common = sum(1 for c in a_lower if c in b_lower)
    return common > 0.6 * max(len(a), len(b))


register(RoutePathVerifier())
register(FilePathVerifier())
