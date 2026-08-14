"""Enum verifiers — count and member checks.

Verifies claims about enum/set size and membership using the pre-built
RepoIndex which has already parsed all Python enums via AST.
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


class EnumCountVerifier(BaseVerifier):
    """Verify that an enum has the claimed number of members."""

    @property
    def name(self) -> str:
        return "enum_count"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.ENUM_COUNT})

    def verify(self, claim: Claim, repo_root: Path, index: RepoIndex) -> VerificationResult:
        obj = claim.object
        if not isinstance(obj, dict) or "count" not in obj:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.ERROR,
                reason="Claim object missing 'count'", verifier=self.name,
            )

        expected_count = obj["count"]
        noun = obj.get("noun", "")

        # Try to match an enum by context
        best_match = None
        for cls_name, enum_def in index.enums.items():
            if self._matches_context(cls_name, claim, noun):
                best_match = (cls_name, enum_def)
                break

        if best_match is None:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.UNVERIFIABLE, confidence=0.3,
                reason=f"Could not find enum matching '{noun}' context",
                expected=expected_count, verifier=self.name,
            )

        cls_name, enum_def = best_match
        actual_count = len(enum_def.members)

        if actual_count == expected_count:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=0.95,
                reason=f"`{cls_name}` has {actual_count} members (matches doc)",
                expected=expected_count, observed=actual_count,
                evidence=[Evidence(file_path=enum_def.file_path, line=enum_def.line, symbol=cls_name)],
                verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id, verdict=Verdict.CONTRADICTED, confidence=0.9,
            severity=Severity.HIGH,
            reason=f"`{cls_name}` has {actual_count} members, doc claims {expected_count}",
            expected=expected_count,
            observed={"count": actual_count, "members": enum_def.members},
            evidence=[Evidence(file_path=enum_def.file_path, line=enum_def.line, symbol=cls_name)],
            verifier=self.name,
        )

    def _matches_context(self, cls_name: str, claim: Claim, noun: str) -> bool:
        name_lower = cls_name.lower()
        if noun and noun.lower() in name_lower:
            return True
        raw_lower = claim.raw_text.lower()
        if cls_name in claim.raw_text or name_lower in raw_lower:
            return True
        if claim.section and name_lower in claim.section.lower():
            return True
        return False


class EnumMembersVerifier(BaseVerifier):
    """Verify that specific enum members exist."""

    @property
    def name(self) -> str:
        return "enum_members"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.ENUM_MEMBERS})

    def verify(self, claim: Claim, repo_root: Path, index: RepoIndex) -> VerificationResult:
        obj = claim.object
        if not isinstance(obj, dict) or "members" not in obj:
            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.ERROR,
                reason="Claim object missing 'members'", verifier=self.name,
            )

        expected_members = set(obj["members"])
        enum_name = obj.get("enum_name", "")

        for cls_name, enum_def in index.enums.items():
            if enum_name and enum_name.lower() not in cls_name.lower():
                continue
            actual_names = set(enum_def.members)
            missing = expected_members - actual_names

            if not missing:
                return VerificationResult(
                    claim_id=claim.id, verdict=Verdict.CONFIRMED, confidence=0.95,
                    reason=f"All claimed members found in `{cls_name}`",
                    verifier=self.name,
                )

            return VerificationResult(
                claim_id=claim.id, verdict=Verdict.CONTRADICTED, confidence=0.9,
                severity=Severity.HIGH,
                reason=f"`{cls_name}` missing: {missing}; extra: {actual_names - expected_members}",
                expected=sorted(expected_members), observed=sorted(actual_names),
                verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id, verdict=Verdict.UNVERIFIABLE,
            reason=f"Enum '{enum_name}' not found", verifier=self.name,
        )


register(EnumCountVerifier())
register(EnumMembersVerifier())
