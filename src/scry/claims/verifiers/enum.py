"""Enum verifiers — count and member checks.

Verifies claims about enum/set size and membership by parsing Python
enum definitions and comparing against documented claims.
"""

from __future__ import annotations

import ast
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


def _find_enum_members(repo_root: Path, enum_hint: str) -> dict[str, list[tuple[str, Path, int]]]:
    """Find Python enum/StrEnum definitions and their members.

    Returns dict of {class_name: [(member_name, file, line), ...]}.
    """
    results: dict[str, list[tuple[str, Path, int]]] = {}

    for fp in repo_root.rglob("*.py"):
        if ".git" in fp.parts or "__pycache__" in fp.parts:
            continue
        try:
            source = fp.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(source, filename=str(fp))
        except (SyntaxError, OSError):
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            # Check if this class inherits from an Enum type
            is_enum = any(
                (isinstance(b, ast.Name) and b.id in ("Enum", "StrEnum", "IntEnum", "Flag"))
                or (isinstance(b, ast.Attribute) and b.attr in ("Enum", "StrEnum", "IntEnum", "Flag"))
                for b in node.bases
            )
            if not is_enum:
                continue

            members: list[tuple[str, Path, int]] = []
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id.isupper():
                            members.append((target.id, fp, item.lineno))
                elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                    if item.target.id.isupper() or (item.value is not None):
                        members.append((item.target.id, fp, item.lineno))

            if members:
                results[node.name] = members

    return results


class EnumCountVerifier(BaseVerifier):
    """Verify that an enum has the claimed number of members."""

    @property
    def name(self) -> str:
        return "enum_count"

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        return frozenset({ClaimType.ENUM_COUNT})

    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult:
        obj = claim.object
        if not isinstance(obj, dict) or "count" not in obj:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.ERROR,
                reason="Claim object missing 'count'",
                verifier=self.name,
            )

        expected_count = obj["count"]
        noun = obj.get("noun", "")

        # Search for enum definitions
        all_enums = _find_enum_members(repo_root, noun)

        # Try to match by noun context
        best_match: tuple[str, list[tuple[str, Path, int]]] | None = None
        for cls_name, members in all_enums.items():
            # Check if the enum name relates to the section/context
            if self._matches_context(cls_name, claim, noun):
                best_match = (cls_name, members)
                break

        if best_match is None:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.UNVERIFIABLE,
                confidence=0.3,
                reason=f"Could not find enum matching '{noun}' context",
                expected=expected_count,
                verifier=self.name,
            )

        cls_name, members = best_match
        actual_count = len(members)

        if actual_count == expected_count:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.CONFIRMED,
                confidence=0.95,
                reason=f"`{cls_name}` has {actual_count} members (matches doc)",
                expected=expected_count,
                observed=actual_count,
                evidence=[Evidence(
                    file_path=str(members[0][1].relative_to(repo_root)),
                    line=members[0][2],
                    symbol=cls_name,
                )],
                verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id,
            verdict=Verdict.CONTRADICTED,
            confidence=0.9,
            severity=Severity.HIGH,
            reason=f"`{cls_name}` has {actual_count} members, doc claims {expected_count}",
            expected=expected_count,
            observed={"count": actual_count, "members": [m[0] for m in members]},
            evidence=[Evidence(
                file_path=str(members[0][1].relative_to(repo_root)),
                line=members[0][2],
                symbol=cls_name,
            )],
            verifier=self.name,
        )

    def _matches_context(self, cls_name: str, claim: Claim, noun: str) -> bool:
        """Check if an enum class name matches the claim context."""
        name_lower = cls_name.lower()
        # Direct noun match
        if noun and noun.lower() in name_lower:
            return True
        # Check section context
        section_lower = claim.section.lower() if claim.section else ""
        raw_lower = claim.raw_text.lower()
        # Check if class name appears in the claim text
        if cls_name in claim.raw_text or name_lower in raw_lower:
            return True
        if name_lower in section_lower:
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

    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult:
        obj = claim.object
        if not isinstance(obj, dict) or "members" not in obj:
            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.ERROR,
                reason="Claim object missing 'members'",
                verifier=self.name,
            )

        expected_members = set(obj["members"])
        enum_name = obj.get("enum_name", "")

        all_enums = _find_enum_members(repo_root, enum_name)

        for cls_name, members in all_enums.items():
            if enum_name and enum_name.lower() not in cls_name.lower():
                continue
            actual_names = {m[0] for m in members}
            missing = expected_members - actual_names
            extra = actual_names - expected_members

            if not missing:
                return VerificationResult(
                    claim_id=claim.id,
                    verdict=Verdict.CONFIRMED,
                    confidence=0.95,
                    reason=f"All claimed members found in `{cls_name}`",
                    verifier=self.name,
                )

            return VerificationResult(
                claim_id=claim.id,
                verdict=Verdict.CONTRADICTED,
                confidence=0.9,
                severity=Severity.HIGH,
                reason=f"`{cls_name}` missing: {missing}; extra: {extra}",
                expected=sorted(expected_members),
                observed=sorted(actual_names),
                verifier=self.name,
            )

        return VerificationResult(
            claim_id=claim.id,
            verdict=Verdict.UNVERIFIABLE,
            reason=f"Enum '{enum_name}' not found",
            verifier=self.name,
        )


register(EnumCountVerifier())
register(EnumMembersVerifier())
