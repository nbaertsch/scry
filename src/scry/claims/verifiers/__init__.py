"""Verifier framework for claim-level documentation checking.

Each verifier handles one or more ``ClaimType`` values and produces
``VerificationResult`` objects.  Verifiers are registered in ``REGISTRY``
and dispatched by claim type.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol, runtime_checkable

from scry.claims.model import (
    Claim,
    ClaimType,
    VerificationResult,
)


@runtime_checkable
class Verifier(Protocol):
    """Protocol for claim verifiers."""

    @property
    def name(self) -> str:
        """Human-readable verifier name."""
        ...

    @property
    def supported_types(self) -> frozenset[ClaimType]:
        """Claim types this verifier can handle."""
        ...

    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult:
        """Verify a single claim against the codebase at repo_root."""
        ...


class BaseVerifier(ABC):
    """Base class for verifiers with common utilities."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_types(self) -> frozenset[ClaimType]: ...

    @abstractmethod
    def verify(self, claim: Claim, repo_root: Path) -> VerificationResult: ...

    def _search_files(
        self,
        repo_root: Path,
        pattern: str,
        *,
        glob_pattern: str = "**/*.py",
    ) -> list[tuple[Path, int, str]]:
        """Search files for a text pattern, return (path, line_no, line_text) tuples."""
        import re

        results: list[tuple[Path, int, str]] = []
        compiled = re.compile(pattern)
        for fp in repo_root.glob(glob_pattern):
            if ".git" in fp.parts or "__pycache__" in fp.parts:
                continue
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeDecodeError):
                continue
            for i, line in enumerate(text.splitlines(), 1):
                if compiled.search(line):
                    results.append((fp, i, line.strip()))
        return results

    def _read_file(self, repo_root: Path, rel_path: str) -> str | None:
        """Read a file relative to repo root, return None if missing."""
        fp = repo_root / rel_path
        if not fp.is_file():
            return None
        try:
            return fp.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None


# ───── Registry ──────────────────────────────────────────────────────

REGISTRY: dict[ClaimType, Verifier] = {}


def register(verifier: Verifier) -> Verifier:
    """Register a verifier for its supported claim types."""
    for ct in verifier.supported_types:
        REGISTRY[ct] = verifier
    return verifier


def get_verifier(claim_type: ClaimType) -> Verifier | None:
    """Look up the verifier for a claim type."""
    return REGISTRY.get(claim_type)


def verify_claim(claim: Claim, repo_root: Path) -> VerificationResult:
    """Verify a single claim using the appropriate registered verifier."""
    from scry.claims.model import Verdict

    verifier = get_verifier(claim.claim_type)
    if verifier is None:
        return VerificationResult(
            claim_id=claim.id,
            verdict=Verdict.UNVERIFIABLE,
            reason=f"No verifier registered for claim type '{claim.claim_type}'",
            verifier="dispatch",
        )
    return verifier.verify(claim, repo_root)
