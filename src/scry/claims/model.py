"""Data models for claim-level documentation verification.

Claims are atomic, falsifiable assertions extracted from documentation.
Each claim has a type that determines which verifier handles it and what
code constructs it should be checked against.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# ───── Claim types ───────────────────────────────────────────────────


class ClaimType(StrEnum):
    """Taxonomy of verifiable documentation claims."""

    SYMBOL_EXISTS = "symbol_exists"
    """A named symbol (function, class, variable) is claimed to exist."""

    SYMBOL_SIGNATURE = "symbol_signature"
    """A symbol has a specific signature, return type, or parameters."""

    NUMERIC_VALUE = "numeric_value"
    """A specific number is claimed (default, limit, count, timeout)."""

    ENUM_COUNT = "enum_count"
    """An enum/set is claimed to have N members."""

    ENUM_MEMBERS = "enum_members"
    """Specific enum/set members are listed."""

    ENV_VAR_NAME = "env_var_name"
    """An environment variable name is referenced."""

    ROUTE_PATH = "route_path"
    """An API route path (and optionally method) is claimed."""

    FILE_PATH = "file_path"
    """A file or directory path is referenced."""

    COVERAGE_ASSERTION = "coverage_assertion"
    """A universal claim ('every', 'all', 'each', 'never', 'always')."""

    BEHAVIORAL = "behavioral"
    """A behavioral/architectural claim requiring LLM or complex analysis."""


# ───── Verdict ───────────────────────────────────────────────────────


class Verdict(StrEnum):
    """Result of verifying a single claim against code."""

    CONFIRMED = "confirmed"
    """Claim is verified as correct."""

    CONTRADICTED = "contradicted"
    """Claim is definitively wrong."""

    INCOMPLETE = "incomplete"
    """Claim is partially correct but missing information."""

    UNVERIFIABLE = "unverifiable"
    """Cannot be mechanically verified (needs human/LLM review)."""

    STALE_TARGET = "stale_target"
    """The code target referenced by the claim no longer exists."""

    ERROR = "error"
    """Verification encountered an error."""


class Severity(StrEnum):
    """How important a verification failure is."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# ───── Span ──────────────────────────────────────────────────────────


class DocSpan(BaseModel):
    """Location of a claim within a document."""

    model_config = ConfigDict(frozen=True)

    start_line: int
    """1-based start line in the document."""

    end_line: int
    """1-based end line (inclusive)."""

    start_char: int | None = None
    """Character offset within start_line (optional)."""

    end_char: int | None = None
    """Character offset within end_line (optional)."""


# ───── Evidence ──────────────────────────────────────────────────────


class Evidence(BaseModel):
    """A piece of code evidence supporting a verification verdict."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    """Repo-relative path to the evidence file."""

    line: int | None = None
    """Line number (1-based) where evidence was found."""

    line_end: int | None = None
    """End line for multi-line evidence."""

    snippet: str | None = None
    """Code snippet showing the evidence."""

    symbol: str | None = None
    """Symbol name if evidence is a specific symbol."""


# ───── Claim ─────────────────────────────────────────────────────────


def _claim_id(doc_path: str, span_start: int, raw_text: str) -> str:
    """Generate a stable claim ID from document location and content."""
    content = f"{doc_path}:{span_start}:{raw_text}"
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"claim_{digest}"


class Claim(BaseModel):
    """An atomic, falsifiable assertion extracted from documentation.

    A claim is the smallest unit of documentation that can be verified
    against source code.  Each claim has a type that routes it to the
    appropriate verifier.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default="", description="Stable claim identifier.")
    doc_path: str = Field(description="Repo-relative path to the doc file.")
    section: str = Field(default="", description="Heading breadcrumb / section title.")
    span: DocSpan = Field(description="Location in the document.")
    raw_text: str = Field(description="Original text the claim was extracted from.")
    claim_text: str = Field(description="Normalized atomic claim statement.")
    claim_type: ClaimType = Field(description="Type determining which verifier to use.")

    # Subject-predicate-object triple
    subject: str = Field(default="", description="What the claim is about.")
    predicate: str = Field(default="", description="What is asserted (e.g. 'equals', 'exists', 'contains').")
    object: Any = Field(default=None, description="The expected value, set, count, or path.")

    # Verification routing
    evidence_selectors: list[str] = Field(
        default_factory=list,
        description="File patterns or symbol names to search for evidence.",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Extraction confidence (how certain we are this is a real claim).",
    )
    needs_llm: bool = Field(
        default=False,
        description="Whether this claim requires LLM adjudication.",
    )

    def model_post_init(self, __context: Any) -> None:
        if not self.id:
            object.__setattr__(
                self,
                "id",
                _claim_id(self.doc_path, self.span.start_line, self.raw_text),
            )


# ───── ClaimDependency ───────────────────────────────────────────────


class DependencyKind(StrEnum):
    """How a claim depends on a code artifact."""

    SYMBOL = "symbol"
    FILE = "file"
    PATTERN = "pattern"


class ClaimDependency(BaseModel):
    """Links a claim to the code it depends on for verification."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    file_path: str
    """Repo-relative file that the claim depends on."""

    symbol: str | None = None
    """Specific symbol within the file (optional)."""

    kind: DependencyKind = DependencyKind.FILE


# ───── VerificationResult ────────────────────────────────────────────


class VerificationResult(BaseModel):
    """Result of verifying a single claim against source code."""

    model_config = ConfigDict(frozen=True)

    claim_id: str
    verdict: Verdict
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="How confident the verifier is in its verdict.",
    )
    severity: Severity = Severity.MEDIUM
    reason: str = Field(default="", description="Human-readable explanation.")

    expected: Any = Field(default=None, description="What the doc claims.")
    observed: Any = Field(default=None, description="What the code actually has.")

    evidence: list[Evidence] = Field(
        default_factory=list,
        description="Code locations supporting the verdict.",
    )
    verifier: str = Field(default="", description="Name of the verifier that produced this.")

    # Caching
    code_fingerprint: str = Field(
        default="",
        description="Hash of the code state when this result was produced.",
    )
    verified_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def failed(self) -> bool:
        """Whether this result represents a verification failure."""
        return self.verdict in (Verdict.CONTRADICTED, Verdict.INCOMPLETE, Verdict.STALE_TARGET)


# ───── VerificationReport ────────────────────────────────────────────


class VerificationReport(BaseModel):
    """Aggregate report from verifying claims in one or more documents."""

    total_claims: int = 0
    verified: int = 0
    confirmed: int = 0
    contradicted: int = 0
    incomplete: int = 0
    unverifiable: int = 0
    stale_target: int = 0
    errors: int = 0

    results: list[VerificationResult] = Field(default_factory=list)

    @property
    def failed_results(self) -> list[VerificationResult]:
        return [r for r in self.results if r.failed]

    @property
    def pass_rate(self) -> float:
        if self.verified == 0:
            return 0.0
        return self.confirmed / self.verified

    def add(self, result: VerificationResult) -> None:
        """Add a verification result and update counters."""
        self.results.append(result)
        self.total_claims += 1
        self.verified += 1
        match result.verdict:
            case Verdict.CONFIRMED:
                self.confirmed += 1
            case Verdict.CONTRADICTED:
                self.contradicted += 1
            case Verdict.INCOMPLETE:
                self.incomplete += 1
            case Verdict.UNVERIFIABLE:
                self.unverifiable += 1
                self.verified -= 1  # not actually verified
            case Verdict.STALE_TARGET:
                self.stale_target += 1
            case Verdict.ERROR:
                self.errors += 1
                self.verified -= 1
