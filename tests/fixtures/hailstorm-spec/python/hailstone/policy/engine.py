"""Policy engine implementation."""

from __future__ import annotations


class PolicyRule:
    """An access-control rule evaluated by the policy engine.

    Attributes:
        name:      Human-readable identifier for the rule.
        condition: Python expression string evaluated against the request context.
        effect:    ``'allow'`` or ``'deny'`` (default ``'allow'``).
        priority:  Evaluation order; lower number = higher precedence.
    """

    def __init__(
        self,
        name: str,
        condition: str,
        effect: str = "allow",
        priority: int = 0,
    ) -> None:
        self.name = name
        self.condition = condition
        self.effect = effect
        self.priority = priority

    def __repr__(self) -> str:
        return f"PolicyRule({self.name!r}, effect={self.effect!r}, priority={self.priority})"


def evaluate(rule: PolicyRule, context: dict) -> str | None:
    """Evaluate *rule* against *context*.

    Returns the rule's effect string when the condition matches, or ``None``
    when the rule does not apply to this context.

    Args:
        rule:    The policy rule to evaluate.
        context: Request attribute dictionary (e.g. ``{'user': 'alice', 'action': 'read'}``).

    Returns:
        ``'allow'``, ``'deny'``, or ``None``.
    """
    try:
        if eval(rule.condition, {}, context):
            return rule.effect
    except Exception:
        return None
    return None
