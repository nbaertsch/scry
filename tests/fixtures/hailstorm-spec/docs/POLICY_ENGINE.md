# Policy Engine

The Policy Engine evaluates access-control rules against request contexts.
Each rule is a predicate over a set of attributes; the engine short-circuits
on the first matching rule.

## Rule Structure

A `PolicyRule` combines a condition predicate with an effect and an optional
priority.

```python
class PolicyRule:
    """An access-control rule evaluated by the policy engine."""

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
```

Rules are stored in priority order; the first matching rule determines the
decision.

## Rule Evaluation

The `evaluate` function tests a rule against a request context and returns
the rule's effect when the condition is satisfied.

```python
def evaluate(rule: PolicyRule, context: dict) -> str | None:
    """Evaluate *rule* against *context*.

    Returns the rule's effect string (``'allow'`` or ``'deny'``) when the
    condition matches, or ``None`` when the rule does not apply.
    """
    try:
        if eval(rule.condition, {}, context):  # noqa: S307
            return rule.effect
    except Exception:
        return None
    return None
```

## Rule Priority Resolution

When multiple rules match a request context, the rule with the lowest
`priority` value wins (lower number = higher precedence).

## Engine Lifecycle

The engine is stateless between evaluations.  Rules are compiled once at
startup and stored in sorted order.  Hot-reload is supported by swapping
the rule list under a read-write lock.
