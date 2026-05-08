# API Documentation

## Foo Class

```python
class Foo:
    """Foo class documentation."""

    def __init__(self, name: str) -> None:
        self.name = name

    def greet(self) -> str:
        return f"Hello, {self.name}"
```

This block contains a class declaration and should become a ``code_in_doc``
anchor with ID suffix ``::foo``.
