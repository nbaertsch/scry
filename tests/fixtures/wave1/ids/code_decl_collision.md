# Configuration

## Config Classes

```python
class Config:
    """First Config class in this section."""

    debug: bool = False
```

```python
class Config:
    """Second Config class in this section — declaration collision."""

    debug: bool = True
    verbose: bool = False
```

Two blocks with the same declaration name in the same section.
The first gets bare name ``config``; subsequent occurrences get ``@2``, ``@3``, …
