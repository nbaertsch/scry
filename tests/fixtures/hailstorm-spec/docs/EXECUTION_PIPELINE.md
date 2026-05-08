# Execution Pipeline

The Execution Pipeline orchestrates request handling from ingestion through
policy evaluation to response emission.

## Pipeline Stages

Requests pass through three ordered stages:

1. **Parse** — deserialise the raw HTTP body and validate required fields.
2. **Evaluate** — run the request context through the Policy Engine.
3. **Emit** — serialise and send the response.

## Stage Contracts

Each stage must implement the `Stage` protocol: a synchronous `process`
method that accepts the current request context dict and returns an updated
context dict.

```python
def process_pipeline(stages: list, context: dict) -> dict:
    """Run *context* through each stage in *stages* in order.

    Each stage receives the context returned by the previous stage.
    If a stage raises, the pipeline aborts and the exception propagates.
    """
    for stage in stages:
        context = stage.process(context)
    return context
```

## Error Handling

Unhandled exceptions from any stage are caught at the pipeline boundary,
logged with full traceback, and converted to a ``500 Internal Server Error``
response.
