> Condensed from [Langfuse DSPy integration docs](https://langfuse.com/docs/integrations/dspy) and [Langfuse Python SDK docs](https://langfuse.com/docs/sdk/python/decorators). Verify against upstream for latest.

# Langfuse DSPy Integration — API Reference

## Installation

```bash
pip install langfuse dspy openinference-instrumentation-dspy -U
```

## DSPyInstrumentor

Auto-instruments all DSPy modules for tracing. Must be called before any DSPy module execution.

```python
from openinference.instrumentation.dspy import DSPyInstrumentor

DSPyInstrumentor().instrument()
```

Captures: LM calls (prompts, responses, tokens, cost, latency), retrievals (queries, passages), module executions (input/output), nested spans (full call tree).

## Langfuse client

```python
from langfuse import get_client

langfuse = get_client()  # Singleton — safe to call multiple times
langfuse.auth_check()    # Verify credentials
```

### Key client methods

| Method | Description |
|--------|-------------|
| `get_client()` | Returns singleton Langfuse client |
| `langfuse.auth_check()` | Verify API credentials |
| `langfuse.flush()` | Send pending traces (required for scripts/notebooks) |
| `langfuse.shutdown()` | Flush and terminate background threads (use on process exit) |
| `langfuse.score(trace_id, name, value, data_type, comment)` | Attach a score to a trace |
| `langfuse.get_current_trace_id()` | Get the active trace ID |
| `langfuse.start_as_current_observation(as_type, name)` | Context manager for manual trace/span creation |

## @observe() decorator

Creates a traced observation around a function.

```python
from langfuse import observe

@observe(name=None, capture_input=True, capture_output=True)
def my_function():
    pass
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | function name | Custom observation name |
| `capture_input` | `bool` | `True` | Capture function input arguments |
| `capture_output` | `bool` | `True` | Capture function return value |

## propagate_attributes()

Context manager that attaches metadata to auto-captured DSPy traces within its scope.

```python
from langfuse import propagate_attributes

with propagate_attributes(
    user_id="user_123",
    session_id="session_abc",
    tags=["production"],
    metadata={"key": "value"},
    version="1.0",
):
    # DSPy calls here inherit these attributes
    result = program(question="...")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `user_id` | `str \| None` | `None` | User identifier for filtering |
| `session_id` | `str \| None` | `None` | Session identifier for grouping multi-turn traces |
| `tags` | `list[str] \| None` | `None` | Tags for filtering and organization |
| `metadata` | `dict \| None` | `None` | Arbitrary key-value metadata |
| `version` | `str \| None` | `None` | Version identifier |

## Scoring

Attach evaluation results to traces. Score types: `NUMERIC`, `CATEGORICAL`, `BOOLEAN`, `TEXT`.

```python
langfuse.score(
    trace_id="trace-id",
    name="accuracy",
    value=0.92,           # float for NUMERIC, bool for BOOLEAN, str for CATEGORICAL/TEXT
    data_type="NUMERIC",  # "NUMERIC" | "CATEGORICAL" | "BOOLEAN" | "TEXT"
    comment="Optional explanation",
)
```

## Environment variables

| Variable | Description |
|----------|-------------|
| `LANGFUSE_PUBLIC_KEY` | Public API key (`pk-lf-...`) |
| `LANGFUSE_SECRET_KEY` | Secret API key (`sk-lf-...`) |
| `LANGFUSE_BASE_URL` | Endpoint URL (see regional endpoints below) |
| `LANGFUSE_DEBUG` | Set to `"True"` for verbose logging |
| `LANGFUSE_OBSERVE_DECORATOR_IO_CAPTURE_ENABLED` | Set to `"false"` to disable I/O capture globally |

### Regional endpoints

| Region | URL |
|--------|-----|
| EU | `https://cloud.langfuse.com` |
| US | `https://us.cloud.langfuse.com` |
| Japan | `https://jp.cloud.langfuse.com` |
| HIPAA | `https://hipaa.cloud.langfuse.com` |
| Self-hosted | Your instance URL |

## Key features beyond tracing

| Feature | Description |
|---------|-------------|
| **Scores** | Numeric, boolean, categorical, or text scores on any trace |
| **Annotation queues** | Structured human review workflows |
| **Experiments** | Compare prompt versions with tag-based filtering |
| **Sessions** | Group multi-turn traces by `session_id` |
| **Environments** | Separate dev/staging/prod via tags or metadata |
| **Prompt management** | Version and manage prompts in the Langfuse dashboard |
