> Condensed from [Langtrace DSPy integration docs](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy). Verify against upstream for latest.

# Langtrace DSPy Integration — API Reference

## Installation

```bash
pip install langtrace-python-sdk
```

## langtrace.init()

Initializes Langtrace and patches DSPy for auto-instrumentation. Must be called before any DSPy imports.

```python
from langtrace_python_sdk import langtrace

langtrace.init(
    api_key=None,       # Langtrace API key (or set LANGTRACE_API_KEY env var)
    api_host=None,      # Custom endpoint for self-hosted (or set LANGTRACE_API_HOST env var)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | env `LANGTRACE_API_KEY` | API key for cloud or self-hosted |
| `api_host` | `str \| None` | `https://app.langtrace.ai` | Custom endpoint URL (for self-hosted instances) |

## Auto-instrumentation

After `langtrace.init()`, all DSPy calls are traced automatically:

| Component | What gets captured |
|-----------|-------------------|
| LM calls | Prompts, responses, token counts, cost, latency |
| Retrievals | Queries, retrieved passages, scores |
| Module executions | Input/output per `forward()` call |
| Nested pipelines | Full call tree with parent-child spans |
| Optimizer internals | LM calls made during optimization |

## @with_langtrace_root_span()

Creates a named parent span to group DSPy calls. Only needed for custom metadata or grouping.

```python
from langtrace_python_sdk import with_langtrace_root_span

@with_langtrace_root_span("span-name")
def my_function():
    # DSPy calls inside are grouped under this span
    pass
```

## inject_additional_attributes()

Adds custom metadata to the current trace for filtering in the UI.

```python
langtrace.inject_additional_attributes({
    "user_id": "user-123",
    "environment": "production",
    "experiment": "mipro-v2-run1",  # for experiment tracking
})
```

### Experiment tracking attributes

| Attribute | Required | Description |
|-----------|----------|-------------|
| `experiment` | Yes | Experiment name identifier |
| `description` | No | Contextual details |
| `run_id` | No | Unique ID for associating traces to evaluation runs |

## Environment variables

| Variable | Description |
|----------|-------------|
| `LANGTRACE_API_KEY` | API key for cloud or self-hosted |
| `LANGTRACE_API_HOST` | Custom endpoint URL (for self-hosted) |
| `TRACE_DSPY_CHECKPOINT` | Set to `false` to disable checkpoint tracing (reduces latency in production) |

## Deployment options

- **Cloud**: Managed at [app.langtrace.ai](https://app.langtrace.ai)
- **Self-hosted**: Docker Compose — `git clone https://github.com/Scale3-Labs/langtrace.git && docker compose up -d`
- **Source**: [github.com/Scale3-Labs/langtrace](https://github.com/Scale3-Labs/langtrace)

## Key behaviors

- **Checkpoint tracing**: Enabled by default. Serializes predictor state at each step. Disable with `TRACE_DSPY_CHECKPOINT=false` for production.
- **DSPy caching**: DSPy caches LM responses by default. Cached calls do not produce new traces. Disable with `dspy.configure_cache(enable=False)` when debugging.
- **ThreadPoolExecutor**: When using parallel execution, wrap calls with `contextvars.copy_context().run()` to propagate tracing context across threads.
