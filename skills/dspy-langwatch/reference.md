> Condensed from [LangWatch DSPy integration docs](https://langwatch.ai/docs/integration/python/integrations/dspy) and [LangWatch Python SDK guide](https://langwatch.ai/docs/integration/python/guide). Verify against upstream for latest.

# LangWatch DSPy Integration — API Reference

## Installation

```bash
pip install langwatch
# Or with DSPy compatibility pinning:
pip install langwatch[dspy]
```

## SDK initialization

```python
import langwatch

langwatch.setup(
    api_key=None,           # defaults to LANGWATCH_API_KEY env var
    project_id=None,        # required for service API keys; reads LANGWATCH_PROJECT_ID env var
    endpoint_url=None,      # defaults to https://app.langwatch.ai (or LANGWATCH_ENDPOINT env var)
    base_attributes=None,   # dict of attributes applied to all spans
    instrumentors=None,     # list of automatic instrumentors
    tracer_provider=None,   # existing OpenTelemetry TracerProvider to integrate with
    debug=False,            # enable debug logging
    disable_sending=False,  # prevent trace transmission (for testing)
    flush_on_exit=True,     # auto-flush spans on program exit
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | env `LANGWATCH_API_KEY` | LangWatch API key |
| `project_id` | `str \| None` | env `LANGWATCH_PROJECT_ID` | Required for service API keys |
| `endpoint_url` | `str \| None` | `https://app.langwatch.ai` | LangWatch endpoint (set for self-hosted) |
| `base_attributes` | `dict \| None` | `None` | Attributes applied to all spans |
| `instrumentors` | `Sequence \| None` | `None` | Automatic instrumentors (e.g., `OpenAIInstrumentor`) |
| `tracer_provider` | `TracerProvider \| None` | `None` | Existing OpenTelemetry provider to integrate with |
| `debug` | `bool` | `False` | Enable debug logging |
| `disable_sending` | `bool` | `False` | Disable trace transmission |
| `flush_on_exit` | `bool` | `True` | Auto-flush on program exit |

## Auto-tracing (inference)

### @langwatch.trace()

Decorator that creates a trace context for a function.

```python
@langwatch.trace(name="optional-name", metadata={"key": "value"})
def my_function():
    langwatch.get_current_trace().autotrack_dspy()
    # DSPy calls are now auto-traced
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str \| None` | function name | Custom trace name |
| `metadata` | `dict \| None` | `None` | Key-value metadata attached to the trace |

### autotrack_dspy()

Enables automatic DSPy tracking within the current trace context.

```python
trace = langwatch.get_current_trace()
trace.autotrack_dspy()
```

Must be called inside a `@langwatch.trace()` decorated function. Captures:
- Module calls (inputs/outputs per `forward()`)
- LM calls (model, messages, response, token counts)
- Retrievals (queries, passages)
- Nested spans with parent-child relationships

### Trace context methods

| Method | Description |
|--------|-------------|
| `langwatch.get_current_trace()` | Get the active trace context |
| `trace.autotrack_dspy()` | Enable DSPy auto-tracking |
| `trace.update(metadata={...})` | Add metadata to the trace |
| `langwatch.get_current_span()` | Get the active span |

### @langwatch.span()

Decorator for custom sub-spans within a trace.

```python
@langwatch.span(type="rag", name="Retrieval")
def retrieve(query):
    # custom retrieval logic
    pass
```

## Optimizer progress tracking

### langwatch.dspy.init()

Patches a DSPy optimizer to stream live progress to the LangWatch dashboard.

```python
import langwatch.dspy

langwatch.dspy.init(
    experiment="experiment-name",
    optimizer=optimizer,
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment` | `str` | Experiment name (appears in dashboard; use unique names per run) |
| `optimizer` | `Teleprompter \| None` | DSPy optimizer instance to patch for progress streaming |
| `evaluator` | `Evaluate \| None` | DSPy `Evaluate` instance to track (alternative to `optimizer`) |
| `run_id` | `str \| None` | Custom run ID (auto-generated if omitted) |
| `slug` | `str \| None` | URL-friendly identifier for the experiment |
| `workflow_id` | `str \| None` | Associate run with a LangWatch workflow |
| `workflow_version_id` | `str \| None` | Pin run to a specific workflow version |

`optimizer` and `evaluator` are mutually exclusive — pass one or the other.

Must be called **before** `optimizer.compile()`. The dashboard shows live scores, predictor states, LM calls, cost, and progress.

### Supported optimizers

| Optimizer | Supported |
|-----------|-----------|
| `dspy.BootstrapFewShot` | Yes |
| `dspy.BootstrapFewShotWithRandomSearch` | Yes |
| `dspy.COPRO` | Yes |
| `dspy.MIPROv2` | Yes |
| Other optimizers | Raises `ValueError` |

## Environment variables

| Variable | Description |
|----------|-------------|
| `LANGWATCH_API_KEY` | API key for cloud or self-hosted |
| `LANGWATCH_ENDPOINT` | Custom endpoint URL (for self-hosted instances) |

## Deployment options

- **Cloud**: Managed at [app.langwatch.ai](https://app.langwatch.ai) (free tier available)
- **Self-hosted**: Docker Compose or Helm chart — see [self-hosting docs](https://langwatch.ai/docs/self-hosting)
