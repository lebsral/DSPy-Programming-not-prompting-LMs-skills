# Tracing and Debugging Reference

> Condensed from [dspy.ai/tutorials/observability](https://dspy.ai/tutorials/observability/) and [dspy.ai/tutorials/streaming](https://dspy.ai/tutorials/streaming/). Verify against upstream for latest.

## inspect_history

The simplest debugging tool. Prints the last N LM calls to stdout.

```python
dspy.inspect_history(n=5)

# Save to file (DSPy 3.2+)
dspy.inspect_history(n=10, file_path="debug_trace.txt")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `n` | int | 1 | Number of recent LM calls to display |
| `file_path` | str | None | Save output to file instead of stdout (DSPy 3.2+) |

**Limitations:** Only logs LM calls. Misses retriever, tool, and custom module data. Not suitable for production — use the callback system or an external trace viewer instead.

## Callback System (BaseCallback)

The official DSPy observability API. Hooks into every module, LM call, tool call, and adapter operation.

```python
from dspy.utils.callback import BaseCallback

class MyCallback(BaseCallback):
    def on_module_start(self, call_id, instance, inputs):
        ...
    def on_module_end(self, call_id, outputs, exception):
        ...

dspy.configure(callbacks=[MyCallback()])
```

### Available hooks

| Hook | Trigger |
|---|---|
| `on_module_start(call_id, instance, inputs)` | Before any DSPy module runs |
| `on_module_end(call_id, outputs, exception)` | After any DSPy module completes |
| `on_lm_start(call_id, instance, inputs)` | Before an LM call |
| `on_lm_end(call_id, outputs, exception)` | After an LM call completes |
| `on_adapter_format_start(call_id, instance, inputs)` | Before adapter formats the prompt |
| `on_adapter_format_end(call_id, outputs, exception)` | After adapter formats the prompt |
| `on_adapter_parse_start(call_id, instance, inputs)` | Before adapter parses LM output |
| `on_adapter_parse_end(call_id, outputs, exception)` | After adapter parses LM output |
| `on_tool_start(call_id, instance, inputs)` | Before a tool is called |
| `on_tool_end(call_id, outputs, exception)` | After a tool completes |
| `on_evaluate_start(call_id, instance, inputs)` | Before evaluation starts |
| `on_evaluate_end(call_id, outputs, exception)` | After evaluation completes |

**Important:** Do not mutate `inputs` or `outputs` in place — these are live references. Copy before modifying.

## StreamListener

Monitors token streaming for specific output fields.

```python
listener = dspy.streaming.StreamListener(
    signature_field_name="answer",
    predict=module.predict1,         # optional: target specific predict
    predict_name="predict1",         # optional: disambiguate duplicate names
    allow_reuse=False,               # True for loops like ReAct
)

stream_module = dspy.streamify(module, stream_listeners=[listener])
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `signature_field_name` | str | required | Output field to stream |
| `predict` | Predict | None | Target a specific predict module |
| `predict_name` | str | None | Disambiguate when multiple modules share field names |
| `allow_reuse` | bool | False | Enable reuse across multiple streaming sessions |

### StreamResponse

Each streamed token is a `StreamResponse` with:
- `predict_name` — name of the predict module
- `signature_field_name` — output field identifier
- `chunk` — the token value

### StatusMessageProvider

Provides real-time execution status updates (tool calls, LM invocations):

```python
from dspy.streaming import StatusMessageProvider

class MyStatusProvider(StatusMessageProvider):
    def lm_start_status_message(self, instance, inputs):
        return "Calling LM..."
    def tool_start_status_message(self, instance, inputs):
        return f"Using tool: {instance.name}"
```

### dspy.streamify()

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stream_listeners` | list[StreamListener] | required | Fields to monitor |
| `status_message_provider` | StatusMessageProvider | None | Custom status messages |
| `async_streaming` | bool | True | Toggle async/sync generators |

Returns an async or sync generator yielding `StreamResponse`, `StatusMessage`, or `Prediction` objects. Cached results skip individual tokens and yield the final `Prediction` directly.

## Caching

DSPy uses a three-layer cache: in-memory (LRU), on-disk (FanoutCache), and provider-side prompt cache. Both in-memory and disk caching are enabled by default.

```python
dspy.configure_cache(
    enable_disk_cache=True,
    enable_memory_cache=True,
    disk_size_limit_bytes=1_000_000_000,
    memory_max_entries=10_000,
)
```

### Provider-side prompt caching

```python
lm = dspy.LM(
    "anthropic/claude-sonnet-4-5-20250929",
    cache_control_injection_points=[{"location": "message", "role": "system"}],
)
```

### Security

Use `restrict_pickle=True` to prevent arbitrary code execution from corrupted cache files:

```python
dspy.configure_cache(restrict_pickle=True, safe_types=[CustomType])
```

## External trace viewers

| Tool | Install | Auto-instrument | Docs |
|---|---|---|---|
| Langtrace | `pip install langtrace-python-sdk` | `langtrace.init(api_key=...)` | See `/dspy-langtrace` |
| Arize Phoenix | `pip install arize-phoenix openinference-instrumentation-dspy` | `DSPyInstrumentor().instrument()` | See `/dspy-phoenix` |
| MLflow | `pip install mlflow>=2.18.0` | `mlflow.dspy.autolog()` | See `/dspy-mlflow` |
| W&B Weave | `pip install weave` | `@weave.op()` decorator | See `/dspy-weave` |
| LangWatch | `pip install langwatch` | `langwatch.dspy.init()` | See `/dspy-langwatch` |
| Langfuse | `pip install langfuse` | `DSPyInstrumentor` or `@observe` | See `/dspy-langfuse` |
