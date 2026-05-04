# Streaming API Reference

> Condensed from [dspy.ai/api/streaming](https://dspy.ai/api/streaming/). Verify against upstream for latest.

## streamify()

```python
from dspy.streaming import streamify

streaming_program = streamify(
    program,                                        # Module -- required
    stream_listeners=None,                          # list[StreamListener] | None
    include_final_prediction_in_output_stream=True, # bool
    is_async_program=False,                         # bool
    async_streaming=True,                           # bool
    status_message_provider=None,                   # Callable | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `program` | `dspy.Module` | required | The DSPy module to wrap for streaming |
| `stream_listeners` | `list[StreamListener]` | `None` | Listeners targeting specific output fields |
| `include_final_prediction_in_output_stream` | `bool` | `True` | Emit final `Prediction` as last stream item |
| `is_async_program` | `bool` | `False` | Set `True` if `program.forward()` is already async |
| `async_streaming` | `bool` | `True` | `True` returns async generator; `False` returns sync |
| `status_message_provider` | `Callable` | `None` | Function returning status strings during processing |

**Returns:** A callable that accepts the same kwargs as the original module and returns a generator (async or sync) yielding chunks.

## StreamListener

```python
from dspy.streaming import StreamListener

listener = StreamListener(
    signature_field_name,   # str -- required
    predict=None,           # Any | None
    predict_name=None,      # str | None
    allow_reuse=False,      # bool
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature_field_name` | `str` | required | Output field name to capture tokens for |
| `predict` | `Any` | `None` | Specific predictor instance to monitor (auto-detected if None) |
| `predict_name` | `str` | `None` | Attribute name of the predictor in the module (disambiguates duplicate field names) |
| `allow_reuse` | `bool` | `False` | Allow listener across multiple stream iterations (required for agents) |

**Methods:**
- `receive(chunk)` -- processes incoming token, manages internal buffer
- `finalize()` -- flushes remaining buffer, returns final chunk with `is_last_chunk=True`
- `flush()` -- flushes buffer immediately, clears it

## StreamResponse

Emitted for each batch of tokens from a listened field.

| Attribute | Type | Description |
|-----------|------|-------------|
| `<field_name>` | `str` | Partial text for the named field |
| `is_last_chunk` | `bool` | `True` when this is the final chunk for the field |

Access the field dynamically: `chunk.<signature_field_name>`.

## StatusMessage

Emitted during multi-step processing (tool calls, retries).

| Attribute | Type | Description |
|-----------|------|-------------|
| `message` | `str` | Human-readable status text |

## Prediction (final)

The last item in the stream (if `include_final_prediction_in_output_stream=True`). Contains all output fields with their complete values, identical to non-streaming module output.

## Chunk type detection pattern

```python
from dspy.streaming import StreamResponse, StatusMessage

async for chunk in streaming_program(**inputs):
    if isinstance(chunk, StreamResponse):
        # Partial tokens
        pass
    elif isinstance(chunk, StatusMessage):
        # Progress update
        pass
    elif isinstance(chunk, dspy.Prediction):
        # Final complete result
        pass
```
