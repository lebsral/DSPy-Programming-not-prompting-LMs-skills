# DSPy Utilities API Reference

> Condensed from [dspy.ai](https://dspy.ai/). Verify against upstream for latest.

## streamify

```python
from dspy.streaming import streamify

streaming_program = streamify(
    program,                                        # Module -- required
    stream_listeners=None,                          # list[StreamListener]
    include_final_prediction_in_output_stream=True, # include Prediction in stream
    is_async_program=False,                         # True if program is already async
    async_streaming=True,                           # True for async generator
    status_message_provider=None,                   # custom status messages
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `program` | `Module` | required | DSPy module to stream |
| `stream_listeners` | `list[StreamListener] \| None` | `None` | Listeners for specific output fields |
| `include_final_prediction_in_output_stream` | `bool` | `True` | Yield final `Prediction` in stream |
| `is_async_program` | `bool` | `False` | Set `True` if program is async |
| `async_streaming` | `bool` | `True` | Return async vs sync generator |

## StreamListener

```python
from dspy.streaming import StreamListener

listener = StreamListener(
    signature_field_name,    # str -- required, output field to stream
    predict=None,            # predictor to monitor (auto-detected)
    predict_name=None,       # name identifier for the predictor
    allow_reuse=False,       # allow reuse across multiple streams
)
```

## inspect_history

```python
dspy.inspect_history(n=1)  # show last n LM calls
```

Shows full prompt sent, raw response, and adapter format.

## save / load

```python
program.save("path.json")   # save learned state (demos, instructions)
program.load("path.json")   # load into a fresh instance of the same class
```

Saves only learned state -- class definition must exist at load time. Call `dspy.configure()` before `load()`.

## asyncify

```python
async_program = dspy.asyncify(program)
result = await async_program(**inputs)
```

Wraps sync DSPy programs for async execution. Captures and propagates `dspy.configure` settings to worker thread.

## configure_cache

```python
dspy.configure_cache(enable=True)   # enable/disable caching globally
dspy.LM("model", cache=False)      # per-LM cache control
```
