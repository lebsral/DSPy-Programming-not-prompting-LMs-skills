# DSPy Utilities API Reference

> Condensed from [dspy.ai/api/utils/configure_cache](https://dspy.ai/api/utils/configure_cache/), [dspy.ai/api/utils/inspect_history](https://dspy.ai/api/utils/inspect_history/), and [dspy.ai/tutorials/saving](https://dspy.ai/tutorials/saving/). Verify against upstream for latest.

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
dspy.inspect_history(n=1, file=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `1` | Number of recent LM calls to display |
| `file` | `TextIO \| None` | `None` | Write to file instead of stdout (disables ANSI color codes) |

Shows full prompt sent, raw response, and adapter format.

## save / load

```python
# Save learned state only (class definition must exist at load time)
program.save("path.json")

# Save entire program including class definition
program.save("path_dir", save_program=True, modules_to_serialize=None)

# Load state into a fresh instance
program = MyProgram()
program.load("path.json", allow_pickle=False)

# Load an entire saved program (requires save_program=True)
program = dspy.load("path_dir", allow_pickle=False)
```

`save()` parameters: `path` (str), `save_program` (bool, default `False`), `modules_to_serialize` (list, optional).
`load()` parameters: `path` (str), `allow_pickle` (bool, default `False` -- set `True` for `.pkl` files).
`dspy.load()` parameters: `path` (str), `allow_pickle` (bool, default `False`).

Call `dspy.configure(lm=lm)` before `program.load()`. Not needed for `dspy.load()`.

## asyncify

```python
async_program = dspy.asyncify(program)
result = await async_program(**inputs)
```

Wraps sync DSPy programs for async execution. Captures and propagates `dspy.configure` settings to worker thread.

## configure_cache

```python
dspy.configure_cache(
    enable_disk_cache=True,         # persistent file-based cache
    enable_memory_cache=True,       # RAM-based cache
    disk_cache_dir=None,            # storage location (defaults to ~/.dspy_cache)
    disk_size_limit_bytes=None,     # max disk storage (defaults to DISK_CACHE_LIMIT)
    memory_max_entries=1000000,     # max in-memory entries
    restrict_pickle=False,          # harden against untrusted pickle payloads (3.2+)
    safe_types=None,                # additional trusted types for restricted pickle
)
dspy.LM("model", cache=False)      # per-LM cache control
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_disk_cache` | `bool \| None` | `True` | Persistent file-based caching |
| `enable_memory_cache` | `bool \| None` | `True` | RAM-based caching |
| `disk_cache_dir` | `str \| None` | `DISK_CACHE_DIR` | Cache directory path |
| `disk_size_limit_bytes` | `int \| None` | `DISK_CACHE_LIMIT` | Max disk storage |
| `memory_max_entries` | `int` | `1000000` | Max in-memory entries |
| `restrict_pickle` | `bool` | `False` | Harden against untrusted payloads |
| `safe_types` | `list[type] \| None` | `None` | Additional trusted types |

Common usage:
```python
# Disable all caching
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# Disable only disk cache
dspy.configure_cache(enable_disk_cache=False)
```
