> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Serving

## dspy.LM

[API docs](https://dspy.ai/api/models/LM/)

```python
dspy.LM(model, model_type="chat", temperature=0.0, max_tokens=1000, cache=True, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Provider/model string (e.g., `"openai/gpt-4o-mini"`) |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `max_tokens` | `int` | `1000` | Max output tokens |
| `cache` | `bool` | `True` | Enable response caching |

## dspy.configure

```python
dspy.configure(lm=None, adapter=None, callbacks=None, async_max_workers=8)
```

Sets **global** state. Call once at startup. Do NOT call inside route handlers — use `dspy.context()` for per-request overrides.

## dspy.context

```python
with dspy.context(lm=override_lm):
    result = program(query=input)
```

Thread-safe context manager for per-request configuration overrides. Use this instead of `dspy.configure()` when handling concurrent requests with different models or temperatures.

## Module.save / Module.load

```python
# Save optimized program
program.save(path, save_program=False, modules_to_serialize=None)

# Load at server startup
program.load(path, allow_pickle=False, allow_unsafe_lm_state=False)
```

| Method | Key Parameters |
|--------|---------------|
| `save(path)` | Saves to `.json` (state only) or directory (`save_program=True` for full pickle) |
| `load(path)` | Loads `.json` or `.pkl`. Set `allow_pickle=True` for pickled programs |

Saves optimized prompts, demos, and weights. No training data or optimizer needed at deploy time.

## dspy.Refine -- output validation in APIs

When using `dspy.Refine` in a served module, it raises an exception if `fail_count` is exhausted without meeting the reward threshold. Catch this as a validation failure and map to HTTP 422:

```python
# dspy.Refine raises a generic Exception when fail_count is exhausted.
# Detect it by checking the error message or wrapping the Refine call.
try:
    result = program(query=request.query)
except Exception as e:
    if "refine" in str(e).lower() or "fail_count" in str(e).lower():
        raise HTTPException(status_code=422, detail=f"Output validation failed: {e}")
    raise
```

Note: `dspy.Assert`/`dspy.Suggest` and `DSPyAssertionError` were removed in DSPy 3.x. Use `dspy.Refine` with a reward function instead.
