> Condensed from [dspy.ai/api/modules/Predict](https://dspy.ai/api/modules/Predict/). Verify against upstream for latest.

# dspy.Predict — API Reference

## Constructor

```python
dspy.Predict(signature, callbacks=None, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | (required) | Inline string (`"question -> answer"`) or a `dspy.Signature` subclass |
| `callbacks` | `list[BaseCallback] \| None` | `None` | Instrumentation callbacks for tracing/logging |
| `**config` | keyword args | — | Forwarded to the LM (e.g., `temperature=0.7`, `max_tokens=500`); overridable per call |

## Key methods

### Execution

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(**kwargs)` | Invoke prediction. Only keyword args — positional args raise `ValueError` |
| `forward` | `(**kwargs)` | Main execution: builds prompt via adapter, calls LM, returns `Prediction` |
| `acall` | `(**kwargs)` | Async version of `__call__` |
| `aforward` | `(**kwargs)` | Async execution with streaming support |

### Batch processing

```python
predict.batch(
    examples,              # list[dspy.Example]
    num_threads=None,      # max parallel threads
    max_errors=None,       # stop after N errors
    return_failed_examples=False,
    provide_traceback=None,
    disable_progress_bar=False,
    timeout=120,           # per-example timeout in seconds
    straggler_limit=3,     # slow-example threshold multiplier
)
```

Returns a list of results. If `return_failed_examples=True`, returns `(results, failed_examples, exceptions)`.

### Save / Load

```python
# Save to file
predict.save("path/to/model.json")

# Load from file
predict.load("path/to/model.json")

# Lower-level state management
state = predict.dump_state()
predict.load_state(state)
```

### Introspection

| Method | Returns | Description |
|--------|---------|-------------|
| `get_lm()` | `LM` | Returns the LM; raises if multiple LMs in use |
| `set_lm(lm)` | — | Sets LM for all predictors recursively |
| `named_predictors()` | `list[(str, Predict)]` | All named Predict instances |
| `inspect_history(n=1)` | — | Print the last `n` LM calls for debugging |
| `reset()` | — | Clears LM, traces, demos, and train data |

### Config

```python
predict.get_config()           # current kwargs dict
predict.update_config(**kw)    # merge new kwargs
```

Override config per call:

```python
result = predict(question="...", config={"temperature": 0.0})
```
