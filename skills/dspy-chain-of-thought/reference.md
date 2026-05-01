> Condensed from [dspy.ai/api/modules/ChainOfThought/](https://dspy.ai/api/modules/ChainOfThought/). Verify against upstream for latest.

# dspy.ChainOfThought — API Reference

## Constructor

```python
dspy.ChainOfThought(
    signature,                          # str | type[Signature] (required)
    rationale_field=None,               # FieldInfo | None
    rationale_field_type=str,           # type
    **config,                           # dict[str, Any]
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str | type[Signature]` | required | Defines input/output fields for the module. |
| `rationale_field` | `FieldInfo | None` | `None` | Custom field info for the reasoning output (e.g., `dspy.OutputField(prefix="Thinking:")`). Changes the prompt prefix but the field is still accessed as `result.reasoning`. |
| `rationale_field_type` | `type` | `str` | Type annotation for the reasoning field. |
| `**config` | `dict[str, Any]` | — | Additional configuration passed to the internal `Predict` module. |

## Inheritance

`ChainOfThought` extends `dspy.Module`. It wraps an internal `dspy.Predict` instance with an augmented signature that prepends a `reasoning` output field before your declared output fields.

## Key methods

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `__call__` | `(**kwargs)` | `Prediction` | Execute with callback support and usage tracking. |
| `forward` | `(**kwargs)` | `Prediction` | Run the chain-of-thought prediction. Delegates to internal `self.predict`. |
| `aforward` | `(**kwargs)` | `Prediction` | Async version of `forward`. |
| `acall` | `(**kwargs)` | `Prediction` | Async version of `__call__`. |
| `batch` | `(examples, ...)` | `list[Prediction]` | Process multiple inputs in parallel. |

## Module management methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `set_lm` | `(lm)` | Override the LM for this module and all sub-predictors. |
| `get_lm` | `()` | Retrieve the current LM. |
| `named_predictors` | `()` | Access all internal Predict modules. |
| `map_named_predictors` | `(func)` | Apply a transformation to all predictors. |

## Persistence methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `save` | `(path, save_program=False)` | Save module state to JSON. |
| `load` | `(path)` | Load a previously saved module. |
| `dump_state` | `(json_mode=True)` | Export state as a dict. |
| `load_state` | `(state)` | Restore state from a dict. |

## How the reasoning field works

ChainOfThought augments your signature by prepending a `reasoning` field:

```
Original:  question -> answer
Augmented: question -> reasoning, answer
```

- The `reasoning` field is always accessible as `result.reasoning`
- It is generated before the output fields, giving the LM space to think
- If `rationale_field` is set, the prompt prefix changes but the attribute name stays `reasoning`
- The reasoning field is included in few-shot demos during optimization
