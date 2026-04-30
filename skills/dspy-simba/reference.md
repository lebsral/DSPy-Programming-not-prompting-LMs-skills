# SIMBA API Reference

> Condensed from [dspy.ai/api/optimizers/SIMBA](https://dspy.ai/api/optimizers/SIMBA). Verify against upstream for latest.

## Constructor

```python
dspy.SIMBA(
    *,
    metric,                          # Callable -- required
    bsize=32,                        # mini-batch size
    num_candidates=6,                # candidates per iteration
    max_steps=8,                     # optimization iterations
    max_demos=4,                     # max demos per predictor
    prompt_model=None,               # dspy.LM for rule generation
    teacher_settings=None,           # teacher model config dict
    demo_input_field_maxlen=100000,  # char limit for demo inputs
    num_threads=None,                # parallel threads
    temperature_for_sampling=0.2,    # trajectory sampling temperature
    temperature_for_candidates=0.2,  # source program selection temperature
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | `(example, prediction_dict) -> float` |
| `bsize` | `int` | `32` | Examples per mini-batch |
| `num_candidates` | `int` | `6` | New candidate programs per step |
| `max_steps` | `int` | `8` | Total optimization iterations |
| `max_demos` | `int` | `4` | Max few-shot demos per predictor |
| `prompt_model` | `dspy.LM \| None` | `None` | LM for introspective rules (falls back to global LM) |
| `teacher_settings` | `dict \| None` | `None` | Teacher model configuration |
| `demo_input_field_maxlen` | `int` | `100000` | Max chars for demo input fields |
| `num_threads` | `int \| None` | `None` | Defaults to `dspy.settings.num_threads` |
| `temperature_for_sampling` | `float` | `0.2` | Temperature for trajectory sampling |
| `temperature_for_candidates` | `float` | `0.2` | Temperature for source program selection |

## Key Methods

### compile()

```python
optimized = optimizer.compile(student, *, trainset, seed=0)
```

Returns an optimized `dspy.Module` with:
- `candidate_programs` -- list of `(program, score)` tuples
- `trial_logs` -- per-step metrics

### get_params()

```python
optimizer.get_params() -> dict[str, Any]
```

Returns optimizer configuration as a dictionary.
