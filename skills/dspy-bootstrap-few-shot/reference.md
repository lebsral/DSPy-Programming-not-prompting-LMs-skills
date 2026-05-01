> Condensed from [dspy.ai/api/optimizers/BootstrapFewShot/](https://dspy.ai/api/optimizers/BootstrapFewShot/). Verify against upstream for latest.

# dspy.BootstrapFewShot — API Reference

## Constructor

```python
dspy.BootstrapFewShot(
    metric=None,                 # Callable | None
    metric_threshold=None,       # float | None
    teacher_settings=None,       # dict | None
    max_bootstrapped_demos=4,    # int
    max_labeled_demos=16,        # int
    max_rounds=1,                # int
    max_errors=None,             # int | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable | None` | `None` | Scoring function `(example, prediction, trace=None) -> bool | float`. Traces where the metric passes become candidate demonstrations. |
| `metric_threshold` | `float | None` | `None` | Numerical threshold for accepting bootstrap examples. When set, only traces scoring above this value are kept as demos. |
| `teacher_settings` | `dict | None` | `None` | Configuration for the teacher model used during bootstrapping (e.g., `{"lm": teacher_lm}`). When set, the teacher generates traces instead of the student. |
| `max_bootstrapped_demos` | `int` | `4` | Maximum number of bootstrapped (program-generated) demonstrations per predictor. |
| `max_labeled_demos` | `int` | `16` | Maximum number of labeled (from trainset) demonstrations per predictor. These are raw input/output pairs without intermediate reasoning. |
| `max_rounds` | `int` | `1` | Number of bootstrap rounds. Each round runs the program with demos from prior rounds and collects new passing traces. |
| `max_errors` | `int | None` | `None` | Error tolerance before stopping. Defaults to `dspy.settings.max_errors` if unset. |

## Inheritance

`BootstrapFewShot` extends `Teleprompter`.

## Methods

### `compile()`

```python
optimizer.compile(
    student,                # dspy.Module (required)
    *,
    teacher=None,           # dspy.Module | None
    trainset,               # list[Example] (required)
) -> Module
```

Orchestrates bootstrapping: prepares student/teacher, maps predictors, runs bootstrap rounds, and attaches the best demonstrations to each predictor.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `dspy.Module` | required | The program whose predictors will receive bootstrapped demos. |
| `teacher` | `dspy.Module | None` | `None` | Optional teacher program. If provided, the teacher generates traces instead of the student. |
| `trainset` | `list[Example]` | required | Labeled training examples. Each must have `.with_inputs()` called. |

**Returns:** The compiled student module with `_compiled = True` and bootstrapped demos attached to each predictor.

### `get_params()`

```python
optimizer.get_params() -> dict[str, Any]
```

Returns all configuration parameters as a dictionary.

## Key behaviors

- Each bootstrap round uses a fresh LM instance with `temperature=1.0` to bypass caches and gather diverse traces
- Successfully bootstrapped examples are accepted immediately; rounds continue only until one succeeds per example
- When `teacher_settings` is provided, the teacher model generates traces while the student receives the resulting demos
- The `compile()` method returns a copy of the student — the original module is not modified
- Bootstrapped demos include intermediate fields (e.g., `reasoning` from ChainOfThought), while labeled demos are raw input/output pairs
