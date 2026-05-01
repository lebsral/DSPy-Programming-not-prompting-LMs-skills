> Condensed from [dspy.ai/api/optimizers/BetterTogether/](https://dspy.ai/api/optimizers/BetterTogether/). Verify against upstream for latest.

# dspy.BetterTogether — API Reference

## Constructor

```python
dspy.BetterTogether(
    metric,                  # Callable (required)
    **optimizers,            # Teleprompter instances as keyword args
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Evaluation function `(example, prediction, trace=None) -> numeric`. Higher is better. |
| `**optimizers` | `Teleprompter` | `{}` | Custom optimizer instances as keyword args. Keys become strategy identifiers (e.g., `p=MIPROv2(...)`, `w=BootstrapFinetune(...)`). If empty, defaults to `p=BootstrapFewShotWithRandomSearch` and `w=BootstrapFinetune`. |

## Inheritance

`BetterTogether` extends `Teleprompter`.

## Methods

### `compile()`

```python
optimizer.compile(
    student,                              # Module (required)
    *,
    trainset,                             # list[Example] (required)
    teacher=None,                         # Module | list[Module] | None
    valset=None,                          # list[Example] | None
    num_threads=None,                     # int | None
    max_errors=None,                      # int | None
    provide_traceback=None,               # bool | None
    seed=None,                            # int | None
    valset_ratio=0.1,                     # float
    shuffle_trainset_between_steps=True,  # bool
    strategy="p -> w -> p",              # str
    optimizer_compile_args=None,          # dict[str, dict[str, Any]] | None
) -> Module
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `Module` | required | Program to optimize. All predictors must have LMs assigned via `set_lm()`. |
| `trainset` | `list[Example]` | required | Training examples. Must not be empty. |
| `teacher` | `Module | list[Module] | None` | `None` | Optional teacher program(s) for bootstrapping. |
| `valset` | `list[Example] | None` | `None` | Validation set. If `None`, splits from trainset using `valset_ratio`. |
| `num_threads` | `int | None` | `None` | Parallel evaluation threads. |
| `max_errors` | `int | None` | `None` | Max tolerated evaluation errors. Defaults to `dspy.settings.max_errors`. |
| `provide_traceback` | `bool | None` | `None` | Show detailed error tracebacks. |
| `seed` | `int | None` | `None` | Random seed for reproducibility. |
| `valset_ratio` | `float` | `0.1` | Fraction of trainset to reserve as validation when `valset=None`. Range [0, 1). |
| `shuffle_trainset_between_steps` | `bool` | `True` | Shuffle trainset before each optimization step. |
| `strategy` | `str` | `"p -> w -> p"` | Optimizer sequence separated by `" -> "`. Keys must match constructor keyword argument names. |
| `optimizer_compile_args` | `dict[str, dict[str, Any]] | None` | `None` | Per-optimizer custom compile arguments. Cannot include `"student"` key. |

**Returns:** Optimized `Module` with two extra attributes:
- `candidate_programs`: List of dicts with `"program"`, `"score"`, `"strategy"` keys, sorted by score descending
- `flag_compilation_error_occurred`: `True` if any optimization step failed

**Raises:**
- `ValueError` if trainset is empty, `valset_ratio` outside [0, 1), strategy keys do not match optimizer names, or `optimizer_compile_args` has invalid keys
- `TypeError` if `optimizer_compile_args` contains `"student"` key

### `get_params()`

```python
optimizer.get_params() -> dict[str, Any]
```

Returns all configuration parameters as a dictionary.

## Key behaviors

- **Strategy execution**: Applies optimizers sequentially per the strategy string. At each step, trainset is optionally shuffled and the result is evaluated on the validation set.
- **Program selection**: With validation data, returns the best-scoring program across all phases. Without validation, returns the latest program. Earlier programs win ties.
- **Error resilience**: If any optimization step fails, logs the error, continues to the next step, and sets `flag_compilation_error_occurred = True`.
- **Model lifecycle**: Automatically launches, kills, and relaunches models between steps (critical for local providers, no-ops for API LMs).
- **Default optimizers**: When no `**optimizers` are provided, uses `BootstrapFewShotWithRandomSearch` (key `p`) and `BootstrapFinetune` (key `w`).
