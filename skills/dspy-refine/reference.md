# Refine API Reference

> Condensed from [dspy.ai/api/modules/Refine](https://dspy.ai/api/modules/Refine/). Verify against upstream for latest.

## Constructor

```python
dspy.Refine(
    module,       # Module -- required
    N,            # int -- required, max attempts
    reward_fn,    # Callable[[dict, Prediction], float] -- required
    threshold,    # float -- required, target reward score
    fail_count=None,  # int | None -- max failures before error (defaults to N)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module whose outputs to refine |
| `N` | `int` | required | Max number of attempts (each at temperature=1.0) |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores a prediction; receives `(args_dict, prediction)` |
| `threshold` | `float` | required | Target score -- returns immediately when met |
| `fail_count` | `int \| None` | `None` (defaults to `N`) | Max allowed failures before raising an error |

## Key Methods

- `forward(**kwargs) -> dspy.Prediction` -- run the refinement loop
- `acall(**kwargs) -> dspy.Prediction` -- async variant
- `batch(examples, num_threads, max_errors, ...) -> list[Prediction]` -- parallel processing

## Inherited Module Methods

| Method | Description |
|--------|-------------|
| `save(path)` | Persist learned state to JSON |
| `load(path)` | Load state into a fresh instance |
| `set_lm(lm)` | Override LM for this module |
| `get_lm()` | Get the current LM |
| `deepcopy()` | Deep copy the module |

## Reward Function Signature

```python
def reward_fn(args: dict, pred: dspy.Prediction) -> float:
    """
    args: dict of inputs passed to the module (e.g., {"question": "..."})
    pred: the module's Prediction object (access fields like pred.answer)
    Returns: float score. Higher is better.
    """
```

## Behavior

1. Runs the module at `temperature=1.0` with varying rollout IDs
2. Scores each attempt with `reward_fn`
3. If score >= `threshold`, returns immediately
4. If below threshold, generates feedback from the failure for the next attempt
5. After N attempts, returns the attempt with the highest reward score
