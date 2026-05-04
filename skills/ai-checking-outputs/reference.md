> Condensed from [dspy.ai/api/modules/Refine](https://dspy.ai/api/modules/Refine/) and [dspy.ai/api/modules/BestOfN](https://dspy.ai/api/modules/BestOfN/). Verify against upstream for latest.

# Refine and BestOfN API Reference

## dspy.Refine

Iterative self-correction: runs a module up to N times, feeding reward feedback back to guide improvement.

### Constructor

```python
dspy.Refine(
    module: dspy.Module,
    N: int,
    reward_fn: Callable[[dict, dspy.Prediction], float],
    threshold: float,
    fail_count: int | None = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to refine |
| `N` | `int` | required | Maximum number of attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores each prediction (0.0-1.0) |
| `threshold` | `float` | required | Stop early if reward meets this score |
| `fail_count` | `int \| None` | `None` (defaults to N) | Max failures before raising error |

### Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(**kwargs) -> Prediction` | Runs module up to N times at temperature=1.0. Returns first prediction exceeding threshold, or highest-scoring attempt. Generates feedback when predictions fall short. |
| `__call__` | `(*args, **kwargs) -> Prediction` | Primary invocation with callback/usage tracking |
| `set_lm` | `(lm) -> None` | Assigns LM to all internal predictors |
| `batch` | `(examples, num_threads=None, ...) -> list` | Processes multiple examples in parallel |

### Behavior

1. Runs module, scores output with `reward_fn`
2. If score >= `threshold`, returns immediately
3. If score < threshold, generates feedback from the reward signal and retries
4. Repeats up to N total attempts
5. Returns the best-scoring output across all attempts
6. Raises if `fail_count` is exceeded

---

## dspy.BestOfN

Sampling approach: runs a module N times independently, returns the highest-scoring output.

### Constructor

```python
dspy.BestOfN(
    module: dspy.Module,
    N: int,
    reward_fn: Callable[[dict, dspy.Prediction], float],
    threshold: float,
    fail_count: int | None = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to run multiple times |
| `N` | `int` | required | Number of independent attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores each prediction (0.0-1.0) |
| `threshold` | `float` | required | Stop early if any attempt meets this score |
| `fail_count` | `int \| None` | `None` (defaults to N) | Max failures before raising error |

### Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `forward` | `(**kwargs) -> Prediction` | Runs module N times independently at temperature=1.0. Returns highest-scoring prediction. |
| `__call__` | `(*args, **kwargs) -> Prediction` | Primary invocation with callback/usage tracking |
| `set_lm` | `(lm) -> None` | Assigns LM to all internal predictors |
| `batch` | `(examples, num_threads=None, ...) -> list` | Processes multiple examples in parallel |

### Behavior

1. Runs module N times independently (no shared feedback between runs)
2. Scores each output with `reward_fn`
3. If any score >= `threshold`, can stop early
4. Returns the output with the highest score
5. Raises if none meet threshold and `fail_count` is exceeded

---

## Refine vs BestOfN

| | Refine | BestOfN |
|---|--------|---------|
| **Feedback** | Yes — prior attempt context feeds into next | No — each attempt is independent |
| **Best for** | Format fixes, constraint satisfaction, iterative improvement | Creative diversity, independent sampling |
| **Cost** | Sequential (slower, but may converge in fewer attempts) | Parallel-friendly (faster wall-clock if parallelized) |
| **When outputs are correlated** | Good — can learn from mistakes | Wasteful — generates similar outputs |
| **When outputs are diverse** | Less effective — feedback may not help | Good — maximizes chance of a great sample |

## reward_fn signature

```python
def my_reward(args: dict, pred: dspy.Prediction) -> float:
    # args = the kwargs passed to the module (e.g., {"question": "..."})
    # pred = the module's output Prediction
    # Return 0.0 for failure, 1.0 for success, partial scores for soft preferences
    ...
```

- Must return `float`, not `bool`
- `args` contains the module's input keyword arguments
- `pred` is a `dspy.Prediction` — access fields like `pred.answer`
- Do not instantiate DSPy modules inside reward_fn — create them once outside
