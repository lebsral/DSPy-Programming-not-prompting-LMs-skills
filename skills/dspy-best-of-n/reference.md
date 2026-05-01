> Condensed from [dspy.ai/api/modules/BestOfN/](https://dspy.ai/api/modules/BestOfN/). Verify against upstream for latest.

# dspy.BestOfN — API Reference

## Constructor

```python
dspy.BestOfN(
    module,          # dspy.Module (required)
    N,               # int (required)
    reward_fn,       # Callable[[dict, Prediction], float] (required)
    threshold,       # float (required)
    fail_count=None, # int | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to execute repeatedly. Deep-copied for each rollout to maintain isolation. |
| `N` | `int` | required | Maximum number of execution attempts. |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scoring function that takes the input args dict and a prediction, returns a scalar reward (higher is better). |
| `threshold` | `float` | required | Early-stop threshold. If any attempt scores >= this value, return immediately without running remaining attempts. |
| `fail_count` | `int | None` | `None` | Maximum number of attempts that can raise exceptions before BestOfN itself raises. Defaults to N (all can fail). |

## Inheritance

`BestOfN` extends `dspy.Module`.

## Methods

### `forward()`

```python
best_of_n.forward(**kwargs) -> Prediction
```

Executes the wrapped module up to N times with `temperature=1.0` and unique rollout IDs for each attempt. Returns the prediction with the highest reward score, or the first prediction that meets the threshold.

**Behavior:**
1. Deep-copies the module for each rollout
2. Calls the module with `temperature=1.0` and a unique rollout ID
3. Scores the result with `reward_fn(kwargs, prediction)`
4. If score >= `threshold`, returns immediately (early stopping)
5. If the attempt raises an exception, increments failure counter
6. After all N attempts, returns the highest-scoring prediction
7. If failures exceed `fail_count`, raises an exception

### `__call__()`

```python
best_of_n(**kwargs) -> Prediction
```

Entry point with callback support and usage tracking. Delegates to `forward()`.

### Module management methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `set_lm` | `(lm)` | Recursively sets the LM for all Predict instances within the module. |
| `get_lm` | `()` | Retrieves the current LM. |
| `batch` | `(examples, ...)` | Process multiple inputs in parallel. |
| `save` | `(path)` | Save module state to JSON. |
| `load` | `(path)` | Load a previously saved module. |
| `named_predictors` | `()` | Access all internal Predict modules. |
| `deepcopy` | `()` | Custom deep copy preserving parameters. |

## Key behaviors

- Module instances are deep-copied for each rollout to maintain isolation between attempts
- Temperature is fixed at 1.0 to maximize output diversity
- Unique rollout IDs ensure the LM produces different outputs even with identical inputs
- Execution traces from the best attempt are preserved
- The reward function signature is `(args_dict, prediction) -> float`, NOT `(example, prediction, trace) -> float`
