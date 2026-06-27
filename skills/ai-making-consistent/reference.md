> Condensed from [dspy.ai/api/](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Consistency

## dspy.LM

[API docs](https://dspy.ai/api/models/LM/)

```python
dspy.LM(model, temperature=0.0, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | LiteLLM model string — `"openai/gpt-4o-mini"`, `"anthropic/claude-sonnet-4-5-20250929"`, etc. |
| `temperature` | `float` | `0.0` | Sampling temperature. Set to `0` for maximum determinism. |

```python
lm = dspy.LM("openai/gpt-4o-mini", temperature=0)
dspy.configure(lm=lm)
```

Temperature controls randomness. At `temperature=0` the model picks the highest-probability token at each step. Some providers still have minor non-determinism due to floating-point differences in GPU arithmetic, but variation is minimal.

## dspy.configure_cache

[Tutorial](https://dspy.ai/tutorials/cache/)

```python
dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_disk_cache` | `bool` | `True` | Persist LM responses to disk. Identical inputs always return the same output. |
| `enable_memory_cache` | `bool` | `True` | Cache responses in memory within a single process run. |

DSPy enables both caches by default. Disable when measuring consistency so you see real LM variation instead of cached replay:

```python
# Disable when measuring — forces live LM calls
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

# Re-enable for production
dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)
```

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

Wraps a module and retries until the reward function score meets the threshold.

```python
dspy.Refine(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The DSPy module to wrap. |
| `N` | `int` | required | Maximum retry attempts. |
| `reward_fn` | `Callable` | required | `(args: dict, pred: dspy.Prediction) -> float` — higher score is better. |
| `threshold` | `float` | required | Accept the output when the reward reaches this value. |
| `fail_count` | `int \| None` | `None` | Raise an error after this many failures (defaults to N). |

Reward function signature — takes the input args dict and the prediction, returns a float:

```python
def reward_fn(args: dict, pred: dspy.Prediction) -> float:
    # args["ticket"] — input field passed to the module
    # pred.summary  — output field produced by the module
    score = 1.0
    if not pred.summary.rstrip().endswith("."):
        score -= 0.1
    return score
```

Use Refine for strict rules where feedback helps the model self-correct. Use `dspy.BestOfN` when sampling diversity is more useful than retrying with feedback.

## dspy.BestOfN

[API docs](https://dspy.ai/api/modules/BestOfN/)

Runs the module N times independently and returns the highest-scoring result.

```python
dspy.BestOfN(module, N, reward_fn, threshold=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The DSPy module to sample from. |
| `N` | `int` | required | Number of independent samples. |
| `reward_fn` | `Callable` | required | Same signature as Refine — `(args, pred) -> float`. |
| `threshold` | `float \| None` | `None` | Early-stop if any sample clears this score. |

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | `(example, prediction, trace=None) -> bool \| float` |
| `max_bootstrapped_demos` | `int` | `4` | Max generated few-shot demos to include in the prompt. |
| `max_labeled_demos` | `int` | `16` | Max labeled demos drawn from the trainset. |
| `max_rounds` | `int` | `1` | Bootstrap iterations. |

Key method:

```python
optimized = optimizer.compile(module, trainset=trainset)
```

Write the metric to penalize inconsistency alongside correctness — for example, return `False` when a correct answer contains hedging words (`"maybe"`, `"possibly"`) that would not appear in a confident consistent response.

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, failure_score=0.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples. |
| `metric` | `Callable \| None` | `None` | Scoring function. |
| `num_threads` | `int \| None` | `None` | Parallel evaluation threads. |
| `display_progress` | `bool` | `False` | Show progress bar. |
| `display_table` | `bool \| int` | `False` | Show results table (int = row count). |

```python
evaluator = dspy.Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(my_program)
```

## Typed output constraints

### Literal (typing module)

```python
from typing import Literal

CATEGORIES = ["billing", "technical", "account"]

class Route(dspy.Signature):
    ticket: str = dspy.InputField()
    team: Literal[tuple(CATEGORIES)] = dspy.OutputField()
```

Always use `Literal[tuple(my_list)]` — the `tuple()` call is required. `Literal[my_list]` is a syntax error.

### Pydantic BaseModel

```python
from pydantic import BaseModel, Field
from typing import Literal

class Output(BaseModel):
    category: Literal["low", "medium", "high"]
    confidence: float = Field(ge=0.0, le=1.0)

class MySignature(dspy.Signature):
    text: str = dspy.InputField()
    result: Output = dspy.OutputField()
```

Pydantic enforces types and field constraints at the DSPy adapter layer. Combine with `dspy.Refine` to enforce logic rules Pydantic cannot express (for example, `correct_answer` must appear in `options`).

## Package version

| Package | Tested version |
|---------|----------------|
| `dspy` | 3.2.1 |

```bash
pip install -U dspy
```
