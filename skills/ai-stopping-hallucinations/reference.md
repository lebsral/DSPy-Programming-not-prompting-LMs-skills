> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Stopping Hallucinations

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to wrap and retry |
| `N` | `int` | required | Max number of attempts |
| `reward_fn` | `Callable[[dict, dspy.Prediction], float]` | required | Scores each attempt 0.0 to 1.0 |
| `threshold` | `float` | required | Short-circuit score — stops early when reached |
| `fail_count` | `int \| None` | `None` | Max allowed failures before raising an error; defaults to N |

Calls the wrapped module up to `N` times. On each attempt, passes the reward score as feedback so the model can improve. Returns the highest-scoring prediction seen. If `threshold` is set, stops early when the score meets or exceeds it.

**Reward function signature — required:**

```python
def my_reward(args: dict, pred: dspy.Prediction) -> float:
    # args: the inputs passed to the module
    # pred: the prediction returned by the module
    return 1.0  # 0.0 to 1.0
```

Use `getattr(pred, "field_name", default)` inside reward functions — if the module raises or returns a partial prediction, direct attribute access crashes the retry loop.

## dspy.BestOfN

[API docs](https://dspy.ai/api/modules/BestOfN/)

```python
dspy.BestOfN(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to sample from |
| `N` | `int` | required | Number of candidates to generate |
| `reward_fn` | `Callable[[dict, dspy.Prediction], float]` | required | Scores each candidate |
| `threshold` | `float` | required | Minimum acceptable reward score |
| `fail_count` | `int \| None` | `None` | Max allowed failures before raising an error; defaults to N |

Always generates all `N` candidates independently (no early exit), then returns the highest-scoring one. Use when you want sampling diversity over iterative correction.

**Refine vs. BestOfN:** `Refine` is sequential with feedback and can exit early; `BestOfN` always runs all N and picks the best. Use `Refine` for citation/faithfulness enforcement, `BestOfN` for high-stakes outputs where consistent sampling matters.

## dspy.Predict and dspy.ChainOfThought

```python
dspy.Predict(signature, **config)
dspy.ChainOfThought(signature, **config)
```

`dspy.Predict` — direct LM call, no reasoning step. Use for the verification/faithfulness-check predictor (classification task, no reasoning needed).

`dspy.ChainOfThought` — adds a `reasoning` field before outputs. Use for the answer-generation predictor. Do not add `reasoning` to your signature — DSPy injects it automatically.

## set_lm on predictors

```python
predictor.set_lm(dspy.LM("openai/gpt-4o-mini"))
```

Sets a per-predictor LM override. Verification is a classification task — smaller, cheaper models handle it well. Call `.set_lm()` on the verify predictor inside `__init__`, not on the whole module.

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, max_bootstrapped_demos=4,
                      max_labeled_demos=16, max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function `(example, prediction, trace) -> float` |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demo examples |
| `max_labeled_demos` | `int` | `16` | Max labeled examples from trainset |
| `max_rounds` | `int` | `1` | Bootstrap iterations |

```python
optimizer = dspy.BootstrapFewShot(metric=faithfulness_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(my_module, trainset=trainset)
optimized.save("optimized.json")

# Reload
module = MyModule()
module.load("optimized.json")
```

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(devset, metric=None, num_threads=None,
              display_progress=False, display_table=False, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[dspy.Example]` | required | Evaluation examples |
| `metric` | `Callable` | `None` | `(example, prediction, trace) -> float \| bool` |
| `num_threads` | `int \| None` | `None` | Parallel evaluation threads |
| `display_table` | `bool \| int` | `False` | Show results table (int = row count) |

```python
evaluator = dspy.Evaluate(devset=devset, metric=faithfulness_metric, num_threads=4)
score = evaluator(my_grounded_qa)
```

## Quick Reference

### Setup

```python
pip install -U dspy           # DSPy 3.2.1+
import dspy
lm = dspy.LM("openai/gpt-4o-mini")   # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

### Pattern summary

| Pattern | Wrapper | LM calls | Threshold |
|---------|---------|----------|-----------|
| Citation enforcement | `dspy.Refine` | 1-N | 0.5 (ratio) |
| Faithfulness verification | `dspy.Refine` | 2×1-N | 1.0 (binary) |
| Cross-check (best of N) | `dspy.BestOfN` | N | 0.0+ |
| Confidence gating | none | 1 | user-defined |
| Cheap verifier | `dspy.Refine` + `.set_lm()` | 1 + 1-N | 1.0 |

`.with_inputs(*field_names)` is required on `dspy.Example` objects passed to optimizers — marks which fields are inputs vs. gold labels.
