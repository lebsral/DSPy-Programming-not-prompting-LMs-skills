> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Sorting

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines inputs/outputs |
| `rationale_field` | `FieldInfo \| None` | `None` | Custom reasoning field |
| `rationale_field_type` | `type` | `str` | Type for the rationale |

Adds a `reasoning` field automatically before the output. Do not add `reasoning` to your signature — DSPy injects it.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

Simplest module — no reasoning step. Use for binary/obvious classifications where reasoning adds cost without improving accuracy.

## dspy.Signature

[API docs](https://dspy.ai/api/signatures/)

```python
class MySignature(dspy.Signature):
    """Docstring becomes the task instruction."""
    input_field: str = dspy.InputField(desc="description")
    output_field: Literal[tuple(CATEGORIES)] = dspy.OutputField(desc="description")
```

Key methods on examples:
- `.with_inputs(*field_names)` — marks which fields are inputs (required for training data)

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `16` | Max labeled demos from trainset |
| `max_rounds` | `int` | `1` | Bootstrap iterations |

Key method:
- `.compile(module, trainset=...)` — returns optimized module

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto='light', prompt_model=None, task_model=None,
             max_bootstrapped_demos=4, max_labeled_demos=4,
             num_candidates=None, num_threads=None, seed=9, verbose=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function |
| `auto` | `'light' \| 'medium' \| 'heavy' \| None` | `'light'` | Optimization intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos |
| `num_candidates` | `int \| None` | `None` | Instruction candidates to try |

Key method:
- `.compile(module, trainset=...)` — returns optimized module

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, failure_score=0.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function |
| `num_threads` | `int \| None` | `None` | Parallel threads |
| `display_progress` | `bool` | `False` | Show progress bar |
| `display_table` | `bool \| int` | `False` | Show results table (int = row count) |

Call the evaluator instance with a module: `score = evaluator(module)`
