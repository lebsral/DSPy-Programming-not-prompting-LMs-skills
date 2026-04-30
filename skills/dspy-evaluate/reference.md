> Condensed from [dspy.ai/api/evaluation/Evaluate/](https://dspy.ai/api/evaluation/Evaluate/), [SemanticF1](https://dspy.ai/api/evaluation/SemanticF1/), and [CompleteAndGrounded](https://dspy.ai/api/evaluation/CompleteAndGrounded/). Verify against upstream for latest.

# dspy.Evaluate — API Reference

## Constructor

```python
dspy.Evaluate(
    *,
    devset,                    # list[Example] (required)
    metric=None,               # Callable | None
    num_threads=None,          # int | None
    display_progress=False,    # bool
    display_table=False,       # bool | int (int truncates columns)
    max_errors=None,           # int | None
    provide_traceback=None,    # bool | None
    failure_score=0.0,         # float
    save_as_csv=None,          # str | None
    save_as_json=None,         # str | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation dataset |
| `metric` | `Callable \| None` | `None` | Metric function `(example, prediction, trace=None) -> float\|bool` |
| `num_threads` | `int \| None` | `None` | Thread count for parallel evaluation |
| `display_progress` | `bool` | `False` | Show progress bar |
| `display_table` | `bool \| int` | `False` | Show results table; int truncates columns |
| `max_errors` | `int \| None` | `None` | Max errors before stopping |
| `failure_score` | `float` | `0.0` | Score assigned on evaluation failure |
| `save_as_csv` | `str \| None` | `None` | Save results to CSV file |
| `save_as_json` | `str \| None` | `None` | Save results to JSON file |

## __call__()

```python
result = evaluator(
    program,                   # dspy.Module (required)
    metric=None,               # override metric
    devset=None,               # override devset
    num_threads=None,          # override threads
    display_progress=None,     # override progress
    display_table=None,        # override table
)
```

**Returns:** `EvaluationResult` with:
- `.score` — aggregate percentage (0-100)
- `.results` — list of `(example, prediction, score)` tuples

**Note:** `return_all_scores` and `return_outputs` are removed. Per-example results are always in `.results`.

## Metric function signature

```python
def metric(example, prediction, trace=None):
    """
    Args:
        example: dspy.Example with ground truth
        prediction: dspy.Prediction from the program
        trace: None during evaluation, set during optimization

    Returns:
        bool, int, or float
    """
```

## Built-in metrics

### SemanticF1

```python
from dspy.evaluate import SemanticF1

semantic_f1 = SemanticF1(threshold=0.66, decompositional=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.66` | Minimum F1 score to accept during optimization |
| `decompositional` | `bool` | `False` | Use decompositional semantic recall/precision |

**Expected fields:** `question` and `response` on both example and prediction.

### CompleteAndGrounded

```python
from dspy.evaluate import CompleteAndGrounded

complete_and_grounded = CompleteAndGrounded(threshold=0.66)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `threshold` | `float` | `0.66` | Minimum score to accept during optimization |

**Expected fields:** `question` and `response` on example; `response` and `context` on prediction.

### Other built-ins

| Metric | Description |
|--------|-------------|
| `answer_exact_match` | Normalized string equality on `answer` field |
| `answer_passage_match` | Substring check on `answer` field |
