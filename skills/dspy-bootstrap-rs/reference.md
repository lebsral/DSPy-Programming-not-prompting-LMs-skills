> Condensed from [dspy.ai/api/optimizers/BootstrapFewShotWithRandomSearch/](https://dspy.ai/api/optimizers/BootstrapFewShotWithRandomSearch/). Verify against upstream for latest.

# dspy.BootstrapFewShotWithRandomSearch — API Reference

## Constructor

```python
dspy.BootstrapFewShotWithRandomSearch(
    metric,
    teacher_settings=None,
    max_bootstrapped_demos=4,
    max_labeled_demos=16,
    max_rounds=1,
    num_candidate_programs=16,
    num_threads=None,
    max_errors=None,
    stop_at_score=None,
    metric_threshold=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function `(example, prediction, trace=None) -> float\|bool` |
| `teacher_settings` | `dict \| None` | `None` | Configuration for the teacher model, e.g. `{"lm": teacher_lm}` |
| `max_bootstrapped_demos` | `int` | `4` | Maximum bootstrapped (program-generated) demos per predictor |
| `max_labeled_demos` | `int` | `16` | Maximum labeled (from trainset) demos per predictor |
| `max_rounds` | `int` | `1` | Bootstrap rounds per candidate. >1 generates diverse traces at temperature=1.0 |
| `num_candidate_programs` | `int` | `16` | Number of random bootstrap attempts to evaluate |
| `num_threads` | `int \| None` | `None` | Threads for evaluation. Falls back to `dspy.settings.num_threads` |
| `max_errors` | `int \| None` | `None` | Error tolerance threshold. Falls back to `dspy.settings.max_errors` |
| `stop_at_score` | `float \| None` | `None` | Stop search early if a candidate reaches this score |
| `metric_threshold` | `float \| None` | `None` | Minimum metric score for a bootstrapped demo to be included |

## Methods

### `compile()`

```python
optimizer.compile(
    student,
    *,
    teacher=None,
    trainset,
    valset=None,
    restrict=None,
    labeled_sample=True,
)
```

Optimizes the student program by running BootstrapFewShot multiple times with different random seeds and returning the best candidate.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `dspy.Module` | required | The program to optimize |
| `teacher` | `dspy.Module \| None` | `None` | Optional teacher module for bootstrapping demos |
| `trainset` | `list[Example]` | required | Training examples |
| `valset` | `list[Example] \| None` | `None` | Validation examples. If `None`, uses a portion of `trainset` |
| `restrict` | `list[int] \| None` | `None` | Restrict to specific seed indices |
| `labeled_sample` | `bool` | `True` | Whether to sample labeled demos |

**Returns:** Optimized `dspy.Module` with a `candidate_programs` attribute containing all scored candidates.

### `get_params()`

```python
optimizer.get_params() -> dict[str, Any]
```

Returns all configuration parameters as a dictionary.

## Candidate initialization strategies

The optimizer evaluates candidates across these strategies:

| Seed | Strategy | Description |
|------|----------|-------------|
| -3 | Zero-shot baseline | Uncompiled program, no demos |
| -2 | Label-only few-shot | `LabeledFewShot` with labeled demos only |
| -1 | Standard bootstrap | `BootstrapFewShot` with unshuffled training data |
| 0+ | Random variants | `BootstrapFewShot` with shuffled training data |

## Result attributes

The returned optimized program has:

| Attribute | Type | Description |
|-----------|------|-------------|
| `candidate_programs` | `list` | All candidate programs with their validation scores |
| `demos` (per predictor) | `list[dict]` | The selected few-shot demos for each predictor |

## Relationship to BootstrapFewShot

`BootstrapFewShotWithRandomSearch` wraps `BootstrapFewShot`. It shares the same bootstrap logic but runs it `num_candidate_programs` times and picks the best result. Key differences:

| | BootstrapFewShot | BootstrapFewShotWithRandomSearch |
|---|---|---|
| Bootstrap runs | 1 | `num_candidate_programs` (default 16) |
| Selection | Single result | Best of N candidates |
| Validation | No | Yes (on valset) |
| `stop_at_score` | No | Yes |
| Cost | 1x | ~Nx |
