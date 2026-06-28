> Condensed from [dspy.ai/api/modules/ChainOfThought](https://dspy.ai/api/modules/ChainOfThought/), [dspy.ai/api/optimizers/BootstrapFewShot](https://dspy.ai/api/optimizers/BootstrapFewShot/), [dspy.ai/api/optimizers/MIPROv2](https://dspy.ai/api/optimizers/MIPROv2/), and [dspy.ai/api/evaluation/Evaluate](https://dspy.ai/api/evaluation/Evaluate/). Verify against upstream for latest.

# DSPy API Reference for Record Matching

## Pairwise Comparison Signature

The canonical output shape for a matching signature has three fields:

```python
class CompareRecords(dspy.Signature):
    """Docstring sets the task — describe what makes two records the same entity."""
    record_a: str = dspy.InputField(desc="First record as field: value pairs")
    record_b: str = dspy.InputField(desc="Second record as field: value pairs")
    is_match: bool = dspy.OutputField(desc="True if both records refer to the same entity")
    confidence: float = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    explanation: str = dspy.OutputField(desc="Brief reason for the match or non-match decision")
```

Field notes:
- `is_match: bool` — required for threshold routing (auto-merge / review / reject)
- `confidence: float` — required; boolean-only output makes threshold routing impossible
- `explanation: str` — aids human review queues and optimizer training signal
- Do not add a `reasoning` field — `dspy.ChainOfThought` injects it automatically

Expand the docstring with domain-specific signal weights:
```python
"""Email match is strong evidence. Name variants like Bob/Robert are common."""
```

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

Use `ChainOfThought` for pairwise scoring — the reasoning step handles edge cases (abbreviations, rebrands, nickname variants) that `Predict` misses.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

Use `Predict` (no reasoning) only for high-volume pre-filters where latency matters more than edge-case accuracy. For final pairwise scoring, prefer `ChainOfThought`.

## Batch Processing

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

All DSPy modules expose `batch()` for parallel execution across candidate pairs:

```python
results = matcher.batch(
    examples,                    # list[dspy.Example]
    num_threads=4,               # parallel LM calls
    max_errors=5,                # abort threshold
    return_failed_examples=False,
)
```

Set `num_threads` to match your LM provider's rate limit (typically 4–8). `batch()` is the right tool for large candidate pair lists — avoid a sequential Python loop.

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function `(example, pred, trace) -> float` |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `16` | Max labeled demos from trainset |
| `max_rounds` | `int` | `1` | Bootstrap iterations |

Key method: `.compile(module, trainset=...)` — returns optimized module.

Start here when you have 20–50 labeled pairs. Weight the metric to penalize false positives more than false negatives — merging distinct records is harder to reverse than missing a duplicate.

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, prompt_model=None, task_model=None, teacher_settings=None,
             max_bootstrapped_demos=4, max_labeled_demos=4,
             auto='light', num_candidates=None, num_threads=None,
             max_errors=None, seed=9, init_temperature=1.0,
             verbose=False, track_stats=True, log_dir=None,
             metric_threshold=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function |
| `auto` | `'light' \| 'medium' \| 'heavy' \| None` | `'light'` | Optimization intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos |
| `teacher_settings` | `dict \| None` | `None` | Config for the teacher model |
| `max_errors` | `int \| None` | `None` | Error limit before termination |
| `init_temperature` | `float` | `1.0` | Initial sampling temperature |
| `track_stats` | `bool` | `True` | Track optimization statistics |
| `log_dir` | `str \| None` | `None` | Directory to write logs |
| `metric_threshold` | `float \| None` | `None` | Score threshold for accepting demos |

Key method: `.compile(module, trainset=...)` — returns optimized module.

Upgrade from `BootstrapFewShot` to `MIPROv2` when accuracy plateaus and you have 100+ labeled pairs. `MIPROv2` also tunes the instruction text, not just the few-shot examples.

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(*, devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, provide_traceback=None,
              failure_score=0.0, save_as_csv=None, save_as_json=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function |
| `num_threads` | `int \| None` | `None` | Parallel threads |
| `display_progress` | `bool` | `False` | Show progress bar |
| `display_table` | `bool \| int` | `False` | Show full table (`True`), suppress (`False`), or limit rows (int) |
| `max_errors` | `int \| None` | `None` | Abort threshold for errors |
| `provide_traceback` | `bool \| None` | `None` | Include tracebacks in error output |
| `failure_score` | `float` | `0.0` | Score assigned to failed examples |
| `save_as_csv` | `str \| None` | `None` | Path to save results as CSV |
| `save_as_json` | `str \| None` | `None` | Path to save results as JSON |

Call the evaluator as a callable: `score = evaluator(module)`.

## Blocking Libraries

Non-DSPy dependencies used in the blocking layer before LM scoring:

| Library | Install | Used for |
|---------|---------|---------|
| `jellyfish` | `pip install jellyfish` | Soundex / phonetic name encoding |
| `scikit-learn` | `pip install scikit-learn` | Cosine similarity for embedding blocking |
| `rapidfuzz` | `pip install rapidfuzz` | Token overlap / fuzzy string pre-filter |
| `itertools` | stdlib | `combinations(records, 2)` for pair generation |

## Quick Reference

### Confidence thresholds

| Confidence | Action |
|-----------|--------|
| >= 0.9 | Auto-merge |
| 0.6 – 0.9 | Human review queue |
| < 0.6 | Reject as distinct |

Tune with labeled pairs via `dspy.Evaluate`. Tighten the auto-merge threshold when false positives are costly.

### Blocking strategy by scale

| Dataset size | Recommended strategy |
|-------------|---------------------|
| < 500 records | All-pairs — skip blocking |
| 500 – 50k | Phonetic (jellyfish Soundex) or token overlap |
| 50k+ | Embedding cosine similarity or sorted neighborhood |

### Save / load

```python
optimized_matcher.save("record_matcher.json")

loaded = RecordMatcher()
loaded.load("record_matcher.json")
```
