> Condensed from [dspy.ai/api/](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Data Cleaning

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Input/output contract for the cleaning task |

The primary module for data cleaning. No reasoning step — direct input-to-output mapping. Use for single-field cleaners, rule inference, and category normalization where adding reasoning adds cost without improving correctness.

## dspy.Signature, InputField, OutputField

[API docs](https://dspy.ai/api/signatures/)

```python
class MySignature(dspy.Signature):
    """Docstring becomes the task instruction."""
    input_field: str = dspy.InputField(desc="description")
    output_field: str = dspy.OutputField(desc="description")
```

### InputField / OutputField options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str` | `""` | Description injected into the prompt |
| `prefix` | `str` | field name | Label prefix displayed in the prompt |

### Output field types used in cleaning signatures

| Type | Usage |
|---|---|
| `str` | Cleaned value, normalized form |
| `float` | Confidence score 0.0–1.0 |
| `bool` | `change_made`, `is_ambiguous`, `meaning_preserved`, `is_recognized` |
| `list[str]` | Rule lists from `InferNormalizationRules` |
| Pydantic `BaseModel` | Typed, validated output with `@field_validator` format checks |

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to wrap with retry logic |
| `N` | `int` | required | Max attempts before returning best result |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Returns 0.0–1.0; called after each attempt. Receives inputs dict and Prediction object |
| `threshold` | `float` | required | Stop early when reward meets or exceeds this value |
| `fail_count` | `int \| None` | `None` | Allowed failures before raising error (defaults to N if None) |

Use for format-validation loops — retry until the cleaned value matches your target regex. `dspy.Assert` and `dspy.Suggest` were removed in DSPy 3.x; `dspy.Refine` is the replacement.

```python
import re
TARGET_RE = r"^\+1 \(\d{3}\) \d{3}-\d{4}$"

def format_reward(args, pred):
    return 1.0 if re.match(TARGET_RE, pred.cleaned_value or "") else 0.0

cleaner = dspy.Refine(dspy.Predict(CleanField), N=3, reward_fn=format_reward, threshold=1.0)
```

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function — use exact match for format-critical fields |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos per predictor |
| `max_labeled_demos` | `int` | `16` | Max labeled demos pulled from trainset |
| `max_rounds` | `int` | `1` | Bootstrap iterations |

Key method: `.compile(module, trainset=...)` returns the optimized module. Requires ~50+ manually-cleaned examples with exact target values. Prefer `MIPROv2` when you have 200+ examples and accuracy is critical.

## module.batch()

Available on all DSPy modules. More efficient than a Python `for` loop for processing hundreds of ambiguous rows that could not be handled by deterministic pre-filters.

```python
results = cleaner.batch(
    examples,                     # list[dspy.Example]
    num_threads=4,                # parallel threads
    max_errors=5,                 # abort after N consecutive errors
    return_failed_examples=False,
)
```

`examples` must be a list of `dspy.Example` objects with `.with_inputs(...)` marking which fields are inputs. Results are returned in input order.

## Quick-Reference

| Task | DSPy API | Key notes |
|---|---|---|
| Single-field cleaning | `dspy.Predict(CleanField)` | Include `confidence`, `change_made` outputs |
| Format validation loop | `dspy.Refine(module, N=3, ...)` | `reward_fn` returns 1.0 on regex match |
| Rule inference from sample | `dspy.Predict(InferNormalizationRules)` | Use 20–50 anomalous sample values |
| Typed/validated output | Pydantic `BaseModel` as `OutputField` | Pair with `@field_validator` for format checks |
| Parallel batch processing | `module.batch(num_threads=N)` | After deterministic pre-filter to minimize LM calls |
| Optimize on gold standard | `dspy.BootstrapFewShot` | 50+ cleaned examples; exact-match metric |
| Stronger prompt optimization | `dspy.MIPROv2(metric, auto="medium")` | 200+ examples; best for high-volume pipelines |
