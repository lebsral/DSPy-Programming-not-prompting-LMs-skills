> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Content Moderation

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

No reasoning step. Use for binary or obvious violations where reasoning adds cost without improving accuracy.

## Moderation signatures

Class-based signatures are required for `Literal`-constrained outputs:

```python
from typing import Literal

VIOLATIONS = Literal["safe", "spam", "hate_speech", "harassment", "violence", "nsfw", "self_harm", "illegal"]

class ModerateContent(dspy.Signature):
    """Assess user-generated content against platform policies."""
    content: str = dspy.InputField(desc="user-generated content to moderate")
    platform_context: str = dspy.InputField(desc="where this content appears")
    violation_type: VIOLATIONS = dspy.OutputField()
    severity: Literal["none", "low", "medium", "high"] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="0.0 to 1.0 — how sure are you?")
    explanation: str = dspy.OutputField(desc="brief reason for the decision")
```

For dynamic violation lists, use `Literal[tuple(categories)]` — not `Literal[list]` (silently fails type validation):

```python
label: Literal[tuple(VIOLATION_TYPES)] = dspy.OutputField()  # correct
```

Multi-label output: `violations: list[str] = dspy.OutputField(desc=f"all that apply from: {VIOLATION_TYPES}")`.

Always clamp `confidence` before routing: `confidence = max(0.0, min(1.0, result.confidence))`. LM self-reported confidence is directionally useful but not a calibrated probability — tune thresholds on your dev set.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to refine |
| `N` | `int` | required | Max retry attempts |
| `reward_fn` | `Callable[[args, pred], float]` | required | Scores output quality; higher is better |
| `threshold` | `float` | `1.0` | Stop early if reward meets or exceeds this |

Use to constrain multi-label outputs to the known violation set:

```python
def multi_label_reward(args, pred):
    return 1.0 if all(v in VIOLATION_TYPES for v in pred.violations) else 0.0

validated = dspy.Refine(module=MultiLabelModerator(), N=3, reward_fn=multi_label_reward, threshold=1.0)
```

## Batch processing

All DSPy modules support `.batch()` for parallel moderation at scale:

```python
results = moderator.batch(examples, num_threads=4, max_errors=5, return_failed_examples=False)
```

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto='light', max_bootstrapped_demos=4, max_labeled_demos=4, num_threads=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function |
| `auto` | `'light' \| 'medium' \| 'heavy'` | `'light'` | Optimization intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos |

Key method: `.compile(module, trainset=...)` — returns optimized module.

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset, metric=None, num_threads=None, display_table=False)
score = evaluator(module)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Labeled evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function |
| `num_threads` | `int \| None` | `None` | Parallel threads |
| `display_table` | `bool \| int` | `False` | Show results table (int = row count) |

## Quick-reference: severity routing

| Condition | Decision |
|-----------|----------|
| `confidence < threshold` | `human_review` |
| `severity == "high"` | `remove` |
| `severity == "medium"` | `human_review` |
| `severity == "low"` | `warn` |
| `severity == "none"` | `approve` |

Default `confidence_threshold=0.7`. For high-volume pipelines, assign a cheaper LM to the assess step: `moderator.assess.set_lm(dspy.LM("openai/gpt-4o-mini"))`.
