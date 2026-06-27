> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Anomaly Detection

## Quick config

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

## Severity levels and score ranges

| Severity   | Score range | Default action          | Blocks |
|------------|-------------|-------------------------|--------|
| `normal`   | 0.0 – 0.2   | Dismiss silently        | No     |
| `low`      | 0.2 – 0.4   | Log for review          | No     |
| `medium`   | 0.4 – 0.6   | Queue for analyst       | No     |
| `high`     | 0.6 – 0.8   | Alert on-call           | No     |
| `critical` | 0.8 – 1.0   | Escalate + block        | Yes    |

Define as a constant and reuse across signatures:

```python
from typing import Literal

SEVERITY_LEVELS = ["normal", "low", "medium", "high", "critical"]
severity_type = Literal[tuple(SEVERITY_LEVELS)]  # use Literal[tuple(list)], not Literal[list]
```

## Core anomaly signature fields

| Field              | Type                          | Role                                                    |
|--------------------|-------------------------------|---------------------------------------------------------|
| `event`            | `str` (InputField)            | The event to analyze, as JSON or structured text        |
| `baseline_summary` | `str` (InputField)            | Compact description of normal behavior                  |
| `severity`         | `Literal[tuple(SEVERITY_LEVELS)]` (OutputField) | Graduated severity — always use five levels |
| `anomaly_score`    | `float` (OutputField)         | Confidence from 0.0 (normal) to 1.0 (anomalous)        |
| `explanation`      | `str` (OutputField)           | Specific deviations with concrete numbers               |
| `risk_factors`     | `list[str]` (OutputField)     | Optional - individual risk signals for session scoring  |

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter             | Type                   | Default | Description                              |
|-----------------------|------------------------|---------|------------------------------------------|
| `signature`           | `str \| type[Signature]` | required | Defines inputs/outputs                 |
| `rationale_field`     | `FieldInfo \| None`    | `None`  | Custom reasoning field                   |
| `rationale_field_type`| `type`                 | `str`   | Type for the rationale                   |

Adds a `reasoning` field automatically before outputs. Do not add `reasoning` to your signature — DSPy injects it. Use `ChainOfThought` for anomaly scoring; the reasoning step materially improves severity accuracy.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

No reasoning step. Use only for cheap pre-filter steps (e.g., quick binary normal/suspicious triage) before passing candidates to a `ChainOfThought` scorer.

## dspy.BestOfN and dspy.Refine

[API docs — BestOfN](https://dspy.ai/api/modules/BestOfN/) | [Refine](https://dspy.ai/api/modules/Refine/)

Use these to enforce valid severity outputs instead of `dspy.Assert`/`dspy.Suggest` (removed in DSPy 3.x).

```python
def severity_reward(args, pred):
    if pred.severity not in SEVERITY_LEVELS:
        return 0.0
    if not pred.explanation or len(pred.explanation) < 20:
        return 0.5  # penalize vague explanations
    return 1.0

# BestOfN - independent samples, pick best
scorer = dspy.BestOfN(module=dspy.ChainOfThought(ScoreAnomaly), N=3, reward_fn=severity_reward)

# Refine - iterative with feedback
scorer = dspy.Refine(module=dspy.ChainOfThought(ScoreAnomaly), N=3, reward_fn=severity_reward, threshold=0.8)
```

| Parameter    | Type       | Default | Description                              |
|--------------|------------|---------|------------------------------------------|
| `module`     | `dspy.Module` | required | Wrapped module to sample              |
| `N`          | `int`      | required | Number of samples                        |
| `reward_fn`  | `Callable` | required | `(args, pred) -> float` — higher is better |
| `threshold`  | `float`    | `None`  | Stop early if a sample exceeds this score |

## dspy.Module (custom pipeline)

```python
class AnomalyDetector(dspy.Module):
    def __init__(self):
        self.baseline_summarizer = dspy.ChainOfThought(SummarizeBaseline)
        self.scorer = dspy.ChainOfThought(ScoreAnomaly)

    def forward(self, event: str, baseline_summary: str):
        return self.scorer(event=event, baseline_summary=baseline_summary)
```

All modules expose `.batch(examples, num_threads=4, max_errors=5)` for parallel event scoring.

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter               | Type       | Default | Description                        |
|-------------------------|------------|---------|------------------------------------|
| `metric`                | `Callable` | `None`  | Scoring function                   |
| `max_bootstrapped_demos`| `int`      | `4`     | Max generated demos                |
| `max_labeled_demos`     | `int`      | `16`    | Max labeled demos from trainset    |
| `max_rounds`            | `int`      | `1`     | Bootstrap iterations               |

```python
optimizer = dspy.BootstrapFewShot(metric=anomaly_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(anomaly_scorer, trainset=trainset)
optimized.save("anomaly_scorer.json")
```

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, failure_score=0.0)
```

| Parameter         | Type           | Default | Description                           |
|-------------------|----------------|---------|---------------------------------------|
| `devset`          | `list[Example]` | required | Labeled evaluation examples          |
| `metric`          | `Callable`     | `None`  | Scoring function                      |
| `num_threads`     | `int \| None`  | `None`  | Parallel threads                      |
| `display_table`   | `bool \| int`  | `False` | Show results table (int = row count)  |

Anomaly metric with partial credit for adjacent severities:

```python
def anomaly_metric(example, prediction, trace=None):
    if prediction.severity == example.severity:
        return 1.0
    order = {s: i for i, s in enumerate(SEVERITY_LEVELS)}
    distance = abs(order[prediction.severity] - order[example.severity])
    return max(0.0, 1.0 - distance * 0.3)
```

Call: `score = evaluator(anomaly_scorer)`
