> Condensed from [dspy.ai/api/](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Scoring

## Setup

```bash
pip install -U dspy          # DSPy 3.2.1+
```

```python
import dspy
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

## dspy.Signature

[API docs](https://dspy.ai/api/signatures/)

```python
class ScoreCriterion(dspy.Signature):
    """Docstring becomes the task instruction."""
    submission: str    = dspy.InputField(desc="The work being evaluated")
    criterion: str     = dspy.InputField(desc="Criterion to score, including scale")
    score: int         = dspy.OutputField(desc="Score from 1 to 5")
    justification: str = dspy.OutputField(desc="Evidence from the submission")
```

Do not add a `reasoning` field — `ChainOfThought` injects it automatically.

## Pydantic score models

```python
from pydantic import BaseModel, Field

class CriterionScore(BaseModel):
    criterion: str
    score: int = Field(ge=1, le=5)           # raises ValidationError on out-of-range
    justification: str = Field(min_length=20) # rejects vague one-word answers
```

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
score_criterion = dspy.ChainOfThought(ScoreCriterion)
result = score_criterion(submission=text, criterion=criterion_str)
print(result.reasoning)     # injected evidence trace
score = int(result.score)   # always cast — adapter may return str
```

Adds a `reasoning` field before the output. Use this for scoring: reasoning through evidence before assigning a number produces better-calibrated results than `Predict`. `ChainOfThought(signature, rationale_field=None, rationale_field_type=str)`.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

No reasoning step. Use for high-volume, low-stakes screening where cost matters more than calibration. Expect ~10-15% lower agreement with human scores versus `ChainOfThought`.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N=3, reward_fn=None, threshold=None, fail_count=1)
```

`reward_fn` signature: `(args, pred) -> float` (higher = better). Use to enforce justification quality — retry when justification is vague or score is out-of-range. Attempts see feedback from prior failures; unlike `BestOfN`, which samples independently.

## dspy.BestOfN

[API docs](https://dspy.ai/api/modules/BestOfN/)

```python
dspy.BestOfN(module, N=5, reward_fn=None, threshold=None)
```

Runs the module N times independently, returns the prediction with the highest reward. Use when you want sampling diversity without feedback across attempts.

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
from dspy.evaluate import Evaluate
evaluator = Evaluate(devset=gold_examples, metric=scoring_metric,
                     num_threads=4, display_progress=True)
score = evaluator(scorer)   # returns aggregate score (0-100)
```

Key parameters: `devset` (required, `list[Example]`), `metric` (`(example, prediction, trace=None) -> float`), `num_threads`, `display_table` (`bool | int`).

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
optimizer = dspy.MIPROv2(metric=scoring_metric, auto="medium")
optimized = optimizer.compile(scorer, trainset=trainset)
```

Key parameters: `metric` (required), `auto` (`"light" | "medium" | "heavy"`), `max_bootstrapped_demos` (default `4`), `max_labeled_demos` (default `4`).

Typical improvement with 30+ gold examples: 50-65% human agreement → 75-90% after MIPROv2.

## Metric patterns

```python
# Mean absolute error — 0.0-1.0 (0 error = 1.0, 4-point error = 0.0)
def scoring_metric(example, prediction, trace=None):
    errors = [abs(cs.score - example.gold_scores[cs.criterion])
              for cs in prediction.criterion_scores
              if cs.criterion in example.gold_scores]
    return max(0.0, 1.0 - sum(errors) / len(errors) / 4.0) if errors else 0.0

# Agreement rate — 1.0 when all criteria within 1 point of gold
def agreement_metric(example, prediction, trace=None):
    return float(all(abs(cs.score - example.gold_scores.get(cs.criterion, cs.score)) <= 1
                     for cs in prediction.criterion_scores))
```

## dspy.Example and batch

```python
# .with_inputs() is required for optimizer trainsets
example = dspy.Example(submission="...", gold_scores={"clarity": 4}).with_inputs("submission")

# All modules support batch() for parallel scoring
results = scorer.batch(examples, num_threads=4, max_errors=5)
```

## Quick-reference

| Task | API | Notes |
|------|-----|-------|
| Score with reasoning | `dspy.ChainOfThought` | Default — adds evidence trace before score |
| High-volume screening | `dspy.Predict` | Faster, ~10-15% lower human agreement |
| Enforce justification quality | `dspy.Refine` | Retry with reward feedback, up to N times |
| Multi-rater sampling | `dspy.BestOfN` | Independent samples, pick highest reward |
| Evaluate scorer vs gold | `dspy.Evaluate` | Pass MAE or agreement rate metric |
| Optimize scorer prompts | `dspy.MIPROv2` | Needs 20-50+ gold-scored examples |
