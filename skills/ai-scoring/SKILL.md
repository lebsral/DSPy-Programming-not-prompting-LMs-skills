---
name: ai-scoring
description: Score, grade, or evaluate things using AI against a rubric. Use when grading essays, scoring code reviews, rating candidate responses, auditing support quality, evaluating compliance, building a quality rubric, running QA checks against criteria, assessing performance, rating content quality, or any task where you need numeric scores with justifications — not just categories.
---

# Build an AI Scorer

Guide the user through building AI that scores, grades, or evaluates work against defined criteria. The pattern: define a rubric, score each criterion independently, calibrate with examples, and validate scorer quality.

## Step 1: Define the rubric

Ask the user:
1. **What are you scoring?** (essays, code, support responses, applications, etc.)
2. **What criteria matter?** (clarity, accuracy, completeness, tone, security, etc.)
3. **What's the scale?** (1-5, 1-10, pass/fail, letter grade)
4. **Are criteria weighted equally?** (e.g., accuracy 50%, clarity 30%, formatting 20%)

A good rubric has:
- **3-7 criteria** — more than that and scorers lose focus
- **Clear scale anchors** — what does a "2" vs a "4" look like?
- **Observable evidence** — criteria should reference things you can point to, not vibes

## Step 2: Build the scoring signature

```python
import dspy
from pydantic import BaseModel, Field

class CriterionScore(BaseModel):
    criterion: str = Field(description="Name of the criterion being scored")
    score: int = Field(ge=1, le=5, description="Score from 1 (poor) to 5 (excellent)")
    justification: str = Field(description="Evidence from the input that supports this score")

class ScoringResult(BaseModel):
    criterion_scores: list[CriterionScore] = Field(description="Score for each criterion")
    overall_score: float = Field(ge=1.0, le=5.0, description="Weighted overall score")
    summary: str = Field(description="Brief overall assessment")
```

Define what's being scored and the criteria:

```python
CRITERIA = [
    "clarity: Is the writing clear and easy to follow? (1=confusing, 5=crystal clear)",
    "argument: Is the argument well-structured and logical? (1=no structure, 5=compelling)",
    "evidence: Does the writing cite relevant evidence? (1=no evidence, 5=strong support)",
]

class ScoreCriterion(dspy.Signature):
    """Score the submission on a single criterion. Be specific — cite evidence from the text."""
    submission: str = dspy.InputField(desc="The work being evaluated")
    criterion: str = dspy.InputField(desc="The criterion to score, including scale description")
    score: int = dspy.OutputField(desc="Score from 1 to 5")
    justification: str = dspy.OutputField(desc="Specific evidence from the submission supporting this score")
```

## Step 3: Score per criterion independently

Scoring all criteria at once causes "halo effect" — a strong first impression biases all scores. Instead, score each criterion in its own call:

```python
class RubricScorer(dspy.Module):
    def __init__(self, criteria: list[str], weights: list[float] = None):
        self.criteria = criteria
        self.weights = weights or [1.0 / len(criteria)] * len(criteria)
        self.score_criterion = dspy.ChainOfThought(ScoreCriterion)

    def forward(self, submission: str):
        criterion_scores = []

        for criterion in self.criteria:
            result = self.score_criterion(
                submission=submission,
                criterion=criterion,
            )

            dspy.Assert(
                1 <= result.score <= 5,
                f"Score must be 1-5, got {result.score}"
            )
            dspy.Assert(
                len(result.justification) > 20,
                "Justification must cite specific evidence from the submission"
            )

            criterion_scores.append(CriterionScore(
                criterion=criterion.split(":")[0],
                score=result.score,
                justification=result.justification,
            ))

        overall = sum(
            cs.score * w for cs, w in zip(criterion_scores, self.weights)
        )

        return dspy.Prediction(
            criterion_scores=criterion_scores,
            overall_score=round(overall, 2),
        )
```

Using `ChainOfThought` here is important — reasoning through the evidence before assigning a score produces more calibrated results than jumping straight to a number.

## Step 4: Calibrate with anchor examples

Without anchors, the scorer doesn't know what a "2" vs a "4" looks like. Provide reference examples at each level:

```python
ANCHORS = """
Score 2 example for clarity: "The thing with the data is that it does stuff and the results are what they are."
→ Vague language, no specific referents, reader can't follow what's being described.

Score 4 example for clarity: "The customer churn model reduced false positives by 30% compared to the rule-based approach, though it still struggles with seasonal patterns."
→ Specific claims with numbers, clear comparison, one caveat noted.
"""

class ScoreCriterionCalibrated(dspy.Signature):
    """Score the submission on a single criterion. Use the anchor examples to calibrate your scoring."""
    submission: str = dspy.InputField(desc="The work being evaluated")
    criterion: str = dspy.InputField(desc="The criterion to score, including scale description")
    anchors: str = dspy.InputField(desc="Reference examples showing what different score levels look like")
    score: int = dspy.OutputField(desc="Score from 1 to 5")
    justification: str = dspy.OutputField(desc="Specific evidence from the submission supporting this score")
```

Then pass anchors per criterion:

```python
class CalibratedScorer(dspy.Module):
    def __init__(self, criteria: list[str], anchors: dict[str, str], weights: list[float] = None):
        self.criteria = criteria
        self.anchors = anchors
        self.weights = weights or [1.0 / len(criteria)] * len(criteria)
        self.score_criterion = dspy.ChainOfThought(ScoreCriterionCalibrated)

    def forward(self, submission: str):
        criterion_scores = []

        for criterion in self.criteria:
            criterion_name = criterion.split(":")[0]
            result = self.score_criterion(
                submission=submission,
                criterion=criterion,
                anchors=self.anchors.get(criterion_name, "No anchors provided."),
            )

            dspy.Assert(1 <= result.score <= 5, f"Score must be 1-5, got {result.score}")

            criterion_scores.append(CriterionScore(
                criterion=criterion_name,
                score=result.score,
                justification=result.justification,
            ))

        overall = sum(cs.score * w for cs, w in zip(criterion_scores, self.weights))
        return dspy.Prediction(
            criterion_scores=criterion_scores,
            overall_score=round(overall, 2),
        )
```

Writing good anchors takes effort, but it's the single biggest lever for scoring quality. Start with 2-3 anchors per criterion at the low, mid, and high ends of the scale.

## Step 5: Handle edge cases

### Validate score consistency

The overall score should be consistent with per-criterion scores:

```python
def validate_scores(criterion_scores, weights, overall_score):
    expected = sum(cs.score * w for cs, w in zip(criterion_scores, weights))
    dspy.Assert(
        abs(expected - overall_score) < 0.1,
        f"Overall score {overall_score} doesn't match weighted criteria ({expected:.2f})"
    )
```

### Handle "not applicable" criteria

Some criteria don't apply to every submission:

```python
class CriterionScoreOptional(BaseModel):
    criterion: str
    score: int = Field(ge=0, le=5, description="Score 1-5, or 0 if not applicable")
    justification: str
    applicable: bool = Field(description="Whether this criterion applies to this submission")
```

### Score ranges for pass/fail decisions

```python
def pass_fail(overall_score: float, threshold: float = 3.0) -> str:
    if overall_score >= threshold:
        return "pass"
    return "fail"

# Or with a "needs review" band
def tiered_decision(overall_score: float) -> str:
    if overall_score >= 4.0:
        return "pass"
    elif overall_score >= 2.5:
        return "needs_review"
    return "fail"
```

## Step 6: Multi-rater ensemble

For high-stakes scoring, run multiple independent scorers and flag disagreements:

```python
class EnsembleScorer(dspy.Module):
    def __init__(self, criteria, anchors, num_raters=3, weights=None):
        self.raters = [
            CalibratedScorer(criteria, anchors, weights)
            for _ in range(num_raters)
        ]

    def forward(self, submission: str):
        all_results = [rater(submission=submission) for rater in self.raters]

        # Check for disagreement per criterion
        flagged = []
        for i, criterion in enumerate(self.raters[0].criteria):
            criterion_name = criterion.split(":")[0]
            scores = [r.criterion_scores[i].score for r in all_results]
            spread = max(scores) - min(scores)
            if spread > 1:
                flagged.append({
                    "criterion": criterion_name,
                    "scores": scores,
                    "spread": spread,
                })

        # Average the overall scores
        avg_overall = sum(r.overall_score for r in all_results) / len(all_results)

        return dspy.Prediction(
            overall_score=round(avg_overall, 2),
            all_results=all_results,
            flagged_disagreements=flagged,
            needs_human_review=len(flagged) > 0,
        )
```

When raters disagree by more than 1 point on any criterion, flag it for human review. This catches the submissions that are genuinely ambiguous — exactly where human judgment matters most.

## Step 7: Evaluate scorer quality

### Prepare gold-standard scores

You need human-scored examples to evaluate your AI scorer:

```python
scored_examples = [
    dspy.Example(
        submission="...",
        gold_scores={"clarity": 4, "argument": 3, "evidence": 5},
        gold_overall=4.0,
    ).with_inputs("submission"),
    # 20-50+ scored examples
]
```

### Mean absolute error metric

```python
def scoring_metric(example, prediction, trace=None):
    """Measures how close AI scores are to human gold scores."""
    errors = []
    for cs in prediction.criterion_scores:
        gold = example.gold_scores.get(cs.criterion)
        if gold is not None:
            errors.append(abs(cs.score - gold))

    if not errors:
        return 0.0

    mae = sum(errors) / len(errors)
    # Convert to 0-1 scale (0 error = 1.0, 4 error = 0.0)
    return max(0.0, 1.0 - mae / 4.0)
```

### Agreement rate metric

```python
def agreement_metric(example, prediction, trace=None):
    """Score is 1.0 if all criteria are within 1 point of gold."""
    for cs in prediction.criterion_scores:
        gold = example.gold_scores.get(cs.criterion)
        if gold is not None and abs(cs.score - gold) > 1:
            return 0.0
    return 1.0
```

### Optimize the scorer

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=scored_examples, metric=scoring_metric, num_threads=4)
baseline = evaluator(scorer)

optimizer = dspy.MIPROv2(metric=scoring_metric, auto="medium")
optimized_scorer = optimizer.compile(scorer, trainset=trainset)

optimized_score = evaluator(optimized_scorer)
print(f"Baseline MAE: {baseline:.1f}%")
print(f"Optimized MAE: {optimized_score:.1f}%")
```

## Key patterns

- **Score per criterion independently** — prevents halo effect where one strong dimension inflates all scores
- **Use anchor examples** — the single biggest lever for calibration quality
- **ChainOfThought for scoring** — reasoning before scoring produces better-calibrated results
- **Require justifications** — forces the scorer to cite evidence, catches lazy scoring
- **Multi-rater for high stakes** — flag disagreements for human review
- **Validate consistency** — overall score should match weighted criterion scores
- **Pydantic for structure** — `Field(ge=1, le=5)` enforces valid score ranges automatically

## Additional resources

- For worked examples (essay grading, code review, support QA), see [examples.md](examples.md)
- Need discrete categories instead of scores? Use `/ai-sorting`
- Need to validate AI output (not score human work)? Use `/ai-checking-outputs`
- Need to improve scorer accuracy? Use `/ai-improving-accuracy`
- Next: `/ai-improving-accuracy` to measure and optimize your scorer
