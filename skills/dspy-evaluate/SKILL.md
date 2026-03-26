---
name: dspy-evaluate
description: "Use when you need to measure how well your DSPy program performs — writing metrics, scoring against a dev set, or comparing before/after optimization. Common scenarios: measuring accuracy before and after optimization, writing custom metrics for your task, scoring a program against a held-out dev set, comparing two prompt strategies, building a test suite for AI quality, or running regression tests on AI outputs. Related: ai-improving-accuracy, ai-scoring, ai-monitoring. Also: "dspy.Evaluate", "dspy.evaluate", "write DSPy metric function", "measure AI accuracy", "evaluate DSPy program", "dev set evaluation", "before and after optimization comparison", "custom scoring function", "test AI quality systematically", "AI regression testing", "metric-driven development", "how to know if my DSPy program improved", "score predictions against labels", "evaluation harness for LLM", "CI/CD for AI quality"."
---

# Evaluate Your DSPy Program

Guide the user through measuring AI quality with DSPy's `Evaluate` class. The pattern: pick a metric, prepare a devset, run the evaluator, interpret results, then feed the same metric into an optimizer.

## What is dspy.Evaluate

`dspy.Evaluate` runs your program on every devset example, scores each with a metric, and reports the aggregate score. It handles threading and progress display. Returns a percentage (0-100).

## Built-in metrics

DSPy provides `answer_exact_match` (normalized string equality) and `answer_passage_match` (substring check). Both expect an `answer` field on example and prediction.

### SemanticF1

Measures token-level overlap between the predicted and expected answer using an F1 score. More forgiving than exact match — it gives partial credit for answers that are close but not identical:

```python
from dspy.evaluate import SemanticF1

semantic_f1 = SemanticF1()

evaluator = Evaluate(devset=devset, metric=semantic_f1, num_threads=4)
score = evaluator(my_program)
```

`SemanticF1` is a good default metric for open-ended QA tasks where exact match is too strict. It returns a float between 0.0 and 1.0.

### CompleteAndGrounded

Checks whether the predicted answer is both complete (covers all key claims in the gold answer) and grounded (doesn't hallucinate facts not in the gold answer):

```python
from dspy.evaluate import CompleteAndGrounded

complete_and_grounded = CompleteAndGrounded()

evaluator = Evaluate(devset=devset, metric=complete_and_grounded, num_threads=4)
score = evaluator(my_program)
```

This is an LM-based metric — it uses the configured LM to judge completeness and groundedness. Useful for summarization and RAG tasks where you care about both recall and precision of facts.

## Custom metrics

A metric is `def metric(example, prediction, trace=None)` returning `bool`, `int`, or `float`. The `trace` parameter is `None` during evaluation but set during optimization (use this to apply stricter requirements during training).

### Multi-field scoring

```python
def metric(example, prediction, trace=None):
    fields = ["name", "email", "phone"]
    correct = sum(
        1 for f in fields
        if getattr(prediction, f, "").strip().lower() == getattr(example, f, "").strip().lower()
    )
    return correct / len(fields)
```

## LM-as-judge

For open-ended tasks (summaries, creative writing, complex QA), use an LM to judge quality. Define a signature for the judge, then call it inside your metric:

```python
class AssessAnswer(dspy.Signature):
    """Assess if the predicted answer correctly addresses the question."""
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField(desc="The reference answer")
    predicted_answer: str = dspy.InputField(desc="The answer to evaluate")
    is_correct: bool = dspy.OutputField(desc="True if the prediction is correct and complete")

def llm_judge_metric(example, prediction, trace=None):
    judge = dspy.Predict(AssessAnswer)
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=prediction.answer,
    )
    return result.is_correct
```

### Use a separate LM for the judge

To avoid the model grading its own work, use a different (often stronger) LM for the judge:

```python
judge_lm = dspy.LM("openai/gpt-4o")

def llm_judge_metric(example, prediction, trace=None):
    judge = dspy.Predict(AssessAnswer)
    with dspy.context(lm=judge_lm):
        result = judge(
            question=example.question,
            gold_answer=example.answer,
            predicted_answer=prediction.answer,
        )
    return result.is_correct
```

### Graded judge (float scores)

Return a float instead of a bool for partial credit:

```python
class GradeAnswer(dspy.Signature):
    """Grade the predicted answer on a scale of 0 to 5."""
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    score: int = dspy.OutputField(desc="Score from 0 (completely wrong) to 5 (perfect)")
    justification: str = dspy.OutputField(desc="Why this score was given")

def graded_metric(example, prediction, trace=None):
    judge = dspy.ChainOfThought(GradeAnswer)
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=prediction.answer,
    )
    return result.score / 5.0  # normalize to 0.0-1.0
```

## Composite metrics

Combine multiple signals into a single score with weights:

```python
def composite_metric(example, prediction, trace=None):
    # Correctness (primary signal)
    correct = float(prediction.answer.strip().lower() == example.answer.strip().lower())

    # Conciseness (prefer shorter answers)
    concise = float(len(prediction.answer.split()) < 50)

    # Has reasoning (check that the model explained its thinking)
    has_reasoning = float(len(getattr(prediction, "reasoning", "")) > 20)

    # Weighted combination
    return 0.7 * correct + 0.2 * concise + 0.1 * has_reasoning
```

### Mixing exact checks with LM judges

```python
def hybrid_metric(example, prediction, trace=None):
    # Fast exact check
    if prediction.answer.strip().lower() == example.answer.strip().lower():
        return 1.0

    # Fall back to LM judge for partial credit
    judge = dspy.Predict(AssessAnswer)
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=prediction.answer,
    )
    return 0.5 if result.is_correct else 0.0
```

## Debugging with per-example scores

Pass `return_all_scores=True` to get a tuple of `(aggregate_score, all_scores)`. Use this to find failing examples and understand failure patterns.

## Common patterns

### Trace-aware metrics for optimization

The `trace` parameter is `None` during evaluation but set during optimization. Use this to apply stricter requirements during training:

```python
def metric(example, prediction, trace=None):
    correct = prediction.answer.strip().lower() == example.answer.strip().lower()
    if trace is not None:
        # During optimization: also require good reasoning
        has_reasoning = len(getattr(prediction, "reasoning", "")) > 50
        return correct and has_reasoning
    # During evaluation: only check correctness
    return correct
```

This makes the optimizer filter for traces where the model both got the answer right and showed its work. The result is more robust few-shot demonstrations.

### Before-and-after comparison

A common workflow for measuring the impact of optimization:

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_table=5)

# Baseline
baseline_score = evaluator(my_program)

# Optimize
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized = optimizer.compile(my_program, trainset=trainset)

# Compare
optimized_score = evaluator(optimized)
print(f"Baseline:  {baseline_score:.1f}%")
print(f"Optimized: {optimized_score:.1f}%")
print(f"Delta:     {optimized_score - baseline_score:+.1f}%")
```

## Gotchas

1. **Metrics must return a `float` or `bool`, not a string** -- returning a string silently breaks scoring.
2. **Use the `trace` parameter to differentiate optimization-time vs evaluation-time behavior** -- during optimization, `trace` is not `None`, so you can require stricter criteria (e.g., good reasoning + correct answer) for selecting few-shot demos.
3. **Small dev sets (<30 examples) give unreliable scores** -- results can swing 10-20% between runs. Aim for 50+ examples for stable evaluation.
4. **`SemanticF1` and `CompleteAndGrounded` call the LM** -- they're slower and cost money, but much better than exact match for open-ended tasks. Budget for the extra API calls.

## Cross-references

- Need to prepare training and evaluation data? Use `/dspy-data`
- Ready to optimize with few-shot examples? Use `/dspy-bootstrap-few-shot`
- Want the best prompt optimization? Use `/dspy-miprov2`
- For the full measure-improve-verify loop, see `/ai-improving-accuracy`
- For **decomposed RAG evaluation** (faithfulness, context precision/recall) see `/dspy-ragas`
- For worked examples (exact match, LM judge, composite), see [examples.md](examples.md)
