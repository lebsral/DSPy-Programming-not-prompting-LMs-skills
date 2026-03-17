---
name: dspy-evaluate
description: "Use DSPy's Evaluate class and built-in metrics to measure AI quality. Use when you want to run dspy.Evaluate, write custom metrics, use SemanticF1 or answer_exact_match, build an LM-as-judge, score predictions against a devset, or measure accuracy before and after optimization."
---

# Evaluate Your DSPy Program

Guide the user through measuring AI quality with DSPy's `Evaluate` class. The pattern: pick a metric, prepare a devset, run the evaluator, interpret results, then feed the same metric into an optimizer.

## What is dspy.Evaluate

`dspy.Evaluate` runs your program on every example in a devset, scores each prediction with a metric function, and reports the aggregate score. It handles threading, progress display, and result tables so you don't have to write evaluation loops by hand.

The evaluator returns a percentage score (0-100) representing how many examples your program got right (or the average metric score if your metric returns floats).

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4)
score = evaluator(my_program)
print(f"Score: {score}%")
```

## Basic usage

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Your program
qa = dspy.ChainOfThought("question -> answer")

# 2. Your devset (examples with expected outputs)
devset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    # 20-100+ examples for reliable evaluation
]

# 3. Your metric
def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# 4. Run evaluation
evaluator = Evaluate(
    devset=devset,
    metric=metric,
    num_threads=4,
    display_progress=True,
    display_table=5,
)
score = evaluator(qa)
print(f"Score: {score}%")
```

## Built-in metrics

DSPy provides several metrics so you don't have to write common ones from scratch.

### answer_exact_match

Returns `True` if the predicted answer exactly matches the expected answer (after normalization):

```python
from dspy.evaluate import answer_exact_match

evaluator = Evaluate(devset=devset, metric=answer_exact_match, num_threads=4)
score = evaluator(my_program)
```

`answer_exact_match` expects the example and prediction to have an `answer` field. It normalizes whitespace and capitalization before comparing.

### answer_passage_match

Returns `True` if the expected answer appears anywhere in the predicted answer:

```python
from dspy.evaluate import answer_passage_match

evaluator = Evaluate(devset=devset, metric=answer_passage_match, num_threads=4)
score = evaluator(my_program)
```

Useful when your program returns a full sentence but the gold answer is a short phrase (e.g., "Paris" should match "The capital of France is Paris.").

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

A metric is a function with this signature:

```python
def metric(example, prediction, trace=None):
    # example: the dspy.Example from your devset (has both inputs and expected outputs)
    # prediction: the dspy.Prediction from your program (has only outputs)
    # trace: None during evaluation, not None during optimization
    # Return: bool, int, or float
    return prediction.answer == example.answer
```

Return values:
- **bool** — `True`/`False` for pass/fail (evaluator reports % that pass)
- **int** — 0 or 1 (same as bool)
- **float** — 0.0 to 1.0 for partial credit (evaluator reports average)

### Normalized match

```python
def metric(example, prediction, trace=None):
    pred = prediction.answer.strip().lower()
    gold = example.answer.strip().lower()
    return pred == gold
```

### Substring match

```python
def metric(example, prediction, trace=None):
    return example.answer.lower() in prediction.answer.lower()
```

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

## Evaluate options

```python
evaluator = Evaluate(
    devset=devset,           # list of dspy.Example — required
    metric=metric,           # metric function — required
    num_threads=4,           # parallel threads for faster evaluation
    display_progress=True,   # show progress bar
    display_table=5,         # show a table of the first N results
    return_all_scores=True,  # return per-example scores (not just aggregate)
)

score = evaluator(my_program)
```

### Getting per-example scores

When `return_all_scores=True`, the evaluator returns a tuple:

```python
evaluator = Evaluate(
    devset=devset,
    metric=metric,
    num_threads=4,
    return_all_scores=True,
)

aggregate_score, all_scores = evaluator(my_program)
print(f"Aggregate: {aggregate_score}%")

# Find failing examples
for i, (example, score) in enumerate(zip(devset, all_scores)):
    if score < 0.5:
        print(f"FAIL [{i}]: {example.question} (score={score})")
```

This is useful for debugging — look at the examples your program gets wrong to understand failure patterns.

## Using with optimizers

The same metric function you use for evaluation is passed to optimizers. This keeps your definition of "good" consistent across measurement and improvement:

```python
from dspy.evaluate import Evaluate

# Define metric once
def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# Use for evaluation
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4)
baseline = evaluator(my_program)

# Use the same metric for optimization
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(my_program, trainset=trainset)

# Re-evaluate to measure improvement
improved = evaluator(optimized)
print(f"Baseline: {baseline:.1f}% -> Optimized: {improved:.1f}%")
```

Never evaluate on your training set — always use a held-out devset. A typical split is 80% train, 20% dev.

## Common patterns

### Partial credit scoring

Give partial credit instead of all-or-nothing:

```python
def partial_credit(example, prediction, trace=None):
    gold_tokens = set(example.answer.lower().split())
    pred_tokens = set(prediction.answer.lower().split())
    if not gold_tokens:
        return float(not pred_tokens)
    overlap = gold_tokens & pred_tokens
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(overlap) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### Penalizing bad behavior

Deduct points for undesirable outputs:

```python
def metric_with_penalties(example, prediction, trace=None):
    score = float(prediction.answer.strip().lower() == example.answer.strip().lower())

    # Penalize overly long answers
    if len(prediction.answer.split()) > 100:
        score -= 0.2

    # Penalize hedging language
    hedges = ["I think", "maybe", "probably", "I'm not sure"]
    if any(h.lower() in prediction.answer.lower() for h in hedges):
        score -= 0.1

    return max(0.0, score)
```

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

## Cross-references

- Need to prepare training and evaluation data? Use `/dspy-data`
- Ready to optimize with few-shot examples? Use `/dspy-bootstrap-few-shot`
- Want the best prompt optimization? Use `/dspy-miprov2`
- For the full measure-improve-verify loop, see `/ai-improving-accuracy`
- For worked examples (exact match, LM judge, composite), see [examples.md](examples.md)
