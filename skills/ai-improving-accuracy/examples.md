# Accuracy Improvement Examples

## Metric Patterns

### Exact match

```python
def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()
```

### Token-level F1 (partial credit)

```python
def metric(example, prediction, trace=None):
    gold = set(example.answer.lower().split())
    pred = set(prediction.answer.lower().split())
    if not gold or not pred:
        return float(gold == pred)
    precision = len(gold & pred) / len(pred)
    recall = len(gold & pred) / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
```

F1 is useful for open-ended answers where word overlap matters — summaries, entity extraction, or tasks where partial credit is meaningful.

### AI-as-judge (for open-ended tasks)

```python
import dspy

class AssessQuality(dspy.Signature):
    """Assess if the predicted answer is correct and complete."""
    question: str = dspy.InputField()
    gold_answer: str = dspy.InputField()
    predicted_answer: str = dspy.InputField()
    is_correct: bool = dspy.OutputField()

def metric(example, prediction, trace=None):
    judge = dspy.Predict(AssessQuality)
    result = judge(
        question=example.question,
        gold_answer=example.answer,
        predicted_answer=prediction.answer,
    )
    return result.is_correct
```

Use AI-as-judge for final validation only — it's slow and expensive during optimization. Prefer exact match or F1 during `compile()`, then validate the final result with AI-as-judge afterward.

### Training-aware metric

The `trace` parameter is `None` during evaluation but set during optimization. Use it to enforce stricter requirements during training:

```python
def metric(example, prediction, trace=None):
    correct = prediction.answer.strip().lower() == example.answer.strip().lower()
    if trace is not None:
        # During optimization, also require substantive reasoning
        has_reasoning = len(prediction.reasoning) > 50
        return correct and has_reasoning
    return correct
```

## Measuring a Baseline

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")

devset = [
    dspy.Example(question="What is the return policy?", answer="30 days").with_inputs("question"),
    dspy.Example(question="Do you ship internationally?", answer="Yes, to 50+ countries").with_inputs("question"),
    # 50-200+ examples for reliable measurement
]

def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True, display_table=5)
baseline = evaluator(qa)
print(f"Baseline: {baseline:.1f}%")  # e.g. Baseline: 61.0%
```

`display_table=5` prints 5 individual predictions so you can spot metric bugs before trusting the aggregate score. A 75% score looks fine until you inspect the table and notice the metric is giving credit for empty answers that happen to match whitespace.

## Optimizing with BootstrapFewShot

Start here — it is fast and often gives a meaningful lift without changing any instructions:

```python
import random

random.seed(42)
random.shuffle(examples)
split = int(0.8 * len(examples))
trainset, devset = examples[:split], examples[split:]

optimizer = dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,   # default is 16 — lower for small datasets
)
optimized = optimizer.compile(qa, trainset=trainset)

improved = evaluator(optimized)
print(f"Baseline:             {baseline:.1f}%")   # 61.0%
print(f"After BootstrapFewShot: {improved:.1f}%") # 76.0%
```

If accuracy has plateaued or you have 200+ examples, upgrade to MIPROv2.

## Upgrading to MIPROv2

MIPROv2 optimizes both instructions and few-shot examples using Bayesian search — it typically outperforms BootstrapFewShot at the cost of more LM calls:

```python
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="medium",   # "light" for quick iteration, "heavy" for best quality
)
optimized_v2 = optimizer.compile(qa, trainset=trainset)

final = evaluator(optimized_v2)
print(f"After MIPROv2: {final:.1f}%")   # 87.0%
```

**Stacking tip:** Compile with BootstrapFewShot first, then pass the result into MIPROv2. Bootstrap finds good demonstrations; MIPROv2 then refines the instructions around them.

## End-to-End: Support Q&A from 61% to 87%

A realistic workflow — data from JSON, baseline evaluation, BootstrapFewShot, then MIPROv2 stacked on top:

```python
import dspy
import json
import random
from dspy.evaluate import Evaluate

# 1. Configure
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# 2. Load and split data — never use the same data for training and evaluation
with open("faq_labeled.json") as f:
    data = json.load(f)   # [{question: ..., answer: ...}, ...]

examples = [
    dspy.Example(question=item["question"], answer=item["answer"]).with_inputs("question")
    for item in data
]

random.seed(42)
random.shuffle(examples)
split = int(0.8 * len(examples))
trainset, devset = examples[:split], examples[split:]

# 3. Define program and metric
class SupportQA(dspy.Signature):
    """Answer the customer support question concisely."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

qa = dspy.ChainOfThought(SupportQA)

def metric(example, prediction, trace=None):
    gold = set(example.answer.lower().split())
    pred = set(prediction.answer.lower().split())
    if not gold or not pred:
        return float(gold == pred)
    precision = len(gold & pred) / len(pred)
    recall = len(gold & pred) / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

# 4. Measure baseline — inspect display_table before trusting the score
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True, display_table=5)
baseline = evaluator(qa)
print(f"Baseline: {baseline:.1f}%")   # ~61%

# 5. BootstrapFewShot (fast first pass)
opt1 = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=4)
step1 = opt1.compile(qa, trainset=trainset)
score1 = evaluator(step1)
print(f"After BootstrapFewShot: {score1:.1f}%")   # ~76%

# 6. MIPROv2 stacked on the BootstrapFewShot result
opt2 = dspy.MIPROv2(metric=metric, auto="medium")
optimized = opt2.compile(step1, trainset=trainset)
final = evaluator(optimized)
print(f"After MIPROv2:          {final:.1f}%")   # ~87%

print(f"Total improvement: +{final - baseline:.1f}pp")   # +26pp

# 7. Save optimized state — load it back without re-running the optimizer
optimized.save("support_qa_optimized.json")

# Load later:
# qa = dspy.ChainOfThought(SupportQA)
# qa.load("support_qa_optimized.json")
```

Expected improvement from this stacked approach: 20-30 percentage points over an unoptimized baseline on most QA tasks. Actual gains depend on label quality, devset size, and task complexity. If gains are smaller than expected, check the troubleshooting table in `SKILL.md` — the most common cause is a metric that doesn't accurately reflect real quality.
