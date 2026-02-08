---
name: ai-improving-accuracy
description: Measure and improve how well your AI works. Use when AI gives wrong answers, accuracy is bad, responses are unreliable, you need to test AI quality, evaluate your AI, write metrics, benchmark performance, optimize prompts, improve results, or systematically make your AI better. Covers DSPy evaluation, metrics, and optimization.
---

# Measure and Improve Your AI

Guide the user through measuring how well their AI works, then systematically improving it. This is a loop: define "good" -> measure -> improve -> verify.

## The Workflow

1. **Define what "good" means** — write a metric
2. **Measure current quality** — run an evaluation
3. **Improve** — choose an optimizer, run it
4. **Verify** — re-evaluate to confirm improvement
5. **Iterate or ship**

## Step 1: Define what "good" means (write a metric)

A metric takes an expected answer and the AI's answer, and returns a score.

### Exact match (simplest)

```python
def metric(example, prediction, trace=None):
    return prediction.answer == example.answer
```

### Normalized match (handles capitalization/whitespace)

```python
def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()
```

### Partial credit (for multi-field outputs)

```python
def metric(example, prediction, trace=None):
    fields = ["name", "email", "phone"]
    correct = sum(
        1 for f in fields
        if getattr(prediction, f, "").lower() == getattr(example, f, "").lower()
    )
    return correct / len(fields)
```

### F1 score (for text overlap)

```python
def metric(example, prediction, trace=None):
    gold_tokens = set(example.answer.lower().split())
    pred_tokens = set(prediction.answer.lower().split())
    if not gold_tokens or not pred_tokens:
        return float(gold_tokens == pred_tokens)
    precision = len(gold_tokens & pred_tokens) / len(pred_tokens)
    recall = len(gold_tokens & pred_tokens) / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
```

### AI-as-judge (for open-ended tasks)

When exact match is too strict (summaries, creative tasks, open-ended Q&A):

```python
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

### Composite metric (multiple criteria)

```python
def metric(example, prediction, trace=None):
    correct = float(prediction.answer.lower() == example.answer.lower())
    concise = float(len(prediction.answer.split()) < 50)
    has_reasoning = float(len(getattr(prediction, 'reasoning', '')) > 20)
    return 0.7 * correct + 0.2 * concise + 0.1 * has_reasoning
```

### Training-aware metric

The `trace` parameter is not `None` during optimization. Use it for stricter requirements during training:

```python
def metric(example, prediction, trace=None):
    correct = prediction.answer == example.answer
    if trace is not None:
        # During optimization, also require good reasoning
        has_reasoning = len(prediction.reasoning) > 50
        return correct and has_reasoning
    return correct
```

## Step 2: Measure current quality (run evaluation)

### Prepare test data

If you don't have enough examples, use `/ai-generating-data` to generate synthetic training data.

```python
import dspy

# Manual creation
devset = [
    dspy.Example(question="What is DSPy?", answer="A framework for LM programs").with_inputs("question"),
    # 20-100+ examples for reliable evaluation
]

# From CSV/JSON
import json
with open("test_data.json") as f:
    data = json.load(f)
devset = [dspy.Example(**x).with_inputs("question") for x in data]

# From HuggingFace
from datasets import load_dataset
dataset = load_dataset("squad", split="validation[:100]")
devset = [
    dspy.Example(question=x["question"], answer=x["answers"]["text"][0]).with_inputs("question")
    for x in dataset
]
```

### Run evaluation

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=devset,
    metric=metric,
    num_threads=4,
    display_progress=True,
    display_table=5,   # show 5 example results
)

baseline_score = evaluator(my_program)
print(f"Baseline: {baseline_score}")
```

## Step 3: Improve (choose an optimizer)

### Quick guide: which optimizer?

| Training examples | Recommended optimizer | Expected improvement | Typical cost |
|------------------|-----------------------|---------------------|-------------|
| <20 | GEPA (instruction tuning) | 5-15% | ~$0.50 |
| 20-50 | BootstrapFewShot | 5-20% | ~$0.50-2 |
| 50-200 | BootstrapFewShot, then MIPROv2 | 15-35% | ~$2-10 |
| 200-500 | MIPROv2 (auto="medium") | 20-40% | ~$5-15 |
| 500+ | MIPROv2 (auto="heavy") or BootstrapFinetune | 25-50% | ~$15-50+ |

```
Start here
|
+- Just getting started (<50 examples)? -> BootstrapFewShot
|   Quick, cheap, usually gives a solid boost.
|
+- Want better prompts (50+ examples)? -> MIPROv2
|   Optimizes both instructions and examples.
|   Best general-purpose prompt optimizer.
|
+- Want to tune instructions only (<50 examples)? -> GEPA
|   Good when you have few examples.
|
+- Need maximum quality (500+ examples)? -> BootstrapFinetune
|   Fine-tunes the model weights.
|   Best for production with smaller/cheaper models.
|
+- Want to combine approaches? -> BetterTogether
    Jointly optimizes prompts and weights.
```

**Stacking tip:** Run BootstrapFewShot first, then MIPROv2 on the result. This often beats either alone — bootstrap finds good examples, then MIPRO refines the instructions.

Optimized prompts are model-specific. If you change models, re-run your optimizer. See `/ai-switching-models`.

### BootstrapFewShot (start here)

Fast, cheap. Finds good examples by running your program and keeping successful traces.

```python
optimizer = dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
optimized = optimizer.compile(my_program, trainset=trainset)
```

**Cost:** Minimal (one pass through trainset). **Expected improvement:** 5-20%.

### MIPROv2 (recommended for most cases)

Optimizes both instructions and examples. Best general-purpose optimizer.

```python
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="medium",    # "light", "medium", "heavy"
)
optimized = optimizer.compile(my_program, trainset=trainset)
```

- `"light"`: Quick, ~$1-2
- `"medium"`: Balanced, ~$5-10
- `"heavy"`: Thorough, ~$15-30

**Expected improvement:** 15-35%.

### GEPA (instruction tuning)

Good with few examples or when you want to focus on instruction quality:

```python
optimizer = dspy.GEPA()
optimized = optimizer.compile(my_program, trainset=trainset, metric=metric)
```

### BootstrapFinetune (maximum quality)

Fine-tunes model weights for the biggest accuracy gains. Requires 500+ examples and a fine-tunable model:

```python
optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
optimized = optimizer.compile(my_program, trainset=trainset)
```

For the full fine-tuning workflow (decision framework, prerequisites, model distillation, BetterTogether), see `/ai-fine-tuning`.

### When optimization plateaus

If your score stops improving, check these common causes:

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Score stuck at 60-70% despite optimization | Task too complex for single step | `/ai-decomposing-tasks` — break into subtasks |
| Optimizer overfits (train score high, dev score flat) | Too little training data | `/ai-generating-data` — generate more examples |
| Score varies wildly between runs | Non-deterministic metric or small devset | Increase devset to 100+, set temperature=0 |
| Diminishing returns from more demos | Prompt is maxed out; model is the limit | `/ai-switching-models` — try a more capable model |
| Score high but real users complain | Metric doesn't match real quality | Rewrite metric based on actual failure patterns |

## Step 4: Verify improvement

```python
optimized_score = evaluator(optimized)
print(f"Baseline: {baseline_score:.1f}%")
print(f"Optimized: {optimized_score:.1f}%")
print(f"Improvement: {optimized_score - baseline_score:.1f}%")
```

## Step 5: Save and ship

```python
optimized.save("optimized_program.json")

# Load later
my_program = MyProgram()
my_program.load("optimized_program.json")
```

## Key patterns

- **Start simple**: exact match metric + BootstrapFewShot, then upgrade if needed
- **Validate your metric**: manually check 10-20 examples to make sure the metric scores correctly
- **More data helps**: optimizers work better with more training examples
- **Never evaluate on trainset**: always use a held-out devset
- **Use `display_table`**: looking at actual predictions reveals metric bugs
- **Iterate**: run optimization, check results, improve metric, re-optimize

## Additional resources

- For optimizer comparison table and metric patterns, see [reference.md](reference.md)
- Once quality is good, use `/ai-cutting-costs` to reduce your AI bill
- Use `/ai-monitoring` to track quality in production after deployment
- Use `/ai-tracking-experiments` to log, compare, and manage multiple optimization runs
- Accuracy plateaued despite optimization? Try `/ai-decomposing-tasks` to restructure your task
- If things are broken, use `/ai-fixing-errors` to diagnose
