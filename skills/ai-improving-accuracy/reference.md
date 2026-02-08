# Accuracy Improvement Reference

## Optimizer Comparison Table

| Optimizer | Tunes | Min Data | Cost | Speed | Best For |
|-----------|-------|----------|------|-------|----------|
| `BootstrapFewShot` | Few-shot examples | 20 | $ | Fast | First attempt |
| `BootstrapFewShotWithRandomSearch` | Few-shot examples | 50 | $$ | Medium | Better few-shot |
| `MIPROv2` | Instructions + few-shot | 100 | $$-$$$ | Medium | Best prompt optimization |
| `GEPA` | Instructions | 20 | $$ | Medium | Instruction tuning |
| `BootstrapFinetune` | LM weights | 500+ | $$$$ | Slow | Maximum quality |
| `BetterTogether` | Instructions + weights | 500+ | $$$$$ | Slow | Combined optimization |

## Choosing by data size

| Data Size | Recommended Optimizer |
|-----------|----------------------|
| 10-20 examples | BootstrapFewShot or GEPA |
| 50-200 examples | MIPROv2 (auto="light" or "medium") |
| 200-500 examples | MIPROv2 (auto="medium" or "heavy") |
| 500+ examples | BootstrapFinetune or BetterTogether |

## Optimizer Details

### BootstrapFewShot

How it works:
1. Runs your program on training examples
2. Keeps traces where the metric scored high
3. Uses those traces as few-shot examples in prompts

```python
dspy.BootstrapFewShot(
    metric=metric,
    max_bootstrapped_demos=4,         # generated few-shot examples
    max_labeled_demos=4,              # labeled examples from trainset
    max_rounds=1,                     # bootstrapping rounds
)
```

### BootstrapFewShotWithRandomSearch

Same as BootstrapFewShot but tries multiple random configurations:

```python
dspy.BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=4,
    num_candidate_programs=8,         # configurations to try
    max_labeled_demos=4,
)
```

### MIPROv2

How it works:
1. Generates candidate instructions using an LM
2. Generates candidate few-shot examples
3. Uses Bayesian optimization to find the best combination

```python
dspy.MIPROv2(
    metric=metric,
    auto="medium",                    # "light" | "medium" | "heavy"
    # Or manual control:
    # num_candidates=10,
    # init_temperature=0.7,
    # num_trials=30,
)
```

Auto settings:
- `"light"`: ~10 trials, good for quick iteration
- `"medium"`: ~30 trials, balanced quality vs cost
- `"heavy"`: ~100 trials, best quality

### GEPA

Generates, evaluates, and proposes alternative instructions using an evolutionary approach.

```python
dspy.GEPA()
# Usage: optimizer.compile(program, trainset=trainset, metric=metric)
```

### BootstrapFinetune

How it works:
1. Bootstraps training data from successful traces
2. Fine-tunes the LM on this data
3. Returns a program using the fine-tuned model

```python
dspy.BootstrapFinetune(
    metric=metric,
    num_threads=24,
)
```

Requirements:
- Sufficient training data (500+)
- A fine-tunable model (OpenAI GPT models, or open-source via Together/Anyscale)
- Budget for fine-tuning API costs

### BetterTogether

Jointly optimizes prompts and fine-tuned weights.

```python
dspy.BetterTogether(metric=metric)
optimized = optimizer.compile(program, trainset=trainset)
```

### Composing optimizers

You can chain optimizers:

```python
# First, optimize instructions with GEPA
opt1 = dspy.GEPA()
step1 = opt1.compile(program, trainset=trainset, metric=metric)

# Then, optimize few-shot examples
opt2 = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
step2 = opt2.compile(step1, trainset=trainset)
```

## Metric Patterns

### Classification metrics

```python
# Accuracy
def accuracy(example, pred, trace=None):
    return pred.label == example.label

# Multi-label F1
def multilabel_f1(example, pred, trace=None):
    gold = set(example.labels)
    predicted = set(pred.labels)
    if not gold and not predicted:
        return 1.0
    tp = len(gold & predicted)
    precision = tp / len(predicted) if predicted else 0
    recall = tp / len(gold) if gold else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
```

### Extraction metrics

```python
# Field-level accuracy
def field_accuracy(example, pred, trace=None):
    fields = ["name", "date", "amount"]
    scores = []
    for f in fields:
        expected = getattr(example, f, None)
        predicted = getattr(pred, f, None)
        if expected is not None:
            scores.append(
                float(str(predicted).strip().lower() == str(expected).strip().lower())
            )
    return sum(scores) / len(scores) if scores else 0.0

# Pydantic model comparison
def model_accuracy(example, pred, trace=None):
    gold = example.extracted
    predicted = pred.extracted
    if type(gold) != type(predicted):
        return 0.0
    fields = gold.model_fields.keys()
    correct = sum(1 for f in fields if getattr(gold, f) == getattr(predicted, f))
    return correct / len(fields)
```

### AI-as-judge variants

```python
# Binary judge
class IsCorrect(dspy.Signature):
    """Is the answer correct?"""
    question: str = dspy.InputField()
    expected: str = dspy.InputField()
    actual: str = dspy.InputField()
    correct: bool = dspy.OutputField()

# Graded judge (0-5 scale)
class GradeAnswer(dspy.Signature):
    """Grade the answer quality on a 0-5 scale."""
    question: str = dspy.InputField()
    expected: str = dspy.InputField()
    actual: str = dspy.InputField()
    grade: int = dspy.OutputField(desc="Score from 0 (wrong) to 5 (perfect)")

def graded_metric(example, pred, trace=None):
    judge = dspy.Predict(GradeAnswer)
    result = judge(question=example.question, expected=example.answer, actual=pred.answer)
    return result.grade / 5.0

# Multi-criteria judge
class AssessMulti(dspy.Signature):
    """Assess the answer on multiple criteria."""
    question: str = dspy.InputField()
    expected: str = dspy.InputField()
    actual: str = dspy.InputField()
    factually_correct: bool = dspy.OutputField()
    well_structured: bool = dspy.OutputField()
    concise: bool = dspy.OutputField()

def multi_criteria_metric(example, pred, trace=None):
    judge = dspy.Predict(AssessMulti)
    result = judge(question=example.question, expected=example.answer, actual=pred.answer)
    return (result.factually_correct * 0.6 + result.well_structured * 0.2 + result.concise * 0.2)
```

## Data Loading Patterns

```python
# From CSV
import csv
with open("data.csv") as f:
    reader = csv.DictReader(f)
    examples = [dspy.Example(**row).with_inputs("question") for row in reader]

# From JSON
import json
with open("data.json") as f:
    data = json.load(f)
examples = [dspy.Example(**item).with_inputs("input_field") for item in data]

# From HuggingFace
from datasets import load_dataset
ds = load_dataset("squad", split="validation[:200]")
examples = [
    dspy.Example(question=x["question"], answer=x["answers"]["text"][0]).with_inputs("question")
    for x in ds
]

# Train/dev split
import random
random.seed(42)
random.shuffle(examples)
split = int(0.8 * len(examples))
trainset, devset = examples[:split], examples[split:]
```

## Evaluate class options

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(
    devset=devset,           # evaluation dataset
    metric=metric,           # scoring function
    num_threads=4,           # parallel threads
    display_progress=True,   # progress bar
    display_table=5,         # show N example results in table
    max_errors=5,            # stop after N errors
)
```

## Troubleshooting

**Optimization doesn't improve score:**
- Check your metric — is it measuring the right thing?
- Check your data — are labels correct?
- Try a different optimizer
- Add more training data

**Optimization is too expensive:**
- Use `auto="light"` with MIPROv2
- Start with BootstrapFewShot
- Use a cheaper LM for optimization, then transfer to the target LM

**Optimized program is worse than baseline:**
- Overfitting — reduce `max_bootstrapped_demos`
- Bad metric — validate metric scores manually
- Use a validation set to check for overfitting

## Evaluation Best Practices

1. **Minimum devset size**: 50 examples for rough estimates, 200+ for reliable comparisons
2. **Statistical significance**: A 2% improvement on 50 examples might be noise; on 500 it's likely real
3. **Error analysis**: Look at failures, not just the score — they reveal what to fix
4. **Metric validation**: Manually score 20 examples and compare to your metric
5. **Version tracking**: Log scores with timestamps, model versions, and optimizer settings
