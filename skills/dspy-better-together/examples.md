# BetterTogether Examples

Worked examples showing how to use `dspy.BetterTogether` for joint prompt and weight optimization.

## Example 1: Combined prompt + weight optimization

Full workflow for a classification task, comparing BetterTogether against prompt-only and fine-tune-only approaches.

### Setup and data

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class ClassifyIntent(dspy.Signature):
    """Classify the user message into an intent category."""
    message: str = dspy.InputField()
    intent: str = dspy.OutputField(
        desc="one of: purchase, refund, support, feedback, account, other"
    )

program = dspy.ChainOfThought(ClassifyIntent)

# IMPORTANT: Assign LM explicitly for BetterTogether
program.set_lm(lm)

# Load labeled data (1000+ examples)
import json
with open("intents.json") as f:
    data = json.load(f)

examples = [
    dspy.Example(message=x["message"], intent=x["intent"]).with_inputs("message")
    for x in data
]

# Split 80/10/10
trainset = examples[:800]
devset = examples[800:900]
testset = examples[900:]

def metric(example, prediction, trace=None):
    return prediction.intent.strip().lower() == example.intent.strip().lower()

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
```

### Step 1: Measure baselines individually

```python
# Baseline (no optimization)
baseline_score = evaluator(program)
print(f"Baseline: {baseline_score:.1f}%")

# Prompt optimization only
prompt_opt = dspy.MIPROv2(metric=metric, auto="medium")
prompt_optimized = prompt_opt.compile(program, trainset=trainset)
prompt_score = evaluator(prompt_optimized)
print(f"Prompt-only (MIPROv2): {prompt_score:.1f}%")

# Fine-tuning only
ft_opt = dspy.BootstrapFinetune(metric=metric, num_threads=24)
finetuned = ft_opt.compile(program, trainset=trainset)
ft_score = evaluator(finetuned)
print(f"Fine-tune-only: {ft_score:.1f}%")
```

### Step 2: Run BetterTogether

```python
optimizer = dspy.BetterTogether(metric=metric)

compiled = optimizer.compile(
    program,
    trainset=trainset,
    valset=devset,
    strategy="p -> w -> p",
)

bt_score = evaluator(compiled)
print(f"BetterTogether: {bt_score:.1f}%")
```

### Step 3: Compare results

```python
test_eval = Evaluate(devset=testset, metric=metric, num_threads=4, display_progress=True)

print("Test set results:")
print(f"  Baseline:         {test_eval(program):.1f}%")
print(f"  Prompt-only:      {test_eval(prompt_optimized):.1f}%")
print(f"  Fine-tune-only:   {test_eval(finetuned):.1f}%")
print(f"  BetterTogether:   {test_eval(compiled):.1f}%")

# Inspect per-step scores
for candidate in compiled.candidate_programs:
    print(f"  Step '{candidate['strategy']}': {candidate['score']:.1f}%")
```

### Expected results

| Approach | Dev accuracy | Notes |
|----------|-------------|-------|
| Baseline | ~68% | No optimization |
| Prompt-only (MIPROv2) | ~82% | +14 pts |
| Fine-tune-only | ~85% | +17 pts |
| BetterTogether (p -> w -> p) | ~91% | +23 pts |

BetterTogether gets +6 pts beyond the best individual approach because the prompt and weight optimization rounds compound on each other.

### Save for production

```python
compiled.save("intent_classifier_bt.json")

# Load later
from my_module import build_program
production = build_program()
production.load("intent_classifier_bt.json")
result = production(message="I want my money back")
print(result.intent)  # "refund"
```

---

## Example 2: Two-phase optimization strategy with custom optimizers

Use GEPA for instruction tuning and BootstrapFinetune for weight optimization in a simpler two-phase strategy. This is cheaper than the default three-phase strategy and works well when you want faster iteration.

### Setup

```python
import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import GEPA, BootstrapFinetune

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class SummarizeReview(dspy.Signature):
    """Summarize the product review into a short, factual summary."""
    review: str = dspy.InputField()
    summary: str = dspy.OutputField(desc="1-2 sentence factual summary")
    sentiment: str = dspy.OutputField(desc="one of: positive, negative, mixed")

program = dspy.ChainOfThought(SummarizeReview)
program.set_lm(lm)

# Load data
import json
with open("reviews.json") as f:
    data = json.load(f)

examples = [
    dspy.Example(
        review=x["review"],
        summary=x["summary"],
        sentiment=x["sentiment"],
    ).with_inputs("review")
    for x in data
]

trainset = examples[:800]
valset = examples[800:900]
testset = examples[900:]

# Composite metric: sentiment accuracy + summary quality
class AssessSummary(dspy.Signature):
    """Assess if the summary accurately captures the review."""
    review: str = dspy.InputField()
    gold_summary: str = dspy.InputField()
    predicted_summary: str = dspy.InputField()
    is_accurate: bool = dspy.OutputField()

def metric(example, prediction, trace=None):
    # Sentiment must be exact match
    sentiment_correct = prediction.sentiment.strip().lower() == example.sentiment.strip().lower()

    # Summary quality via LM judge
    judge = dspy.Predict(AssessSummary)
    assessment = judge(
        review=example.review,
        gold_summary=example.summary,
        predicted_summary=prediction.summary,
    )
    summary_good = float(assessment.is_accurate)

    return 0.5 * float(sentiment_correct) + 0.5 * summary_good

evaluator = Evaluate(devset=valset, metric=metric, num_threads=4, display_progress=True)
```

### Run two-phase BetterTogether

```python
# Use GEPA for instruction tuning (good with composite metrics)
# and BootstrapFinetune for weight optimization
optimizer = dspy.BetterTogether(
    metric=metric,
    p=GEPA(metric=metric, auto="medium"),
    w=BootstrapFinetune(metric=metric),
)

compiled = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset,
    strategy="p -> w",  # Two phases only -- cheaper and faster
)

score = evaluator(compiled)
print(f"BetterTogether (p -> w): {score:.1f}%")
```

### Pass custom arguments to individual optimizers

Use `optimizer_compile_args` to control each optimizer's behavior independently:

```python
optimizer = dspy.BetterTogether(
    metric=metric,
    p=GEPA(metric=metric, auto="medium"),
    w=BootstrapFinetune(metric=metric),
)

compiled = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset,
    strategy="p -> w",
    optimizer_compile_args={
        "p": {"num_threads": 8},
        "w": {"num_threads": 24},
    },
)
```

### Compare two-phase vs three-phase

```python
# Two-phase: cheaper, faster
two_phase = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset,
    strategy="p -> w",
)
two_phase_score = evaluator(two_phase)

# Three-phase: potentially better quality
three_phase = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset,
    strategy="p -> w -> p",
)
three_phase_score = evaluator(three_phase)

print(f"Two-phase (p -> w):     {two_phase_score:.1f}%")
print(f"Three-phase (p -> w -> p): {three_phase_score:.1f}%")
```

### Expected results

| Strategy | Quality | Cost | Time |
|----------|---------|------|------|
| `"p -> w"` | ~87% | Lower | Faster |
| `"p -> w -> p"` | ~91% | Higher | Slower |

The third phase (re-optimizing prompts) typically adds 2-5 percentage points. Whether the extra cost is worth it depends on your quality requirements.

### When to choose two-phase

- You're iterating quickly and want faster feedback
- The quality gap between two-phase and three-phase is small for your task
- You want to save on compute costs
- You plan to run BetterTogether multiple times with different configurations
