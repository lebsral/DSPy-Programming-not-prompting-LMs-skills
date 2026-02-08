# Fine-Tuning Examples

Worked examples showing the full fine-tuning workflow for different use cases.

## Example 1: Classification fine-tuning (ticket sorting)

Train a small model to sort support tickets as well as an expensive model.

### Setup and data

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class ClassifyTicket(dspy.Signature):
    """Classify the support ticket into a category."""
    text: str = dspy.InputField()
    category: str = dspy.OutputField(
        desc="one of: billing, technical, account, feature_request, other"
    )

program = dspy.ChainOfThought(ClassifyTicket)

# Load labeled data (1000 tickets)
import json
with open("tickets.json") as f:
    data = json.load(f)

examples = [
    dspy.Example(text=x["text"], category=x["category"]).with_inputs("text")
    for x in data
]

# Split 80/10/10
trainset = examples[:800]
devset = examples[800:900]
testset = examples[900:]

def metric(example, prediction, trace=None):
    return prediction.category.strip().lower() == example.category.strip().lower()

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
```

### Step 1: Measure baseline

```python
baseline_score = evaluator(program)
print(f"Baseline (GPT-4o-mini, no optimization): {baseline_score:.1f}%")
# Expected: ~65-75%
```

### Step 2: Optimize prompts (comparison point)

```python
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
prompt_optimized = optimizer.compile(program, trainset=trainset)
prompt_score = evaluator(prompt_optimized)
print(f"Prompt-optimized: {prompt_score:.1f}%")
# Expected: ~80-88%
```

### Step 3: Fine-tune

```python
ft_optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
finetuned = ft_optimizer.compile(program, trainset=trainset)
finetuned_score = evaluator(finetuned)
print(f"Fine-tuned: {finetuned_score:.1f}%")
# Expected: ~88-95%
```

### Results comparison

```python
test_eval = Evaluate(devset=testset, metric=metric, num_threads=4, display_progress=True)
print(f"Test set results:")
print(f"  Baseline:         {test_eval(program):.1f}%")
print(f"  Prompt-optimized: {test_eval(prompt_optimized):.1f}%")
print(f"  Fine-tuned:       {test_eval(finetuned):.1f}%")
```

| Stage | Dev accuracy | Notes |
|-------|-------------|-------|
| Baseline | ~70% | No optimization |
| Prompt-optimized | ~84% | MIPROv2 medium |
| Fine-tuned | ~92% | BootstrapFinetune |

---

## Example 2: RAG distillation (GPT-4o to GPT-4o-mini)

Distill an expensive RAG pipeline into a cheap model that runs 15x cheaper.

### Build teacher with expensive model

```python
import dspy
from dspy.evaluate import Evaluate

teacher_lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=teacher_lm)

class AnswerFromDocs(dspy.Signature):
    """Answer the question based on the provided context."""
    context: str = dspy.InputField()
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class RAG(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought(AnswerFromDocs)

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

teacher = RAG()

# Load data
import json
with open("qa_data.json") as f:
    data = json.load(f)

examples = [
    dspy.Example(question=x["question"], answer=x["answer"]).with_inputs("question")
    for x in data
]

trainset = examples[:800]
devset = examples[800:900]
testset = examples[900:]

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

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)

# Optimize teacher
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
teacher_optimized = optimizer.compile(teacher, trainset=trainset)
teacher_score = evaluator(teacher_optimized)
print(f"Teacher (GPT-4o, optimized): {teacher_score:.1f}%")
# Expected: ~82%
```

### Distill to student

```python
# Switch to cheap model
student_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=student_lm)

student = RAG()
ft_optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
student_finetuned = ft_optimizer.compile(student, trainset=trainset, teacher=teacher_optimized)

student_score = evaluator(student_finetuned)
print(f"Student (GPT-4o-mini, fine-tuned): {student_score:.1f}%")
# Expected: ~76%
```

### Results

| Model | Quality (F1) | Cost per 1M tokens | Relative cost |
|-------|-------------|-------------------|---------------|
| GPT-4o (teacher) | ~82% | ~$5.00 | 1x |
| GPT-4o-mini (no tuning) | ~62% | ~$0.15 | 0.03x |
| GPT-4o-mini (fine-tuned) | ~76% | ~$0.15 | 0.03x |

93% quality retention at 33x lower cost.

---

## Example 3: BetterTogether for maximum quality

Use alternating prompt + weight optimization for a multi-step reasoning task.

### Setup

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class SolveWord(dspy.Signature):
    """Solve the math word problem step by step."""
    problem: str = dspy.InputField()
    answer: float = dspy.OutputField()

program = dspy.ChainOfThought(SolveWord)

# Load math problems
import json
with open("math_problems.json") as f:
    data = json.load(f)

examples = [
    dspy.Example(problem=x["problem"], answer=x["answer"]).with_inputs("problem")
    for x in data
]

trainset = examples[:800]
devset = examples[800:900]
testset = examples[900:]

def metric(example, prediction, trace=None):
    try:
        return abs(float(prediction.answer) - float(example.answer)) < 0.01
    except (ValueError, TypeError):
        return False

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
```

### Run all three approaches

```python
# 1. Baseline
baseline_score = evaluator(program)
print(f"Baseline: {baseline_score:.1f}%")

# 2. Prompt optimization only
prompt_opt = dspy.MIPROv2(metric=metric, auto="medium")
prompt_optimized = prompt_opt.compile(program, trainset=trainset)
prompt_score = evaluator(prompt_optimized)
print(f"Prompt-only: {prompt_score:.1f}%")

# 3. Fine-tuning only
ft_opt = dspy.BootstrapFinetune(metric=metric, num_threads=24)
finetuned = ft_opt.compile(program, trainset=trainset)
ft_score = evaluator(finetuned)
print(f"Fine-tune-only: {ft_score:.1f}%")

# 4. BetterTogether (prompt + weight optimization)
bt_opt = dspy.BetterTogether(
    metric=metric,
    prompt_optimizer=dspy.MIPROv2,
    weight_optimizer=dspy.BootstrapFinetune,
)
best = bt_opt.compile(program, trainset=trainset)
best_score = evaluator(best)
print(f"BetterTogether: {best_score:.1f}%")
```

### Results

| Approach | Dev accuracy | Notes |
|----------|-------------|-------|
| Baseline | ~42% | No optimization |
| Prompt-only (MIPROv2) | ~58% | +16 pts |
| Fine-tune-only | ~61% | +19 pts |
| BetterTogether | ~71% | +29 pts |

BetterTogether gets +10 pts beyond the best individual approach because prompt optimization and weight optimization complement each other.

---

## Example 4: Troubleshooting low bootstrap success

When your base model is too weak to bootstrap enough successful traces.

### Problem: 30% baseline, bootstrapping fails

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class ExtractEntities(dspy.Signature):
    """Extract all named entities from the text."""
    text: str = dspy.InputField()
    entities: list[str] = dspy.OutputField()

program = dspy.ChainOfThought(ExtractEntities)

# Baseline is only ~30% — too weak for bootstrapping
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
print(f"Baseline: {evaluator(program):.1f}%")  # ~30%

# BootstrapFinetune will struggle — only ~30% of traces pass the metric
# Not enough successful traces to fine-tune well
```

### Fix 1: Use a stronger model for bootstrapping

Use an expensive model to generate traces, then fine-tune the cheap model on them:

```python
# Bootstrap with the strong model
strong_lm = dspy.LM("openai/gpt-4o")
dspy.configure(lm=strong_lm)

strong_program = dspy.ChainOfThought(ExtractEntities)

# Now bootstrap — GPT-4o will succeed on ~70% of examples
# giving us plenty of traces to fine-tune GPT-4o-mini
weak_lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=weak_lm)

optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
finetuned = optimizer.compile(program, trainset=trainset, teacher=strong_program)
print(f"Fine-tuned (with strong teacher): {evaluator(finetuned):.1f}%")
# Expected: ~55% (up from 30%)
```

### Fix 2: Relax metric for bootstrapping

Use a lenient metric during trace collection, strict metric for evaluation:

```python
def lenient_metric(example, prediction, trace=None):
    """Accept partial matches during bootstrapping."""
    gold = set(e.lower() for e in example.entities)
    pred = set(e.lower() for e in prediction.entities)
    if not gold:
        return float(len(pred) == 0)
    # Accept if at least half the entities are found
    recall = len(gold & pred) / len(gold)
    return recall >= 0.5

def strict_metric(example, prediction, trace=None):
    """Require exact match for evaluation."""
    gold = set(e.lower() for e in example.entities)
    pred = set(e.lower() for e in prediction.entities)
    return gold == pred

# Bootstrap with lenient metric (more traces pass)
optimizer = dspy.BootstrapFinetune(metric=lenient_metric, num_threads=24)
finetuned = optimizer.compile(program, trainset=trainset)

# Evaluate with strict metric
strict_evaluator = Evaluate(devset=devset, metric=strict_metric, num_threads=4, display_progress=True)
print(f"Fine-tuned (lenient bootstrap): {strict_evaluator(finetuned):.1f}%")
# Expected: ~48% (up from 30%)
```

### Summary: when bootstrapping fails

| Problem | Fix | Expected gain |
|---------|-----|--------------|
| Weak base model (~30%) | Use stronger teacher model | +20-30 pts |
| Too-strict metric | Relax metric for bootstrapping | +15-25 pts |
| Complex multi-step task | Break into simpler sub-tasks | Varies |
| Not enough data | Collect more labeled examples | Varies |
