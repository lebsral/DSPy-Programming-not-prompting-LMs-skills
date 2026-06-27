# Kickoff Examples

## Minimal first program

The smallest working DSPy program — configure an LM, define a signature, run it:

```python
import dspy

# 1. Configure the LM
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# 2. Define inputs and outputs
class Summarize(dspy.Signature):
    """Summarize the key point in one sentence."""
    text: str = dspy.InputField(desc="Text to summarize")
    summary: str = dspy.OutputField(desc="One-sentence summary")

# 3. Run it
program = dspy.ChainOfThought(Summarize)
result = program(text="DSPy lets you program language models instead of prompting them.")
print(result.summary)
print(result.reasoning)  # ChainOfThought exposes its step-by-step reasoning
```

---

## Adding a first evaluation

Measure a baseline before optimizing. A handful of labeled examples is enough to get a score:

```python
import dspy
from typing import Literal
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class ClassifyIntent(dspy.Signature):
    """Classify the customer message intent."""
    message: str = dspy.InputField()
    intent: Literal["question", "complaint", "praise", "cancellation"] = dspy.OutputField()

program = dspy.ChainOfThought(ClassifyIntent)

devset = [
    dspy.Example(message="How do I reset my password?", intent="question").with_inputs("message"),
    dspy.Example(message="This broke after one week, I want a refund.", intent="complaint").with_inputs("message"),
    dspy.Example(message="Your support team is fantastic!", intent="praise").with_inputs("message"),
    dspy.Example(message="Please cancel my subscription.", intent="cancellation").with_inputs("message"),
]

def metric(example, pred, trace=None):
    return pred.intent == example.intent

evaluator = Evaluate(devset=devset, metric=metric, num_threads=1, display_progress=True)
print(f"Baseline: {evaluator(program):.1f}%")
```

---

## Running a first optimization

Add 20+ labeled examples to `trainset` and run `BootstrapFewShot` to automatically improve prompts:

```python
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(program, trainset=trainset)

print(f"Optimized: {evaluator(optimized):.1f}%")  # typically +10-20pp over baseline
optimized.save("intent_classifier.json")
```

Upgrade to `dspy.MIPROv2` when you have 200+ examples and want the best possible accuracy.

---

## End-to-end: email triage starter

A complete six-file project for routing support emails to the right team. This is what the scaffold looks like after adapting placeholder field names to a real task.

### `requirements.txt`

```
dspy>=3.0
```

### `data.py`

```python
import dspy

def load_data():
    examples = [
        dspy.Example(email="I can't log in, my account is locked.", team="account").with_inputs("email"),
        dspy.Example(email="I was charged twice this month.", team="billing").with_inputs("email"),
        dspy.Example(email="The API keeps returning 500 errors.", team="technical").with_inputs("email"),
        dspy.Example(email="Can I upgrade to the enterprise plan?", team="sales").with_inputs("email"),
        # Add 30+ examples across all teams for meaningful optimization
    ]
    split = int(0.8 * len(examples))
    return examples[:split], examples[split:]
```

### `program.py`

```python
import dspy
from typing import Literal

TEAMS = ["account", "billing", "technical", "sales"]

class TriageEmail(dspy.Signature):
    """Route this customer support email to the correct team."""
    email: str = dspy.InputField(desc="Customer support email body")
    team: Literal[tuple(TEAMS)] = dspy.OutputField(desc="Support team to handle this")

class EmailTriageProgram(dspy.Module):
    def __init__(self):
        self.classify = dspy.ChainOfThought(TriageEmail)

    def forward(self, email):
        return self.classify(email=email)
```

### `metrics.py`

```python
def metric(example, prediction, trace=None):
    return prediction.team == example.team
```

### `evaluate.py`

```python
import dspy
from dspy.evaluate import Evaluate
from program import EmailTriageProgram
from metrics import metric
from data import load_data

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

_, devset = load_data()
program = EmailTriageProgram()

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
print(f"Baseline accuracy: {evaluator(program):.1f}%")
```

### `optimize.py`

```python
import dspy
from dspy.evaluate import Evaluate
from program import EmailTriageProgram
from metrics import metric
from data import load_data

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

trainset, devset = load_data()
program = EmailTriageProgram()

optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(program, trainset=trainset)

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
print(f"Optimized accuracy: {evaluator(optimized):.1f}%")
optimized.save("optimized.json")
```

### `main.py`

```python
import dspy
from program import EmailTriageProgram

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

program = EmailTriageProgram()
try:
    program.load("optimized.json")
    print("Loaded optimized program")
except FileNotFoundError:
    print("Running unoptimized — run optimize.py first")

result = program(email="I've been double-charged for two months now.")
print(f"Route to: {result.team}")
print(f"Reasoning: {result.reasoning}")
```
