# Consistency Examples

## Measure inconsistency first

Before applying fixes, turn off the cache so you see real variation, then run the same input several times:

```python
import dspy
from collections import Counter

lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0)  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Disable cache — otherwise repeated calls return identical cached results
dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)

qa = dspy.ChainOfThought("question -> answer")

results = []
for i in range(5):
    result = qa(question="Should I use Python or JavaScript for data analysis?")
    results.append(result.answer.strip().lower()[:60])
    print(f"Run {i+1}: {result.answer[:80]}")

counts = Counter(results)
top, count = counts.most_common(1)[0]
print(f"\nConsistency: {count/5:.0%} ({len(counts)} unique answers in 5 runs)")
```

Expected output (high variation, low consistency):
```
Run 1: Python is the better choice for data analysis due to libraries like pandas...
Run 2: For data analysis, Python is generally preferred because of numpy, pandas...
Run 3: Python is typically recommended for data analysis. JavaScript has limited...
Run 4: Both languages can work, but Python has a significant advantage in data...
Run 5: Python is usually the go-to for data analysis because of its ecosystem...
Consistency: 0% (5 unique answers in 5 runs)
```

## Fix 1: temperature=0

The single biggest consistency fix. Set temperature when constructing the LM:

```python
lm = dspy.LM("openai/gpt-4o-mini", temperature=0)  # or any other provider
dspy.configure(lm=lm)
```

Re-run the same loop with caching disabled — you will typically see 90–99% consistency on classification tasks, somewhat less on open-ended generation.

## Fix 2: Constrain output with Literal types

Free-form string outputs drift even at temperature=0 — the model might return `"high"`, `"HIGH"`, `"High priority"`, or `"urgent"` for the same input. Lock the output to exact values:

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini", temperature=0)
dspy.configure(lm=lm)

PRIORITIES = ["low", "medium", "high", "critical"]

class PrioritizeTicket(dspy.Signature):
    """Assign urgency priority to a support ticket."""
    ticket: str = dspy.InputField(desc="Support ticket text")
    priority: Literal[tuple(PRIORITIES)] = dspy.OutputField()

classifier = dspy.Predict(PrioritizeTicket)

result = classifier(ticket="Our entire production API is returning 500 errors")
print(result.priority)  # critical — locked to one of the four values, every run
```

## Fix 3: Pydantic model for multi-field outputs

When the output has several fields, a Pydantic model enforces types and constraints on all of them simultaneously:

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini", temperature=0)
dspy.configure(lm=lm)

class TicketAnalysis(BaseModel):
    priority: Literal["low", "medium", "high", "critical"]
    team: Literal["billing", "technical", "account", "security"]
    summary: str = Field(max_length=120, description="One-sentence summary")

class AnalyzeTicket(dspy.Signature):
    """Analyze a support ticket and route it to the right team."""
    ticket: str = dspy.InputField()
    analysis: TicketAnalysis = dspy.OutputField()

analyzer = dspy.ChainOfThought(AnalyzeTicket)
result = analyzer(ticket="I was charged twice for my subscription this month")
print(result.analysis.priority)  # high
print(result.analysis.team)      # billing
print(result.analysis.summary)   # "Customer was double-charged for their subscription."
```

## Fix 4: Refine for format and length constraints

Types alone cannot enforce rules like "exactly one sentence" or "between 10 and 40 words." Use `dspy.Refine` for those:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini", temperature=0)
dspy.configure(lm=lm)

class Summarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought("article -> summary")

    def forward(self, article):
        return self.summarize(article=article)

def summary_reward(args: dict, pred: dspy.Prediction) -> float:
    summary = pred.summary
    # Hard constraints — disqualify if broken
    if summary.count(".") > 2:
        return 0.0
    words = len(summary.split())
    if not (10 <= words <= 40):
        return 0.0
    # Soft constraint — prefer a terminal period
    score = 1.0
    if not summary.rstrip().endswith("."):
        score -= 0.1
    return score

consistent_summarizer = dspy.Refine(
    Summarizer(),
    N=3,
    reward_fn=summary_reward,
    threshold=0.9,
)

result = consistent_summarizer(article="OpenAI released a new model today...")
print(result.summary)  # One sentence, 10–40 words, ends with period — every run
```

## End-to-end: production support ticket classifier

All four consistency layers together — temperature, Literal types, BootstrapFewShot optimization, and measurement:

```python
import dspy
from typing import Literal
from collections import Counter

# 1. Configure with temperature=0
lm = dspy.LM("openai/gpt-4o-mini", temperature=0)  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# 2. Locked output types
PRIORITIES = ["low", "medium", "high", "critical"]
TEAMS = ["billing", "technical", "account", "security"]

class TriageTicket(dspy.Signature):
    """Route the support ticket to the right team and set its priority."""
    ticket: str = dspy.InputField(desc="Customer support message")
    priority: Literal[tuple(PRIORITIES)] = dspy.OutputField()
    team: Literal[tuple(TEAMS)] = dspy.OutputField()

triager = dspy.ChainOfThought(TriageTicket)

# 3. Training data
trainset = [
    dspy.Example(ticket="Charged twice this month", priority="high", team="billing").with_inputs("ticket"),
    dspy.Example(ticket="App crashes on login", priority="high", team="technical").with_inputs("ticket"),
    dspy.Example(ticket="How do I export my data?", priority="low", team="account").with_inputs("ticket"),
    dspy.Example(ticket="Suspicious login from unknown country", priority="critical", team="security").with_inputs("ticket"),
    dspy.Example(ticket="Invoice does not match my plan", priority="medium", team="billing").with_inputs("ticket"),
    dspy.Example(ticket="API rate limit hit", priority="medium", team="technical").with_inputs("ticket"),
    # Add 30+ examples for best results
]

# 4. Consistency metric — penalizes hedging language alongside correctness
def triage_metric(example, pred, trace=None):
    correct_priority = pred.priority == example.priority
    correct_team = pred.team == example.team
    no_hedging = not any(w in str(pred).lower() for w in ["maybe", "possibly", "unclear"])
    return correct_priority and correct_team and no_hedging

# 5. Optimize to lock in consistent routing patterns
optimizer = dspy.BootstrapFewShot(metric=triage_metric, max_bootstrapped_demos=4)
optimized_triager = optimizer.compile(triager, trainset=trainset)

# 6. Measure consistency on a fixed input (cache disabled to see real behavior)
def measure_consistency(program, inputs, n_runs=10):
    dspy.configure_cache(enable_disk_cache=False, enable_memory_cache=False)
    results = []
    for _ in range(n_runs):
        r = program(**inputs)
        results.append(f"{r.priority}|{r.team}")
    counts = Counter(results)
    top, count = counts.most_common(1)[0]
    print(f"Consistency: {count/n_runs:.0%} ({len(counts)} unique in {n_runs} runs)")
    print(f"Most common output: {top}")
    return count / n_runs

measure_consistency(
    optimized_triager,
    {"ticket": "Cannot log into my account after the password reset"},
)
# temperature=0 alone:                     ~80% consistent
# temperature=0 + Literal types:           ~95% consistent
# temperature=0 + Literal + BootstrapFewShot: ~99% consistent

# 7. Save for production
optimized_triager.save("ticket_triager.json")
```
