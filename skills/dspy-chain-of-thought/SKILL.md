---
name: dspy-chain-of-thought
description: "Use DSPy's ChainOfThought module for step-by-step reasoning. Use when you want to use dspy.ChainOfThought, add intermediate reasoning to predictions, improve accuracy on complex tasks, or access the reasoning field to understand how the LM arrived at its answer."
---

# Step-by-Step Reasoning with dspy.ChainOfThought

Guide the user through using DSPy's `ChainOfThought` module -- the go-to module for tasks that benefit from intermediate reasoning before producing an answer.

## What is ChainOfThought

`dspy.ChainOfThought` is a drop-in replacement for `dspy.Predict` that automatically adds a `reasoning` field to the output. Before generating any output fields, the LM first produces a step-by-step reasoning trace. You don't modify your signature -- DSPy injects the reasoning field for you.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Same signature, different module
predict = dspy.Predict("question -> answer")
cot = dspy.ChainOfThought("question -> answer")

# Predict: goes straight to the answer
result = predict(question="Is 17 a prime number?")
print(result.answer)  # "Yes"

# ChainOfThought: reasons first, then answers
result = cot(question="Is 17 a prime number?")
print(result.reasoning)  # "To check if 17 is prime, I need to test divisibility..."
print(result.answer)     # "Yes"
```

The `reasoning` field is always available on the result, even though your signature only declared `answer`. DSPy adds it automatically.

## When CoT helps

ChainOfThought improves accuracy when the task requires the LM to work through intermediate steps before arriving at an answer:

- **Multi-step logic** -- math, puzzles, conditional reasoning
- **Analysis and judgment** -- "Is this code buggy?", "Should we approve this loan?"
- **Classification with nuance** -- when the label depends on weighing multiple factors
- **Explanation-heavy tasks** -- anything where you want to see *why* the LM chose its answer
- **Complex extraction** -- parsing ambiguous data where the LM needs to resolve conflicts

**Rule of thumb:** If a human would need to think through the problem before answering, use ChainOfThought.

## When NOT to use CoT

ChainOfThought adds latency and token cost because the LM generates extra reasoning text. Skip it when:

- **Simple lookups** -- "What is the capital of France?" No reasoning needed.
- **Direct extraction** -- pulling a name or date from structured text. `Predict` is enough.
- **Speed-critical paths** -- if you need sub-second responses and the task is straightforward, use `Predict`.
- **High-volume, low-complexity** -- processing thousands of simple items where reasoning adds cost without improving accuracy.

When in doubt, start with `ChainOfThought` and switch to `Predict` later if profiling shows the reasoning is unnecessary.

## Basic usage

### With inline signatures

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Single input, single output
solver = dspy.ChainOfThought("question -> answer")
result = solver(question="What is 23 * 47?")
print(result.reasoning)  # step-by-step multiplication
print(result.answer)     # "1081"

# Multiple inputs
grader = dspy.ChainOfThought("essay, rubric -> score: int")
result = grader(essay="...", rubric="...")
print(result.reasoning)  # how the essay was evaluated
print(result.score)      # 85

# Multiple outputs
analyzer = dspy.ChainOfThought("code -> has_bug: bool, explanation: str")
result = analyzer(code="def add(a, b): return a - b")
print(result.reasoning)    # traces through the logic
print(result.has_bug)      # True
print(result.explanation)  # "The function subtracts instead of adding"
```

### With class-based signatures

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class TriageBug(dspy.Signature):
    """Triage a bug report by severity and assign to the right team."""
    title: str = dspy.InputField(desc="Bug report title")
    description: str = dspy.InputField(desc="Bug report body")
    severity: Literal["critical", "high", "medium", "low"] = dspy.OutputField()
    team: str = dspy.OutputField(desc="Team name to assign the bug to")

triager = dspy.ChainOfThought(TriageBug)
result = triager(
    title="Login page crashes on Safari",
    description="Users on Safari 17 see a white screen when clicking Sign In. Affects ~20% of users.",
)
print(result.reasoning)  # "Safari 17 affecting 20% of users is significant..."
print(result.severity)   # "critical"
print(result.team)       # "frontend"
```

The docstring in your signature class acts as the task instruction. ChainOfThought uses it to guide the reasoning.

## Accessing the reasoning field

The `reasoning` field is a string containing the LM's step-by-step thought process. You can use it for:

### Logging and debugging

```python
result = cot(question="Should we retry this failed API call?")
print(result.reasoning)  # see the LM's thought process
print(result.answer)

# Log the reasoning for auditing
import logging
logger = logging.getLogger(__name__)
logger.info(f"Decision: {result.answer}, Reasoning: {result.reasoning}")
```

### Passing reasoning downstream

```python
class ReviewDecision(dspy.Module):
    def __init__(self):
        self.analyze = dspy.ChainOfThought("application -> decision: str, risk_level: str")
        self.summarize = dspy.Predict("decision, reasoning -> summary")

    def forward(self, application):
        analysis = self.analyze(application=application)
        # Pass the reasoning to the next step
        summary = self.summarize(
            decision=analysis.decision,
            reasoning=analysis.reasoning,
        )
        return dspy.Prediction(
            decision=analysis.decision,
            risk_level=analysis.risk_level,
            reasoning=analysis.reasoning,
            summary=summary.summary,
        )
```

### Surfacing reasoning to end users

```python
result = cot(question="Why did my deployment fail?")

# Show reasoning in a UI
response = {
    "answer": result.answer,
    "show_work": result.reasoning,  # display in an expandable section
}
```

## Predict vs ChainOfThought -- side by side

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Same signature, same inputs
signature = "code_snippet -> has_bug: bool, fix: str"

predict = dspy.Predict(signature)
cot = dspy.ChainOfThought(signature)

code = "def factorial(n): return n * factorial(n)"

# Predict: fast, no reasoning
r1 = predict(code_snippet=code)
print(r1.has_bug)  # True
print(r1.fix)      # might miss the subtle issue

# ChainOfThought: slower, reasons through the code
r2 = cot(code_snippet=code)
print(r2.reasoning)  # "The function calls factorial(n) without decrementing..."
print(r2.has_bug)    # True
print(r2.fix)        # "Change factorial(n) to factorial(n - 1) and add base case"
```

| | `dspy.Predict` | `dspy.ChainOfThought` |
|---|---|---|
| **Output fields** | Only what the signature declares | Signature fields + `reasoning` |
| **Latency** | Lower | Higher (generates reasoning tokens) |
| **Cost** | Lower | Higher (more output tokens) |
| **Accuracy on complex tasks** | Lower | Higher |
| **Accuracy on simple tasks** | Same | Same (but wastes tokens) |
| **Best for** | Lookups, extraction, simple classification | Analysis, judgment, multi-step problems |

## Combining CoT with typed outputs

ChainOfThought works with all the same type constraints as Predict -- `Literal`, `int`, `float`, `bool`, `list[str]`, and Pydantic models.

```python
import dspy
from pydantic import BaseModel
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class RiskAssessment(BaseModel):
    risk_score: float
    factors: list[str]
    recommendation: Literal["approve", "review", "deny"]

class AssessRisk(dspy.Signature):
    """Assess the risk level of a financial transaction."""
    transaction_details: str = dspy.InputField()
    assessment: RiskAssessment = dspy.OutputField()

assessor = dspy.ChainOfThought(AssessRisk)
result = assessor(
    transaction_details="Wire transfer of $50,000 to a new recipient in a high-risk jurisdiction"
)
print(result.reasoning)                   # detailed risk analysis
print(result.assessment.risk_score)       # 0.85
print(result.assessment.factors)          # ["high amount", "new recipient", "high-risk jurisdiction"]
print(result.assessment.recommendation)   # "review"
```

The LM reasons through the problem first, then produces the structured output. The reasoning happens before type enforcement, so the LM has space to think before committing to typed fields.

## Optimizing CoT with few-shot examples

ChainOfThought benefits significantly from optimization. When you run an optimizer, DSPy discovers high-quality reasoning traces and uses them as few-shot demonstrations:

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Your CoT module
classifier = dspy.ChainOfThought("ticket_text -> priority: str, team: str")

# Training data
trainset = [
    dspy.Example(
        ticket_text="Site is down for all users",
        priority="critical",
        team="infrastructure",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Typo on the pricing page",
        priority="low",
        team="content",
    ).with_inputs("ticket_text"),
    # ... more examples
]

# Metric
def ticket_metric(example, prediction, trace=None):
    priority_correct = prediction.priority == example.priority
    team_correct = prediction.team == example.team
    return priority_correct + team_correct

# Optimize -- the optimizer generates and selects good reasoning traces
optimizer = dspy.BootstrapFewShot(metric=ticket_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(classifier, trainset=trainset)

# The optimized program now includes few-shot examples with reasoning
result = optimized(ticket_text="Users can't upload files larger than 10MB")
print(result.reasoning)  # higher quality reasoning, guided by learned demos
print(result.priority)
print(result.team)

# Save for production
optimized.save("ticket_classifier.json")
```

What optimization does for ChainOfThought:
- **Bootstraps reasoning traces** -- the optimizer runs the module on training examples, keeps the traces that led to correct answers, and includes them as few-shot demonstrations
- **Improves consistency** -- the LM sees examples of good reasoning patterns before generating its own
- **Works with all optimizers** -- `BootstrapFewShot`, `MIPROv2`, `BootstrapFewShotWithRandomSearch` all support CoT modules

## Using CoT inside custom modules

ChainOfThought is a sub-module like any other. Use it in `dspy.Module` for multi-step pipelines:

```python
import dspy

class CodeReviewer(dspy.Module):
    def __init__(self):
        self.find_issues = dspy.ChainOfThought("code -> issues: list[str], severity: str")
        self.suggest_fix = dspy.ChainOfThought("code, issues -> fixed_code: str")

    def forward(self, code):
        analysis = self.find_issues(code=code)

        if analysis.severity == "none":
            return dspy.Prediction(
                issues=[],
                severity="none",
                fixed_code=code,
                reasoning=analysis.reasoning,
            )

        fix = self.suggest_fix(code=code, issues=analysis.issues)

        return dspy.Prediction(
            issues=analysis.issues,
            severity=analysis.severity,
            fixed_code=fix.fixed_code,
            reasoning=analysis.reasoning,
        )
```

Both sub-modules use CoT because code review and fix suggestion both benefit from step-by-step thinking. When optimized, DSPy tunes each sub-module's reasoning independently.

## Cross-references

- **Predict** for simple calls without reasoning -- see `/dspy-predict`
- **Signatures** for defining input/output contracts -- see `/dspy-signatures`
- **Modules** for building multi-step programs with CoT sub-modules -- see `/dspy-modules`
- **Reasoning patterns** for broader strategies (decomposition, self-correction) -- see `/ai-reasoning`
- For worked examples, see [examples.md](examples.md)
