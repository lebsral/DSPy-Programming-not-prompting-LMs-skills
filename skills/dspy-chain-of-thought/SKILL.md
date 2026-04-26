---
name: dspy-chain-of-thought
description: Use when the task benefits from intermediate reasoning before producing an answer — multi-step logic, analysis, math, or complex classification where direct prediction fails. Common scenarios: classification tasks where the model needs to reason about edge cases, math word problems, multi-step analysis, complex question answering, legal or medical reasoning, any task where thinking before answering improves quality. Related: ai-reasoning, dspy-predict, dspy-multi-chain-comparison. Also: dspy.ChainOfThought, CoT prompting in DSPy, think step by step, show your reasoning, intermediate reasoning steps, LLM gives wrong answer without thinking, reasoning before output, make AI explain its logic, step-by-step problem solving, when to use ChainOfThought vs Predict, add reasoning to any DSPy module, let the model think, chain of thought for classification.
---

# Step-by-Step Reasoning with dspy.ChainOfThought

Guide the user through using DSPy's `ChainOfThought` module -- the go-to module for tasks that benefit from intermediate reasoning before producing an answer.

## What is ChainOfThought

`dspy.ChainOfThought` is a drop-in replacement for `dspy.Predict` that automatically injects a `reasoning` field before your output fields. Same signature, one-word swap -- the LM reasons step-by-step before answering. The `reasoning` field is always available on the result even though your signature doesn't declare it.

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

## Passing reasoning downstream

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

## Predict vs ChainOfThought

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

## Gotchas

1. **Don't add `reasoning` to your signature** -- DSPy injects it automatically. Declaring it yourself creates a duplicate field that confuses the prompt.
2. **CoT adds ~100-300 tokens of overhead per call** -- measure whether the accuracy gain justifies the extra latency and cost for your task.
3. **Reasoning quality degrades with very short `max_tokens`** -- leave room for the reasoning trace before the actual output fields.
4. **Not all tasks benefit** -- classification with fewer than 5 clear categories often does better with plain `Predict`. CoT shines on tasks requiring multi-step deduction.

## Cross-references

- **Predict** for simple calls without reasoning -- see `/dspy-predict`
- **Signatures** for defining input/output contracts -- see `/dspy-signatures`
- **Modules** for building multi-step programs with CoT sub-modules -- see `/dspy-modules`
- **Reasoning patterns** for broader strategies (decomposition, self-correction) -- see `/ai-reasoning`
- For worked examples, see [examples.md](examples.md)
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
