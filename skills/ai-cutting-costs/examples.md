# Cost-Cutting Examples

## Switch to a Cheaper Model

The simplest cost reduction: swap the model and run your evaluator to confirm quality holds.

```python
import dspy
from dspy.evaluate import Evaluate

cheap_lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-haiku-4-5-20251001", etc.
dspy.configure(lm=cheap_lm)

classify = dspy.Predict("text -> sentiment: str")
result = classify(text="The shipping was slow but the product is great.")
print(result.sentiment)  # mixed

dspy.inspect_history(n=1)  # shows prompt, response, and token counts
```

Measure quality on your devset with `dspy.Evaluate` before and after switching. If scores drop more than ~5%, re-optimize (see `/ai-switching-models`).

## Verify Caching Is Active

DSPy caches by default — identical inputs return instantly, with no API call and no charge.

```python
import time, dspy

lm = dspy.LM("openai/gpt-4o-mini")  # cache=True by default
dspy.configure(lm=lm)
qa = dspy.Predict("question -> answer")

start = time.time(); qa(question="What is the capital of France?")
print(f"First call:  {time.time() - start:.2f}s")   # ~0.5–2s (live API call)

start = time.time(); qa(question="What is the capital of France?")
print(f"Second call: {time.time() - start:.4f}s")  # ~0.001s (cache hit)
```

If both calls take the same time, caching is off — check that you did not pass `cache=False` to `dspy.LM`.

## Per-Module LM Assignment

Not every pipeline step needs the expensive model. Assign cheap models to simple steps with `set_lm`:

```python
import dspy

cheap_lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-haiku-4-5-20251001", etc.
expensive_lm = dspy.LM("openai/gpt-4o")   # or "anthropic/claude-sonnet-4-5-20250929", etc.

dspy.configure(lm=expensive_lm)

class SupportPipeline(dspy.Module):
    def __init__(self):
        self.categorize = dspy.Predict("message -> category: str")
        self.draft_reply = dspy.ChainOfThought("message, category -> reply: str")

    def forward(self, message):
        category = self.categorize(message=message)
        return self.draft_reply(message=message, category=category.category)

pipeline = SupportPipeline()
pipeline.categorize.set_lm(cheap_lm)      # simple classification → cheap
pipeline.draft_reply.set_lm(expensive_lm) # complex generation → expensive

result = pipeline(message="I was charged twice for my order last Tuesday.")
print(result.reply)  # billing-aware apology + next steps
```

`set_lm` persists through optimizer compilation. For temporary per-call overrides, use `dspy.context(lm=...)` instead.

## Cascading — Try Cheap, Escalate if Unsure

Try the cheap model first; escalate only if confidence is low. Saves 60-80% on real-world traffic where most inputs are routine.

```python
import dspy

cheap_lm = dspy.LM("openai/gpt-4o-mini")  # or any cheap model
expensive_lm = dspy.LM("openai/gpt-4o")   # or any capable model

class AnswerQuestion(dspy.Signature):
    """Answer the user's question accurately."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class CheckConfidence(dspy.Signature):
    """Is this answer confident and complete, or should we escalate to a better model?"""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    is_confident: bool = dspy.OutputField()

class CascadingQA(dspy.Module):
    def __init__(self):
        self.answer = dspy.Predict(AnswerQuestion)
        self.verify = dspy.Predict(CheckConfidence)  # Predict not ChainOfThought — see gotchas

    def forward(self, question):
        with dspy.context(lm=cheap_lm):
            result = self.answer(question=question)
            check = self.verify(question=question, answer=result.answer)

        if not check.is_confident:
            with dspy.context(lm=expensive_lm):
                result = self.answer(question=question)

        return result

qa = CascadingQA()
print(qa(question="What is 2 + 2?").answer)                  # cheap → "4"
print(qa(question="Explain the Riemann hypothesis").answer)  # escalates to expensive
```

## Route by Complexity

Classify inputs upfront and route easy ones cheap, hard ones expensive. Lower latency than cascading when complexity is predictable from the input.

```python
from typing import Literal
import dspy

cheap_lm = dspy.LM("openai/gpt-4o-mini")  # or any cheap model
expensive_lm = dspy.LM("openai/gpt-4o")   # or any capable model

class AssessComplexity(dspy.Signature):
    """Assess whether this question needs a powerful model or a simple one can handle it."""
    question: str = dspy.InputField()
    complexity: Literal["simple", "complex"] = dspy.OutputField()  # simple = factual, complex = reasoning

class AnswerQuestion(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField()
    answer: str = dspy.OutputField()

class ComplexityRouter(dspy.Module):
    def __init__(self):
        self.assess = dspy.Predict(AssessComplexity)  # Predict, not ChainOfThought
        self.handler = dspy.Predict(AnswerQuestion)

    def forward(self, question):
        with dspy.context(lm=cheap_lm):
            assessment = self.assess(question=question)

        target_lm = cheap_lm if assessment.complexity == "simple" else expensive_lm
        with dspy.context(lm=target_lm):
            return self.handler(question=question)

router = ComplexityRouter()
print(router(question="What year was the Eiffel Tower built?").answer)        # simple → cheap
print(router(question="Compare Keynesian and Modern Monetary Theory").answer) # complex → expensive
```

## End-to-End: Cost-Optimized Ticket Triage

Baseline on expensive, optimize on cheap, compare, save.

```python
import dspy
from dspy.evaluate import Evaluate
from typing import Literal

cheap_lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-haiku-4-5-20251001", etc.
expensive_lm = dspy.LM("openai/gpt-4o")   # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=expensive_lm)

rows = [
    ("Reset my password", "account", "low"),
    ("App crashes on login — production down", "technical", "critical"),
    ("I was charged twice", "billing", "high"),
    ("Suspicious login from unknown location", "security", "critical"),
    # 50+ examples in practice
]
dataset = [dspy.Example(message=m, category=c, urgency=u).with_inputs("message") for m, c, u in rows]
trainset, devset = dataset[:3], dataset[3:]

class TriageTicket(dspy.Signature):
    """Route and prioritize the support ticket."""
    message: str = dspy.InputField()
    category: Literal["billing", "technical", "account", "security"] = dspy.OutputField()
    urgency: Literal["low", "medium", "high", "critical"] = dspy.OutputField()

class TriagePipeline(dspy.Module):
    def __init__(self):
        self.triage = dspy.Predict(TriageTicket)

    def forward(self, message):
        return self.triage(message=message)

def metric(example, pred, trace=None):
    return pred.category == example.category and pred.urgency == example.urgency

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)

# Baseline on expensive model
baseline = evaluator(TriagePipeline())
print(f"Baseline (gpt-4o):       {baseline:.0f}%")  # e.g. 100%

# Optimize on cheap model
cheap_pipeline = TriagePipeline()
cheap_pipeline.triage.set_lm(cheap_lm)
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=2, max_labeled_demos=2)
optimized = optimizer.compile(cheap_pipeline, trainset=trainset)

cheap_score = evaluator(optimized)
print(f"Optimized (gpt-4o-mini): {cheap_score:.0f}%")  # e.g. 90%+ at ~5% of the cost

optimized.save("triage_cheap_optimized.json")
```
