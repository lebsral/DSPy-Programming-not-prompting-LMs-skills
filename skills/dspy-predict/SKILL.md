---
name: dspy-predict
description: "Use when the mapping from input to output is straightforward and doesn't need reasoning steps — simple classification, extraction, formatting, or Q&A where minimal latency matters."
---

# Direct LM Calls with dspy.Predict

Guide the user through using `dspy.Predict` -- the simplest and fastest DSPy module for calling a language model. It takes inputs, calls the LM, and returns typed outputs. No intermediate reasoning, no extra steps.

## What is dspy.Predict

`dspy.Predict` is the atomic building block of every DSPy program -- one LM call, no reasoning chain, no tool loops. It takes a signature and calls the LM once to produce the output fields. Every other DSPy module (`ChainOfThought`, `ReAct`, etc.) builds on top of it.

## When to use Predict vs ChainOfThought

| Use `dspy.Predict` when... | Use `dspy.ChainOfThought` when... |
|---|---|
| The task is straightforward (classification, extraction, formatting) | The task benefits from step-by-step reasoning |
| You want minimal latency and token usage | Accuracy matters more than speed |
| The mapping from input to output is direct | The LM needs to "think through" intermediate steps |
| You're building a simple sub-step inside a larger pipeline | You need to inspect the model's reasoning |

**Rule of thumb:** Start with `Predict`. If accuracy is too low, switch to `ChainOfThought` -- it's a one-word change.

## Predict with Pydantic output types

For complex structured outputs, use a Pydantic `BaseModel` as the output type:

```python
import dspy
from pydantic import BaseModel

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class ContactInfo(BaseModel):
    name: str
    email: str
    phone: str

class ExtractContact(dspy.Signature):
    """Extract contact information from the text."""
    text: str = dspy.InputField()
    contact: ContactInfo = dspy.OutputField()

extractor = dspy.Predict(ExtractContact)
result = extractor(text="Reach out to Jane Doe at jane@example.com or 555-0123")
print(result.contact.name)   # Jane Doe
print(result.contact.email)  # jane@example.com
print(result.contact.phone)  # 555-0123
```

## Gotchas

1. **Predict is not "dumb"** -- optimizers can add few-shot demos and tuned instructions, making `Predict` surprisingly powerful. Don't underestimate it.
2. **If accuracy is low, try `ChainOfThought` before reaching for complex solutions** -- it's a one-word swap (`dspy.ChainOfThought` instead of `dspy.Predict`) and often gets you 10-20% accuracy gains on reasoning-heavy tasks.
3. **For batch processing, use `dspy.Parallel` rather than a Python loop** -- it handles concurrency and is significantly faster for large batches.

## Cross-references

- **Defining signatures** (inline and class-based, typed fields, Pydantic outputs) -- see `/dspy-signatures`
- **Adding step-by-step reasoning** -- see `/dspy-chain-of-thought`
- **Building multi-step programs** with modules that compose Predict calls -- see `/dspy-modules`
- **Classification and sorting** with real-world patterns -- see `/ai-sorting`
- For worked examples, see [examples.md](examples.md)
