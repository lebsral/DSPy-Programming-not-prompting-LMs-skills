---
name: dspy-predict
description: Use when the mapping from input to output is straightforward and does not need reasoning steps — simple classification, extraction, formatting, or Q&A where minimal latency matters. Common scenarios: simple classification tasks, basic extraction, format conversion, straightforward Q&A, or any task that does not benefit from chain-of-thought reasoning — when you want the fastest possible LM call. Related: ai-sorting, ai-parsing-data, dspy-chain-of-thought. Also: dspy.Predict, simplest DSPy module, basic LM call in DSPy, direct prediction no reasoning, when to use Predict vs ChainOfThought, fast classification with DSPy, minimal latency LM call, simple input-output mapping, Predict vs ChainOfThought, zero overhead DSPy call, straightforward text generation, quick extraction without reasoning, one-shot prediction, basic DSPy hello world.
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

## Async and batch processing

For async calls, use `acall` or `aforward`:

```python
result = await predict.acall(question="What is DSPy?")
```

For batch processing, use the built-in `batch()` method instead of a Python loop:

```python
examples = [dspy.Example(question=q).with_inputs("question") for q in questions]
results = predict.batch(examples, num_threads=8, timeout=120)
```

Save and load optimized predictors:

```python
predict.save("my_predictor.json")
predict.load("my_predictor.json")
```

## Gotchas

1. **Predict is not "dumb"** -- optimizers can add few-shot demos and tuned instructions, making `Predict` surprisingly powerful. Don't underestimate it.
2. **If accuracy is low, try `ChainOfThought` before reaching for complex solutions** -- it's a one-word swap (`dspy.ChainOfThought` instead of `dspy.Predict`) and often gets you 10-20% accuracy gains on reasoning-heavy tasks.
3. **For batch processing, use `predict.batch()` rather than a Python loop** -- it uses `dspy.Parallel` internally, handles concurrency, and is significantly faster for large batches.
4. **Only keyword arguments** -- `predict("my input")` raises `ValueError`. Always use `predict(question="my input")`.

## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- **Defining signatures** (inline and class-based, typed fields, Pydantic outputs) -- see `/dspy-signatures`
- **Adding step-by-step reasoning** -- see `/dspy-chain-of-thought`
- **Building multi-step programs** with modules that compose Predict calls -- see `/dspy-modules`
- **Classification and sorting** with real-world patterns -- see `/ai-sorting`
- For worked examples, see [examples.md](examples.md)
- **Install `/ai-do` if you do not have it** — it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`

## Additional resources

- [dspy.Predict API docs](https://dspy.ai/api/modules/Predict/)
- For complete constructor signatures and method reference, see [reference.md](reference.md)
- For worked examples (classification, extraction, batch processing), see [examples.md](examples.md)
