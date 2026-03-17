---
name: dspy-predict
description: "Use DSPy's Predict module for direct LM calls. Use when you want to use dspy.Predict, make simple input-to-output predictions, don't need chain-of-thought reasoning, or want the fastest and simplest DSPy inference module."
---

# Direct LM Calls with dspy.Predict

Guide the user through using `dspy.Predict` -- the simplest and fastest DSPy module for calling a language model. It takes inputs, calls the LM, and returns typed outputs. No intermediate reasoning, no extra steps.

## What is dspy.Predict

`dspy.Predict` is a module that takes a signature (an input/output spec) and calls the LM once to produce the output fields. It is the atomic building block of every DSPy program.

- **One LM call** -- no reasoning chain, no tool loops, no multi-step logic
- **Fastest module** -- minimal token overhead because it only generates the output fields
- **Optimizable** -- DSPy optimizers can add few-shot demos and tune instructions, just like any other module
- **Composable** -- use it inside `dspy.Module` subclasses alongside other modules

Every other DSPy module (`ChainOfThought`, `ReAct`, `ProgramOfThought`) builds on top of `Predict`. If you don't need their extra capabilities, `Predict` is the right choice.

## When to use Predict vs ChainOfThought

| Use `dspy.Predict` when... | Use `dspy.ChainOfThought` when... |
|---|---|
| The task is straightforward (classification, extraction, formatting) | The task benefits from step-by-step reasoning |
| You want minimal latency and token usage | Accuracy matters more than speed |
| The mapping from input to output is direct | The LM needs to "think through" intermediate steps |
| You're building a simple sub-step inside a larger pipeline | You need to inspect the model's reasoning |

**Rule of thumb:** Start with `Predict`. If accuracy is too low, switch to `ChainOfThought` -- it's a one-word change.

## Basic usage with inline signatures

Inline signatures are strings with `->` separating inputs from outputs.

```python
import dspy

# Configure any LM provider
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Create a Predict module with an inline signature
classify = dspy.Predict("text -> label")
result = classify(text="I love this product!")
print(result.label)  # positive
```

You can add type annotations to inline signatures:

```python
# Float output
scorer = dspy.Predict("question, answer -> score: float")
result = scorer(question="What is 2+2?", answer="4")
print(result.score)  # 1.0

# Boolean output
checker = dspy.Predict("claim, evidence -> is_supported: bool")
result = checker(claim="Water boils at 100C", evidence="At sea level, water boils at 100 degrees Celsius.")
print(result.is_supported)  # True

# List output
tagger = dspy.Predict("text -> tags: list[str]")
result = tagger(text="Python web scraping tutorial with BeautifulSoup")
print(result.tags)  # ['python', 'web scraping', 'beautifulsoup']
```

## Basic usage with class-based signatures

Class-based signatures give you type constraints, field descriptions, and docstrings that act as task instructions.

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

class ClassifyTicket(dspy.Signature):
    """Classify a support ticket into a department for routing."""
    ticket: str = dspy.InputField(desc="Customer support ticket text")
    department: Literal["billing", "technical", "account", "general"] = dspy.OutputField()

classify = dspy.Predict(ClassifyTicket)
result = classify(ticket="I can't log into my account after resetting my password")
print(result.department)  # account
```

The docstring is the task instruction -- DSPy uses it to guide the LM. Write it as a clear directive.

## Accessing output fields

The result of a `Predict` call is a `dspy.Prediction` object. Access output fields as attributes:

```python
# Single output field
result = dspy.Predict("text -> summary")(text="Long article text here...")
print(result.summary)

# Multiple output fields
class AnalyzeText(dspy.Signature):
    """Analyze the given text."""
    text: str = dspy.InputField()
    language: str = dspy.OutputField(desc="Detected language")
    word_count: int = dspy.OutputField(desc="Approximate word count")
    is_question: bool = dspy.OutputField(desc="Whether the text is asking a question")

result = dspy.Predict(AnalyzeText)(text="Wie spat ist es?")
print(result.language)     # German
print(result.word_count)   # 4
print(result.is_question)  # True
```

You can also inspect all fields at once:

```python
print(result)  # shows all output fields and their values
```

## Batch predictions

To process multiple items, call `Predict` in a loop. Each call is independent.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

classify = dspy.Predict("text -> sentiment: str")

reviews = [
    "Absolutely love it, best purchase ever!",
    "Terrible quality, broke after one day.",
    "It's okay, nothing special.",
]

results = [classify(text=review) for review in reviews]
for review, result in zip(reviews, results):
    print(f"{result.sentiment}: {review}")
```

For structured batch processing inside a module, wrap the loop in `forward()`:

```python
class BatchClassifier(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> label")

    def forward(self, texts: list[str]):
        labels = [self.classify(text=t).label for t in texts]
        return dspy.Prediction(labels=labels)
```

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

## When Predict is enough

`dspy.Predict` is the right choice for most simple, direct tasks:

- **Classification** -- map text to a category (`Literal` types work great)
- **Extraction** -- pull structured data from unstructured text
- **Formatting** -- convert data from one format to another
- **Labeling** -- tag items with labels from a fixed set
- **Simple Q&A** -- factual lookups that don't need reasoning
- **Scoring / judging** -- assign a numeric score or boolean judgment
- **Translation** -- convert text between languages

These tasks share a common trait: the mapping from input to output is direct and doesn't require intermediate reasoning steps.

## When to upgrade to ChainOfThought

Switch from `Predict` to `ChainOfThought` when:

- **Accuracy drops** -- the task is harder than a direct mapping (e.g., multi-hop questions, nuanced classification)
- **You need to inspect reasoning** -- `ChainOfThought` exposes a `reasoning` field so you can see why the LM chose its answer
- **The task has multiple valid approaches** -- step-by-step thinking helps the LM explore before committing to an answer
- **You're getting inconsistent results** -- reasoning often stabilizes outputs

The change is minimal:

```python
# Before
classify = dspy.Predict("text -> label")

# After -- one word changes
classify = dspy.ChainOfThought("text -> label")
result = classify(text="...")
print(result.reasoning)  # now available
print(result.label)
```

Everything else stays the same -- same signature, same field access, same optimization. `ChainOfThought` just adds a `reasoning` field that the LM fills in before generating the output.

## Cross-references

- **Defining signatures** (inline and class-based, typed fields, Pydantic outputs) -- see `/dspy-signatures`
- **Adding step-by-step reasoning** -- see `/dspy-chain-of-thought`
- **Building multi-step programs** with modules that compose Predict calls -- see `/dspy-modules`
- **Classification and sorting** with real-world patterns -- see `/ai-sorting`
- For worked examples, see [examples.md](examples.md)
