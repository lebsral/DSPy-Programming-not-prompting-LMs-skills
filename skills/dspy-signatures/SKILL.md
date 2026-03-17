---
name: dspy-signatures
description: "Use DSPy's Signature, InputField, and OutputField to declare typed input/output specs for LM calls. Use when you want to define a dspy.Signature, create typed fields, use inline signatures, build class-based signatures, add Pydantic models as output types, or constrain LM outputs with type annotations."
---

# DSPy Signatures

Guide the user through defining DSPy Signatures — typed declarations of what goes into and comes out of an LM call.

## What is a Signature

A Signature is DSPy's way of declaring the input/output contract for a language model call. Instead of writing a prompt, you describe the shape of the data: what fields go in, what fields come out, and what types they should be. DSPy compiles this into an optimized prompt automatically.

Signatures separate *what* you want the LM to do from *how* it does it. You define the I/O spec; DSPy handles the prompting.

## When to use each style

| Style | When to use | Example |
|-------|-------------|---------|
| **Inline** | Quick one-liner, 1-2 inputs, 1 output, string types | `"question -> answer"` |
| **Class-based** | Multiple fields, type constraints, descriptions, Pydantic outputs | `class Classify(dspy.Signature)` |

**Rule of thumb:** Start inline for prototyping. Switch to class-based when you need type constraints, field descriptions, or more than one output.

## Inline signatures

Inline signatures are strings with `->` separating inputs from outputs. Field names become the parameter names.

```python
import dspy

# Configure any LM provider
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Basic: string in, string out
qa = dspy.Predict("question -> answer")
result = qa(question="What is the capital of France?")
print(result.answer)  # Paris

# With type annotation
scorer = dspy.Predict("question, answer -> score: float")
result = scorer(question="What is 2+2?", answer="4")
print(result.score)  # 1.0

# List output
tagger = dspy.Predict("text -> labels: list[str]")
result = tagger(text="Python is great for data science and web development")
print(result.labels)  # ['python', 'data science', 'web development']

# Boolean output
checker = dspy.Predict("claim, evidence -> is_supported: bool")
result = checker(claim="The sky is blue", evidence="The sky appears blue due to Rayleigh scattering")
print(result.is_supported)  # True
```

Inline signatures support these type suffixes: `str` (default), `int`, `float`, `bool`, `list[str]`.

## Class-based signatures

Class-based signatures give you full control over field types, descriptions, and documentation.

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

class AssessTone(dspy.Signature):
    """Assess the tone of the given message for a customer support context."""
    message: str = dspy.InputField(desc="Customer message to analyze")
    tone: Literal["friendly", "neutral", "frustrated", "angry"] = dspy.OutputField(desc="Detected tone")
    confidence: float = dspy.OutputField(desc="Confidence score between 0 and 1")

assessor = dspy.ChainOfThought(AssessTone)
result = assessor(message="I've been waiting 3 days and nobody has responded!")
print(result.tone)        # frustrated
print(result.confidence)  # 0.9
```

Key elements of a class-based signature:

- **Docstring** — acts as the task instruction. DSPy uses it to guide the LM. Write it as a clear directive.
- **InputField** — declares an input parameter. The LM receives these values.
- **OutputField** — declares an output parameter. The LM generates these values.
- **Type annotation** — constrains what the LM can return. DSPy enforces these at parse time.

## Field options

Both `InputField` and `OutputField` accept these parameters:

```python
class Example(dspy.Signature):
    """Demonstrate field options."""
    # desc: describes the field to the LM (shows up in the prompt)
    text: str = dspy.InputField(desc="The document to analyze")

    # prefix: overrides the field label in the prompt (default is the field name)
    summary: str = dspy.OutputField(prefix="Summary:")

    # desc + type constraint combined
    category: str = dspy.OutputField(
        desc="The document category",
        type_=Literal["news", "blog", "research"]
    )
```

- **`desc`** — a natural language description. Helps the LM understand what the field means. Use this when the field name alone is ambiguous.
- **`prefix`** — overrides the label shown in the prompt. Defaults to the field name followed by a colon.
- **`type_`** — alternative way to set type constraint directly on OutputField (instead of using Python type annotation).

## Typed outputs

Use Python type annotations to constrain what the LM returns:

```python
from typing import Literal

class TypeDemo(dspy.Signature):
    """Demonstrate typed output fields."""
    text: str = dspy.InputField()

    # String (default)
    summary: str = dspy.OutputField()

    # Integer
    word_count: int = dspy.OutputField(desc="Number of words in the text")

    # Float
    readability_score: float = dspy.OutputField(desc="Score from 0.0 to 1.0")

    # Boolean
    is_english: bool = dspy.OutputField(desc="Whether the text is in English")

    # Constrained string (one of the listed values)
    difficulty: Literal["easy", "medium", "hard"] = dspy.OutputField()

    # List
    keywords: list[str] = dspy.OutputField(desc="Key topics mentioned")
```

`Literal` is especially useful for classification tasks — it constrains the LM to return only one of the listed values.

## Pydantic models as output types

For complex or nested structured output, use a Pydantic `BaseModel` as the output type. DSPy handles serialization and validation automatically.

```python
import dspy
from pydantic import BaseModel, Field
from typing import Optional

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

class LineItem(BaseModel):
    description: str
    quantity: int
    unit_price: float

class Invoice(BaseModel):
    vendor: str
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    total: float
    items: list[LineItem]
    notes: Optional[str] = None

class ParseInvoice(dspy.Signature):
    """Extract structured invoice data from the raw text."""
    text: str = dspy.InputField(desc="Raw invoice text")
    invoice: Invoice = dspy.OutputField(desc="Parsed invoice data")

parser = dspy.ChainOfThought(ParseInvoice)
result = parser(text="Invoice from Acme Corp, Jan 15 2025. 2x Widget ($10 each), 1x Gadget ($25). Total: $45.")
print(result.invoice.vendor)       # Acme Corp
print(result.invoice.items[0])     # LineItem(description='Widget', quantity=2, unit_price=10.0)
print(result.invoice.total)        # 45.0
```

**When to use Pydantic outputs:**
- You need nested objects (addresses, line items, etc.)
- You want automatic validation (Pydantic enforces types)
- The output maps to a database model or API response
- You need `Optional` fields for data that may not be present

## Multiple outputs

Signatures can have any number of output fields. Each becomes an attribute on the result.

```python
class AnalyzeReview(dspy.Signature):
    """Analyze a product review across multiple dimensions."""
    review: str = dspy.InputField(desc="Product review text")
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = dspy.OutputField()
    topics: list[str] = dspy.OutputField(desc="Product aspects mentioned")
    summary: str = dspy.OutputField(desc="One-sentence summary")
    would_recommend: bool = dspy.OutputField(desc="Whether the reviewer would recommend the product")

analyzer = dspy.ChainOfThought(AnalyzeReview)
result = analyzer(review="The camera is fantastic but battery dies in 2 hours. Not worth the price.")
print(result.sentiment)        # mixed
print(result.topics)           # ['camera', 'battery', 'price']
print(result.summary)          # "Great camera quality undermined by poor battery life and high price."
print(result.would_recommend)  # False
```

## Common patterns

### Docstrings as task instructions

The docstring is the most important part of a class-based signature. DSPy uses it as the primary instruction to the LM.

```python
# Vague — the LM doesn't know what "classify" means in your context
class Bad(dspy.Signature):
    """Classify the text."""
    text: str = dspy.InputField()
    label: str = dspy.OutputField()

# Clear — tells the LM exactly what to do
class Good(dspy.Signature):
    """Classify the customer support message into a department for routing.
    Consider the primary intent, not just keywords."""
    message: str = dspy.InputField(desc="Customer support message")
    department: Literal["billing", "technical", "account", "general"] = dspy.OutputField()
```

### Descriptions to disambiguate fields

```python
class ExtractDate(dspy.Signature):
    """Extract the most relevant date from the text."""
    text: str = dspy.InputField()
    # Without desc, the LM might return any date format
    date: str = dspy.OutputField(desc="Date in ISO 8601 format (YYYY-MM-DD)")
```

### Multiple inputs

```python
class CompareProducts(dspy.Signature):
    """Compare two products and recommend which one to buy."""
    product_a: str = dspy.InputField(desc="Description of first product")
    product_b: str = dspy.InputField(desc="Description of second product")
    budget: float = dspy.InputField(desc="Maximum budget in dollars")
    recommendation: str = dspy.OutputField(desc="Which product to buy and why")
    within_budget: bool = dspy.OutputField()
```

## Cross-references

- **Using signatures with modules** — see `/dspy-predict` (Predict), `/dspy-chain-of-thought` (ChainOfThought), `/dspy-modules` (custom modules)
- **Parsing structured data from text** — see `/ai-parsing-data`
- **Classification and sorting** — see `/ai-sorting`
- **Full working examples** — see [examples.md](examples.md)
