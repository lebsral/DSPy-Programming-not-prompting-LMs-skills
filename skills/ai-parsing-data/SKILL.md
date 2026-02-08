---
name: ai-parsing-data
description: Pull structured data from messy text using AI. Use when parsing invoices, extracting fields from emails, scraping entities from articles, converting unstructured text to JSON, extracting contact info, parsing resumes, reading forms, or any task where messy text goes in and clean structured data comes out. Powered by DSPy extraction.
---

# Build an AI Data Parser

Guide the user through building AI that pulls structured data out of messy text. Uses DSPy extraction — define what you want, and the AI extracts it.

## Step 1: Define what to extract

Ask the user:
1. **What are you parsing?** (emails, invoices, resumes, articles, forms, etc.)
2. **What fields do you need?** (names, dates, amounts, entities, etc.)
3. **What's the output format?** (flat fields, list of objects, nested structure)

## Step 2: Build the parser

### Simple field extraction

```python
import dspy

class ParseContact(dspy.Signature):
    """Extract contact information from the text."""
    text: str = dspy.InputField(desc="Text containing contact information")
    name: str = dspy.OutputField(desc="Person's full name")
    email: str = dspy.OutputField(desc="Email address")
    phone: str = dspy.OutputField(desc="Phone number")

parser = dspy.ChainOfThought(ParseContact)
```

### Structured output with Pydantic

For complex or nested output, use Pydantic models:

```python
from pydantic import BaseModel, Field

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: int
    address: Address
    skills: list[str]

class ParsePerson(dspy.Signature):
    """Extract person details from the text."""
    text: str = dspy.InputField()
    person: Person = dspy.OutputField()

parser = dspy.ChainOfThought(ParsePerson)
result = parser(text="John Doe, 32, lives at 123 Main St, Springfield IL 62701. Expert in Python and SQL.")
print(result.person)  # Person(name='John Doe', age=32, ...)
```

### List extraction

Extract multiple items from text:

```python
class Entity(BaseModel):
    name: str
    type: str = Field(description="Type: person, organization, location, or date")

class ParseEntities(dspy.Signature):
    """Extract all named entities from the text."""
    text: str = dspy.InputField()
    entities: list[Entity] = dspy.OutputField(desc="All entities found in the text")

parser = dspy.ChainOfThought(ParseEntities)
```

## Step 3: Handle messy data

Use assertions to catch bad extractions:

```python
class ValidatedParser(dspy.Module):
    def __init__(self):
        self.parse = dspy.ChainOfThought(ParseContact)

    def forward(self, text):
        result = self.parse(text=text)
        dspy.Suggest(
            "@" in result.email,
            "Email should contain @"
        )
        dspy.Suggest(
            len(result.phone) >= 10,
            "Phone number should have at least 10 digits"
        )
        return result
```

## Step 4: Test the quality

```python
def parsing_metric(example, prediction, trace=None):
    """Score based on field-level accuracy."""
    correct = 0
    total = 0
    for field in ["name", "email", "phone"]:
        expected = getattr(example, field, None)
        predicted = getattr(prediction, field, None)
        if expected is not None:
            total += 1
            if predicted and expected.lower().strip() == predicted.lower().strip():
                correct += 1
    return correct / total if total > 0 else 0.0
```

For Pydantic outputs, compare the model objects directly or field-by-field.

## Step 5: Improve accuracy

```python
optimizer = dspy.BootstrapFewShot(metric=parsing_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(parser, trainset=trainset)
```

## Key patterns

- **Use Pydantic models** for complex structured output — DSPy handles serialization
- **Use `list[Model]`** to extract variable-length lists of items
- **`ChainOfThought` helps** — reasoning through which text maps to which fields improves accuracy
- **Validate with assertions** — `dspy.Suggest` and `dspy.Assert` catch malformed extractions
- **Partial credit metrics** — score field-by-field rather than all-or-nothing

## Additional resources

- For worked examples (invoices, resumes, entities), see [examples.md](examples.md)
- Need summaries instead of structured data? Use `/ai-summarizing`
- AI missing items on complex inputs? Use `/ai-decomposing-tasks` to break extraction into reliable subtasks
- Next: `/ai-improving-accuracy` to measure and improve your parser
