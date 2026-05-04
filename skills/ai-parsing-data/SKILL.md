---
name: ai-parsing-data
description: Pull structured data from messy text using AI. Use when parsing invoices, extracting fields from emails, scraping entities from articles, converting unstructured text to JSON, extracting contact info, parsing resumes, reading forms, pulling data from transcripts (VTT, LiveKit, Recall), extracting fields from Langfuse traces, or any task where messy text goes in and clean structured data comes out. Also use when emails are messy and lack structure, or structured data extraction from unstructured content is unreliable., extract entities from text, parse PDF with AI, structured extraction from unstructured text, OCR plus AI extraction, convert email to structured data, pull fields from documents automatically, AI data entry automation, invoice parsing, resume parsing with AI, medical record extraction.
---

# Build an AI Data Parser

Guide the user through building AI that pulls structured data out of messy text. Uses DSPy extraction — define the output shape, and the AI fills it in.

## Step 1: Define what to extract

Ask the user:
1. **What are you parsing?** (emails, invoices, resumes, transcripts, articles, forms, etc.)
2. **What fields do you need?** (names, dates, amounts, entities, etc.)
3. **Are any fields optional?** (some documents might not have every field)
4. **What's the output format?** (flat fields, list of objects, nested structure)
5. **Do you have examples of correct extractions?** (even a few help with optimization)

## Step 2: Build the parser

### Simple field extraction

For pulling a known set of fields from text:

```python
import dspy

# Configure any LM provider
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

class ParseContact(dspy.Signature):
    """Extract contact information from the text."""
    text: str = dspy.InputField(desc="Text containing contact information")
    name: str = dspy.OutputField(desc="Person's full name")
    email: str = dspy.OutputField(desc="Email address")
    phone: str = dspy.OutputField(desc="Phone number")

parser = dspy.ChainOfThought(ParseContact)
```

`ChainOfThought` adds reasoning before extraction, which helps the model think through which text maps to which field — typically 5-15% more accurate than bare `Predict` on ambiguous inputs.

### Structured output with Pydantic

For complex or nested output, use Pydantic models. DSPy handles the serialization automatically:

```python
from pydantic import BaseModel, Field
from typing import Optional

class Address(BaseModel):
    street: str
    city: str
    state: str
    zip_code: str

class Person(BaseModel):
    name: str
    age: Optional[int] = None
    email: Optional[str] = None
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

Use `Optional` for fields that might not appear in every document — this tells the model it's OK to return `None` instead of guessing.

### List extraction

When you need to pull a variable number of items (entities, line items, experiences):

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

## Step 3: Load your data

### From files

```python
from pathlib import Path

# Single file
text = Path("document.txt").read_text()
result = parser(text=text)

# Directory of files
documents = []
for path in Path("documents/").glob("*.txt"):
    documents.append({"file": path.name, "text": path.read_text()})
```

### From a CSV

```python
import pandas as pd

df = pd.read_csv("emails.csv")  # column: body
results = []
for _, row in df.iterrows():
    result = parser(text=row["body"])
    results.append(result.person.model_dump())  # Pydantic → dict

# Save extracted data
pd.DataFrame(results).to_csv("extracted.csv", index=False)
```

### From transcripts (VTT, LiveKit, Recall)

Transcripts are a common parsing source — extracting caller info, action items, decisions, or structured summaries from conversations.

**WebVTT (.vtt) files:**

```python
import re

def load_vtt(path):
    """Extract text from a VTT transcript, stripping timestamps."""
    text = open(path).read()
    lines = [line.strip() for line in text.split("\n")
             if line.strip() and not line.startswith("WEBVTT")
             and not re.match(r"\d{2}:\d{2}", line)
             and not line.strip().isdigit()]
    return " ".join(lines)
```

**LiveKit transcripts:**

```python
import json

def load_livekit_transcript(path):
    """Extract text from a LiveKit transcript JSON export."""
    data = json.load(open(path))
    segments = data.get("segments", data.get("results", []))
    return " ".join(seg.get("text", "") for seg in segments)
```

**Recall.ai transcripts:**

```python
def load_recall_transcript(transcript_data):
    """Extract text from a Recall.ai transcript response."""
    return " ".join(
        entry["words"] for entry in transcript_data if entry.get("words")
    )
```

**Example: extracting structured data from a call transcript:**

```python
class CallSummary(BaseModel):
    caller_name: Optional[str] = None
    issue_summary: str
    resolution: Optional[str] = None
    follow_up_needed: bool
    action_items: list[str]

class ParseCallTranscript(dspy.Signature):
    """Extract structured information from a customer call transcript."""
    transcript: str = dspy.InputField(desc="Full call transcript text")
    summary: CallSummary = dspy.OutputField()

parser = dspy.ChainOfThought(ParseCallTranscript)
transcript = load_livekit_transcript("call_001.json")
result = parser(transcript=transcript)
```

### From Langfuse traces

Extract structured data from AI interactions logged in Langfuse:

```python
from langfuse import Langfuse

langfuse = Langfuse()
traces = langfuse.fetch_traces(limit=100).data

# Parse each trace's input/output for structured fields
for trace in traces:
    if trace.input:
        text = trace.input.get("message", str(trace.input))
        result = parser(text=text)
```

## Step 4: Handle messy data

Real-world text is messy. Use a reward function with `dspy.Refine` to catch bad extractions and retry:

```python
class ValidatedParser(dspy.Module):
    def __init__(self):
        self.parse = dspy.ChainOfThought(ParseContact)

    def forward(self, text):
        return self.parse(text=text)

def contact_reward(args, pred):
    score = 1.0
    if not pred.email or "@" not in pred.email:
        score -= 0.4  # Email should contain @
    phone_digits = pred.phone.replace("-", "").replace(" ", "") if pred.phone else ""
    if len(phone_digits) < 10:
        score -= 0.3  # Phone number should have at least 10 digits
    return max(score, 0.0)

validated_parser = dspy.Refine(
    module=ValidatedParser(),
    N=3,
    reward_fn=contact_reward,
    threshold=0.7,
)
```

`dspy.Refine` retries the extraction up to N times, keeping the attempt with the highest reward score. Penalize each failed constraint proportionally to its importance.

### Handling missing fields

When a field genuinely isn't in the text, you want the model to say so rather than hallucinate a value. Use `Optional` types in your Pydantic model, and add a validation note in the signature docstring:

```python
class ParseContact(dspy.Signature):
    """Extract contact info from the text. Return None for fields not present — do not guess."""
    text: str = dspy.InputField()
    name: str = dspy.OutputField(desc="Person's full name")
    email: Optional[str] = dspy.OutputField(desc="Email address, or None if not found")
    phone: Optional[str] = dspy.OutputField(desc="Phone number, or None if not found")
```

## Step 5: Evaluate quality

```python
from dspy.evaluate import Evaluate

def parsing_metric(example, prediction, trace=None):
    """Score based on field-level accuracy (partial credit)."""
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

evaluator = Evaluate(devset=devset, metric=parsing_metric, num_threads=4, display_progress=True)
score = evaluator(parser)
print(f"Baseline accuracy: {score}%")
```

For Pydantic outputs, compare field-by-field or use the model's `.model_dump()` to compare dicts. Partial credit (scoring each field independently) is better than all-or-nothing for extraction tasks — it tells you which specific fields are causing problems.

## Step 6: Optimize and deploy

```python
# Optimize
optimizer = dspy.BootstrapFewShot(metric=parsing_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(parser, trainset=trainset)

# Evaluate improvement
improved = evaluator(optimized)
print(f"Optimized accuracy: {improved}%")

# Save for production
optimized.save("parser.json")

# Load later
parser = dspy.ChainOfThought(ParseContact)
parser.load("parser.json")
```

## Batch processing

For parsing many documents at once:

```python
import json

results = []
errors = []

for doc in documents:
    try:
        result = optimized(text=doc["text"])
        results.append({
            "source": doc["file"],
            **result.person.model_dump()  # flatten Pydantic fields
        })
    except Exception as e:
        errors.append({"source": doc["file"], "error": str(e)})

# Save results
with open("extracted.json", "w") as f:
    json.dump(results, f, indent=2)

if errors:
    print(f"{len(errors)} documents failed to parse — check errors list")
```

## Additional resources

- For worked examples (invoices, resumes, entities, relations, forms), see [examples.md](examples.md)
- Need summaries instead of structured data? Use `/ai-summarizing`
- AI missing items on complex inputs? Use `/ai-decomposing-tasks`
- Want to measure and improve further? Use `/ai-improving-accuracy`
- Need to generate training data? Use `/ai-generating-data`

## Gotchas

- **Pydantic models must be JSON-serializable** — avoid custom types, datetime objects, or complex validators in output models. Stick to `str`, `int`, `float`, `bool`, `list`, `dict`, and nested Pydantic models.
- **Optional fields need explicit `None` defaults** — use `field: Optional[str] = dspy.OutputField(default=None)` or the model will hallucinate values for missing fields instead of returning None.
- **List extraction undercounts by default** — when extracting lists of items (e.g., "all people mentioned"), the LM tends to stop early. Set `max_tokens` higher and add a "be exhaustive" instruction in the signature docstring.
- **Long inputs get truncated silently** — if your input text exceeds the model's context window, DSPy doesn't warn you. Chunk long documents before parsing, or use a model with a larger context window.
- **Nested Pydantic models increase failure rate** — each level of nesting adds extraction difficulty. Flatten where possible, or break into multiple extraction steps (extract outer structure first, then fill in nested fields).
- **Install `/ai-do` if you do not have it** — it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`
