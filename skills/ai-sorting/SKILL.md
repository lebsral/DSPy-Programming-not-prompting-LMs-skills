---
name: ai-sorting
description: Auto-sort, categorize, or label content using AI. Use when sorting tickets into categories, auto-tagging content, labeling emails, detecting sentiment, routing messages to the right team, triaging support requests, building a spam filter, intent detection, topic classification, or any task where text goes in and a category comes out.
---

# Build an AI Content Sorter

Guide the user through building an AI that sorts, tags, or categorizes content using DSPy. This skill covers the full lifecycle: defining categories, building the sorter, loading real data, evaluating quality, optimizing accuracy, and deploying.

## Step 1: Define the sorting task

Ask the user:
1. **What are you sorting?** (tickets, emails, reviews, messages, comments, etc.)
2. **What are the categories?** (list all labels/buckets)
3. **One category per item, or multiple?** (e.g., "priority" vs "all applicable tags")
4. **Do you have labeled examples already?** (a CSV, database, spreadsheet with items + their correct category)

The answers determine which pattern to use below.

## Step 2: Build the sorter

### Single category (most common)

```python
import dspy
from typing import Literal

# Configure your LM — works with any provider
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Define your categories
CATEGORIES = ["billing", "technical", "account", "feature_request", "general"]

class SortContent(dspy.Signature):
    """Sort the customer message into the correct support category."""
    message: str = dspy.InputField(desc="The content to sort")
    category: Literal[tuple(CATEGORIES)] = dspy.OutputField(desc="The assigned category")

sorter = dspy.ChainOfThought(SortContent)
```

`Literal` locks the output to valid categories — the model can't invent labels. `ChainOfThought` adds reasoning before the answer, which typically improves classification accuracy by 5-15% over bare `Predict`.

**When to use `Predict` instead:** If your categories are very obvious (spam vs not-spam, yes vs no) and you're optimizing for speed/cost, `dspy.Predict(SortContent)` skips the reasoning step. Start with `ChainOfThought` and drop to `Predict` only if the reasoning isn't helping.

### Multiple tags

When items can belong to several categories at once (e.g., a news article that's both "technology" and "business"):

```python
class TagContent(dspy.Signature):
    """Assign all applicable tags to the content."""
    message: str = dspy.InputField(desc="The content to tag")
    tags: list[Literal[tuple(CATEGORIES)]] = dspy.OutputField(desc="All applicable tags")

tagger = dspy.ChainOfThought(TagContent)
```

### Handling "none of the above"

If real-world content might not fit any category, add an explicit catch-all rather than hoping the model picks the least-bad option:

```python
CATEGORIES = ["billing", "technical", "account", "feature_request", "other"]
```

This gives the model a safe escape hatch and makes it easy to filter out uncategorized items for human review.

### Sorting with context

Sometimes classification depends on extra context — a customer's plan tier, previous interactions, or business rules. Add those as input fields:

```python
class SortWithContext(dspy.Signature):
    """Sort the ticket considering the customer's context."""
    message: str = dspy.InputField(desc="The support message")
    customer_tier: str = dspy.InputField(desc="Customer plan: free, pro, or enterprise")
    category: Literal[tuple(CATEGORIES)] = dspy.OutputField()
    priority: Literal["low", "medium", "high", "urgent"] = dspy.OutputField()
```

## Step 3: Load your data

If the user has labeled data, help them load it. The key step is converting their data into `dspy.Example` objects and marking which fields are inputs (what the model sees) vs outputs (what it should predict).

### From a CSV or DataFrame

```python
import pandas as pd

df = pd.read_csv("labeled_tickets.csv")  # columns: message, category

dataset = [
    dspy.Example(message=row["message"], category=row["category"]).with_inputs("message")
    for _, row in df.iterrows()
]

# Split into train/dev sets
trainset, devset = dataset[:len(dataset)*4//5], dataset[len(dataset)*4//5:]
```

### From a list of dicts

```python
data = [
    {"message": "I was charged twice", "category": "billing"},
    {"message": "Can't log in", "category": "technical"},
    # ...
]

dataset = [dspy.Example(**d).with_inputs("message") for d in data]
```

### From transcripts (VTT, LiveKit, Recall)

Transcripts are a common source for sorting — classifying call topics, tagging meeting segments, routing conversations. The key is extracting the text content from whatever format you have.

**WebVTT (.vtt) files:**

```python
import re

def load_vtt(path):
    """Extract text lines from a VTT transcript, stripping timestamps."""
    text = open(path).read()
    # Remove VTT header and timestamp lines
    lines = [line.strip() for line in text.split("\n")
             if line.strip() and not line.startswith("WEBVTT")
             and not re.match(r"\d{2}:\d{2}", line)
             and not line.strip().isdigit()]
    return " ".join(lines)

# Sort entire transcripts by topic
transcript = load_vtt("meeting.vtt")
dataset = [dspy.Example(message=transcript, category="standup").with_inputs("message")]
```

**LiveKit transcripts** (from LiveKit Agents egress or webhook data):

```python
import json

def load_livekit_transcript(path):
    """Extract text from a LiveKit transcript JSON export."""
    data = json.load(open(path))
    # LiveKit transcription segments have text + timestamps
    segments = data.get("segments", data.get("results", []))
    return " ".join(seg.get("text", "") for seg in segments)

transcript = load_livekit_transcript("call_transcript.json")
```

**Recall.ai transcripts:**

```python
def load_recall_transcript(transcript_data):
    """Extract text from a Recall.ai transcript response.
    transcript_data is the JSON from Recall's /transcript endpoint."""
    return " ".join(
        entry["words"]
        for entry in transcript_data
        if entry.get("words")
    )
```

**Sorting transcript segments** — often you want to classify individual segments rather than whole transcripts (e.g., tag each speaker turn by topic):

```python
def vtt_to_segments(path):
    """Parse VTT into individual segments for per-segment sorting."""
    import webvtt  # pip install webvtt-py
    return [
        dspy.Example(message=caption.text, category="").with_inputs("message")
        for caption in webvtt.read(path)
        if caption.text.strip()
    ]
```

### From Langfuse traces

If you're sorting AI interactions logged in Langfuse — classifying traces by quality, topic, failure mode, etc.:

```python
from langfuse import Langfuse

langfuse = Langfuse()

# Fetch traces to classify
traces = langfuse.fetch_traces(limit=200).data

dataset = [
    dspy.Example(
        message=trace.input.get("message", str(trace.input)),
        # If traces are already scored/tagged in Langfuse, use that as the label
        category=trace.tags[0] if trace.tags else ""
    ).with_inputs("message")
    for trace in traces
    if trace.input
]

# Filter out unlabeled ones for training, keep them for batch classification
labeled = [ex for ex in dataset if ex.category]
unlabeled = [ex for ex in dataset if not ex.category]
```

### No labeled data yet

If the user doesn't have labeled examples, they have two options:

1. **Label a small set by hand** — even 20-30 examples helps. Suggest they pick representative examples from each category.
2. **Use `/ai-generating-data`** — generate synthetic training data from category descriptions.

## Step 4: Evaluate quality

Before optimizing, measure how the baseline performs:

```python
from dspy.evaluate import Evaluate

def sorting_metric(example, prediction, trace=None):
    return prediction.category == example.category

evaluator = Evaluate(
    devset=devset,
    metric=sorting_metric,
    num_threads=4,
    display_progress=True,
    display_table=5,  # show 5 example results
)
score = evaluator(sorter)
print(f"Baseline accuracy: {score}%")
```

### Multi-label metric

For multi-tag classification, exact match is too strict. Use Jaccard similarity (intersection over union):

```python
def multilabel_metric(example, pred, trace=None):
    gold = set(example.tags)
    predicted = set(pred.tags)
    if not gold and not predicted:
        return 1.0
    return len(gold & predicted) / len(gold | predicted)
```

## Step 5: Optimize accuracy

Start with `BootstrapFewShot` — it's fast and typically gives a meaningful accuracy bump by finding good few-shot examples from your training data:

```python
optimizer = dspy.BootstrapFewShot(
    metric=sorting_metric,
    max_bootstrapped_demos=4,
)
optimized_sorter = optimizer.compile(sorter, trainset=trainset)

# Re-evaluate
score = evaluator(optimized_sorter)
print(f"Optimized accuracy: {score}%")
```

If that's not enough, upgrade to `MIPROv2` which also optimizes the instructions:

```python
optimizer = dspy.MIPROv2(
    metric=sorting_metric,
    auto="medium",  # "light", "medium", or "heavy"
)
optimized_sorter = optimizer.compile(sorter, trainset=trainset)
```

### Training hints for tricky examples

If certain examples are ambiguous ("I want to cancel" — is that billing or account?), add a `hint` field that's only present during training:

```python
class SortWithHint(dspy.Signature):
    """Sort the message into the correct category."""
    message: str = dspy.InputField()
    hint: str = dspy.InputField(desc="Clarifying context for ambiguous cases")
    category: Literal[tuple(CATEGORIES)] = dspy.OutputField()

# In training data, provide hints
trainset = [
    dspy.Example(
        message="I want to cancel",
        hint="Customer is asking about canceling their subscription billing",
        category="billing"
    ).with_inputs("message", "hint"),
]
# At inference time, pass hint="" or omit it
```

## Step 6: Use it

### Single item

```python
result = optimized_sorter(message="I was charged twice on my credit card last month")
print(f"Category: {result.category}")
print(f"Reasoning: {result.reasoning}")
```

### Batch processing

For sorting many items at once, use `dspy.Evaluate` with your data or a simple loop. The evaluator handles threading automatically:

```python
# Quick batch with a loop
results = []
for item in items:
    result = optimized_sorter(message=item["text"])
    results.append({"text": item["text"], "category": result.category})

# Or use pandas
df["category"] = df["message"].apply(
    lambda msg: optimized_sorter(message=msg).category
)
```

### Confidence-based routing

When you need to know how sure the model is — for example, to escalate low-confidence items to a human:

```python
class SortWithConfidence(dspy.Signature):
    """Sort the content and rate your confidence."""
    message: str = dspy.InputField()
    category: Literal[tuple(CATEGORIES)] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="Confidence between 0.0 and 1.0")

sorter = dspy.ChainOfThought(SortWithConfidence)
result = sorter(message="I think there might be an issue")

if result.confidence < 0.7:
    # Flag for human review
    print(f"Low confidence ({result.confidence}) — needs human review")
else:
    print(f"Category: {result.category} (confidence: {result.confidence})")
```

### Save and load

Persist your optimized sorter so you don't have to re-optimize every time:

```python
# Save
optimized_sorter.save("ticket_sorter.json")

# Load later
sorter = dspy.ChainOfThought(SortContent)
sorter.load("ticket_sorter.json")
```

## Additional resources

- For worked examples (sentiment, intent routing, topics, hierarchical), see [examples.md](examples.md)
- Need scores instead of categories? Use `/ai-scoring`
- Want to measure and improve further? Use `/ai-improving-accuracy`
- Need to generate training data? Use `/ai-generating-data`
