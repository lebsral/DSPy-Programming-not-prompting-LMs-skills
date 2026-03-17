# dspy-data Examples

Three complete, copy-paste-ready examples for working with DSPy data.

## Example 1: Load from a CSV file

Load a CSV of customer support tickets, create DSPy Examples, and split into train/dev sets.

**Assumes a CSV file `tickets.csv` with columns: `message`, `category`, `priority`**

```python
import dspy
import csv
import random

# --- Load CSV into dspy.Example objects ---

def load_csv_as_examples(filepath, input_fields):
    """Load a CSV file into a list of dspy.Example objects."""
    examples = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Skip rows with missing required fields
            if all(row.get(field) for field in reader.fieldnames):
                ex = dspy.Example(**row).with_inputs(*input_fields)
                examples.append(ex)
    return examples

all_examples = load_csv_as_examples("tickets.csv", input_fields=["message"])
print(f"Loaded {len(all_examples)} examples")
print(f"Fields: {all_examples[0].keys()}")
print(f"First example: {all_examples[0]}")

# --- Split into train and dev sets ---

random.seed(42)
random.shuffle(all_examples)
split_idx = int(len(all_examples) * 0.8)
trainset = all_examples[:split_idx]
devset = all_examples[split_idx:]
print(f"Train: {len(trainset)}, Dev: {len(devset)}")

# --- Check label distribution ---

from collections import Counter
train_labels = Counter(ex.category for ex in trainset)
dev_labels = Counter(ex.category for ex in devset)
print(f"Train label distribution: {train_labels}")
print(f"Dev label distribution:   {dev_labels}")

# --- Use with an optimizer ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

from typing import Literal

CATEGORIES = list(set(ex.category for ex in trainset))

class ClassifyTicket(dspy.Signature):
    """Classify the support ticket into the correct category."""
    message: str = dspy.InputField(desc="The customer support message")
    category: Literal[tuple(CATEGORIES)] = dspy.OutputField(desc="The ticket category")

classifier = dspy.ChainOfThought(ClassifyTicket)

def metric(example, prediction, trace=None):
    return prediction.category == example.category

optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(classifier, trainset=trainset)

# Evaluate on dev set
from dspy.evaluate import Evaluate
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(optimized)
print(f"Dev accuracy: {score}%")
```

## Example 2: Load from HuggingFace

Load the Stanford Sentiment Treebank (SST-2) dataset from HuggingFace and convert to DSPy Examples for a sentiment classification task.

```python
import dspy
from datasets import load_dataset

# --- Load the dataset ---

dataset = load_dataset("glue", "sst2")

# --- Convert to DSPy Examples ---
# SST-2 has "sentence" and "label" (0=negative, 1=positive)

def convert_sst2(split, max_examples=None):
    """Convert HuggingFace SST-2 split to DSPy Examples."""
    items = list(split)
    if max_examples:
        items = items[:max_examples]
    return [
        dspy.Example(
            text=x["sentence"],
            label="positive" if x["label"] == 1 else "negative",
        ).with_inputs("text")
        for x in items
    ]

# Use a subset for speed — DSPy optimizers don't need huge datasets
trainset = convert_sst2(dataset["train"], max_examples=200)
devset = convert_sst2(dataset["validation"], max_examples=100)

print(f"Train: {len(trainset)}, Dev: {len(devset)}")
print(f"First train example: {trainset[0]}")

# --- Check distribution ---

from collections import Counter
print(f"Train labels: {Counter(ex.label for ex in trainset)}")
print(f"Dev labels:   {Counter(ex.label for ex in devset)}")

# --- Build a classifier and optimize ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

from typing import Literal

class SentimentClassify(dspy.Signature):
    """Classify the sentiment of the text."""
    text: str = dspy.InputField(desc="The text to classify")
    label: Literal["positive", "negative"] = dspy.OutputField(desc="The sentiment")

classifier = dspy.ChainOfThought(SentimentClassify)

def metric(example, prediction, trace=None):
    return prediction.label == example.label

# Quick optimization with BootstrapFewShot
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(classifier, trainset=trainset)

# Evaluate
from dspy.evaluate import Evaluate
score = Evaluate(devset=devset, metric=metric, num_threads=4)(optimized)
print(f"Dev accuracy: {score}%")
```

## Example 3: Manual creation for a classification task

Hand-craft examples when you're prototyping or have a small number of known cases. Demonstrates `with_inputs()`, `.inputs()`, and `.labels()`.

```python
import dspy

# --- Hand-craft examples ---

examples = [
    dspy.Example(
        email="Hi, I can't log into my account. I've tried resetting my password twice.",
        intent="account_access",
        urgency="high",
    ),
    dspy.Example(
        email="When is the next billing cycle? I want to upgrade my plan.",
        intent="billing",
        urgency="low",
    ),
    dspy.Example(
        email="Your API is returning 500 errors on the /users endpoint since 3am.",
        intent="bug_report",
        urgency="critical",
    ),
    dspy.Example(
        email="Could you add dark mode? Would really help with late night coding.",
        intent="feature_request",
        urgency="low",
    ),
    dspy.Example(
        email="I was charged twice for my subscription this month.",
        intent="billing",
        urgency="high",
    ),
    dspy.Example(
        email="The dashboard loads really slowly, takes over 30 seconds.",
        intent="bug_report",
        urgency="medium",
    ),
    dspy.Example(
        email="Thanks for the quick fix on that export bug!",
        intent="feedback",
        urgency="low",
    ),
    dspy.Example(
        email="URGENT: Our entire team is locked out of the platform right now.",
        intent="account_access",
        urgency="critical",
    ),
    dspy.Example(
        email="Is there documentation for the new webhooks feature?",
        intent="question",
        urgency="low",
    ),
    dspy.Example(
        email="We need SSO integration before we can renew our enterprise contract.",
        intent="feature_request",
        urgency="high",
    ),
]

# --- Mark input fields ---
# "email" is the input; "intent" and "urgency" are the expected outputs

examples = [ex.with_inputs("email") for ex in examples]

# --- Verify input/output split ---

first = examples[0]
print("Full example:", first)
print("Inputs only: ", first.inputs())   # Example(email="Hi, I can't log into...")
print("Labels only: ", first.labels())   # Example(intent="account_access", urgency="high")

# --- Split into train and dev ---

trainset = examples[:8]
devset = examples[8:]
print(f"\nTrain: {len(trainset)}, Dev: {len(devset)}")

# --- Build and test a classifier ---

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

from typing import Literal

INTENTS = ["account_access", "billing", "bug_report", "feature_request", "feedback", "question"]
URGENCY_LEVELS = ["low", "medium", "high", "critical"]

class ClassifyEmail(dspy.Signature):
    """Classify the customer email by intent and urgency level."""
    email: str = dspy.InputField(desc="The customer email to classify")
    intent: Literal[tuple(INTENTS)] = dspy.OutputField(desc="The primary intent")
    urgency: Literal[tuple(URGENCY_LEVELS)] = dspy.OutputField(desc="The urgency level")

classifier = dspy.ChainOfThought(ClassifyEmail)

# Test on one example
result = classifier(email=devset[0].email)
print(f"\nPredicted intent: {result.intent}, urgency: {result.urgency}")
print(f"Expected  intent: {devset[0].intent}, urgency: {devset[0].urgency}")

# --- Metric that checks both fields ---

def metric(example, prediction, trace=None):
    intent_correct = prediction.intent == example.intent
    urgency_correct = prediction.urgency == example.urgency
    # Score: 1.0 if both right, 0.5 if one right, 0.0 if neither
    return (intent_correct + urgency_correct) / 2.0

# Optimize
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
optimized = optimizer.compile(classifier, trainset=trainset)

# Evaluate
from dspy.evaluate import Evaluate
score = Evaluate(devset=devset, metric=metric, num_threads=1)(optimized)
print(f"\nDev score: {score}")
```
