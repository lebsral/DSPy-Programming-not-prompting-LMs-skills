# Sorting Examples

## Sentiment Analysis

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

SENTIMENTS = ["positive", "negative", "neutral", "mixed"]

class SortSentiment(dspy.Signature):
    """Classify the sentiment of a product review."""
    review: str = dspy.InputField(desc="Product review text")
    sentiment: Literal[tuple(SENTIMENTS)] = dspy.OutputField(desc="Overall sentiment")

sorter = dspy.ChainOfThought(SortSentiment)

result = sorter(review="The battery life is amazing but the screen is too dim.")
print(f"Sentiment: {result.sentiment}")  # mixed
print(f"Reasoning: {result.reasoning}")

# Training data
trainset = [
    dspy.Example(review="Love this product!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Broke after one week.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It works as expected.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="Great camera but terrible battery.", sentiment="mixed").with_inputs("review"),
    # Add 20-50+ examples for optimization
]
```

## Hierarchical Sorting (Category + Subcategory)

When items need both a broad category and a specific subcategory:

```python
DEPARTMENTS = ["electronics", "clothing", "home", "food"]

class SortProduct(dspy.Signature):
    """Sort the product into a department and specific subcategory."""
    product_description: str = dspy.InputField()
    department: Literal[tuple(DEPARTMENTS)] = dspy.OutputField(desc="Broad department")
    subcategory: str = dspy.OutputField(desc="Specific subcategory within the department")

sorter = dspy.ChainOfThought(SortProduct)
result = sorter(product_description="Wireless noise-cancelling headphones with 30hr battery")
print(f"{result.department} > {result.subcategory}")  # electronics > headphones
```

Note: `subcategory` is a free-form `str` here because subcategories often differ per department. If your subcategories are fixed, use `Literal` for those too.

## Priority Triage with Urgency Detection

Sorting isn't always about topic — sometimes you need to assess urgency:

```python
class TriageTicket(dspy.Signature):
    """Assess the urgency of this support ticket and route it."""
    message: str = dspy.InputField(desc="Customer support message")
    department: Literal["billing", "technical", "account", "security"] = dspy.OutputField()
    urgency: Literal["low", "medium", "high", "critical"] = dspy.OutputField()

triager = dspy.ChainOfThought(TriageTicket)

# "Critical" should trigger for security issues, data loss, outages
result = triager(message="I think someone accessed my account — I see logins from a country I've never been to")
print(f"Route to: {result.department}, Urgency: {result.urgency}")
# department: security, urgency: critical
```

## End-to-End: From CSV to Optimized Sorter

A complete workflow showing data loading, evaluation, optimization, and saving:

```python
import dspy
import pandas as pd
from typing import Literal
from dspy.evaluate import Evaluate

# Setup
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Load data
df = pd.read_csv("support_tickets.csv")  # columns: message, category
CATEGORIES = df["category"].unique().tolist()

dataset = [
    dspy.Example(message=row["message"], category=row["category"]).with_inputs("message")
    for _, row in df.iterrows()
]
trainset, devset = dataset[:len(dataset)*4//5], dataset[len(dataset)*4//5:]

# 2. Define sorter
class SortTicket(dspy.Signature):
    """Route the support ticket to the correct team."""
    message: str = dspy.InputField(desc="Customer support message")
    category: Literal[tuple(CATEGORIES)] = dspy.OutputField(desc="Support category")

sorter = dspy.ChainOfThought(SortTicket)

# 3. Baseline evaluation
def metric(example, pred, trace=None):
    return pred.category == example.category

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
baseline = evaluator(sorter)
print(f"Baseline: {baseline}%")

# 4. Optimize
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(sorter, trainset=trainset)

improved = evaluator(optimized)
print(f"Optimized: {improved}%")

# 5. Save for production
optimized.save("ticket_sorter.json")
```

## Multi-Stage: Sort Then Act

Sorting is often just the first step. Here's a pattern where classification drives downstream behavior:

```python
class SortIntent(dspy.Signature):
    """Identify the customer's intent."""
    message: str = dspy.InputField()
    intent: Literal["question", "complaint", "praise", "request"] = dspy.OutputField()

class GenerateResponse(dspy.Signature):
    """Write a response appropriate to the customer's intent."""
    message: str = dspy.InputField()
    intent: str = dspy.InputField()
    response: str = dspy.OutputField()

class SortAndRespond(dspy.Module):
    def __init__(self):
        self.sorter = dspy.ChainOfThought(SortIntent)
        self.responder = dspy.ChainOfThought(GenerateResponse)

    def forward(self, message):
        classification = self.sorter(message=message)
        return self.responder(message=message, intent=classification.intent)

pipeline = SortAndRespond()
result = pipeline(message="Your product ruined my entire project")
print(f"Intent: {result.intent}")
print(f"Response: {result.response}")
```
