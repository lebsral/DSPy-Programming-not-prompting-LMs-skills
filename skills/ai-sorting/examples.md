# Sorting Examples

## Sentiment Analysis

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

SENTIMENTS = ["positive", "negative", "neutral"]

class SortSentiment(dspy.Signature):
    """Sort the product review by sentiment."""
    review: str = dspy.InputField(desc="Product review text")
    sentiment: Literal[tuple(SENTIMENTS)] = dspy.OutputField(desc="Sentiment of the review")

sorter = dspy.ChainOfThought(SortSentiment)

# Example usage
result = sorter(review="The battery life is amazing but the screen is too dim.")
print(f"Sentiment: {result.sentiment}")
print(f"Reasoning: {result.reasoning}")

# Training data
trainset = [
    dspy.Example(review="Love this product!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Broke after one week.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It works as expected.", sentiment="neutral").with_inputs("review"),
    # Add 20-50+ examples for optimization
]

# Optimize
def metric(example, pred, trace=None):
    return pred.sentiment == example.sentiment

optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(sorter, trainset=trainset)
```

## Support Ticket Routing

```python
INTENTS = [
    "billing_question",
    "technical_support",
    "account_management",
    "feature_request",
    "general_inquiry",
]

class RouteTicket(dspy.Signature):
    """Route the support ticket to the right team."""
    message: str = dspy.InputField(desc="Customer support message")
    team: Literal[tuple(INTENTS)] = dspy.OutputField(desc="Team to handle this ticket")

router = dspy.ChainOfThought(RouteTicket)

result = router(message="I was charged twice for my subscription last month")
print(f"Route to: {result.team}")  # billing_question
```

## Multi-Tag Topic Classification

```python
TOPICS = ["politics", "technology", "sports", "entertainment", "science", "business"]

class TagTopics(dspy.Signature):
    """Assign all relevant topic tags to a news article."""
    article: str = dspy.InputField(desc="News article text")
    topics: list[Literal[tuple(TOPICS)]] = dspy.OutputField(desc="All relevant topics")

tagger = dspy.ChainOfThought(TagTopics)

result = tagger(
    article="The tech company announced a new AI chip, causing its stock to surge 15%."
)
print(f"Topics: {result.topics}")  # ["technology", "business"]

# Multi-label metric (Jaccard similarity)
def multilabel_metric(example, pred, trace=None):
    gold = set(example.topics)
    predicted = set(pred.topics)
    if not gold and not predicted:
        return 1.0
    return len(gold & predicted) / len(gold | predicted)
```

## Sorting with Confidence Scores

```python
class SortWithConfidence(dspy.Signature):
    """Sort the content and rate your confidence."""
    text: str = dspy.InputField()
    category: Literal[tuple(CATEGORIES)] = dspy.OutputField(desc="Category label")
    confidence: float = dspy.OutputField(desc="Confidence between 0.0 and 1.0")

sorter = dspy.ChainOfThought(SortWithConfidence)
result = sorter(text="I think this might be okay")
print(f"Category: {result.category}, Confidence: {result.confidence}")
```

## Hierarchical Sorting (Category + Subcategory)

```python
class SortProduct(dspy.Signature):
    """Sort the product into category and subcategory."""
    product_description: str = dspy.InputField()
    category: Literal["electronics", "clothing", "food", "home"] = dspy.OutputField()
    subcategory: str = dspy.OutputField(desc="Specific subcategory within the main category")

sorter = dspy.ChainOfThought(SortProduct)
result = sorter(product_description="Wireless noise-cancelling headphones with 30hr battery")
print(f"Category: {result.category}, Subcategory: {result.subcategory}")
```
