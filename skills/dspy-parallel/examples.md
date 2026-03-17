# dspy-parallel Examples

## Example 1: Batch classification in parallel

Classify a batch of support tickets by urgency and category, all at once:

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class ClassifyTicket(dspy.Signature):
    """Classify a support ticket by urgency and category."""
    ticket: str = dspy.InputField(desc="Customer support ticket text")
    urgency: Literal["low", "medium", "high", "critical"] = dspy.OutputField()
    category: Literal["billing", "technical", "account", "feature_request", "other"] = dspy.OutputField()


# Simulated batch of support tickets
tickets = [
    "My account was charged twice for the same subscription. Please refund ASAP.",
    "Would be cool if you added dark mode to the dashboard.",
    "I can't log in. Password reset emails aren't arriving. I have a demo in 30 minutes!",
    "How do I export my data to CSV?",
    "Our production integration is down. API returns 500 errors on every request.",
    "Can you change the email address on my account?",
    "The mobile app crashes when I open the settings page.",
    "I'd like to upgrade from the free plan to the team plan.",
]

# Build execution pairs
classify = dspy.Predict(ClassifyTicket)
exec_pairs = [(classify, {"ticket": t}) for t in tickets]

# Run classification in parallel
parallel = dspy.Parallel(num_threads=4)
results = parallel(exec_pairs)

# Print results
print(f"{'Ticket':<70} {'Urgency':<10} {'Category'}")
print("-" * 100)
for ticket, result in zip(tickets, results):
    print(f"{ticket[:67]+'...' if len(ticket)>67 else ticket:<70} {result.urgency:<10} {result.category}")

# Route critical tickets
critical_tickets = [
    (ticket, result)
    for ticket, result in zip(tickets, results)
    if result.urgency == "critical"
]
print(f"\n{len(critical_tickets)} critical ticket(s) need immediate attention:")
for ticket, result in critical_tickets:
    print(f"  [{result.category}] {ticket[:80]}")
```

Expected output:

```
Ticket                                                                 Urgency    Category
----------------------------------------------------------------------------------------------------
My account was charged twice for the same subscription. Please refu... high       billing
Would be cool if you added dark mode to the dashboard.                 low        feature_request
I can't log in. Password reset emails aren't arriving. I have a de... critical   account
How do I export my data to CSV?                                        low        technical
Our production integration is down. API returns 500 errors on every... critical   technical
Can you change the email address on my account?                        low        account
The mobile app crashes when I open the settings page.                  medium     technical
I'd like to upgrade from the free plan to the team plan.               low        billing

2 critical ticket(s) need immediate attention:
  [account] I can't log in. Password reset emails aren't arriving. I have a demo in 30 minutes!
  [technical] Our production integration is down. API returns 500 errors on every request.
```

### With error handling

For production workloads, handle failures gracefully:

```python
parallel = dspy.Parallel(
    num_threads=4,
    max_errors=3,
    return_failed_examples=True,
    provide_traceback=True,
)

results, failed, errors = parallel(exec_pairs)

print(f"Classified: {len(results)}, Failed: {len(failed)}")

# Retry failures one at a time as a fallback
for (module, inputs), error in zip(failed, errors):
    print(f"Failed on: {inputs['ticket'][:50]}... Error: {error}")
    try:
        result = module(**inputs)
        results.append(result)
    except Exception:
        print("  Retry also failed, skipping.")
```

---

## Example 2: Parallel multi-aspect analysis

Analyze a single piece of text from multiple angles simultaneously -- sentiment, topics, and named entities -- then merge the results:

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)


class SentimentAnalysis(dspy.Signature):
    """Analyze the overall sentiment of the text."""
    text: str = dspy.InputField()
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="Confidence score from 0.0 to 1.0")
    explanation: str = dspy.OutputField(desc="Brief explanation of the sentiment")


class TopicExtraction(dspy.Signature):
    """Extract the main topics discussed in the text."""
    text: str = dspy.InputField()
    topics: list[str] = dspy.OutputField(desc="List of main topics, max 5")
    primary_topic: str = dspy.OutputField(desc="The single most prominent topic")


class EntityExtraction(dspy.Signature):
    """Extract named entities from the text."""
    text: str = dspy.InputField()
    people: list[str] = dspy.OutputField(desc="People mentioned")
    organizations: list[str] = dspy.OutputField(desc="Organizations mentioned")
    locations: list[str] = dspy.OutputField(desc="Locations mentioned")


# Three specialized modules
sentiment_module = dspy.Predict(SentimentAnalysis)
topic_module = dspy.Predict(TopicExtraction)
entity_module = dspy.Predict(EntityExtraction)

# Text to analyze
text = """
Apple announced its latest Vision Pro headset at WWDC in Cupertino yesterday.
CEO Tim Cook demonstrated the device's mixed reality capabilities to a packed audience.
While analysts from Goldman Sachs praised the innovation, consumer reviews on social
media were mixed -- many cited the $3,499 price tag as a significant barrier. Samsung
and Meta are expected to respond with competing products by Q2 2025.
"""

# Fan out: run all three analyses in parallel on the same text
exec_pairs = [
    (sentiment_module, {"text": text}),
    (topic_module, {"text": text}),
    (entity_module, {"text": text}),
]

parallel = dspy.Parallel(num_threads=3)
results = parallel(exec_pairs)

sentiment_result = results[0]
topic_result = results[1]
entity_result = results[2]

# Merge into a single analysis report
analysis = {
    "sentiment": {
        "label": sentiment_result.sentiment,
        "confidence": sentiment_result.confidence,
        "explanation": sentiment_result.explanation,
    },
    "topics": {
        "all": topic_result.topics,
        "primary": topic_result.primary_topic,
    },
    "entities": {
        "people": entity_result.people,
        "organizations": entity_result.organizations,
        "locations": entity_result.locations,
    },
}

print("=== Multi-Aspect Analysis ===\n")
print(f"Sentiment: {analysis['sentiment']['label']} "
      f"(confidence: {analysis['sentiment']['confidence']:.0%})")
print(f"  {analysis['sentiment']['explanation']}\n")
print(f"Topics: {', '.join(analysis['topics']['all'])}")
print(f"  Primary: {analysis['topics']['primary']}\n")
print(f"People: {', '.join(analysis['entities']['people'])}")
print(f"Organizations: {', '.join(analysis['entities']['organizations'])}")
print(f"Locations: {', '.join(analysis['entities']['locations'])}")
```

Expected output:

```
=== Multi-Aspect Analysis ===

Sentiment: mixed (confidence: 85%)
  Analysts praised the innovation but consumers criticized the high price

Topics: mixed reality, Vision Pro, consumer pricing, competition, WWDC
  Primary: Vision Pro

People: Tim Cook
Organizations: Apple, Goldman Sachs, Samsung, Meta
Locations: Cupertino
```

### Wrapping it in a reusable module

For cleaner code, wrap the fan-out pattern in a `dspy.Module`:

```python
class MultiAspectAnalyzer(dspy.Module):
    def __init__(self, num_threads=3):
        self.sentiment = dspy.Predict(SentimentAnalysis)
        self.topics = dspy.Predict(TopicExtraction)
        self.entities = dspy.Predict(EntityExtraction)
        self.num_threads = num_threads

    def forward(self, text: str):
        parallel = dspy.Parallel(num_threads=self.num_threads)
        results = parallel([
            (self.sentiment, {"text": text}),
            (self.topics, {"text": text}),
            (self.entities, {"text": text}),
        ])

        return dspy.Prediction(
            sentiment=results[0].sentiment,
            confidence=results[0].confidence,
            topics=results[1].topics,
            primary_topic=results[1].primary_topic,
            people=results[2].people,
            organizations=results[2].organizations,
            locations=results[2].locations,
        )


# Clean single-call interface
analyzer = MultiAspectAnalyzer()
result = analyzer(text=text)
print(result.sentiment, result.topics, result.people)
```

### Scaling to a batch of documents

Combine both patterns -- parallelize across documents, with each document getting multi-aspect analysis:

```python
documents = [text_1, text_2, text_3, ...]  # many documents

analyzer = MultiAspectAnalyzer(num_threads=3)

# Outer parallel: process documents concurrently
# Each call to analyzer internally fans out 3 modules in parallel
outer_parallel = dspy.Parallel(num_threads=4)
exec_pairs = [(analyzer, {"text": doc}) for doc in documents]
all_results = outer_parallel(exec_pairs)

for doc, result in zip(documents, all_results):
    print(f"Doc: {doc[:50]}... -> {result.sentiment}, {result.primary_topic}")
```

Note: the inner `Parallel` (3 threads per document) and outer `Parallel` (4 documents at once) combine for up to 12 concurrent LM calls. Make sure your provider rate limits can handle this.
