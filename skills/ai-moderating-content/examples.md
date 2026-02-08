# Content Moderation Examples

## Example 1: Community forum moderation

A community forum needs to auto-moderate user comments. Categories: safe, spam, toxic, off-topic. Human reviewers handle uncertain cases.

### Set up the moderator

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

class ModerateComment(dspy.Signature):
    """Classify a forum comment for moderation."""
    comment: str = dspy.InputField(desc="user comment to moderate")
    thread_topic: str = dspy.InputField(desc="what the thread is about")
    category: Literal["safe", "spam", "toxic", "off_topic"] = dspy.OutputField()
    severity: Literal["none", "low", "medium", "high"] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="0.0 to 1.0")
    explanation: str = dspy.OutputField(desc="brief reason")

class ForumModerator(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought(ModerateComment)

    def forward(self, comment, thread_topic="general discussion"):
        result = self.assess(comment=comment, thread_topic=thread_topic)

        # Confidence-based routing
        if result.confidence < 0.7:
            decision = "human_review"
        elif result.severity == "high":
            decision = "remove"
        elif result.severity == "medium":
            decision = "human_review"
        elif result.severity == "low":
            decision = "warn"
        else:
            decision = "approve"

        return dspy.Prediction(
            category=result.category,
            severity=result.severity,
            confidence=result.confidence,
            decision=decision,
            explanation=result.explanation,
        )
```

### Prepare labeled training data

```python
trainset = [
    dspy.Example(
        comment="Has anyone tried the new update? It fixed the crash I was having.",
        thread_topic="software updates",
        category="safe",
        severity="none",
    ).with_inputs("comment", "thread_topic"),
    dspy.Example(
        comment="BUY CHEAP WATCHES AT www.spam-site.com BEST PRICES!!!",
        thread_topic="software updates",
        category="spam",
        severity="medium",
    ).with_inputs("comment", "thread_topic"),
    dspy.Example(
        comment="Anyone who uses this software is a complete moron",
        thread_topic="software updates",
        category="toxic",
        severity="high",
    ).with_inputs("comment", "thread_topic"),
    dspy.Example(
        comment="Hey does anyone know a good recipe for banana bread?",
        thread_topic="software updates",
        category="off_topic",
        severity="low",
    ).with_inputs("comment", "thread_topic"),
    dspy.Example(
        comment="I disagree with the previous poster. The old version was more stable.",
        thread_topic="software updates",
        category="safe",
        severity="none",
    ).with_inputs("comment", "thread_topic"),
    dspy.Example(
        comment="Check out my profile for amazing deals on electronics!",
        thread_topic="software updates",
        category="spam",
        severity="low",
    ).with_inputs("comment", "thread_topic"),
    dspy.Example(
        comment="You're all idiots if you think this feature is good",
        thread_topic="software updates",
        category="toxic",
        severity="medium",
    ).with_inputs("comment", "thread_topic"),
    dspy.Example(
        comment="This is somewhat related but has anyone compared it to CompetitorApp?",
        thread_topic="software updates",
        category="safe",
        severity="none",
    ).with_inputs("comment", "thread_topic"),
    # ... 200 labeled examples total for production quality
]

# Split into train/dev
split = int(len(trainset) * 0.8)
train, dev = trainset[:split], trainset[split:]
```

### Evaluate baseline and optimize

```python
from dspy.evaluate import Evaluate

def moderation_metric(example, prediction, trace=None):
    category_correct = float(prediction.category == example.category)
    severity_correct = float(prediction.severity == example.severity)
    return 0.7 * category_correct + 0.3 * severity_correct

evaluator = Evaluate(devset=dev, metric=moderation_metric, num_threads=4, display_table=5)

moderator = ForumModerator()
baseline = evaluator(moderator)
print(f"Baseline: {baseline:.1f}%")
# Output: Baseline: 72.0%

# Optimize
optimizer = dspy.MIPROv2(metric=moderation_metric, auto="medium")
optimized = optimizer.compile(moderator, trainset=train)

optimized_score = evaluator(optimized)
print(f"Optimized: {optimized_score:.1f}%")
# Output: Optimized: 89.0%
```

### Check per-category performance

```python
for category in ["safe", "spam", "toxic", "off_topic"]:
    cat_examples = [e for e in dev if e.category == category]
    if cat_examples:
        cat_evaluator = Evaluate(devset=cat_examples, metric=moderation_metric, num_threads=4)
        score = cat_evaluator(optimized)
        print(f"  {category}: {score:.1f}%")

# Output:
#   safe: 95.0%
#   spam: 88.0%
#   toxic: 85.0%
#   off_topic: 78.0%
```

### Deploy with confidence routing

```python
# In production, track routing distribution
stats = {"approve": 0, "warn": 0, "human_review": 0, "remove": 0}

for comment_text in incoming_comments:
    result = optimized(comment=comment_text, thread_topic=thread_topic)
    stats[result.decision] += 1

    if result.decision == "remove":
        hide_comment(comment_text)
    elif result.decision == "human_review":
        queue_for_review(comment_text, result.explanation)
    elif result.decision == "warn":
        add_warning_label(comment_text)

print(f"Routing: {stats}")
# Typical: approve 75%, warn 10%, human_review 10%, remove 5%
```

---

## Example 2: Marketplace listing moderation

An online marketplace needs to moderate product listings for prohibited items, misleading claims, and PII in descriptions. Listings can violate multiple policies at once.

### Set up multi-label moderation with hard blocks

```python
import dspy
import re
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

LISTING_VIOLATIONS = [
    "clean", "prohibited_item", "misleading_claims",
    "counterfeit", "pii_exposed", "inappropriate_images_described",
]

class ModerateListing(dspy.Signature):
    """Moderate a marketplace product listing. Flag all policy violations."""
    title: str = dspy.InputField(desc="product listing title")
    description: str = dspy.InputField(desc="product listing description")
    price: str = dspy.InputField(desc="listed price")
    violations: list[str] = dspy.OutputField(desc=f"all that apply from: {LISTING_VIOLATIONS}")
    severity: Literal["none", "low", "medium", "high"] = dspy.OutputField()
    explanation: str = dspy.OutputField()

class ListingModerator(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought(ModerateListing)

    def forward(self, title, description, price):
        full_text = f"{title} {description}"

        # Hard blocks: PII patterns (instant, no LM needed)
        dspy.Assert(
            not re.search(r"\b\d{3}-\d{2}-\d{4}\b", full_text),
            "Listing contains SSN — auto-reject and notify seller",
        )
        dspy.Assert(
            not re.search(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                full_text,
            ),
            "Listing contains email — ask seller to remove before publishing",
        )
        dspy.Assert(
            not re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", full_text),
            "Listing contains phone number — ask seller to remove",
        )

        # LM-based assessment
        result = self.assess(
            title=title,
            description=description,
            price=price,
        )

        # Validate violations are from allowed set
        dspy.Assert(
            all(v in LISTING_VIOLATIONS for v in result.violations),
            f"Violations must be from: {LISTING_VIOLATIONS}",
        )

        # Route
        if result.severity == "high" or "prohibited_item" in result.violations:
            decision = "reject"
        elif result.severity == "medium" or len(result.violations) > 1:
            decision = "human_review"
        elif result.severity == "low":
            decision = "request_edit"
        else:
            decision = "approve"

        return dspy.Prediction(
            violations=result.violations,
            severity=result.severity,
            decision=decision,
            explanation=result.explanation,
        )
```

### Training data with multi-label examples

```python
trainset = [
    dspy.Example(
        title="Vintage Leather Jacket - Size M",
        description="Genuine leather, barely worn. Great condition. Smoke-free home.",
        price="$85",
        violations=["clean"],
        severity="none",
    ).with_inputs("title", "description", "price"),
    dspy.Example(
        title="GUARANTEED Weight Loss Pills - Lose 30lbs in 1 Week!",
        description="Doctor-recommended miracle supplement. 100% guaranteed results or your money back. FDA approved.",
        price="$29.99",
        violations=["misleading_claims"],
        severity="high",
    ).with_inputs("title", "description", "price"),
    dspy.Example(
        title="Designer Handbag - Looks Just Like Gucci",
        description="High quality replica. Indistinguishable from the real thing. Same materials and craftsmanship.",
        price="$45",
        violations=["counterfeit"],
        severity="high",
    ).with_inputs("title", "description", "price"),
    dspy.Example(
        title="Used Textbook - Calculus 101",
        description="Some highlighting. Contact me at seller@email.com for bundle deals. Call 555-123-4567.",
        price="$30",
        violations=["pii_exposed"],
        severity="medium",
    ).with_inputs("title", "description", "price"),
    dspy.Example(
        title="AMAZING Deal Electronics - Best Price EVER!!!",
        description="Buy now before they're gone! Limited stock! We beat ANY price! Not sold in stores!",
        price="$9.99",
        violations=["misleading_claims"],
        severity="low",
    ).with_inputs("title", "description", "price"),
    dspy.Example(
        title="Replica Rolex + Weight Loss Combo",
        description="Get a luxury watch AND lose weight! Both guaranteed authentic and effective.",
        price="$99",
        violations=["counterfeit", "misleading_claims"],
        severity="high",
    ).with_inputs("title", "description", "price"),
    # ... 100+ examples for production
]
```

### Evaluate and optimize

```python
from dspy.evaluate import Evaluate

def listing_metric(example, prediction, trace=None):
    """Multi-label metric: check violation overlap and severity."""
    expected = set(example.violations)
    predicted = set(prediction.violations)

    if not expected and not predicted:
        violation_score = 1.0
    elif not expected or not predicted:
        violation_score = 0.0
    else:
        intersection = expected & predicted
        union = expected | predicted
        violation_score = len(intersection) / len(union)  # Jaccard similarity

    severity_score = float(prediction.severity == example.severity)
    return 0.6 * violation_score + 0.4 * severity_score

split = int(len(trainset) * 0.8)
train, dev = trainset[:split], trainset[split:]

evaluator = Evaluate(devset=dev, metric=listing_metric, num_threads=4, display_table=5)

moderator = ListingModerator()
baseline = evaluator(moderator)
print(f"Baseline: {baseline:.1f}%")

optimizer = dspy.MIPROv2(metric=listing_metric, auto="medium")
optimized = optimizer.compile(moderator, trainset=train)

optimized_score = evaluator(optimized)
print(f"Optimized: {optimized_score:.1f}%")

# Save for production
optimized.save("listing_moderator.json")
```

### Production integration

```python
# Process incoming listings
moderator = ListingModerator()
moderator.load("listing_moderator.json")

def moderate_new_listing(listing):
    try:
        result = moderator(
            title=listing["title"],
            description=listing["description"],
            price=listing["price"],
        )
    except dspy.primitives.assertions.DSPyAssertionError as e:
        # Hard block triggered (PII pattern match)
        return {
            "decision": "reject",
            "reason": str(e),
            "action": "notify_seller_to_remove_pii",
        }

    return {
        "decision": result.decision,
        "violations": result.violations,
        "severity": result.severity,
        "explanation": result.explanation,
    }
```

**Result:** The multi-label moderator catches listings that violate multiple policies simultaneously (e.g., counterfeit + misleading claims), while pattern-based hard blocks instantly catch PII before the LM even runs. Confidence-based routing sends ~12% of listings to human reviewers — the ones where the moderator is least certain.
