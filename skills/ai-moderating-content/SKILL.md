---
name: ai-moderating-content
description: Auto-moderate what users post on your platform. Use when you need content moderation, flag harmful comments, detect spam, filter hate speech, catch NSFW content, block harassment, moderate user-generated content, review community posts, filter marketplace listings, or route bad content to human reviewers. Covers DSPy classification with severity scoring, confidence-based routing, and Assert-based policy enforcement.
---

# Auto-Moderate What Users Post

Guide the user through building AI content moderation — classify user-generated content, score severity, and route decisions (auto-approve, human-review, auto-reject). The pattern: classify, score, route.

## When you need content moderation

- User-generated content (comments, posts, reviews, messages)
- Community platforms and forums
- Marketplace listings (product descriptions, seller profiles)
- Chat and messaging features
- Any surface where users create content that others see

## Step 1: Define your moderation policy

Ask the user:
1. **What content do you need to catch?** (hate speech, spam, NSFW, harassment, self-harm, illegal activity, PII)
2. **What are the severity levels?** (warning, remove, ban)
3. **What's the tolerance for false positives?** (over-moderating frustrates users)
4. **Is human review in the loop?** (auto-only vs. auto + human escalation)

## Step 2: Build the moderator

Classification + severity scoring + routing decision:

```python
import dspy
from typing import Literal

VIOLATIONS = Literal[
    "safe", "spam", "hate_speech", "harassment",
    "violence", "nsfw", "self_harm", "illegal",
]

class ModerateContent(dspy.Signature):
    """Assess user-generated content against platform policies."""
    content: str = dspy.InputField(desc="user-generated content to moderate")
    platform_context: str = dspy.InputField(desc="where this content appears, e.g. 'product review'")
    violation_type: VIOLATIONS = dspy.OutputField()
    severity: Literal["none", "low", "medium", "high"] = dspy.OutputField()
    explanation: str = dspy.OutputField(desc="brief reason for the decision")

class ContentModerator(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought(ModerateContent)

    def forward(self, content, platform_context="social media post"):
        result = self.assess(content=content, platform_context=platform_context)

        # Route based on severity
        if result.severity == "high":
            decision = "remove"
        elif result.severity == "medium":
            decision = "human_review"
        elif result.severity == "low":
            decision = "warn"
        else:
            decision = "approve"

        return dspy.Prediction(
            violation_type=result.violation_type,
            severity=result.severity,
            decision=decision,
            explanation=result.explanation,
        )

# Usage
moderator = ContentModerator()
result = moderator(content="Great product, works exactly as described!")
print(result.decision)  # "approve"

result = moderator(content="This seller is a scammer, I'll find where they live")
print(result.decision)  # "remove"
print(result.violation_type)  # "harassment"
```

## Step 3: Multi-label moderation

Content can violate multiple policies at once (e.g., spam *and* contains PII):

```python
VIOLATION_TYPES = ["safe", "spam", "hate_speech", "harassment", "violence", "nsfw", "self_harm", "illegal"]

class MultiLabelModerate(dspy.Signature):
    """Flag all policy violations in user content. Content may have multiple violations."""
    content: str = dspy.InputField()
    platform_context: str = dspy.InputField()
    violations: list[str] = dspy.OutputField(desc=f"all that apply from: {VIOLATION_TYPES}")
    severity: Literal["none", "low", "medium", "high"] = dspy.OutputField(
        desc="overall severity based on the worst violation"
    )
    explanation: str = dspy.OutputField()

class MultiLabelModerator(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought(MultiLabelModerate)

    def forward(self, content, platform_context=""):
        result = self.assess(content=content, platform_context=platform_context)

        # Validate that returned violations are from the allowed set
        dspy.Assert(
            all(v in VIOLATION_TYPES for v in result.violations),
            f"Violations must be from: {VIOLATION_TYPES}",
        )

        return result
```

## Step 4: Hard blocks with assertions

For zero-tolerance patterns, don't even ask the LM — block instantly with pattern matching:

```python
import re

class StrictModerator(dspy.Module):
    def __init__(self):
        self.assess = dspy.ChainOfThought(ModerateContent)

    def forward(self, content, platform_context=""):
        # Pattern-based hard blocks (instant, no LM needed)
        dspy.Assert(
            not re.search(r"\b\d{3}-\d{2}-\d{4}\b", content),
            "Content contains SSN pattern — auto-reject",
        )
        dspy.Assert(
            not re.search(
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                content,
            ),
            "Content contains email addresses — redact before posting",
        )
        dspy.Assert(
            not re.search(r"\b\d{16}\b", content),
            "Content contains potential credit card number — auto-reject",
        )

        # LM-based assessment for everything else
        return self.assess(content=content, platform_context=platform_context)
```

Pattern-based blocks are faster, cheaper, and more reliable than LM-based detection for well-defined patterns (SSNs, credit cards, emails). Use regex for structure, LMs for semantics.

## Step 5: Confidence-based routing

Route uncertain decisions to human reviewers instead of making bad calls:

```python
class ConfidentModerate(dspy.Signature):
    """Moderate content and rate your confidence in the assessment."""
    content: str = dspy.InputField()
    platform_context: str = dspy.InputField()
    violation_type: VIOLATIONS = dspy.OutputField()
    severity: Literal["none", "low", "medium", "high"] = dspy.OutputField()
    confidence: float = dspy.OutputField(desc="0.0 to 1.0 — how sure are you about this assessment?")
    explanation: str = dspy.OutputField()

class ConfidentModerator(dspy.Module):
    def __init__(self, confidence_threshold=0.7):
        self.assess = dspy.ChainOfThought(ConfidentModerate)
        self.confidence_threshold = confidence_threshold

    def forward(self, content, platform_context=""):
        result = self.assess(content=content, platform_context=platform_context)

        # Validate confidence range
        dspy.Assert(
            0.0 <= result.confidence <= 1.0,
            "Confidence must be between 0.0 and 1.0",
        )

        # Route based on confidence + severity
        if result.confidence < self.confidence_threshold:
            decision = "human_review"  # uncertain → always escalate
        elif result.severity == "high":
            decision = "remove"
        elif result.severity == "medium":
            decision = "human_review"
        elif result.severity == "low":
            decision = "warn"
        else:
            decision = "approve"

        return dspy.Prediction(
            violation_type=result.violation_type,
            severity=result.severity,
            confidence=result.confidence,
            decision=decision,
            explanation=result.explanation,
        )
```

## Step 6: Metrics and optimization

### Define moderation metrics

```python
def moderation_metric(example, prediction, trace=None):
    """Weighted score: type matters more than severity."""
    type_correct = float(prediction.violation_type == example.violation_type)
    severity_correct = float(prediction.severity == example.severity)
    return 0.7 * type_correct + 0.3 * severity_correct
```

### Per-category metrics (more useful than overall accuracy)

```python
def make_category_metric(category):
    """Create a precision metric for a specific violation category."""
    def metric(example, prediction, trace=None):
        # Did we correctly identify this category?
        if example.violation_type == category:
            return float(prediction.violation_type == category)  # recall
        else:
            return float(prediction.violation_type != category)  # precision
    return metric

# Track each category separately
hate_speech_metric = make_category_metric("hate_speech")
spam_metric = make_category_metric("spam")
```

### Optimize the moderator

```python
# Prepare labeled training data
trainset = [
    dspy.Example(
        content="Buy cheap watches at spam-site.com!!!",
        platform_context="product review",
        violation_type="spam",
        severity="medium",
    ).with_inputs("content", "platform_context"),
    dspy.Example(
        content="This product changed my life, highly recommend!",
        platform_context="product review",
        violation_type="safe",
        severity="none",
    ).with_inputs("content", "platform_context"),
    # 50-200 labeled examples for good optimization
]

optimizer = dspy.MIPROv2(metric=moderation_metric, auto="medium")
optimized = optimizer.compile(moderator, trainset=trainset)
```

## Step 7: Handle tricky cases

Brief notes on content that's hard to moderate correctly:

- **Sarcasm and satire** — "Oh sure, what a *great* product" isn't hate speech. Context matters. The `platform_context` field helps here.
- **Quoting to criticize** — "The seller said 'you're an idiot'" is reporting harassment, not committing it. Include instructions in your signature to distinguish.
- **Code snippets** — Variable names or test strings might contain offensive words. If your platform has code, add a code-detection step before moderation.
- **Non-English content** — LMs handle major languages well but may miss nuance in less-common languages. Consider language-specific test sets.
- **Adversarial evasion** — Users will try to bypass moderation (leetspeak, Unicode tricks, word splitting). Test your moderator with `/ai-testing-safety`.

## Tips

- **False positives hurt more than false negatives.** Over-moderation kills user engagement. Tune your confidence threshold to minimize false positives, especially for borderline content.
- **Build separate metrics per category.** You care more about catching hate speech (high harm) than catching mild spam (low harm).
- **Use a stronger model for uncertain cases.** Route low-confidence decisions from GPT-4o-mini to GPT-4o for a second opinion.
- **Log every decision.** Moderator decisions should be reviewable and auditable. See `/ai-monitoring` for production logging patterns.
- **Test your moderator adversarially.** Users *will* try to evade moderation. Run `/ai-testing-safety` against your moderator.
- **Start permissive, tighten later.** It's easier to add restrictions than to regain user trust after over-moderating.

## Additional resources

- Use `/ai-sorting` for general classification patterns (moderation is classification + routing)
- Use `/ai-checking-outputs` for output guardrails on your own AI's responses
- Use `/ai-testing-safety` to adversarially test your moderator
- Use `/ai-monitoring` to track moderation quality in production
- See `examples.md` for complete worked examples
