# dspy-chain-of-thought -- Worked Examples

## Example 1: Bug analysis with reasoning trace

Analyze a bug report, reason through the likely root cause, and suggest next steps. The reasoning trace makes it possible to audit why the system reached its conclusion.

```python
import dspy
from typing import Literal


class AnalyzeBug(dspy.Signature):
    """Analyze a bug report to identify the likely root cause and suggest debugging steps."""
    title: str = dspy.InputField(desc="Bug report title")
    description: str = dspy.InputField(desc="Bug report description with reproduction steps")
    stack_trace: str = dspy.InputField(desc="Error stack trace, if available")
    root_cause: str = dspy.OutputField(desc="Most likely root cause of the bug")
    component: str = dspy.OutputField(desc="Software component where the bug likely originates")
    debugging_steps: list[str] = dspy.OutputField(desc="Ordered steps to confirm and fix the bug")
    severity: Literal["critical", "high", "medium", "low"] = dspy.OutputField()


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

analyzer = dspy.ChainOfThought(AnalyzeBug)

result = analyzer(
    title="Orders stuck in 'processing' state after payment",
    description=(
        "Since deploying v2.14, ~5% of orders stay in 'processing' after successful "
        "payment. Customers are charged but never receive confirmation. Happens more "
        "during peak hours (2-4pm EST). Rolling back to v2.13 resolves the issue."
    ),
    stack_trace=(
        "TimeoutError: Task timed out after 30s\n"
        "  at OrderService.finalizeOrder(OrderService.java:142)\n"
        "  at PaymentCallback.onSuccess(PaymentCallback.java:67)\n"
        "  at EventLoop.processQueue(EventLoop.java:201)"
    ),
)

# The reasoning trace shows how the LM arrived at its diagnosis
print("=== Reasoning ===")
print(result.reasoning)
# "The bug appeared after v2.14 and resolves on rollback, so the root cause
#  is in v2.14 changes. The TimeoutError at OrderService.finalizeOrder suggests
#  the order finalization step is taking too long. The correlation with peak hours
#  points to a concurrency or resource contention issue..."

print(f"\nRoot cause: {result.root_cause}")
# "Race condition or resource contention in OrderService.finalizeOrder introduced in v2.14"

print(f"Component: {result.component}")
# "OrderService"

print(f"Severity: {result.severity}")
# "critical"

print("\nDebugging steps:")
for i, step in enumerate(result.debugging_steps, 1):
    print(f"  {i}. {step}")
# 1. Diff OrderService.java between v2.13 and v2.14 to identify the change
# 2. Check connection pool and thread pool sizes under load
# 3. Add timing instrumentation to finalizeOrder to find the slow path
# 4. Reproduce under load in staging with v2.14
# 5. Check database locks during peak-hour order finalization


# --- Logging the reasoning for audit trail ---

import json

audit_record = {
    "bug_title": "Orders stuck in 'processing' state after payment",
    "diagnosis": result.root_cause,
    "severity": result.severity,
    "reasoning_trace": result.reasoning,  # keep for auditing
    "steps": result.debugging_steps,
}
print(json.dumps(audit_record, indent=2))
```

Key points:
- The `reasoning` field gives a full trace of the LM's diagnostic thought process
- Logging the reasoning creates an audit trail -- useful for post-mortems and quality reviews
- Typed outputs (`Literal` for severity, `list[str]` for steps) ensure structured results alongside free-form reasoning


## Example 2: Decision-making with visible logic

Make a go/no-go decision on a feature release, with the reasoning visible to stakeholders. The reasoning field serves as the justification document.

```python
import dspy
from typing import Literal
from pydantic import BaseModel


class ReleaseMetrics(BaseModel):
    test_pass_rate: float
    error_rate_delta: float
    p99_latency_ms: float
    rollback_plan: bool
    affected_users_percent: float


class ReleaseDecision(dspy.Signature):
    """Decide whether a feature is safe to release to production based on metrics and context."""
    feature_name: str = dspy.InputField()
    metrics: ReleaseMetrics = dspy.InputField(desc="Current release metrics")
    context: str = dspy.InputField(desc="Additional context about the release")
    decision: Literal["go", "no-go", "conditional"] = dspy.OutputField()
    conditions: list[str] = dspy.OutputField(
        desc="Conditions that must be met before release (empty if decision is 'go')"
    )
    risk_summary: str = dspy.OutputField(desc="One-paragraph risk summary for stakeholders")


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

decider = dspy.ChainOfThought(ReleaseDecision)

metrics = ReleaseMetrics(
    test_pass_rate=0.97,
    error_rate_delta=0.3,
    p99_latency_ms=450,
    rollback_plan=True,
    affected_users_percent=15.0,
)

result = decider(
    feature_name="New checkout flow",
    metrics=metrics,
    context=(
        "Holiday shopping season starts in 3 days. The new checkout flow reduces "
        "cart abandonment by 12% in A/B testing. Error rate increased 0.3% vs baseline "
        "but all errors are non-blocking validation warnings."
    ),
)

# The reasoning shows the decision logic step by step
print("=== Decision Reasoning ===")
print(result.reasoning)
# "Let me evaluate each metric against release criteria:
#  - Test pass rate 97% meets the 95% threshold
#  - Error rate delta of 0.3% is slightly elevated, but context says these are
#    non-blocking validation warnings, not payment failures
#  - P99 latency of 450ms is within the 500ms SLA
#  - Rollback plan exists, which is required
#  - 15% of users affected is significant, especially before holiday season
#  - The 12% reduction in cart abandonment is a strong business case
#  - The timing risk (3 days before holidays) is notable but rollback plan mitigates it..."

print(f"\nDecision: {result.decision}")
# "conditional"

print("\nConditions:")
for condition in result.conditions:
    print(f"  - {condition}")
# - Monitor error rate for 2 hours after staged rollout to first 5% of users
# - Confirm all validation warnings are truly non-blocking in production logs
# - Have on-call engineer available during the first 24 hours

print(f"\nRisk summary: {result.risk_summary}")


# --- Share with stakeholders ---

report = f"""
# Release Decision: {result.feature_name}

**Decision:** {result.decision.upper()}

## Reasoning
{result.reasoning}

## Conditions
{"".join(f"- {c}" + chr(10) for c in result.conditions) if result.conditions else "None -- clear to proceed."}

## Risk Summary
{result.risk_summary}
"""
print(report)
```

Key points:
- The `reasoning` field acts as a decision justification that stakeholders can review
- Pydantic input models (`ReleaseMetrics`) let you pass structured data cleanly
- `Literal["go", "no-go", "conditional"]` constrains the decision to valid options
- The reasoning is generated before the decision, so the LM weighs all factors before committing


## Example 3: Classification with justification

Classify support tickets with a written justification for each classification. The justification helps human reviewers verify the routing and catch misclassifications quickly.

```python
import dspy
from typing import Literal


class ClassifyTicket(dspy.Signature):
    """Classify a customer support ticket and justify the classification.
    Consider the customer's primary intent, not just keywords."""
    ticket_subject: str = dspy.InputField()
    ticket_body: str = dspy.InputField()
    category: Literal[
        "billing", "technical", "account", "feature_request", "other"
    ] = dspy.OutputField()
    priority: Literal["urgent", "high", "normal", "low"] = dspy.OutputField()
    justification: str = dspy.OutputField(
        desc="One-sentence explanation of why this category and priority were chosen"
    )


# --- Usage ---

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

classifier = dspy.ChainOfThought(ClassifyTicket)

result = classifier(
    ticket_subject="Can't access my account since password reset",
    ticket_body=(
        "I reset my password yesterday using the forgot password link. "
        "Since then I get 'invalid credentials' every time I try to log in. "
        "I have a client presentation in 2 hours and all my files are in there. "
        "Please help ASAP."
    ),
)

# reasoning: the LM's internal thought process (for debugging/logging)
print("=== Internal Reasoning ===")
print(result.reasoning)
# "The customer reset their password but can't log in. This is an account access issue,
#  not a billing or feature request. The mention of a presentation in 2 hours and 'ASAP'
#  indicates time pressure. The password reset flow may have a bug (could be technical),
#  but the primary intent is regaining account access..."

# justification: the user-facing explanation (for the support queue)
print(f"\nCategory: {result.category}")
# "account"
print(f"Priority: {result.priority}")
# "urgent"
print(f"Justification: {result.justification}")
# "Account access blocked after password reset with a time-sensitive deadline in 2 hours."


# --- Batch processing with justifications ---

tickets = [
    {
        "subject": "Charge for plan I didn't sign up for",
        "body": "I was charged $49 for a Pro plan but I'm on the free tier. Please refund.",
    },
    {
        "subject": "API returns 500 on large payloads",
        "body": "When sending payloads > 5MB to /api/upload, we get 500 errors. Works fine under 5MB.",
    },
    {
        "subject": "Would love dark mode",
        "body": "Any plans for a dark mode? Would be easier on the eyes for late-night coding sessions.",
    },
]

for ticket in tickets:
    result = classifier(
        ticket_subject=ticket["subject"],
        ticket_body=ticket["body"],
    )
    print(f"\n[{result.priority.upper()}] [{result.category}] {ticket['subject']}")
    print(f"  Justification: {result.justification}")


# --- Optimization with reasoning ---

trainset = [
    dspy.Example(
        ticket_subject="Can't access my account since password reset",
        ticket_body="I reset my password yesterday...",
        category="account",
        priority="urgent",
        justification="Account access blocked after password reset with time-sensitive deadline.",
    ).with_inputs("ticket_subject", "ticket_body"),
    # ... more labeled examples
]

def ticket_metric(example, prediction, trace=None):
    cat_match = prediction.category == example.category
    pri_match = prediction.priority == example.priority
    has_justification = len(prediction.justification.strip()) > 10
    return cat_match + 0.5 * pri_match + 0.25 * has_justification

# optimizer = dspy.BootstrapFewShot(metric=ticket_metric, max_bootstrapped_demos=4)
# optimized = optimizer.compile(classifier, trainset=trainset)
# optimized.save("ticket_classifier.json")
```

Key points:
- **Two levels of explanation**: `reasoning` is the internal trace for developers; `justification` is a clean, user-facing explanation declared in the signature
- ChainOfThought generates `reasoning` automatically; `justification` is an explicit output field you control
- The `justification` field gives human reviewers a quick way to verify the classification without reading the full ticket
- Optimization teaches the LM to produce better reasoning traces, which improves both the classification accuracy and the quality of justifications
