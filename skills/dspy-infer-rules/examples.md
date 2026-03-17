# dspy.InferRules Examples

## Example 1: Extracting classification rules from labeled data

A support ticket classifier that uses InferRules to discover the decision logic behind priority labels. After compilation, the program's instructions contain explicit rules like "tickets mentioning outages or data loss are urgent."

```python
import dspy
from typing import Literal

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Define a typed signature for ticket classification
class ClassifyTicket(dspy.Signature):
    """Classify a support ticket by priority level."""
    ticket_text: str = dspy.InputField(desc="The support ticket text")
    priority: Literal["critical", "high", "normal", "low"] = dspy.OutputField(
        desc="Priority level for the ticket"
    )

# Build the program
classifier = dspy.ChainOfThought(ClassifyTicket)

# Prepare labeled training data
trainset = [
    # Critical -- system-wide outages, data loss
    dspy.Example(
        ticket_text="Production database is down. All users getting 500 errors. Revenue impact.",
        priority="critical",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Complete site outage. No pages loading for any customer.",
        priority="critical",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Data corruption detected in user accounts table. Backups may be affected.",
        priority="critical",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Payment processing system is completely unresponsive.",
        priority="critical",
    ).with_inputs("ticket_text"),

    # High -- degraded service, security issues
    dspy.Example(
        ticket_text="API response times spiked to 10s. Some requests timing out.",
        priority="high",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Possible unauthorized access detected on admin panel.",
        priority="high",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Image upload feature broken. Users can't attach files to tickets.",
        priority="high",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="SSL certificate expiring in 2 days. Needs immediate renewal.",
        priority="high",
    ).with_inputs("ticket_text"),

    # Normal -- feature requests, non-urgent bugs
    dspy.Example(
        ticket_text="Would like to export reports as PDF in addition to CSV.",
        priority="normal",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Typo on the pricing page. 'Annualy' should be 'Annually'.",
        priority="normal",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Can we add dark mode to the dashboard?",
        priority="normal",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="The date picker widget doesn't work well on mobile Safari.",
        priority="normal",
    ).with_inputs("ticket_text"),

    # Low -- questions, cosmetic issues
    dspy.Example(
        ticket_text="How do I change my notification preferences?",
        priority="low",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="The font on the settings page looks slightly different from the rest.",
        priority="low",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Is there documentation for the API rate limits?",
        priority="low",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Can you update the copyright year in the footer?",
        priority="low",
    ).with_inputs("ticket_text"),
]

# Separate validation set for better evaluation
valset = [
    dspy.Example(
        ticket_text="All microservices in us-east-1 are unreachable. Full region outage.",
        priority="critical",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Login is intermittently failing for about 30% of users.",
        priority="high",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="Could we add two-factor authentication as an option?",
        priority="normal",
    ).with_inputs("ticket_text"),
    dspy.Example(
        ticket_text="What browsers do you officially support?",
        priority="low",
    ).with_inputs("ticket_text"),
]

# Define the metric
def priority_match(example, pred, trace=None):
    return pred.priority.strip().lower() == example.priority.strip().lower()

# Compile with InferRules
optimizer = dspy.InferRules(
    metric=priority_match,
    num_rules=10,         # extract up to 10 rules
    num_candidates=5,     # try 5 different rule sets
)
compiled_classifier = optimizer.compile(
    classifier,
    trainset=trainset,
    valset=valset,
)

# Inspect the discovered rules
for name, predictor in compiled_classifier.named_predictors():
    print(f"--- Predictor: {name} ---")
    print(predictor.signature.instructions)
    print()

# Use the compiled classifier
test_tickets = [
    "Entire checkout flow is broken. Customers cannot complete purchases.",
    "Would be nice to have keyboard shortcuts for common actions.",
    "Memory leak in the worker process causing gradual slowdown.",
    "Where can I find the changelog for the latest release?",
]

for ticket in test_tickets:
    result = compiled_classifier(ticket_text=ticket)
    print(f"Ticket: {ticket[:60]}...")
    print(f"Priority: {result.priority}")
    print(f"Reasoning: {result.reasoning}")
    print()
```

What this demonstrates:

- **Typed classification** -- `Literal` output field constrains predictions to valid priority levels
- **Separate validation set** -- avoids the automatic 50/50 split, giving more training data for rule induction
- **Inspecting discovered rules** -- after compilation, reading the enhanced instructions shows exactly what patterns InferRules found
- **Practical ticket triage** -- the kind of task where explicit rules are valuable for auditing and stakeholder communication

## Example 2: Rule discovery for content moderation

A content moderation pipeline that uses InferRules to extract moderation policies from labeled examples. The discovered rules become an explicit, auditable moderation policy that can be reviewed by a trust-and-safety team.

```python
import dspy
from typing import Literal
from pydantic import BaseModel

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))


# Structured output for moderation decisions
class ModerationDecision(BaseModel):
    action: str           # "allow", "flag", "remove"
    policy_reason: str    # which policy applies


class ModerateContent(dspy.Signature):
    """Decide whether user-generated content should be allowed, flagged for review, or removed."""
    content: str = dspy.InputField(desc="User-generated content to moderate")
    context: str = dspy.InputField(desc="Where the content was posted (e.g., 'product review', 'forum post', 'profile bio')")
    decision: ModerationDecision = dspy.OutputField(desc="Moderation decision with action and policy reason")


class ContentModerator(dspy.Module):
    def __init__(self):
        self.moderate = dspy.ChainOfThought(ModerateContent)

    def forward(self, content, context):
        return self.moderate(content=content, context=context)


# Labeled moderation examples
trainset = [
    # Allow -- normal content
    dspy.Example(
        content="This product works great for cleaning kitchen counters. Highly recommend!",
        context="product review",
        decision_action="allow",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="Has anyone tried using this library with Python 3.12? I'm getting import errors.",
        context="forum post",
        decision_action="allow",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="Software engineer based in Portland. Love hiking and open source.",
        context="profile bio",
        decision_action="allow",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="I disagree with the previous reviewer. The battery life is actually quite poor.",
        context="product review",
        decision_action="allow",
    ).with_inputs("content", "context"),

    # Flag -- borderline content needing human review
    dspy.Example(
        content="This is the WORST company ever. They should be SHUT DOWN. Total scam artists!!!",
        context="product review",
        decision_action="flag",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="I can show you how to get around the paywall. DM me for details.",
        context="forum post",
        decision_action="flag",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="Check out my amazing crypto investment opportunity! 10x guaranteed returns!",
        context="forum post",
        decision_action="flag",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="The CEO is personally responsible for this disaster. Name and shame!",
        context="forum post",
        decision_action="flag",
    ).with_inputs("content", "context"),

    # Remove -- clear policy violations
    dspy.Example(
        content="Buy cheap followers and likes at spamsite.example.com. Best prices!",
        context="forum post",
        decision_action="remove",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="Here is John Smith's home address and phone number: 123 Main St...",
        context="forum post",
        decision_action="remove",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="You are an absolute idiot and I hope terrible things happen to you.",
        context="product review",
        decision_action="remove",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="CLICK HERE FOR FREE iPHONE >>> spamlink.example.com <<< ACT NOW!!!",
        context="profile bio",
        decision_action="remove",
    ).with_inputs("content", "context"),
]

valset = [
    dspy.Example(
        content="Solid product. Does exactly what the description says. 4/5 stars.",
        context="product review",
        decision_action="allow",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="This competitor's product is way better. Don't waste your money here.",
        context="product review",
        decision_action="flag",
    ).with_inputs("content", "context"),
    dspy.Example(
        content="Visit my profile for adult content links and premium subscriptions.",
        context="profile bio",
        decision_action="remove",
    ).with_inputs("content", "context"),
]


# Metric: check the action field of the structured output
def moderation_match(example, pred, trace=None):
    try:
        predicted_action = pred.decision.action.strip().lower()
    except AttributeError:
        return 0.0
    return predicted_action == example.decision_action.strip().lower()


# Compile with InferRules
optimizer = dspy.InferRules(
    metric=moderation_match,
    num_rules=15,         # more rules to capture nuanced moderation policies
    num_candidates=8,     # more candidates for a high-stakes task
)
compiled_moderator = optimizer.compile(
    ContentModerator(),
    trainset=trainset,
    valset=valset,
)

# Extract and display the discovered moderation policy
print("=== Discovered Moderation Policy ===\n")
for name, predictor in compiled_moderator.named_predictors():
    print(predictor.signature.instructions)
    print()

# Test on new content
test_cases = [
    ("Great tutorial! Saved me hours of debugging.", "forum post"),
    ("Everyone in this thread is so dumb. You all deserve to fail.", "forum post"),
    ("I found a way to exploit the referral system. Here's how...", "forum post"),
    ("Honest review: decent product, overpriced for what it is.", "product review"),
]

for content, context in test_cases:
    result = compiled_moderator(content=content, context=context)
    print(f"Content: {content[:60]}...")
    print(f"Context: {context}")
    print(f"Action: {result.decision.action}")
    print(f"Reason: {result.decision.policy_reason}")
    print()
```

What this demonstrates:

- **Structured moderation output** -- uses a Pydantic `BaseModel` to return both the action and the policy reason, making decisions auditable
- **Three-tier moderation** (allow/flag/remove) -- InferRules discovers the boundary between each tier
- **Context-aware rules** -- the `context` field lets the model learn that the same content might be handled differently in a product review vs. a profile bio
- **Higher `num_rules` and `num_candidates`** -- content moderation is high-stakes, so investing more LM calls to find better rules is worthwhile
- **Extracting an auditable policy** -- after compilation, the discovered rules can be printed, reviewed by a trust-and-safety team, and edited before deployment
- **Custom metric on structured output** -- `moderation_match` extracts the `action` field from the Pydantic model to compare against the labeled ground truth
