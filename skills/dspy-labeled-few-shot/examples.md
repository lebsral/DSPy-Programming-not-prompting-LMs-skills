# dspy-labeled-few-shot -- Worked Examples

## Example 1: Curated demos for customer support classification

Use hand-picked examples to teach the LM your company's specific classification rules. Each example demonstrates a category with a representative ticket, giving the LM clear patterns to follow.

```python
import dspy
from typing import Literal


class ClassifyTicket(dspy.Signature):
    """Classify a customer support ticket into a department and urgency level."""
    ticket: str = dspy.InputField(desc="Customer support ticket text")
    department: Literal["billing", "technical", "account", "shipping", "general"] = dspy.OutputField()
    urgency: Literal["low", "medium", "high", "critical"] = dspy.OutputField()


# --- Curated training examples ---
# Pick examples that cover each category and show edge cases your team
# has discussed. Quality matters more than quantity here.

trainset = [
    # Billing -- clear examples of payment and invoice issues
    dspy.Example(
        ticket="I was charged twice for my subscription this month",
        department="billing",
        urgency="high",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Can I get a copy of last quarter's invoices?",
        department="billing",
        urgency="low",
    ).with_inputs("ticket"),

    # Technical -- bugs, crashes, integration problems
    dspy.Example(
        ticket="The API returns 500 errors whenever I send a batch request over 100 items",
        department="technical",
        urgency="high",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Is there a way to export data as Parquet instead of CSV?",
        department="technical",
        urgency="low",
    ).with_inputs("ticket"),

    # Account -- login, permissions, profile changes
    dspy.Example(
        ticket="I can't log in after resetting my password, it says account locked",
        department="account",
        urgency="critical",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Please add my colleague as an admin on our team workspace",
        department="account",
        urgency="medium",
    ).with_inputs("ticket"),

    # Shipping -- delivery, tracking, address changes
    dspy.Example(
        ticket="My order was marked delivered but I never received it",
        department="shipping",
        urgency="high",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="Can I change the delivery address for order #4821?",
        department="shipping",
        urgency="medium",
    ).with_inputs("ticket"),

    # General -- everything else
    dspy.Example(
        ticket="Do you offer discounts for nonprofits?",
        department="general",
        urgency="low",
    ).with_inputs("ticket"),
    dspy.Example(
        ticket="What are your support hours over the holidays?",
        department="general",
        urgency="low",
    ).with_inputs("ticket"),
]

# --- Compile and use ---

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

classifier = dspy.Predict(ClassifyTicket)

# Use k=6 to include a good spread without overloading the prompt
optimizer = dspy.LabeledFewShot(k=6)
optimized = optimizer.compile(classifier, trainset=trainset)

# Classify new tickets
test_tickets = [
    "My credit card was declined but the charge still shows as pending",
    "The dashboard keeps showing a blank page on Firefox",
    "I need to transfer ownership of the account to my business partner",
    "When will my replacement item ship?",
]

for ticket in test_tickets:
    result = optimized(ticket=ticket)
    print(f"[{result.urgency}] {result.department}: {ticket}")

# Save for production use
optimized.save("ticket_classifier.json")
```

Key points:
- Each category has at least two examples showing different urgency levels, so the LM learns both fields
- Examples are chosen to represent real ambiguous cases your team has resolved (e.g., "account locked" is critical, not just high)
- `k=6` gives enough variety without burning too many tokens per classification call
- `with_inputs("ticket")` marks which field is the input -- the rest are treated as labels for demonstrations
- Save the compiled program so you do not need to recompile on every server restart


## Example 2: Hand-picked examples for consistent formatting

Use curated demonstrations to enforce a specific output format that the LM should follow consistently. This is useful when you need structured, predictable output that matches your application's conventions.

```python
import dspy
from pydantic import BaseModel, Field


class ChangelogEntry(BaseModel):
    title: str = Field(description="Short imperative-mood title, max 60 chars")
    category: str = Field(description="One of: added, changed, fixed, removed, security")
    description: str = Field(description="One-sentence user-facing description")
    breaking: bool = Field(description="Whether this is a breaking change")


class FormatChangelog(dspy.Signature):
    """Convert a raw git commit message into a structured changelog entry.
    Use imperative mood for the title (e.g., 'Add export button' not 'Added export button').
    Descriptions should be written for end users, not developers."""
    commit_message: str = dspy.InputField(desc="Raw git commit message")
    entry: ChangelogEntry = dspy.OutputField()


# --- Curated demonstrations ---
# These examples define the exact formatting conventions you want.
# The LM learns your style from these patterns.

trainset = [
    dspy.Example(
        commit_message="feat: add CSV export to the analytics dashboard\n\nUsers have been requesting CSV downloads for months. This adds an export button to the top-right of the dashboard that downloads the current filtered view.",
        entry=ChangelogEntry(
            title="Add CSV export to analytics dashboard",
            category="added",
            description="You can now export your analytics data as a CSV file directly from the dashboard.",
            breaking=False,
        ),
    ).with_inputs("commit_message"),

    dspy.Example(
        commit_message="fix: resolve race condition in webhook delivery\n\nWebhooks were occasionally delivered out of order when multiple events fired within the same millisecond. Added a sequence counter to ensure ordering.",
        entry=ChangelogEntry(
            title="Fix webhook delivery ordering",
            category="fixed",
            description="Webhooks are now guaranteed to arrive in the correct order, even when multiple events fire simultaneously.",
            breaking=False,
        ),
    ).with_inputs("commit_message"),

    dspy.Example(
        commit_message="BREAKING: remove legacy v1 API endpoints\n\nv1 has been deprecated since March 2024. All remaining v1 callers have been migrated. Removing to reduce maintenance burden.",
        entry=ChangelogEntry(
            title="Remove legacy v1 API endpoints",
            category="removed",
            description="The deprecated v1 API has been removed. Please use v2 endpoints instead.",
            breaking=True,
        ),
    ).with_inputs("commit_message"),

    dspy.Example(
        commit_message="refactor: switch password hashing from bcrypt to argon2id\n\nArgon2id is the current OWASP recommendation. Existing passwords will be rehashed on next login.",
        entry=ChangelogEntry(
            title="Upgrade password hashing to Argon2id",
            category="security",
            description="Password hashing has been upgraded to Argon2id for improved security. No action required -- your password will be updated automatically on next login.",
            breaking=False,
        ),
    ).with_inputs("commit_message"),

    dspy.Example(
        commit_message="feat: redesign settings page with tabbed navigation\n\nReplaces the long scrolling settings page with a tabbed layout. Tabs: General, Notifications, Integrations, Security, Billing.",
        entry=ChangelogEntry(
            title="Redesign settings page with tabbed layout",
            category="changed",
            description="The settings page now uses tabs for easier navigation between General, Notifications, Integrations, Security, and Billing sections.",
            breaking=False,
        ),
    ).with_inputs("commit_message"),
]

# --- Compile and use ---

lm = dspy.LM("openai/gpt-4o-mini")  # or any LiteLLM-supported provider
dspy.configure(lm=lm)

formatter = dspy.Predict(FormatChangelog)

# Use sample=False to include the examples in the exact order above,
# ensuring the LM sees a representative spread of categories
optimizer = dspy.LabeledFewShot(k=5)
optimized = optimizer.compile(formatter, trainset=trainset, sample=False)

# Format new commit messages
commits = [
    "feat: add team-wide notification preferences\n\nAdmins can now set default notification settings for the entire team. Individual users can still override.",
    "fix(auth): SSO login fails when email contains a plus sign\n\nThe email parser was treating '+' as a special character. Now properly URL-decodes before matching.",
    "BREAKING: change /api/users response from array to paginated object\n\nResponses now include { data: [...], pagination: { page, per_page, total } }. Clients must update to handle the new envelope format.",
]

for commit in commits:
    result = optimized(commit_message=commit)
    entry = result.entry
    prefix = "BREAKING: " if entry.breaking else ""
    print(f"[{entry.category}] {prefix}{entry.title}")
    print(f"  {entry.description}")
    print()

# Save for use in CI pipeline
optimized.save("changelog_formatter.json")
```

Key points:
- Demonstrations define your formatting conventions by example -- imperative mood, user-facing language, correct categorization
- `sample=False` preserves the deliberate ordering so the LM sees one example of each category
- Pydantic `BaseModel` output ensures the LM returns structured, validated data
- The `breaking` boolean flag shows that demonstrations can teach nuanced classification alongside formatting
- This pattern works well in CI pipelines where you need consistent, machine-readable changelog entries from raw commits
