# AI Generating Notifications — Worked Examples

## Example 1: Push notification generator

Turn app events into concise iOS/Android push notification copy.

### Setup

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

### Signatures and module

```python
from pydantic import BaseModel, Field

class PushNotification(BaseModel):
    title: str = Field(description="Push notification title, max 50 chars")
    body: str = Field(description="Push notification body, max 100 chars")
    urgency_level: Literal["low", "medium", "high", "critical"]

class GeneratePush(dspy.Signature):
    """Write a push notification for a mobile app. Title must be under 50 chars. Body must be under 100 chars. Be direct — the user sees this on their lock screen."""
    event: str = dspy.InputField(desc="App event that triggered the notification")
    recipient_name: str = dspy.InputField(desc="User's first name for personalization")
    notification: PushNotification = dspy.OutputField()


def push_length_reward(args, pred):
    """Hard fail if title or body exceeds character limits."""
    title_ok = len(pred.notification.title) <= 50
    body_ok = len(pred.notification.body) <= 100
    if not title_ok and not body_ok:
        return 0.0
    if not title_ok or not body_ok:
        return 0.5  # partial: one field out of spec
    return 1.0

push_notifier = dspy.Refine(
    module=dspy.ChainOfThought(GeneratePush),
    N=4,
    reward_fn=push_length_reward,
    threshold=1.0,
)
```

### Usage

```python
events = [
    ("New comment on your post 'Launching v2.0'", "Alex"),
    ("Your export is ready to download", "Maria"),
    ("Payment failed - subscription renewal", "Sam"),
    ("Someone mentioned you in #general", "Jordan"),
]

for event, name in events:
    result = push_notifier(event=event, recipient_name=name)
    n = result.notification
    print(f"[{n.urgency_level.upper()}] {n.title}")
    print(f"  {n.body}")
    print()

# [LOW] Alex, new comment waiting
#   "Great post! When does v2.0 ship?" — view now
#
# [LOW] Maria, your export is ready
#   Download your file before it expires in 24 hours
#
# [CRITICAL] Payment failed, Sam
#   Your subscription renewal failed. Update your card now.
#
# [MEDIUM] Jordan, you were mentioned
#   Someone tagged you in #general — tap to read
```

### Metric and optimization

```python
def push_metric(example, prediction, trace=None):
    n = prediction.notification
    title_within = len(n.title) <= 50
    body_within = len(n.body) <= 100
    # Both fields must be within limits
    if not (title_within and body_within):
        return 0.0
    # Urgency must match expected level if provided
    if hasattr(example, "expected_urgency") and n.urgency_level != example.expected_urgency:
        return 0.5
    return 1.0

optimizer = dspy.BootstrapFewShot(metric=push_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(dspy.ChainOfThought(GeneratePush), trainset=trainset)
```

---

## Example 2: Weekly digest email

Aggregate many events from the past week into a single scannable email digest.

### Signatures and module

```python
from pydantic import BaseModel, Field

class DigestSection(BaseModel):
    group_label: str = Field(description="Category label, e.g. 'Comments (4)', 'Deployments (2)'")
    summary: str = Field(description="One sentence summarizing this group")

class WeeklyDigest(BaseModel):
    subject_line: str = Field(description="Email subject, max 60 chars, no clickbait")
    headline: str = Field(description="Most important thing that happened this week, one sentence")
    sections: list[DigestSection] = Field(description="Event groups, 2-5 sections")
    call_to_action: str = Field(description="What the user should do, or empty string if none")

class GroupEvents(dspy.Signature):
    """Group a list of app events by type. Return a list of group labels with counts, e.g. '5 new comments', '2 deployments'."""
    events: list[str] = dspy.InputField(desc="Raw event list")
    grouped: list[str] = dspy.OutputField(desc="Grouped summaries, one per event type")

class WriteDigest(dspy.Signature):
    """Write a weekly digest email from grouped event summaries. Keep it scannable - the user is skimming."""
    grouped_events: list[str] = dspy.InputField(desc="Event groups from GroupEvents step")
    recipient_name: str = dspy.InputField()
    total_event_count: int = dspy.InputField(desc="Total number of raw events this week")
    digest: WeeklyDigest = dspy.OutputField()

class WeeklyDigestNotifier(dspy.Module):
    def __init__(self):
        self.group = dspy.ChainOfThought(GroupEvents)
        self.write = dspy.ChainOfThought(WriteDigest)

    def forward(self, events, recipient_name):
        grouped = self.group(events=events).grouped
        return self.write(
            grouped_events=grouped,
            recipient_name=recipient_name,
            total_event_count=len(events),
        )


def digest_reward(args, pred):
    """Encourage concise subject line and at least 2 event sections."""
    score = 1.0
    if len(pred.digest.subject_line) > 60:
        score -= 0.3  # soft: subject is too long for email clients
    if len(pred.digest.sections) < 2:
        score -= 0.4  # soft: digest needs multiple sections to be useful
    return score

digest_notifier = dspy.Refine(
    module=WeeklyDigestNotifier(),
    N=3,
    reward_fn=digest_reward,
    threshold=0.8,
)
```

### Usage

```python
weekly_events = [
    "User commented on post #42",
    "User commented on post #45",
    "User commented on post #48",
    "Deployment to production succeeded",
    "New follower: @devrel_team",
    "API key rotated",
    "User liked post #42",
    "User liked post #50",
    "Report export completed",
]

result = digest_notifier(events=weekly_events, recipient_name="Taylor")
d = result.digest

print(d.subject_line)
# "Your week in review — 3 comments, 2 likes, 1 deployment"

print(d.headline)
# "Three new comments on your posts this week, including post #42 which got the most engagement."

for section in d.sections:
    print(f"  {section.group_label} - {section.summary}")
# Comments (3) - Readers engaged with posts #42, #45, and #48
# Likes (2) - Posts #42 and #50 received likes
# Deployments (1) - Production deployment succeeded
# Other (2) - API key rotated, report export ready

print(d.call_to_action)
# "Check your comments and reply to keep the conversation going."
```

---

## Example 3: Incident alert generator

Turn system events and log snippets into structured Slack alerts with severity and context.

### Signatures and module

```python
from pydantic import BaseModel, Field
from typing import Literal

class IncidentAlert(BaseModel):
    severity: Literal["info", "warning", "error", "critical"]
    title: str = Field(description="Short incident title, max 60 chars")
    summary: str = Field(description="What happened and what is affected, 1-2 sentences")
    impact: str = Field(description="Who or what is impacted, e.g. 'US-East users', 'checkout API'")
    suggested_action: str = Field(description="Immediate next step for on-call engineer")
    runbook_hint: str = Field(description="Which runbook or playbook to check, or empty string")

class GenerateIncidentAlert(dspy.Signature):
    """Generate a Slack incident alert for an on-call engineer. Include severity, what happened, impact, and the immediate next action. Be precise - no filler text."""
    system_event: str = dspy.InputField(desc="Raw system event, error, or log snippet")
    service_name: str = dspy.InputField(desc="Name of the affected service")
    environment: Literal["production", "staging", "dev"] = dspy.InputField()
    alert: IncidentAlert = dspy.OutputField()

class IncidentNotifier(dspy.Module):
    def __init__(self):
        self.generate = dspy.ChainOfThought(GenerateIncidentAlert)

    def forward(self, system_event, service_name, environment):
        return self.generate(
            system_event=system_event,
            service_name=service_name,
            environment=environment,
        )

    def format_slack_block(self, alert: IncidentAlert) -> str:
        """Format the alert as a Slack markdown block."""
        severity_emoji = {
            "info": ":information_source:",
            "warning": ":warning:",
            "error": ":x:",
            "critical": ":rotating_light:",
        }
        icon = severity_emoji.get(alert.severity, ":bell:")
        lines = [
            f"{icon} *[{alert.severity.upper()}] {alert.title}*",
            f"> {alert.summary}",
            f"*Impact* - {alert.impact}",
            f"*Action* - {alert.suggested_action}",
        ]
        if alert.runbook_hint:
            lines.append(f"*Runbook* - {alert.runbook_hint}")
        return "\n".join(lines)


def incident_reward(args, pred):
    """Hard fail on empty summary or action. Soft penalty for overly long blocks."""
    if not pred.alert.summary or not pred.alert.suggested_action:
        return 0.0  # hard: both fields are required
    slack_text = pred.alert.summary + pred.alert.suggested_action
    if len(slack_text) > 400:
        return 0.6  # soft: Slack blocks should stay readable
    return 1.0

incident_notifier_enforced = dspy.Refine(
    module=IncidentNotifier(),
    N=3,
    reward_fn=incident_reward,
    threshold=0.9,
)
```

### Usage

```python
notifier = IncidentNotifier()

result = notifier(
    system_event="""
    ERROR [2026-05-04T14:32:11Z] checkout-service: Connection pool exhausted
    java.sql.SQLException: Timeout waiting for connection from pool (30000ms)
    at com.example.checkout.OrderRepository.findById(OrderRepository.java:142)
    Active connections: 100/100, Pending requests: 847
    """,
    service_name="checkout-service",
    environment="production",
)

slack_message = notifier.format_slack_block(result.alert)
print(slack_message)
# :rotating_light: *[CRITICAL] checkout-service DB pool exhausted*
# > Connection pool fully exhausted with 847 pending requests; all checkout transactions are failing.
# *Impact* - All users attempting to complete a purchase in production
# *Action* - Scale up DB connection pool or restart checkout-service pods immediately
# *Runbook* - db-connection-pool-runbook.md
```

### Metric

```python
class JudgeIncidentAlert(dspy.Signature):
    """Judge whether an incident alert gives an on-call engineer everything they need to act."""
    system_event: str = dspy.InputField()
    alert_text: str = dspy.InputField()
    has_clear_action: bool = dspy.OutputField(desc="Does the alert say exactly what to do?")
    severity_appropriate: bool = dspy.OutputField(desc="Is the severity correct for the event?")
    no_filler: bool = dspy.OutputField(desc="Is the alert free of vague or generic text?")

def incident_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeIncidentAlert)
    notifier = IncidentNotifier()
    slack_text = notifier.format_slack_block(prediction.alert)
    result = judge(system_event=example.system_event, alert_text=slack_text)
    return (result.has_clear_action + result.severity_appropriate + result.no_filler) / 3
```
