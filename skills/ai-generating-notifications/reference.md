> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Notification Generation

## Channel limits quick reference

| Channel | Combined limit | Title | Body | Format |
|---------|---------------|-------|------|--------|
| `push` | 150 chars | 50 chars | 100 chars | Plain text |
| `email` | 500 chars | ~60 chars subject | 1-3 paragraphs | HTML or plain |
| `slack` | 500 chars | N/A | ~500 chars | Markdown |
| `sms` | 160 chars | N/A | 160 chars | Plain text only |

```python
CHANNEL_LIMITS = {
    "push": 150,
    "email": 500,
    "slack": 500,
    "sms": 160,
}
```

## dspy.Signature

[API docs](https://dspy.ai/api/signatures/)

```python
class MySignature(dspy.Signature):
    """Docstring becomes the task instruction."""
    input_field: type = dspy.InputField(desc="description")
    output_field: type = dspy.OutputField(desc="description")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `desc` | `str` | Field description passed to the LM |
| `prefix` | `str` | Override label in the prompt |

Use `Literal["a", "b", "c"]` for constrained string outputs (channel, urgency). Use a Pydantic `BaseModel` as the output type to enforce structured fields (title + body + urgency as one object).

Do not add a `reasoning` field — `dspy.ChainOfThought` injects it automatically.

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Inputs/outputs contract |
| `rationale_field` | `FieldInfo \| None` | `None` | Custom reasoning field |
| `rationale_field_type` | `type` | `str` | Type for the rationale |

Default choice for notification generation. Adds a `reasoning` field before the output.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

No reasoning step. Use for LM-as-judge steps (`JudgeNotification`, `JudgeIncidentAlert`) where you want direct scoring without intermediate reasoning.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=0.8, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to refine |
| `N` | `int` | required | Max retry attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores output; 1.0 = perfect |
| `threshold` | `float` | `0.8` | Stop retrying once score >= threshold |
| `fail_count` | `int \| None` | `None` | Max failures before raising |

The reward function signature is `reward_fn(args: dict, pred: Prediction) -> float`. `args` holds the module's input keyword arguments; `pred` holds the prediction object. Return `1.0` for pass, `0.0` for hard fail, or a value between for graduated penalties.

Use `dspy.Refine` (not `dspy.Assert`/`dspy.Suggest`, which were removed in DSPy 3.x) for all hard character-limit enforcement and urgency constraints.

## Reward function patterns

```python
# Hard length limit (push)
def push_length_reward(args, pred):
    title_ok = len(pred.notification.title) <= 50
    body_ok = len(pred.notification.body) <= 100
    if not title_ok and not body_ok:
        return 0.0
    if not title_ok or not body_ok:
        return 0.5
    return 1.0

# Graduated penalty (combined text)
def channel_length_reward(args, pred):
    limit = CHANNEL_LIMITS.get(args["channel"], 300)
    text_len = len(pred.notification_text)
    if text_len <= limit:
        return 1.0
    if text_len > limit * 2:
        return 0.0
    return max(0.0, 1.0 - (text_len - limit) / limit)

# Hard fail on missing required fields
def incident_reward(args, pred):
    if not pred.alert.summary or not pred.alert.suggested_action:
        return 0.0
    if len(pred.alert.summary + pred.alert.suggested_action) > 400:
        return 0.6
    return 1.0
```

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `16` | Max labeled demos from trainset |

Key method - `.compile(module, trainset=...)` - returns the optimized module.

Metric signature for notifications: `metric(example, prediction, trace=None) -> float`. Return `0.0` immediately if hard constraints (length limits, required fields) are violated before checking quality dimensions.

## Pydantic typed outputs

DSPy supports Pydantic `BaseModel` as an output field type. DSPy enforces the schema and retries on parse failures.

```python
from pydantic import BaseModel, Field

class PushNotification(BaseModel):
    title: str = Field(description="Push notification title, max 50 chars")
    body: str = Field(description="Push notification body, max 100 chars")
    urgency_level: Literal["low", "medium", "high", "critical"]

class GeneratePush(dspy.Signature):
    """..."""
    event: str = dspy.InputField()
    notification: PushNotification = dspy.OutputField()
```

Access nested fields as `pred.notification.title`, `pred.notification.body`.

## dspy.Module (multi-step pipelines)

```python
class DigestNotifier(dspy.Module):
    def __init__(self):
        self.group = dspy.ChainOfThought(GroupEvents)
        self.write = dspy.ChainOfThought(WriteDigest)

    def forward(self, events, recipient_name):
        grouped = self.group(events=events).grouped
        return self.write(grouped_events=grouped, recipient_name=recipient_name,
                          total_event_count=len(events))
```

Wrap `dspy.Refine` around the full module instance, not just inner steps, when the reward depends on the final composed output.

## Setup

```python
pip install -U dspy       # DSPy 3.2.1+
```

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```
