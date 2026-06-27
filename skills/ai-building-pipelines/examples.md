# Pipeline Examples

## Minimal pipeline: classify then generate

Two stages wired in `forward()` — the simplest possible multi-step pipeline:

```python
import dspy
from typing import Literal

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-haiku-4-5-20251001", etc.
dspy.configure(lm=lm)

INTENTS = ["question", "complaint", "praise", "cancellation"]

class ClassifyIntent(dspy.Signature):
    """Identify the customer's intent."""
    message: str = dspy.InputField()
    intent: Literal[tuple(INTENTS)] = dspy.OutputField()

class DraftReply(dspy.Signature):
    """Write a concise reply appropriate to the intent."""
    message: str = dspy.InputField()
    intent: str = dspy.InputField()
    reply: str = dspy.OutputField()

class IntentThenReply(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyIntent)    # simple: no reasoning needed
        self.draft    = dspy.ChainOfThought(DraftReply) # nuanced: reasoning helps

    def forward(self, message):
        cls = self.classify(message=message)
        return self.draft(message=message, intent=cls.intent)

pipeline = IntentThenReply()
result = pipeline(message="I've been waiting 3 weeks for my order. This is unacceptable.")
print(f"Intent: {result.intent}")  # complaint
print(f"Reply:  {result.reply}")
```

## Classify → Route → Specialize

Route each input to a handler matched to its category. `forward()` is plain Python — use any control flow:

```python
class RouteAndAnswer(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyInput)
        self.handlers = {
            "simple":  dspy.Predict(QuickAnswer),           # fast path
            "complex": dspy.ChainOfThought(DetailedAnswer), # reasoning path
            "policy":  dspy.ChainOfThought(PolicyAnswer),   # compliance path
        }

    def forward(self, question):
        cls = self.classify(question=question)
        handler = self.handlers.get(cls.category, self.handlers["simple"])
        return handler(question=question)
```

The optimizer traces through whichever branch actually runs on each training example, so all branches get optimized proportionally.

## Validate intermediate output with dspy.Refine

Wrap a key stage to retry when output quality is low. Replaces `dspy.Assert` / `dspy.Suggest` (removed in DSPy 3.x):

```python
def summary_quality(args, pred):
    """Reward function: returns a score in [0, 1]."""
    words = pred.summary.split()
    if len(words) < 20:
        return 0.0   # too short
    if len(words) > 100:
        return 0.5   # too long, partial credit
    return 1.0

class SummarizeThenTag(dspy.Module):
    def __init__(self):
        raw_summarizer = dspy.ChainOfThought(Summarize)
        self.summarize = dspy.Refine(       # retry up to 3× if reward < 0.8
            module=raw_summarizer,
            N=3,
            reward_fn=summary_quality,
            threshold=0.8,
        )
        self.tag = dspy.Predict(TagContent)

    def forward(self, document):
        summary = self.summarize(document=document)
        return self.tag(summary=summary.summary)
```

## End-to-end: production content moderation pipeline

A realistic three-stage pipeline with per-stage LM assignment, end-to-end optimization, and persistence.

```python
import dspy
from typing import Literal
from dspy.evaluate import Evaluate

cheap_lm   = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-haiku-4-5-20251001"
quality_lm = dspy.LM("openai/gpt-4o")        # or "anthropic/claude-sonnet-4-5-20250929"
dspy.configure(lm=cheap_lm)

# --- Signatures ---

VIOLATION_TYPES = ["spam", "hate", "misinformation", "self_harm", "benign"]

class ClassifyContent(dspy.Signature):
    """Classify user-submitted content for policy violations."""
    content: str = dspy.InputField()
    violation_type: Literal[tuple(VIOLATION_TYPES)] = dspy.OutputField()
    confidence: Literal["low", "medium", "high"] = dspy.OutputField()

class ExtractSignals(dspy.Signature):
    """Extract specific phrases or patterns that support the classification."""
    content: str = dspy.InputField()
    violation_type: str = dspy.InputField()
    signals: list[str] = dspy.OutputField(desc="Verbatim phrases that indicate the violation type")

class DecideAction(dspy.Signature):
    """Decide the moderation action given violation type, confidence, and supporting signals."""
    content: str = dspy.InputField()
    violation_type: str = dspy.InputField()
    confidence: str = dspy.InputField()
    signals: list[str] = dspy.InputField()
    action: Literal["allow", "flag_for_review", "remove", "escalate"] = dspy.OutputField()
    reason: str = dspy.OutputField(desc="One-sentence explanation for the action")

# --- Pipeline ---

class ContentModerationPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict(ClassifyContent)       # cheap: classification
        self.extract  = dspy.Predict(ExtractSignals)        # cheap: structured extraction
        self.decide   = dspy.ChainOfThought(DecideAction)   # quality: nuanced judgment

    def forward(self, content):
        cls     = self.classify(content=content)
        signals = self.extract(content=content, violation_type=cls.violation_type)
        return self.decide(
            content=content,
            violation_type=cls.violation_type,
            confidence=cls.confidence,
            signals=signals.signals,
        )

pipeline = ContentModerationPipeline()

# Cheap model for simple stages, quality model where judgment matters
pipeline.classify.lm = cheap_lm
pipeline.extract.lm  = cheap_lm
pipeline.decide.lm   = quality_lm

# --- Training data ---

trainset = [
    dspy.Example(content="Buy cheap meds online!! click here!!!", action="remove").with_inputs("content"),
    dspy.Example(content="I hate [group] so much", action="escalate").with_inputs("content"),
    dspy.Example(content="Anyone else struggling with anxiety lately?", action="allow").with_inputs("content"),
    dspy.Example(content="COVID vaccines contain microchips — spread the word", action="remove").with_inputs("content"),
    # Add 50–200 examples for meaningful optimization
]

# --- Baseline evaluation ---

def metric(example, pred, trace=None):
    return pred.action == example.action

devset = trainset[-1:]  # use a real held-out split in production
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)

baseline = evaluator(pipeline)
print(f"Baseline: {baseline:.1f}%")

# --- End-to-end optimization ---
# MIPROv2 tunes all three stages together, not each in isolation

optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized = optimizer.compile(pipeline, trainset=trainset[:-1])

improved = evaluator(optimized)
print(f"Optimized: {improved:.1f}%")
# Typical gain: 15–25% accuracy improvement over unoptimized baseline

# --- Save for production ---
optimized.save("content_moderation.json")

# Reload
fresh = ContentModerationPipeline()
fresh.load("content_moderation.json")
fresh.classify.lm = cheap_lm   # re-assign LMs after loading — not persisted
fresh.extract.lm  = cheap_lm
fresh.decide.lm   = quality_lm
```

## Inspect what each stage produced

```python
result = pipeline(content="Some flagged content here")
print(result.action)           # final decision
print(result.reason)           # explanation from decide stage
print(result.violation_type)   # intermediate from classify stage

dspy.inspect_history(n=3)      # raw LM calls for all three stages
```
