> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Content Generation

## Quick config

```python
pip install -U dspy          # DSPy 3.2.1+

import dspy
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

## dspy.Signature

[API docs](https://dspy.ai/api/signatures/)

```python
class MySignature(dspy.Signature):
    """Docstring becomes the task instruction."""
    input_name: str = dspy.InputField(desc="description")
    output_name: str = dspy.OutputField(desc="description")
```

| Field constructor | Key parameter | Description |
|-------------------|--------------|-------------|
| `dspy.InputField` | `desc` | Human-readable description passed in the prompt |
| `dspy.OutputField` | `desc` | Describes what the LM should produce |

Typed outputs (Pydantic) work directly as output fields — DSPy enforces the schema:

```python
from pydantic import BaseModel, Field

class ContentOutline(BaseModel):
    title: str
    sections: list[Section]

class GenerateOutline(dspy.Signature):
    """Create a structured outline for the content."""
    topic: str = dspy.InputField(desc="Topic or brief to write about")
    content_type: str = dspy.InputField(desc="blog post, report, product description, etc.")
    audience: str = dspy.InputField(desc="Who will read this content")
    outline: ContentOutline = dspy.OutputField()
```

## dspy.ChainOfThought and dspy.Predict

[ChainOfThought docs](https://dspy.ai/api/modules/ChainOfThought/) — [Predict docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
dspy.Predict(signature, **config)
```

| Module | Adds reasoning field | When to use |
|--------|---------------------|-------------|
| `dspy.ChainOfThought` | Yes (injected automatically) | Outline planning, section writing, critique, improve |
| `dspy.Predict` | No | AI-as-judge scoring where reasoning adds unnecessary cost |

Do not add a `reasoning` field to your signature — `ChainOfThought` injects it automatically.

## Content pipeline stages

| Stage | Signature inputs | Output type | Typical module |
|-------|-----------------|-------------|----------------|
| Outline | `topic`, `content_type`, `audience` | `ContentOutline` (Pydantic) | `ChainOfThought` |
| Write section | `topic`, `section_heading`, `key_points`, `previous_sections`, `tone` | `str` | `ChainOfThought` |
| Research queries | `topic`, `key_points` | `list[str]` | `ChainOfThought` |
| Critique | `content`, `content_type`, `audience` | `bool` + `str` (feedback) | `ChainOfThought` |
| Improve | `content`, `feedback` | `str` | `ChainOfThought` |
| Judge | `content`, `content_type`, `topic` | `float` fields | `Predict` |

Pass `previous_sections=running_text[-2000:]` (last 2000 chars) to each section writer to maintain continuity without overflowing context.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to wrap |
| `N` | `int` | required | Max attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Returns 0.0–1.0; stops when score >= threshold |
| `threshold` | `float` | required | Early-stop threshold; returns first prediction that meets it or best overall |
| `fail_count` | `int \| None` | `None` | Failures allowed before raising error; defaults to N if not set |

Use `dspy.Refine` (not `dspy.Assert`/`dspy.Suggest`, which were removed in DSPy 3.x) to enforce brand voice, length constraints, or forbidden words. The reward function receives the original call args and the prediction:

```python
def brand_reward(args, prediction) -> float:
    score = 1.0
    for word in ["utilize", "leverage", "synergy"]:
        if word in prediction.article.lower():
            score -= 0.2
    return max(score, 0.0)

refined_writer = dspy.Refine(module=writer, N=3, reward_fn=brand_reward, threshold=0.8)
```

## Batch generation

All DSPy modules expose `.batch()` for parallel execution:

```python
results = my_module.batch(
    examples,                     # list[dspy.Example]
    num_threads=4,                # parallel threads
    max_errors=5,                 # abort threshold
    return_failed_examples=False,
)
```

For simple loops (e.g., catalog generation), call the module directly per item — `.batch()` is most useful when you have a pre-built `list[dspy.Example]` and want thread pooling.

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, max_bootstrapped_demos=4,
                      max_labeled_demos=16, max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | `(example, prediction, trace) -> float` |
| `max_bootstrapped_demos` | `int` | `4` | Generated few-shot demos per predictor |
| `max_labeled_demos` | `int` | `16` | Labeled demos drawn from trainset |

```python
optimizer = dspy.BootstrapFewShot(metric=content_quality_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(QualityWriter(), trainset=trainset)
optimized.save("optimized_writer.json")
```

After optimization, wrap the optimized module with `dspy.Refine` for runtime quality enforcement — optimization tunes prompts, `Refine` enforces hard constraints at inference time.

## AI-as-judge metric pattern

```python
class JudgeContent(dspy.Signature):
    """Judge the quality of generated content."""
    content: str = dspy.InputField()
    content_type: str = dspy.InputField()
    topic: str = dspy.InputField()
    relevance: float = dspy.OutputField(desc="0.0–1.0")
    coherence: float = dspy.OutputField(desc="0.0–1.0")
    engagement: float = dspy.OutputField(desc="0.0–1.0")

def content_quality_metric(example, prediction, trace=None):
    judge = dspy.Predict(JudgeContent)
    result = judge(content=prediction.article, content_type=example.content_type, topic=example.topic)
    return (result.relevance + result.coherence + result.engagement) / 3
```

Metric signature is always `(example, prediction, trace=None) -> float`. Reward function signature for `Refine`/`BestOfN` is `(args, prediction) -> float` — different interface.
