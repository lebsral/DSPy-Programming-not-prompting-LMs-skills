> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Image Understanding

## dspy.Image

[API docs](https://dspy.ai/api/primitives/)

```python
dspy.Image(url: str)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | `str` | HTTPS URL, local file path, or base64 data URI |

The `url` parameter accepts three forms:

```python
dspy.Image(url="https://example.com/photo.jpg")         # HTTPS URL
dspy.Image(url="photo.jpg")                             # Local file path
dspy.Image(url=f"data:image/jpeg;base64,{b64_string}")  # Base64 data URI
```

The old `from_url()` and `from_file()` classmethods are deprecated — use `dspy.Image(url=...)` for all forms.

Declare image inputs in signatures with `dspy.Image` as the type annotation:

```python
class MySignature(dspy.Signature):
    """Docstring becomes the task instruction."""
    image: dspy.Image = dspy.InputField(desc="Image to analyze")
    result: str = dspy.OutputField(desc="Analysis result")
```

## dspy.InputField / dspy.OutputField

[API docs](https://dspy.ai/api/signatures/)

```python
dspy.InputField(desc="", prefix=None)
dspy.OutputField(desc="", prefix=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str` | `""` | Guidance shown in the prompt for this field |
| `prefix` | `str \| None` | `None` | Custom label prefix in the prompt |

`desc` is the primary lever for steering extraction quality — use it to constrain format, scope, and fallback behavior (e.g., `"Use 0.0 if not present or legible"`).

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `signature` | `str \| type[Signature]` | Defines inputs and outputs |

Use for direct extraction — OCR, field parsing, attribute categorization, alt text. No reasoning step; lower latency and cost than `ChainOfThought`.

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines inputs and outputs |
| `rationale_field` | `FieldInfo \| None` | `None` | Custom reasoning field |
| `rationale_field_type` | `type` | `str` | Type for the rationale |

Adds a `reasoning` field automatically before outputs. Do not add `reasoning` to your signature — DSPy injects it. Use for tasks requiring multi-step inference: UI analysis, chart interpretation, diagnosing visual bugs.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=None)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `dspy.Module` | Module to wrap |
| `N` | `int` | Max retry attempts |
| `reward_fn` | `Callable` | Scores output quality; returns float (higher = better) |
| `threshold` | `float \| None` | Stop early if reward meets this value |

Use when initial extraction quality is low — crumpled receipts, low-contrast screenshots, ambiguous product photos:

```python
refining_extractor = dspy.Refine(dspy.Predict(ReceiptData), N=2, reward_fn=receipt_reward)
```

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto="light", max_bootstrapped_demos=4, max_labeled_demos=4, ...)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Reward/scoring function |
| `auto` | `'light' \| 'medium' \| 'heavy'` | `'light'` | Optimization intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos from trainset |

Key method: `.compile(module, trainset=...)` — returns an optimized module. Reward functions for vision tasks return `float` in `[0.0, 1.0]`.

## Quick Reference

### Setup

```bash
pip install -U dspy
```

```python
import dspy
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

### Vision model selection

| Model | Strengths | Notes |
|-------|-----------|-------|
| `openai/gpt-4o` | Best overall quality | Higher cost |
| `openai/gpt-4o-mini` | Fast, low cost | Weaker on complex layouts |
| `anthropic/claude-sonnet-4-5-20250929` | Strong reasoning, balanced cost | Good for production |
| `google/gemini-2.5-flash` | Long context, PDF support | Verify availability |

### Image sizing recommendations

| Use case | Max longest side | Format |
|----------|-----------------|--------|
| General analysis | 1024 px | JPEG q=85 |
| Receipt / invoice OCR | 2048 px | JPEG q=92 |
| Screenshots with text | 1024 px | PNG |

### Module selection

| Task | Module |
|------|--------|
| OCR, attribute extraction, categorization | `dspy.Predict` |
| Chart analysis, UI reasoning, VQA | `dspy.ChainOfThought` |
| Retry on low-confidence or low-quality inputs | `dspy.Refine` |
| Batch processing | list comprehension over `dspy.Image(url=p)` per item |
