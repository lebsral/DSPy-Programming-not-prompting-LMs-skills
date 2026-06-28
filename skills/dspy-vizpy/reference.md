# VizPy API Reference

> Condensed from [vizpy-docs.vizops.ai](https://vizpy-docs.vizops.ai). Verify against upstream for latest.

## Setup

```bash
pip install vizpy  # current version: 2.4.0
```

```python
import vizpy
vizpy.api_key = "your-key"  # or set VIZPY_API_KEY env var
```

## vizpy.Score

VizPy metrics must return a `vizpy.Score` object — not a plain float, bool, or dict.

```python
vizpy.Score(
    value=float,     # 0.0 to 1.0 — overall quality score
    is_success=bool, # True if this example is considered passing
    feedback=str,    # explanation of failure; empty string if correct
    error_type=str,  # optional — enables stratified batch sampling
)
```

ContraPrompt uses `feedback` to generate contrastive improvement rules. Without it, optimization degrades silently.

## ContraPromptOptimizer (classification)

```python
optimizer = vizpy.ContraPromptOptimizer(
    metric=metric,                    # required — returns vizpy.Score
    config=vizpy.ContraPromptConfig(  # optional
        max_iterations=5,             # optimization passes
        max_attempts=3,               # retries per iteration
        validate_rules=True,          # validate generated rules
    ),
    feedback_generator=None,          # optional custom feedback logic
    example_formatter=None,           # optional custom formatter
)
optimized = optimizer.optimize(module, train_examples=trainset, val_examples=devset)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | `(example, prediction, trace=None) -> vizpy.Score` |
| `config` | `ContraPromptConfig` | None | Optional tuning config |
| `feedback_generator` | `Callable` | None | Custom failure explanation logic |
| `example_formatter` | `Callable` | None | Custom example formatting |

**optimize() parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | DSPy module to optimize |
| `train_examples` | `list[dict]` | required | Labeled training data |
| `val_examples` | `list[dict]` | None | Optional held-out validation set |

Uses contrastive examples to generate instructions that distinguish confusing categories.

## PromptGradOptimizer (generation)

```python
optimizer = vizpy.PromptGradOptimizer(
    metric=metric,                     # required — returns vizpy.Score
    config=vizpy.PromptGradConfig(),   # optional
    base_prompt_source="module",       # optional — where to read initial prompt
    example_formatter=None,            # optional custom formatter
    rule_acceptor=None,                # optional custom rule acceptance logic
)
optimized = optimizer.optimize(module, train_examples=trainset, val_examples=devset)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | `(example, prediction, trace=None) -> vizpy.Score` |
| `config` | `PromptGradConfig` | None | Optional tuning config |
| `base_prompt_source` | `str` | `"module"` | Where to read the starting instruction |
| `example_formatter` | `Callable` | None | Custom example formatting |
| `rule_acceptor` | `Callable` | None | Custom rule acceptance logic |

optimize() parameters are the same as ContraPromptOptimizer.

Uses gradient-inspired optimization to iteratively improve instructions across epochs.

## What Gets Tuned

- Instructions only (same scope as `dspy.GEPA`)
- Does NOT tune: few-shot demos, Pydantic field descriptions, model weights

## Metric format comparison

| Optimizer | Metric returns |
|-----------|----------------|
| VizPy both | `vizpy.Score(value, is_success, feedback)` |
| dspy.GEPA | `{"score": float, "feedback": str}` |
| dspy.MIPROv2 | `float` or `bool` |

## Pricing

| Tier | Runs/month | Cost |
|------|-----------|------|
| Free | 10 | $0 |
| Pro | 200 | $20/mo |
| Enterprise | 1,000 | $200/mo |
