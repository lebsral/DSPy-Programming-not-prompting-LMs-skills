# VizPy API Reference

> Condensed from [vizpy.vizops.ai](https://vizpy.vizops.ai). Verify against upstream for latest.

## Setup

```bash
pip install vizpy
```

```python
import vizpy
vizpy.api_key = "your-key"  # or set VIZPY_API_KEY env var
```

## ContraPromptOptimizer (classification)

```python
optimizer = vizpy.ContraPromptOptimizer(metric=metric)
optimized = optimizer.compile(program, trainset=trainset)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | `(example, prediction, trace=None) -> float` |

Uses contrastive examples to generate instructions that distinguish confusing categories.

## PromptGradOptimizer (generation)

```python
optimizer = vizpy.PromptGradOptimizer(metric=metric)
optimized = optimizer.compile(program, trainset=trainset)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | `(example, prediction, trace=None) -> float` |

Uses gradient-inspired optimization to iteratively improve instructions.

## Common Interface

Both optimizers share the same `.compile()` interface as DSPy native optimizers:

```python
optimized = optimizer.compile(program, trainset=trainset)
optimized.save("optimized.json")   # standard DSPy save
optimized.load("optimized.json")   # standard DSPy load
```

## What Gets Tuned

- Instructions only (same scope as `dspy.GEPA`)
- Does NOT tune: few-shot demos, Pydantic field descriptions, model weights

## Pricing

| Tier | Runs/month | Cost |
|------|-----------|------|
| Free | 10 | $0 |
| Pro | Unlimited | $20/mo |
