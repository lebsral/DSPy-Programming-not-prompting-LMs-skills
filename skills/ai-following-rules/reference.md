> Condensed from [dspy.ai/api/modules/Refine/](https://dspy.ai/api/modules/Refine/) and [dspy.ai/api/modules/BestOfN/](https://dspy.ai/api/modules/BestOfN/). Verify against upstream for latest.

# DSPy API Reference for Following Rules

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module whose outputs to refine |
| `N` | `int` | required | Maximum number of attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores each prediction; higher is better |
| `threshold` | `float` | required | Accept output immediately when score >= this value |
| `fail_count` | `int \| None` | `N` | Raise an error after this many failures |

**Behavior:** Sequential. Each attempt runs at `temperature=1.0` with a different rollout ID. When an attempt fails to meet `threshold`, Refine generates natural-language feedback and includes it in the next attempt's prompt. Returns the first attempt that meets or exceeds `threshold`, or the highest-scoring attempt after all N runs.

## dspy.BestOfN

[API docs](https://dspy.ai/api/modules/BestOfN/)

```python
dspy.BestOfN(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to run N times |
| `N` | `int` | required | Maximum number of attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores each prediction; higher is better |
| `threshold` | `float` | required | Early-stop when any attempt scores >= this value |
| `fail_count` | `int \| None` | `N` | Raise an error after this many failures |

**Behavior:** Independent attempts. Each runs at `temperature=1.0` with a unique rollout ID. No cross-attempt feedback — each attempt is unaware of previous results. Returns the attempt with the highest score, or the first to meet `threshold`.

## Comparing Refine and BestOfN

| | `dspy.Refine` | `dspy.BestOfN` |
|-|--------------|----------------|
| **Attempts** | Sequential with feedback | Independent |
| **Cross-attempt learning** | Yes — failures inform the next attempt | No |
| **Best for** | Format compliance, iterative correction | Sampling diversity, creative generation |
| **On threshold met** | Returns immediately | Returns immediately |
| **N sweet spot** | 3–5 | 3–5 (above 5: diminishing returns) |

## Reward function signature

Both `dspy.Refine` and `dspy.BestOfN` require a `reward_fn` with this exact signature:

```python
def reward_fn(args: dict, pred: dspy.Prediction) -> float:
    # args — keyword arguments passed to the module call (e.g. args["question"])
    # pred — the module's output (e.g. pred.answer, pred.quiz)
    # return — float; higher is better; 0.0 = fail, 1.0 = perfect
    ...
```

This is **not** the `(example, prediction, trace=None)` signature used by `dspy.Evaluate` metrics. The `args` dict contains only inputs, not gold labels.

## Structured outputs with Pydantic

Use a `pydantic.BaseModel` as a typed `OutputField` to enforce structure. DSPy validates the output automatically — no reward logic needed for structural constraints.

```python
from pydantic import BaseModel, Field
from typing import Literal

class Report(BaseModel):
    title: str = Field(min_length=5)
    body: str = Field(min_length=50)
    sentiment: Literal["positive", "neutral", "negative"]

class Analyze(dspy.Signature):
    """Analyze the document and produce a structured report."""
    document: str = dspy.InputField()
    report: Report = dspy.OutputField()
```

Pydantic handles structural rules (types, field presence, enum values, min/max lengths). Add a reward function with Refine only for **logic** rules Pydantic cannot express — for example, `correct_answer must be one of the options`.

## Removed in DSPy 3.x

`dspy.Assert` and `dspy.Suggest` were removed in DSPy 3.x. Do not use them. Use `dspy.Refine` or `dspy.BestOfN` with a reward function instead.

## Quick reference

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

# Refine — retry with feedback until threshold met
enforced = dspy.Refine(module=my_module, N=3, reward_fn=reward_fn, threshold=0.8)

# BestOfN — run N times independently, return highest-scoring result
best = dspy.BestOfN(module=my_module, N=5, reward_fn=reward_fn, threshold=1.0)

# Both wrap any dspy.Module and return dspy.Prediction — same call interface
result = enforced(question="...")
result = best(question="...")

# Optimize the BASE module, then wrap for production enforcement
optimized = optimizer.compile(my_module, trainset=trainset)
production = dspy.Refine(optimized, N=3, reward_fn=reward_fn, threshold=0.8)
```

### Threshold guidance

| Goal | Threshold |
|------|-----------|
| Strict binary rule — must pass or error | `1.0` with binary reward |
| Multi-criteria quality bar | `0.8` with graduated reward |
| Best-effort — always return something | `0.0` (no early stopping) |
