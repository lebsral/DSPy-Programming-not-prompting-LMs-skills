# W&B Weave API Reference

> Condensed from [docs.wandb.ai/weave](https://docs.wandb.ai/weave/) and [DSPy integration guide](https://docs.wandb.ai/weave/guides/integrations/dspy). Verify against upstream for latest.
> Weave version: 0.52.43 (2026-06-28)

## Setup

```bash
pip install weave
```

```python
import weave
weave.init("project-name")  # creates project at wandb.ai; auto-traces all DSPy calls
```

Set `WANDB_API_KEY` env var for non-interactive auth.

## weave.init

```python
weave.init(project_name: str)
```

Initializes Weave tracing for a project. Call once at startup before any DSPy calls. For DSPy projects, this is sufficient — all `dspy.Module`, `dspy.Signature`, and optimizer calls are automatically traced.

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_name` | `str` | W&B project name (created if it does not exist). Format: `"project"` or `"team/project"` |

## DSPy auto-tracing

After `weave.init()`, Weave automatically captures traces for:
- All `dspy.Module` subclasses and their `forward()` calls
- All `dspy.Signature` inputs/outputs
- `dspy.Retrieve` and retrieval module calls
- Optimizer runs (`dspy.BootstrapFewShot`, `dspy.MIPROv2`, etc.)
- LM calls and token counts

No decorators required on DSPy code.

## @weave.op()

```python
@weave.op()
def my_function(x: str) -> str:
    ...
```

Decorator that traces a non-DSPy function's inputs, outputs, latency, and cost. Use for custom business logic, preprocessing, or API calls you want in the trace tree alongside DSPy calls.

Key behavior:
- Nested `@weave.op()` calls and DSPy sub-calls both appear in the call tree
- Must be the outermost decorator when combined with others
- Apply to regular functions, not DSPy module classes (DSPy modules are auto-traced)

## weave.Evaluation

```python
weave.Evaluation(dataset, scorers)
```

Built-in evaluation harness that logs results to Weave. For DSPy, typically use `dspy.evaluate.Evaluate` instead and capture results via a `@weave.op()` wrapper.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | API key from wandb.ai/settings |
| `WANDB_ENTITY` | Team name (optional) |
| `WANDB_PROJECT` | Default project name (optional) |

## Dashboard

View traces at [wandb.ai](https://wandb.ai) under your project's "Traces" tab. Compare runs side-by-side, sort by score, share with team.
