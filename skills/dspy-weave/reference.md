# W&B Weave API Reference

> Condensed from [docs.wandb.ai/weave](https://docs.wandb.ai/weave/). Verify against upstream for latest.

## Setup

```bash
pip install weave
```

```python
import weave
weave.init("project-name")  # creates project at wandb.ai
```

Set `WANDB_API_KEY` env var for non-interactive auth.

## weave.init

```python
weave.init(project_name: str)
```

Initializes Weave tracing for a project. Call once at startup, before any `@weave.op()` calls.

| Parameter | Type | Description |
|-----------|------|-------------|
| `project_name` | `str` | W&B project name (created if it doesn't exist) |

## @weave.op()

```python
@weave.op()
def my_function(x: str) -> str:
    ...
```

Decorator that traces a function's inputs, outputs, latency, and cost. Apply to regular functions, not DSPy module classes.

Key behavior:
- Nested `@weave.op()` calls create a call tree in the dashboard
- Must be the outermost decorator when combined with others
- Each function needs its own decorator (no global auto-instrumentation)

## Environment Variables

| Variable | Description |
|----------|-------------|
| `WANDB_API_KEY` | API key from wandb.ai/settings |
| `WANDB_ENTITY` | Team name (optional) |
| `WANDB_PROJECT` | Default project name (optional) |

## Dashboard

View traces at [wandb.ai](https://wandb.ai) under your project's "Traces" tab. Compare runs side-by-side, sort by score, share with team.
