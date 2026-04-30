# ReAct API Reference

> Condensed from [dspy.ai/api/modules/ReAct](https://dspy.ai/api/modules/ReAct/). Verify against upstream for latest.

## Constructor

```python
dspy.ReAct(
    signature,      # str | type[Signature] -- required
    tools,          # list[Callable | dspy.Tool] -- required
    max_iters=20,   # int -- max reasoning-action cycles
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines input/output contract |
| `tools` | `list[Callable \| dspy.Tool]` | required | Functions the agent can call |
| `max_iters` | `int` | `20` | Max Thought-Action-Observation cycles |

A "finish" tool is added automatically -- the agent calls it to signal task completion.

## Key Methods

- `forward(**input_args) -> dspy.Prediction` -- run the agent loop synchronously
- `aforward(**input_args) -> dspy.Prediction` -- async variant
- `truncate_trajectory(trajectory)` -- removes oldest tool calls when context exceeds limits; override for custom truncation logic

## Inherited Module Methods

| Method | Description |
|--------|-------------|
| `batch(examples, num_threads, max_errors, ...)` | Parallel processing |
| `save(path)` | Persist learned state (demos, instructions) |
| `load(path)` | Load state into a fresh instance |
| `set_lm(lm)` | Override LM for this module |
| `get_lm()` | Get the current LM |
| `named_predictors()` | Access internal `dspy.Predict` instances |

## Return Value

`dspy.Prediction` with:
- Output fields matching your signature (e.g., `.answer`)
- `.trajectory` -- dict mapping the full Thought-Action-Observation trace

## Internal Behavior

1. Constructs an internal `react_signature` extending your signature with trajectory tracking and tool selection fields
2. Each iteration: agent produces a Thought, selects a tool (or "finish"), executes the tool, records the Observation
3. When "finish" is called, a fallback `ChainOfThought` extraction step produces the final output fields
4. All iterations run at the configured LM temperature
