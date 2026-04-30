> Condensed from [dspy.ai/api/modules/MultiChainComparison/](https://dspy.ai/api/modules/MultiChainComparison/). Verify against upstream for latest.

# dspy.MultiChainComparison — API Reference

## Constructor

```python
dspy.MultiChainComparison(
    signature,                    # Task signature (str or dspy.Signature class)
    M: int = 3,                  # Number of reasoning chains to generate and compare
    temperature: float = 0.7,    # Sampling temperature for chain generation
    **config,                    # Additional config passed to internal Predict
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| dspy.Signature` | required | The task signature defining inputs and outputs |
| `M` | `int` | `3` | Number of independent reasoning chains to generate |
| `temperature` | `float` | `0.7` | Sampling temperature. Higher values produce more diverse chains |
| `**config` | `dict` | `{}` | Additional configuration passed to the underlying Predict module |

## How it works

The constructor dynamically builds a comparison signature that:

1. Appends M input fields for each reasoning attempt (`reasoning_attempt_1` through `reasoning_attempt_M`)
2. Prepends an output field for synthesized "Accurate Reasoning"
3. The `forward()` method receives all chain completions and uses a final Predict call to select the best answer

## Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(**kwargs) -> Prediction` | Generate M chains and compare them, returning the best answer |
| `acall` | `async (**kwargs) -> Prediction` | Async version of `__call__` |
| `forward` | `(completions, **kwargs) -> Prediction` | Internal: processes pre-generated completions and synthesizes the best answer |
| `batch` | `(examples, num_threads=None, ...) -> list` | Process multiple inputs in parallel |
| `set_lm` | `(lm) -> None` | Set the language model for all internal predictors |
| `get_lm` | `() -> LM` | Returns the LM if all predictors use the same one |
| `save` | `(path) -> None` | Save the module state (including optimized prompts) to JSON |
| `load` | `(path) -> Module` | Load a saved module state |

## Cost model

With M chains, each call makes M+1 LM calls:

| M | LM calls per invocation | Relative cost vs ChainOfThought |
|---|------------------------|--------------------------------|
| 2 | 3 | 3x |
| 3 | 4 | 4x |
| 5 | 6 | 6x |
