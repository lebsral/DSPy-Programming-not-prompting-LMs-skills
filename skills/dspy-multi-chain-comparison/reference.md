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
| `M` | `int` | `3` | Number of pre-generated reasoning chains the module expects to receive |
| `temperature` | `float` | `0.7` | Sampling temperature for the comparison step. Higher values give more variation |
| `**config` | `dict` | `{}` | Additional configuration passed to the underlying Predict module |

## How it works

`MultiChainComparison` does NOT generate the reasoning chains. The caller generates M completions first (one `dspy.ChainOfThought(signature, n=M)` call) and passes the resulting `.completions` list to the module. The constructor dynamically builds a comparison signature that:

1. Appends M input fields for each reasoning attempt (`reasoning_attempt_1` through `reasoning_attempt_M`)
2. Prepends an output field for synthesized "Accurate Reasoning"
3. The `forward()` method receives the M pre-generated completions and uses a single Predict call to synthesize/select the best answer

The module itself makes exactly **1 LM call** -- the synthesis step.

## Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(completions, **kwargs) -> Prediction` | Compare the M pre-generated completions and return the best answer (1 synthesis LM call) |
| `acall` | `async (completions, **kwargs) -> Prediction` | Async version of `__call__` |
| `forward` | `(completions, **kwargs) -> Prediction` | Synthesizes the best answer from the required `completions` list of M pre-generated chains |
| `batch` | `(examples, num_threads=None, ...) -> list` | Process multiple inputs in parallel |
| `set_lm` | `(lm) -> None` | Set the language model for the comparison predictor |
| `get_lm` | `() -> LM` | Returns the LM if all predictors use the same one |
| `save` | `(path) -> None` | Save the module state (including optimized prompts) to JSON |
| `load` | `(path) -> Module` | Load a saved module state |

`completions` is the **required** first positional argument: a list of M pre-generated completions, obtained from `dspy.ChainOfThought(signature, n=M)(...).completions`.

## Cost model

`MultiChainComparison` itself makes **1 LM call** (the synthesis). Generating the M chains is a separate step -- 1 `ChainOfThought` call with `n=M` -- so the full pattern is 2 LM calls. Token cost still scales with M because the generate call samples M completions:

| M | LM calls (generate + synthesis) | Sampled completions | Relative token cost vs ChainOfThought |
|---|---------------------------------|---------------------|---------------------------------------|
| 2 | 2 (1 + 1) | 2 | ~3x |
| 3 | 2 (1 + 1) | 3 | ~4x |
| 5 | 2 (1 + 1) | 5 | ~6x |
