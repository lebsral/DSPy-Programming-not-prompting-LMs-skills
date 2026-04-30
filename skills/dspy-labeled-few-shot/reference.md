> Condensed from [dspy.ai/api/optimizers/LabeledFewShot](https://dspy.ai/api/optimizers/LabeledFewShot/). Verify against upstream for latest.

# dspy.LabeledFewShot — API Reference

## Constructor

```python
dspy.LabeledFewShot(k=16)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `16` | Maximum number of demonstration examples to include per predictor |

## compile()

```python
optimizer.compile(student, *, trainset, sample=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `dspy.Module` | required | The DSPy program to optimize (a copy is made) |
| `trainset` | `list[dspy.Example]` | required | Labeled examples to use as demonstrations |
| `sample` | `bool` | `True` | `True` = randomly sample `k` examples (fixed seed 0); `False` = take first `k` in order |

**Returns:** A deep copy of `student` with up to `k` demonstrations attached to each predictor's `demos` attribute. If `trainset` is empty, returns the student unmodified.

## Key methods

| Method | Description |
|--------|-------------|
| `compile(student, *, trainset, sample=True)` | Attach demos to all predictors in the student program |
| `get_params()` | Returns list of `(name, param)` tuples for all named parameters |

## Behavior details

- Uses `random.Random(0)` for reproducible sampling when `sample=True`
- Iterates over all `NamedPredictors` in the student and assigns the same set of demos to each
- No metric is required — examples are used as-is without evaluation
- No LM calls during compilation — this is purely a data-copying step
- `compile()` creates a `deepcopy` of the student before modifying it

## Save / Load

```python
# Save optimized program
optimized.save("my_program.json")

# Load later
program = dspy.Predict(MySignature)
program.load("my_program.json")
```
