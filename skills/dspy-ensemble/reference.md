> Condensed from [dspy.ai/api/optimizers/Ensemble/](https://dspy.ai/api/optimizers/Ensemble/). Verify against upstream for latest.

# dspy.Ensemble -- API Reference

## Constructor

```python
dspy.Ensemble(
    reduce_fn=None,       # Callable | None
    size=None,            # int | None
    deterministic=False,  # bool (must be False -- not yet implemented)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reduce_fn` | `Callable | None` | `None` | Aggregation function applied to the list of predictions. If `None`, returns the raw list. |
| `size` | `int | None` | `None` | Number of programs to randomly sample per inference call. `None` = use all programs. |
| `deterministic` | `bool` | `False` | Reserved for future use. Must be `False` -- setting `True` raises `NotImplementedError`. |

## compile()

```python
ensemble_optimizer.compile(programs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `programs` | `list[dspy.Module]` | List of optimized DSPy programs to combine |

**Returns:** `EnsembledProgram` -- a callable module that runs the selected programs and applies `reduce_fn`.

## EnsembledProgram

The object returned by `compile()`. When called:

1. If `size` is set, randomly samples `size` programs from the list
2. Runs each selected program on the same inputs
3. Applies `reduce_fn` to the list of predictions
4. Returns the reduced result (or the raw list if `reduce_fn=None`)

```python
# Use like any DSPy module
result = ensemble_program(question="What is DSPy?")
print(result.answer)
```

## Built-in reduce functions

### dspy.majority

Picks the most frequent output value across all predictions (majority voting).

```python
ensemble = dspy.Ensemble(reduce_fn=dspy.majority)
```

Best for categorical outputs: classification labels, short factual answers, yes/no.

## Custom reduce function protocol

A reduce function receives a list of `dspy.Prediction` objects and must return a single `dspy.Prediction` (or compatible object with the same output fields).

```python
def my_reduce(predictions: list[dspy.Prediction]) -> dspy.Prediction:
    # Aggregate predictions
    # Return a Prediction-like object
    ...
```

### Example: averaging numeric outputs

```python
def average_scores(predictions):
    scores = [float(p.score) for p in predictions]
    avg = sum(scores) / len(scores)
    return predictions[0].__class__(score=str(avg))
```

### Example: weighted voting

```python
from collections import Counter

def weighted_vote(predictions):
    votes = Counter(p.answer for p in predictions)
    winner = votes.most_common(1)[0][0]
    return predictions[0].__class__(answer=winner)
```

## Key behaviors

- Ensemble does **not** modify the constituent programs -- it only combines their outputs at inference time
- Each program retains its own LM context from when it was optimized
- Programs are run sequentially (not parallelized by default)
- With `size=None`, all programs run on every call; with `size=N`, a random subset of N is sampled each time
