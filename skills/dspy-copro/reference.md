> Condensed from [dspy.ai/api/optimizers/COPRO/](https://dspy.ai/api/optimizers/COPRO/). Verify against upstream for latest.

# dspy.COPRO — API Reference

## Constructor

```python
dspy.COPRO(
    prompt_model=None,      # LM | None
    metric=None,            # Callable | None
    breadth=10,             # int (must be >1)
    depth=3,                # int
    init_temperature=1.4,   # float
    track_stats=False,      # bool
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt_model` | `dspy.LM | None` | `None` | LM used to generate instruction candidates. If `None`, uses the globally configured LM. |
| `metric` | `Callable | None` | `None` | Scoring function `(example, prediction, trace=None) -> float|bool`. Required for optimization. |
| `breadth` | `int` | `10` | Number of candidate instructions generated per iteration. Must be >1. Higher = wider search, more LM calls. |
| `depth` | `int` | `3` | Number of optimization rounds. Each round refines candidates from the previous round. |
| `init_temperature` | `float` | `1.4` | Temperature for generating candidates. Higher = more diverse candidates. |
| `track_stats` | `bool` | `False` | When `True`, collects per-iteration statistics (max, average, min, std dev of scores). |

## compile()

```python
optimized = optimizer.compile(
    student,                # dspy.Module (required, modified in-place)
    *,
    trainset,               # list[dspy.Example] (required, keyword-only)
    eval_kwargs,            # dict (required, keyword-only)
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `student` | `dspy.Module` | The program to optimize. **Modified in-place** and also returned. |
| `trainset` | `list[dspy.Example]` | Training examples for evaluating candidates. |
| `eval_kwargs` | `dict` | Passed to `dspy.Evaluate`. Common keys: `num_threads`, `display_progress`, `display_table`. |

**Note:** `trainset` and `eval_kwargs` are keyword-only arguments (after `*`). Always pass them as `trainset=...`, `eval_kwargs=...`.

**Returns:** The optimized program (same object as `student`) with additional attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `candidate_programs` | `dict` | All evaluated candidates with their scores, keyed by predictor name. |
| `total_calls` | `int` | Total number of LM API calls made during optimization. |

## get_params()

```python
optimizer.get_params() -> dict[str, Any]
```

Returns the optimizer's configuration parameters as a dictionary.

## Optimization process

1. **Seed phase (iteration 0):** Takes existing instruction from each signature. Generates `breadth - 1` alternative instructions using temperature-controlled sampling.
2. **Evaluate phase:** Scores every candidate by swapping it into the program and running the metric against the full training set. Duplicate (instruction, prefix) pairs are skipped.
3. **Refine phase (iterations 1 through depth-1):** Generates new candidates informed by the best performers from previous rounds.
4. **Multi-predictor handling:** Optimizes predictors sequentially — locks in the best instruction for predictor 1 before moving to predictor 2.
5. **Selection:** After all iterations, the instruction with the highest metric score is selected for each predictor.

## Cost estimation

Total evaluation calls ≈ `breadth * depth * len(trainset)` per predictor, plus candidate generation calls.

| breadth | depth | trainset size | Approx. eval calls |
|---------|-------|---------------|-------------------|
| 5 | 2 | 50 | ~500 |
| 10 | 3 | 100 | ~3,000 |
| 25 | 3 | 100 | ~7,500 |
| 50 | 3 | 200 | ~30,000 |
