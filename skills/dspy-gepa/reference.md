> Condensed from [dspy.ai/api/optimizers/GEPA/](https://dspy.ai/api/optimizers/GEPA/). Verify against upstream for latest.

# dspy.GEPA — API Reference

**Inherits from:** `Teleprompter`

## Constructor

```python
dspy.GEPA(
    metric,                              # GEPAFeedbackMetric (required)
    *,
    # Budget (exactly one required)
    auto=None,                           # "light", "medium", or "heavy"
    max_full_evals=None,                 # int -- full validation passes allowed
    max_metric_calls=None,               # int -- total metric invocations allowed

    # Reflection
    reflection_lm=None,                  # LM for proposing new instructions
    reflection_minibatch_size=3,         # examples per reflection step
    candidate_selection_strategy="pareto",  # "pareto" or "current_best"
    skip_perfect_score=True,             # skip examples already scoring perfectly
    add_format_failure_as_feedback=False, # include format errors in feedback
    instruction_proposer=None,           # custom ProposalFn
    component_selector="round_robin",    # which predictor to improve next

    # Merging
    use_merge=True,                      # merge successful variants
    max_merge_invocations=5,             # merge attempt limit

    # Evaluation
    num_threads=None,                    # parallel evaluation threads
    failure_score=0.0,                   # score for failed examples
    perfect_score=1.0,                   # score that counts as perfect

    # Logging
    log_dir=None,                        # directory for optimization logs
    track_stats=False,                   # return detailed metadata
    track_best_outputs=False,            # retain best outputs per task
    use_wandb=False,                     # W&B integration
    use_mlflow=False,                    # MLflow integration

    # Reproducibility
    seed=0,                              # random seed
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `GEPAFeedbackMetric` | required | Returns float or `{"score": float, "feedback": str}` |
| `auto` | `Literal["light","medium","heavy"] \| None` | `None` | Budget preset. Exactly one of `auto`, `max_full_evals`, `max_metric_calls` required |
| `max_full_evals` | `int \| None` | `None` | Number of full passes over the validation set |
| `max_metric_calls` | `int \| None` | `None` | Total metric invocations allowed |
| `reflection_lm` | `LM \| None` | `None` | LM that proposes new instructions. Use a strong model |
| `reflection_minibatch_size` | `int` | `3` | Examples per reflection step. Larger = better proposals, more cost |
| `candidate_selection_strategy` | `str` | `"pareto"` | `"pareto"` maintains diverse candidates; `"current_best"` mutates top scorer |
| `skip_perfect_score` | `bool` | `True` | Skip examples already scoring `perfect_score` |
| `add_format_failure_as_feedback` | `bool` | `False` | Include format errors as feedback |
| `instruction_proposer` | `ProposalFn \| None` | `None` | Custom proposal function |
| `component_selector` | `str` | `"round_robin"` | Which predictor to improve next |
| `use_merge` | `bool` | `True` | Merge best modules from different lineages |
| `max_merge_invocations` | `int \| None` | `5` | Cap on merge attempts |
| `num_threads` | `int \| None` | `None` | Parallel evaluation threads |
| `failure_score` | `float` | `0.0` | Score assigned to failed examples |
| `perfect_score` | `float` | `1.0` | Score that counts as perfect |
| `log_dir` | `str \| None` | `None` | Directory for optimization logs |
| `track_stats` | `bool` | `False` | Attach `DspyGEPAResult` to `optimized.detailed_results` |
| `track_best_outputs` | `bool` | `False` | Retain best outputs per task |
| `seed` | `int \| None` | `0` | Random seed for reproducibility |

## compile()

```python
optimized = gepa.compile(
    student,              # dspy.Module (required)
    *,
    trainset,             # list[Example] (required)
    valset=None,          # list[Example] | None -- auto-uses trainset if None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `dspy.Module` | required | The program to optimize |
| `trainset` | `list[Example]` | required | Training examples |
| `valset` | `list[Example] \| None` | `None` | Validation examples. If `None`, uses trainset for both |

**Returns:** Optimized `dspy.Module` with improved instructions. If `track_stats=True`, result has a `detailed_results` attribute of type `DspyGEPAResult`.

**Note:** The `teacher` parameter exists in the signature but is not currently supported (`assert teacher is None`).

## Key methods

| Method | Description |
|--------|-------------|
| `compile(student, *, trainset, valset=None)` | Optimize instructions via reflective evolution |
| `get_params()` | Returns optimizer parameters as dict |
| `auto_budget(num_preds, num_candidates, valset_size, minibatch_size=35, full_eval_steps=5)` | Calculate budget for auto mode |

## GEPAFeedbackMetric protocol

```python
def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """
    Args:
        gold: Expected Example (ground truth)
        pred: Model Prediction
        trace: Full program execution trace
        pred_name: Name of the predictor being optimized
        pred_trace: Sub-trace for this predictor

    Returns:
        float -- simple score
        OR {"score": float, "feedback": str} -- score with textual feedback
    """
```

## DspyGEPAResult (when track_stats=True)

Key properties:
- `best_idx` — index of best candidate
- `best_candidate` — the best optimized module
- `candidates` — all candidate modules
- `val_aggregate_scores` — scores per candidate
- `best_outputs_valset` — best outputs per task (if `track_best_outputs=True`)

## What GEPA does NOT optimize

| Element | Optimized? |
|---------|-----------|
| Signature docstring | Yes |
| `InputField(desc=...)` | No |
| `OutputField(desc=...)` | No |
| Pydantic `Field(description=...)` | No |
| Field names | No |
| Type constraints | No |
| Few-shot demos | No |
