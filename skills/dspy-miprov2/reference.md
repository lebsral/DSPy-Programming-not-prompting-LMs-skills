> Condensed from [dspy.ai/api/optimizers/MIPROv2/](https://dspy.ai/api/optimizers/MIPROv2/). Verify against upstream for latest.

# dspy.MIPROv2 — API Reference

## Constructor

```python
dspy.MIPROv2(
    metric,
    prompt_model=None,
    task_model=None,
    teacher_settings=None,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    auto="light",
    num_candidates=None,
    num_threads=None,
    max_errors=None,
    seed=9,
    init_temperature=1.0,
    verbose=False,
    track_stats=True,
    log_dir=None,
    metric_threshold=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Metric function `(example, prediction, trace=None) -> float\|bool` |
| `prompt_model` | `LM \| None` | `None` | LM for generating instruction candidates (defaults to configured LM) |
| `task_model` | `LM \| None` | `None` | LM for running the task during bootstrapping (defaults to configured LM) |
| `teacher_settings` | `dict \| None` | `None` | Settings dict for the teacher model during bootstrapping |
| `max_bootstrapped_demos` | `int` | `4` | Max bootstrapped (generated) demos per module |
| `max_labeled_demos` | `int` | `4` | Max labeled (from trainset) demos per module |
| `auto` | `str \| None` | `"light"` | Preset: `"light"`, `"medium"`, `"heavy"`, or `None` for manual config |
| `num_candidates` | `int \| None` | `None` | Number of instruction candidates per module (required when `auto=None`) |
| `num_threads` | `int \| None` | `None` | Threads for parallel evaluation |
| `max_errors` | `int \| None` | `None` | Max errors before stopping optimization |
| `seed` | `int` | `9` | Random seed for reproducibility |
| `init_temperature` | `float` | `1.0` | Temperature for generating instruction candidates |
| `verbose` | `bool` | `False` | Print detailed optimization progress |
| `track_stats` | `bool` | `True` | Track optimization statistics |
| `log_dir` | `str \| None` | `None` | Directory for optimization logs |
| `metric_threshold` | `float \| None` | `None` | Early stopping threshold — stop if metric reaches this value |

## compile()

```python
optimizer.compile(
    student,
    *,
    trainset,
    teacher=None,
    valset=None,
    num_trials=None,
    max_bootstrapped_demos=None,
    max_labeled_demos=None,
    seed=None,
    minibatch=True,
    minibatch_size=35,
    minibatch_full_eval_steps=5,
    program_aware_proposer=True,
    data_aware_proposer=True,
    view_data_batch_size=10,
    tip_aware_proposer=True,
    fewshot_aware_proposer=True,
    provide_traceback=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `Module` | required | The DSPy program to optimize |
| `trainset` | `list` | required | Training examples (keyword-only) |
| `teacher` | `Module \| None` | `None` | Optional teacher program for bootstrapping |
| `valset` | `list \| None` | `None` | Validation set (defaults to trainset subset) |
| `num_trials` | `int \| None` | `None` | Bayesian optimization trials (required when `auto=None`) |
| `max_bootstrapped_demos` | `int \| None` | `None` | Override constructor value for this run |
| `max_labeled_demos` | `int \| None` | `None` | Override constructor value for this run |
| `seed` | `int \| None` | `None` | Override constructor seed |
| `minibatch` | `bool` | `True` | Use minibatch evaluation for faster search |
| `minibatch_size` | `int` | `35` | Examples per minibatch |
| `minibatch_full_eval_steps` | `int` | `5` | Full evaluation every N steps |
| `program_aware_proposer` | `bool` | `True` | Use program structure to propose instructions |
| `data_aware_proposer` | `bool` | `True` | Use training data to propose instructions |
| `view_data_batch_size` | `int` | `10` | Examples shown to proposer per batch |
| `tip_aware_proposer` | `bool` | `True` | Include optimization tips in proposals |
| `fewshot_aware_proposer` | `bool` | `True` | Consider few-shot demos when proposing |
| `provide_traceback` | `bool \| None` | `None` | Include tracebacks in error reporting |

Returns the optimized `Module` with tuned instructions and demonstrations.

## Other methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `get_params` | `() -> dict` | Returns teleprompter parameters as a dictionary |

## Deprecated parameters

- `requires_permission_to_run` — removed from `compile()`. Passing `True` raises `ValueError`; `False` triggers a deprecation warning.

## Zeroshot mode

Setting both `max_bootstrapped_demos=0` and `max_labeled_demos=0` makes MIPROv2 optimize instructions only, discarding all few-shot demonstrations. Useful when you want shorter prompts or when demonstrations do not help.
