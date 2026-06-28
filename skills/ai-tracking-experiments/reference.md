> DSPy APIs condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Experiment Tracking

## Package versions

| Package | Install | Notes |
|---------|---------|-------|
| `dspy` | `pip install -U dspy` | DSPy 3.2.1+ |
| `weave` | `pip install weave` | W&B Weave — team dashboards |
| `langwatch` | `pip install langwatch` | Real-time optimizer progress |
| `mlflow` | `pip install mlflow` | Full ML lifecycle — see `/dspy-mlflow` |

## program.save() / program.load()

[Saving and Loading tutorial](https://dspy.ai/tutorials/saving/)

```python
program.save(path, save_program=False, modules_to_serialize=None)
program.load(path, allow_pickle=False)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `path` | required | File path string, e.g., `"artifacts/mipro_medium.json"` |
| `save_program` | `False` | If `True`, saves the entire program to a directory (requires dspy >= 2.6.0). If `False`, saves state only (demos, instructions) to a JSON file. |
| `modules_to_serialize` | `None` | Optional list of custom modules to include when `save_program=True` |
| `allow_pickle` | `False` | Set `True` when loading a pickle file (non-serializable objects like images) |

- For state-only loading (`save_program=False`), recreate the program instance first: `prog = MyProgram(); prog.load("...")`
- For whole-program loading (`save_program=True`), use `dspy.load(path)` directly — no instance needed
- Log the artifact path in your experiment record — without it the run cannot be reproduced or deployed

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
from dspy.evaluate import Evaluate

Evaluate(devset, metric=None, num_threads=None, display_progress=False,
         display_table=False, max_errors=None, provide_traceback=None,
         failure_score=0.0, save_as_csv=None, save_as_json=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function |
| `num_threads` | `int \| None` | `None` | Parallel threads |
| `display_progress` | `bool` | `False` | Show progress bar |
| `display_table` | `bool \| int` | `False` | Show results table |
| `max_errors` | `int \| None` | `None` | Max errors before stopping |
| `provide_traceback` | `bool \| None` | `None` | Include traceback on evaluation errors |
| `failure_score` | `float` | `0.0` | Score assigned on LM error |
| `save_as_csv` | `str \| None` | `None` | File path to save results as CSV |
| `save_as_json` | `str \| None` | `None` | File path to save results as JSON |

Call the evaluator instance with a module: `score = evaluator(program)`

All experiments being compared must use the **exact same devset** — scores across different devsets are not comparable.

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function |
| `max_bootstrapped_demos` | `int` | `4` | Max generated few-shot demos |
| `max_labeled_demos` | `int` | `16` | Max labeled demos from trainset |
| `max_rounds` | `int` | `1` | Bootstrap iterations |

Key method: `.compile(module, trainset=...)` — returns optimized module.

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto='light', prompt_model=None, task_model=None,
             teacher_settings=None, max_bootstrapped_demos=4, max_labeled_demos=4,
             num_candidates=None, num_threads=None, max_errors=None,
             seed=9, init_temperature=1.0, verbose=False, track_stats=True,
             log_dir=None, metric_threshold=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function |
| `auto` | `'light' \| 'medium' \| 'heavy' \| None` | `'light'` | Optimization intensity |
| `prompt_model` | `LM \| None` | `None` | LM for generating instructions (defaults to configured LM) |
| `task_model` | `LM \| None` | `None` | LM for task evaluation (defaults to configured LM) |
| `teacher_settings` | `dict \| None` | `None` | Settings for teacher model |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos |
| `num_candidates` | `int \| None` | `None` | Instruction candidates to try |
| `seed` | `int` | `9` | Random seed |
| `init_temperature` | `float` | `1.0` | Initial sampling temperature |
| `track_stats` | `bool` | `True` | Log optimization statistics |
| `log_dir` | `str \| None` | `None` | Directory for optimization logs |
| `metric_threshold` | `float \| None` | `None` | Minimum metric score threshold |

Default `auto="light"` runs fewer trials — explicitly set `"medium"` or `"heavy"` for thorough optimization. Key method: `.compile(module, trainset=...)`.

## dspy.GEPA

```python
dspy.GEPA(metric, auto='light')
```

Pass `metric` in the **constructor only**. Unlike BootstrapFewShot and MIPROv2, GEPA does not accept `metric` in `compile()` — passing it there raises a `TypeError`. Key method: `.compile(module, trainset=...)`.

## W&B Weave quick reference

```python
import weave

weave.init("project-name")   # authenticate and create/connect project

@weave.op()
def run_optimization(...):
    ...  # Weave logs inputs, outputs, and cost for every call
```

- View runs at `wandb.ai`
- Decorate the function that wraps your optimization loop — Weave captures LM calls, latency, and cost inside it
- For detailed Weave setup, see `/dspy-weave`

## LangWatch quick reference

```python
import langwatch

langwatch.init()   # reads LANGWATCH_API_KEY from env

optimizer = dspy.MIPROv2(metric=metric, auto="heavy")
optimized = optimizer.compile(program, trainset=trainset)
# Optimizer steps stream to app.langwatch.ai in real-time
```

- For full setup (self-hosted, optimizer dashboard), see `/dspy-langwatch`

## Quick reference

| Task | Call |
|------|------|
| Save optimized program | `program.save("artifacts/name.json")` |
| Load in production | `prog = MyProgram(); prog.load("production/optimized.json")` |
| Evaluate a program | `Evaluate(devset=devset, metric=metric, num_threads=4)(program)` |
| BootstrapFewShot | `dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)` |
| MIPROv2 medium | `dspy.MIPROv2(metric=metric, auto="medium")` |
| GEPA | `dspy.GEPA(metric=metric)` — metric in constructor, not compile() |
| W&B Weave init | `weave.init("project-name")` then decorate with `@weave.op()` |
| LangWatch init | `langwatch.init()` before calling `optimizer.compile()` |
| MLflow tracking | See `/dspy-mlflow` for `mlflow.dspy.autolog()` and model registry |
