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

[API docs](https://dspy.ai/api/)

```python
program.save(path)   # write compiled state (demos, instructions) to JSON
program.load(path)   # restore saved state into a program instance
```

- `path` is a string file path, e.g., `"artifacts/mipro_medium.json"`
- `load()` requires an instance of the same class: `prog = MyProgram(); prog.load("...")`
- Log the artifact path in your experiment record — without it the run cannot be reproduced or deployed

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
from dspy.evaluate import Evaluate

Evaluate(devset, metric=None, num_threads=None, display_progress=False,
         display_table=False, max_errors=None, failure_score=0.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function |
| `num_threads` | `int \| None` | `None` | Parallel threads |
| `display_progress` | `bool` | `False` | Show progress bar |
| `display_table` | `bool \| int` | `False` | Show results table |
| `failure_score` | `float` | `0.0` | Score assigned on LM error |

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
             max_bootstrapped_demos=4, max_labeled_demos=4,
             num_candidates=None, num_threads=None, seed=9, verbose=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function |
| `auto` | `'light' \| 'medium' \| 'heavy' \| None` | `'light'` | Optimization intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos |
| `num_candidates` | `int \| None` | `None` | Instruction candidates to try |

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
