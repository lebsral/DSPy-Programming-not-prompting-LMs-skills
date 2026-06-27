> Condensed from [dspy.ai/api/evaluation/Evaluate/](https://dspy.ai/api/evaluation/Evaluate/) and [dspy.ai/api/optimizers/](https://dspy.ai/api/optimizers/). Verify against upstream for latest.

# DSPy API Reference for Watching Optimization

## Monitoring options at a glance

| Tool | Package | Optimizers | Live? | Dashboard |
|------|---------|-----------|-------|-----------|
| `dspy.inspect_history(n)` | `dspy` built-in | All (post-hoc) | No | Console |
| `BaseCallback.on_evaluate_end` | `dspy` built-in | All | Yes (console) | No |
| `GEPA(track_stats=True)` | `dspy` built-in | GEPA only | No (dict after compile) | No |
| `GEPALogger` | `dspy-gepa-logger` | GEPA only | Yes | Web (port 3000) |
| `langwatch.dspy.init(...)` | `langwatch` | BFS / BRS / COPRO / MIPROv2 | Yes | Cloud |
| `mlflow.dspy.autolog(...)` | `mlflow` | All | No (post-compile) | Local web |

## dspy.Evaluate

```python
dspy.Evaluate(devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, failure_score=0.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation dataset |
| `metric` | `Callable \| None` | `None` | `(example, pred, trace=None) -> float\|bool` |
| `num_threads` | `int \| None` | `None` | Parallel evaluation threads |
| `display_progress` | `bool` | `False` | Show tqdm progress bar |
| `display_table` | `bool \| int` | `False` | Show per-example table (int = row count) |
| `max_errors` | `int \| None` | `None` | Errors before stopping |
| `failure_score` | `float` | `0.0` | Score assigned to failed examples |

Call: `score = evaluator(module)` — returns a float on the 0–100 scale.

Run before AND after optimization. Without a baseline score, there is no way to tell whether the optimizer helped.

## dspy.BaseCallback

```python
class MyCallback(dspy.BaseCallback):
    def on_evaluate_end(self, instance, inputs, outputs, exception):
        score = outputs.get("score", None)
        ...

dspy.configure(callbacks=[MyCallback()])
```

Key `on_evaluate_end` parameters - `outputs` (dict, includes `"score"` key), `exception` (`None` on success). Called for every candidate evaluation. Works with all optimizers. Register before calling `optimizer.compile()`.

## dspy.inspect_history

```python
dspy.inspect_history(n=1)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `n` | `int` | `1` | Number of recent LM calls to show |

Prints raw prompt + completion for the last `n` LM calls to stdout. Post-hoc only — not live during optimization. Use to verify which model was called and what instructions the optimizer tried.

## GEPA — track_stats

```python
optimizer = dspy.GEPA(
    metric=metric,
    task_lm=task_lm,
    reflection_lm=reflection_lm,
    track_stats=True,          # GEPA-only parameter
)
optimized = optimizer.compile(program, trainset=trainset)
stats = optimizer.detailed_results  # list[dict], one entry per iteration
for i, r in enumerate(stats):
    print(f"Iteration {i}: score={r['score']:.3f}")
```

`track_stats` has no effect on MIPROv2 or BootstrapFewShot. `optimizer.detailed_results` is populated after `compile()` returns.

## dspy-gepa-logger (GEPALogger)

```bash
pip install dspy-gepa-logger
```

```python
from dspy_gepa_logger import GEPALogger

logger = GEPALogger()               # starts web server on port 3000
optimizer.register_observer(logger) # attach to a GEPA instance before compile()
```

Dashboard shows real-time scores, prompt diffs between iterations, per-example pass/fail, and an event timeline (SeedValidation, Reflection, ValsetEval, and others). Requires Node.js; uses SQLite. GEPA only.

## langwatch — LangWatch

```bash
pip install langwatch
```

```python
langwatch.dspy.init(experiment="run-name")  # call before constructing optimizer
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `experiment` | `str` | Label for this run in the LangWatch dashboard |

Streams live scores, cost, and predictor states to app.langwatch.ai. Requires an API key (free tier available). Call before constructing the optimizer.

Supported - BootstrapFewShot, BootstrapFewShotWithRandomSearch, COPRO, MIPROv2. Not supported - GEPA.

## mlflow — MLflow autolog

```bash
pip install mlflow
```

```python
mlflow.dspy.autolog(log_compiles=True)
mlflow.set_experiment("name")
with mlflow.start_run():
    optimized = optimizer.compile(program, trainset=trainset)
# then: mlflow ui  →  http://localhost:5000
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_compiles` | `bool` | `True` | Log each compile as a parent MLflow run with child runs per candidate |

Captures metrics per candidate, LM call traces, and the final program artifact. Works with all optimizers.

## Convergence signals

| Score pattern | Meaning | Next step |
|---------------|---------|-----------|
| Climbing then plateau | Found improvements, saturated | Stop — use the best candidate |
| Climbing steadily | Still finding gains | Continue |
| Flat from the start | No improvement found | Check metric validity; try more budget or a different optimizer |
| Train up, val flat or falling | Overfitting | Larger trainset; reduce optimization budget |
| Same score 5+ iterations | Stuck | MIPROv2 - try `auto="medium"`. GEPA - stronger reflection LM. Any - check metric granularity (float beats bool) |
