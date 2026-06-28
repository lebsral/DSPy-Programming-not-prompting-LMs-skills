> Condensed from [dspy.ai/api/evaluation/Evaluate/](https://dspy.ai/api/evaluation/Evaluate/), [dspy.ai/api/optimizers/MIPROv2/](https://dspy.ai/api/optimizers/MIPROv2/), [dspy.ai/api/optimizers/BootstrapFewShot/](https://dspy.ai/api/optimizers/BootstrapFewShot/), and [dspy.ai/api/optimizers/GEPA/](https://dspy.ai/api/optimizers/GEPA/overview/). Verify against upstream for latest.

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
dspy.Evaluate(*, devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, provide_traceback=None,
              failure_score=0.0, save_as_csv=None, save_as_json=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation dataset (keyword-only) |
| `metric` | `Callable \| None` | `None` | `(example, pred, trace=None) -> float\|bool` |
| `num_threads` | `int \| None` | `None` | Parallel evaluation threads |
| `display_progress` | `bool` | `False` | Show tqdm progress bar |
| `display_table` | `bool \| int` | `False` | Show per-example table (int = row count) |
| `max_errors` | `int \| None` | `None` | Errors before stopping |
| `provide_traceback` | `bool \| None` | `None` | Include traceback on failures |
| `failure_score` | `float` | `0.0` | Score assigned to failed examples |
| `save_as_csv` | `str \| None` | `None` | Save results to this CSV path |
| `save_as_json` | `str \| None` | `None` | Save results to this JSON path |

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

GEPA has no `task_lm` constructor parameter. Set the task LM via `dspy.configure(lm=task_lm)` before constructing the optimizer. `track_stats` defaults to `False` in GEPA (unlike MIPROv2 where it defaults to `True`).

```python
dspy.configure(lm=task_lm)  # task LM is global, not a GEPA constructor param

optimizer = dspy.GEPA(
    metric=metric,
    reflection_lm=reflection_lm,
    track_stats=True,          # GEPA-specific behavior: populates detailed_results
    # use_mlflow=True,         # alternatively: native MLflow integration
    # use_wandb=True,          # alternatively: native W&B integration
)
optimized = optimizer.compile(program, trainset=trainset)
stats = optimized.detailed_results  # DspyGEPAResult on the compiled program (not the optimizer)
```

`optimized.detailed_results` is a `DspyGEPAResult` object containing candidates, validation scores, Pareto frontiers, and lineage. It is available after `compile()` returns. GEPA also has built-in `use_mlflow=True` and `use_wandb=True` flags as alternatives to dspy-gepa-logger.

## MIPROv2 — verbose and logging params

MIPROv2 has built-in verbosity and logging parameters relevant to watching optimization progress.

```python
optimizer = dspy.MIPROv2(
    metric=metric,
    auto="light",              # "light" | "medium" | "heavy" | None
    verbose=False,             # True = print detailed progress to console (zero extra setup)
    track_stats=True,          # default True; controls internal stats tracking (not user-accessible)
    log_dir=None,              # str path to write optimizer logs to disk
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `verbose` | `bool` | `False` | Print detailed optimizer progress to console |
| `track_stats` | `bool` | `True` | Track internal stats (default on; does not expose `detailed_results`) |
| `log_dir` | `str \| None` | `None` | Directory to write optimizer run logs |
| `auto` | `Literal \| None` | `"light"` | Preset budget: light / medium / heavy / None (manual) |

For live per-candidate scores, use `verbose=True` (console) or LangWatch (cloud dashboard). `track_stats` in MIPROv2 is not the same as in GEPA — it does not produce a `detailed_results` attribute.

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
