> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Production Monitoring

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset, metric=None, num_threads=None,
                     display_progress=False, display_table=False,
                     max_errors=None, failure_score=0.0)
score = evaluator(program)  # returns float (0–100)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function — see metric signature below |
| `num_threads` | `int \| None` | `None` | Parallel threads; `4` is a practical default |
| `display_progress` | `bool` | `False` | Show tqdm progress bar |
| `failure_score` | `float` | `0.0` | Score assigned when metric raises |

**Returns:** `float` in range 0–100 (percentage of examples that passed the metric).

Normalize to 0–1 for comparison: `score / 100`.

## Metric function signature

```python
def my_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float | bool:
    ...
```

| Argument | Type | Description |
|----------|------|-------------|
| `example` | `dspy.Example` | Ground truth — access fields via `example.field_name` |
| `prediction` | `dspy.Prediction` | Program output — access fields via `prediction.field_name` |
| `trace` | `list \| None` | DSPy execution trace; `None` at eval time, non-None during optimization |

Return `True`/`False` or a `float` (0.0–1.0 or higher for weighted metrics). `Evaluate` averages the results and multiplies by 100.

## Built-in metrics

```python
from dspy.evaluate import SemanticF1, answer_exact_match, answer_passage_match

# LM-based semantic F1 (handles paraphrase and partial credit)
metric = SemanticF1(threshold=0.66, decompositional=False)
score = metric(example, prediction)  # returns float 0–1

# Exact match helpers (direct call, not for Evaluate)
answer_exact_match(example, prediction)   # bool
answer_passage_match(example, prediction) # bool
```

`SemanticF1` makes an LM call — use it when phrasing varies. Use `answer_exact_match` for factual fields with a fixed correct string.

## dspy.Predict (LM-as-judge)

[API docs](https://dspy.ai/api/modules/Predict/)

Use `dspy.Predict` (not `dspy.ChainOfThought`) for judge calls inside metrics — reasoning adds cost without improving binary verdicts.

```python
class AssessQuality(dspy.Signature):
    """Is this a high-quality response to the question?"""
    question: str = dspy.InputField()
    response: str = dspy.InputField()
    is_high_quality: bool = dspy.OutputField()

def quality_metric(example, prediction, trace=None):
    judge = dspy.Predict(AssessQuality)
    result = judge(question=example.question, response=prediction.answer)
    return float(result.is_high_quality)
```

**Instantiate the judge inside the metric function** — metrics run in parallel threads inside `Evaluate`; shared module state is not thread-safe.

## dspy.Example

[API docs](https://dspy.ai/api/primitives/)

```python
ex = dspy.Example(question="...", answer="...").with_inputs("question")
```

`.with_inputs(*field_names)` marks which fields are inputs vs. labels. Omitting it causes `Evaluate` to pass all fields (including labels) into the program.

## dspy.Module (monitoring wrapper)

Subclass `dspy.Module` so the wrapper stays composable — `.save()`, `.load()`, and `.batch()` work on the outer class:

```python
class MonitoredProgram(dspy.Module):
    def forward(self, **kwargs):
        result = self.program(**kwargs)
        # write JSON entry to log_path
        return result
```

## dspy.MIPROv2 (re-optimization after degradation)

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
# metric — same callable used for monitoring
# auto — "light" | "medium" | "heavy" (optimization budget)
optimizer = dspy.MIPROv2(metric=quality_metric, auto="medium")
re_optimized = optimizer.compile(program, trainset=trainset)
```

After re-optimizing, update the baseline: `baseline["quality"] = new_score / 100`.

## Observability integrations quick-reference

| Platform | Install | Init | Auto-instruments DSPy |
|----------|---------|------|-----------------------|
| Langtrace | `pip install langtrace-python-sdk` | `langtrace.init(api_key="...")` | Yes |
| Arize Phoenix | `pip install arize-phoenix openinference-instrumentation-dspy` | `px.launch_app(); DSPyInstrumentor().instrument()` | Yes |
| W&B Weave | `pip install weave` | `weave.init("project")` | Via `@weave.op()` decorator |

For per-request debugging (not aggregate), see `/ai-tracing-requests`.

## Config quick-reference

```python
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)

program.save("optimized.json")  # persist after optimization
program.load("optimized.json")  # restore on next run
dspy.inspect_history(n=5)       # ad-hoc LM call inspection
```

## Alerting thresholds (recommended defaults)

| Metric category | Recommended threshold | Rationale |
|----------------|----------------------|-----------|
| Safety | `> 0.01` drop (1%) | Zero-tolerance; alert fast |
| Quality | `> 0.05` drop (5%) | Normal variance; avoid noise |
| Cost | `> 0.20` increase (20%) | Budget headroom |

Tighten safety thresholds for regulated domains (financial, medical); loosen quality thresholds for high-variance tasks like open-ended generation.
