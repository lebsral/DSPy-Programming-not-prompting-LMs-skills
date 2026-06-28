# VizPy Examples

## Side-by-side: Classify sentiment with GEPA vs VizPy

This example optimizes the same sentiment classifier with both GEPA and VizPy ContraPromptOptimizer, then compares results.

### Setup (shared)

```python
import dspy
import vizpy
from dspy.evaluate import Evaluate

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# Program
classify = dspy.ChainOfThought("review -> sentiment")

# Data
trainset = [
    dspy.Example(review="Absolutely love this product!", sentiment="positive").with_inputs("review"),
    dspy.Example(review="Worst purchase I've ever made.", sentiment="negative").with_inputs("review"),
    dspy.Example(review="It's okay, nothing special.", sentiment="neutral").with_inputs("review"),
    dspy.Example(review="The build quality is great but the software is buggy.", sentiment="mixed").with_inputs("review"),
    # ... add 50+ examples for real optimization
]

devset = trainset[:20]  # hold out for evaluation (use separate data in practice)
```

### Metrics (different format for each optimizer)

VizPy requires `vizpy.Score`. GEPA requires a dict. Both can share an underlying correctness check:

```python
def _correct(example, prediction):
    return prediction.sentiment.strip().lower() == example.sentiment.strip().lower()

# For VizPy optimization
def vizpy_metric(example, prediction, trace=None):
    correct = _correct(example, prediction)
    return vizpy.Score(
        value=1.0 if correct else 0.0,
        is_success=correct,
        feedback="" if correct else f"Expected '{example.sentiment}', got '{prediction.sentiment}'.",
    )

# For GEPA optimization
def gepa_metric(gold, pred, trace=None, **kw):
    correct = _correct(gold, pred)
    return {"score": float(correct), "feedback": "" if correct else f"Expected '{gold.sentiment}'."}

# For dspy.Evaluate (must return float, not vizpy.Score)
eval_metric = lambda ex, pred, trace=None: float(_correct(ex, pred))

evaluator = Evaluate(devset=devset, metric=eval_metric, num_threads=4, display_table=5)
```

### Baseline

```python
baseline_score = evaluator(classify)
print(f"Baseline: {baseline_score:.1f}%")
```

### Optimize with GEPA

```python
gepa = dspy.GEPA(
    metric=gepa_metric,
    reflection_lm=dspy.LM("openai/gpt-4o", temperature=1.0, max_tokens=4096),
    auto="light",
)
gepa_optimized = gepa.compile(classify, trainset=trainset)

gepa_score = evaluator(gepa_optimized)
print(f"GEPA: {gepa_score:.1f}%")
```

### Optimize with VizPy

```python
vizpy_optimizer = vizpy.ContraPromptOptimizer(metric=vizpy_metric)
vizpy_optimized = vizpy_optimizer.optimize(classify, train_examples=trainset)

vizpy_score = evaluator(vizpy_optimized)
print(f"VizPy: {vizpy_score:.1f}%")
```

### Compare

```python
print(f"\nResults:")
print(f"  Baseline:  {baseline_score:.1f}%")
print(f"  GEPA:      {gepa_score:.1f}%")
print(f"  VizPy:     {vizpy_score:.1f}%")

# Save the winner
if gepa_score >= vizpy_score:
    gepa_optimized.save("best_classifier.json")
    print("Winner: GEPA")
else:
    vizpy_optimized.save("best_classifier.json")
    print("Winner: VizPy")
```

### Key differences in practice

| Aspect | GEPA | VizPy ContraPrompt |
|--------|------|--------------------|
| Metric format | Returns `{"score": float, "feedback": str}` | Returns `vizpy.Score(value, is_success, feedback)` |
| Optimizer method | `.compile(program, trainset=trainset)` | `.optimize(module, train_examples=trainset)` |
| Requires | `reflection_lm` (strong model) | VizPy API key |
| Runs locally | Yes (LM calls only) | No (SaaS optimization) |
| Budget control | `auto="light"/"medium"/"heavy"` | Runs count against monthly quota |
