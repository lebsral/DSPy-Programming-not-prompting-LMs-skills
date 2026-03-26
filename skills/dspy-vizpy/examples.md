# VizPy Examples

## Side-by-side: Classify sentiment with GEPA vs VizPy

This example optimizes the same sentiment classifier with both GEPA and VizPy ContraPromptOptimizer, then compares results.

### Setup (shared)

```python
import dspy
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

# Metric
def metric(example, prediction, trace=None):
    return prediction.sentiment.strip().lower() == example.sentiment.strip().lower()

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_table=5)
```

### Baseline

```python
baseline_score = evaluator(classify)
print(f"Baseline: {baseline_score:.1f}%")
```

### Optimize with GEPA

```python
def gepa_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    score = float(pred.sentiment.strip().lower() == gold.sentiment.strip().lower())
    feedback = "" if score == 1.0 else f"Expected '{gold.sentiment}', got '{pred.sentiment}'."
    return {"score": score, "feedback": feedback}

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
import vizpy

vizpy_optimizer = vizpy.ContraPromptOptimizer(metric=metric)
vizpy_optimized = vizpy_optimizer.compile(classify, trainset=trainset)

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
| Metric format | Returns `{"score": float, "feedback": str}` | Returns float (standard DSPy metric) |
| Requires | `reflection_lm` (strong model) | VizPy API key |
| Runs locally | Yes (LM calls only) | No (SaaS optimization) |
| Budget control | `auto="light"/"medium"/"heavy"` | Runs count against monthly quota |
