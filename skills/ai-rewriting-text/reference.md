> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Text Rewriting

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines inputs and outputs |

No reasoning step. Use for rewriter and judge modules — both are direct input-to-output mappings that do not benefit from added reasoning overhead.

## dspy.Signature

[API docs](https://dspy.ai/api/signatures/)

```python
class RewriteText(dspy.Signature):
    """Docstring becomes the task instruction."""
    source_text: str = dspy.InputField(desc="description")
    rewritten_text: str = dspy.OutputField(desc="description")
```

Rewriting signatures used in this skill:

| Signature | Key inputs | Key output |
|-----------|-----------|------------|
| `RewriteText` | `source_text`, `target_tone`, `target_audience` | `rewritten_text` |
| `RewriteWithExamples` | + `style_examples` | `rewritten_text` |
| `RewriteToReadingLevel` | `source_text`, `target_reading_level` | `rewritten_text` |
| `BrandVoiceRewriter` | `source_text`, `brand_guidelines`, `brand_examples` | `rewritten_text` |
| `FidelityJudge` | `source_text`, `rewritten_text` | `fidelity_score: float`, `reasoning` |
| `StyleJudge` | `rewritten_text`, `target_tone`, `target_audience` | `style_score: float`, `reasoning` |

## dspy.InputField / dspy.OutputField

```python
dspy.InputField(desc="description", prefix="Label:")
dspy.OutputField(desc="description")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str` | `""` | Field description shown in the prompt |
| `prefix` | `str` | `None` | Override the field label in the prompt |

For judge signatures, declare score outputs as `float` — DSPy parses the LM's numeric output into the correct type.

```python
fidelity_score: float = dspy.OutputField(desc="Float 0-1")
```

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function — receives `(example, prediction, trace)` |
| `max_bootstrapped_demos` | `int` | `4` | Max auto-generated few-shot demos |
| `max_labeled_demos` | `int` | `16` | Max labeled demos pulled from trainset |
| `max_rounds` | `int` | `1` | Bootstrap iterations |

Key method: `.compile(module, trainset=list[dspy.Example])` — returns an optimized module.

The composite metric for rewriting multiplies the two judge scores:

```python
def rewrite_metric(example, prediction, trace=None):
    fidelity = fidelity_judge(source_text=example.source_text,
                              rewritten_text=prediction.rewritten_text)
    style = style_judge(rewritten_text=prediction.rewritten_text,
                        target_tone=example.target_tone,
                        target_audience=example.target_audience)
    return float(fidelity.fidelity_score) * float(style.style_score)
```

## dspy.Example

[API docs](https://dspy.ai/api/primitives/)

```python
dspy.Example(**fields).with_inputs(*field_names)
```

`with_inputs()` is required on every training example — it tells DSPy which fields are inputs vs. expected outputs.

```python
dspy.Example(
    source_text="...",
    target_tone="casual, friendly",
    target_audience="non-technical users",
    rewritten_text="...",
).with_inputs("source_text", "target_tone", "target_audience")
```

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function |
| `num_threads` | `int \| None` | `None` | Parallel threads |
| `display_progress` | `bool` | `False` | Show progress bar |
| `display_table` | `bool \| int` | `False` | Show results table (`int` = row count) |

Call the instance to run: `score = evaluator(module)`

## textstat (reading level measurement)

```bash
pip install textstat
```

```python
import textstat

textstat.flesch_kincaid_grade(text)  # returns grade level as float
```

| Function | Returns | Notes |
|----------|---------|-------|
| `flesch_kincaid_grade(text)` | `float` | U.S. grade level (8.0 = 8th grade) |
| `flesch_reading_ease(text)` | `float` | 0-100 scale; higher = easier |

Tolerance of ±1.5 grade levels is a practical acceptance threshold — the model cannot hit a precise decimal grade consistently.

## Quick reference

| Goal | Signature inputs | Module | Notes |
|------|-----------------|--------|-------|
| Tone / audience change | `source_text`, `target_tone`, `target_audience` | `dspy.Predict` | Add `style_examples` for anchor |
| Reading level | `source_text`, `target_reading_level` | `dspy.Predict` | Measure with `textstat` |
| Brand voice | `source_text`, `brand_guidelines`, `brand_examples` | `dspy.Predict` | Examples dominate guidelines |
| Fidelity check | `source_text`, `rewritten_text` | `dspy.Predict` | Output `fidelity_score: float` |
| Style check | `rewritten_text`, `target_tone`, `target_audience` | `dspy.Predict` | Output `style_score: float` |
| Optimization | trainset of ~20-50 examples | `BootstrapFewShot` | Metric = fidelity * style |
