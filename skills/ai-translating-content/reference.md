> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Translating Content

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, callbacks=None, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines inputs and outputs |
| `callbacks` | `list[BaseCallback] \| None` | `None` | Optional instrumentation callbacks |

Direct input-to-output mapping — no reasoning step. Use for straightforward translation signatures and judge modules.

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

Adds a `reasoning` field automatically before the output. Use when translating idiom-heavy or ambiguous content where explicit reasoning improves confidence scoring. Do not add `reasoning` to your signature — DSPy injects it.

## dspy.Signature

[API docs](https://dspy.ai/api/signatures/)

```python
class TranslateText(dspy.Signature):
    """Docstring becomes the task instruction."""
    source_text: str = dspy.InputField(desc="description")
    translated_text: str = dspy.OutputField(desc="description")
```

Translation signatures used in this skill:

| Signature | Key inputs | Key output | Module |
|-----------|-----------|------------|--------|
| `Translate` | `source_text`, `target_language` | `translated_text: str` | `dspy.Predict` |
| `TranslateWithGlossary` | + `glossary: list[str]` | `result: TranslationResult` | `dspy.Predict` |
| `TranslateLocaleAware` | + `tone: str`, `glossary: list[str]` | `translated_text: str` | `dspy.Predict` |
| `TranslateWithQuality` | `source_text`, `target_language`, `glossary` | `result: TranslationWithQuality` | `dspy.Predict` |
| `TranslateI18nString` | `source_text`, `target_language`, `glossary` | `translated_text: str` | `dspy.Predict` |
| `TranslateSupportTicket` | `ticket_text` | `result: SupportTranslationResult` | `dspy.ChainOfThought` |
| `MeaningPreservationJudge` | `source_text`, `translated_text`, `target_language` | `score: float` | `dspy.Predict` |

## dspy.InputField / dspy.OutputField

```python
dspy.InputField(desc="description", prefix="Label:")
dspy.OutputField(desc="description")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str` | `""` | Field description shown in the prompt |
| `prefix` | `str` | `None` | Override the field label in the prompt |

Declare glossary as `list[str]` with `default=[]` so callers can omit it when no glossary is needed. Declare confidence and score outputs as `float` — DSPy parses the LM's numeric output into the correct type.

## Pydantic output models

Use `pydantic.BaseModel` for structured translation outputs. DSPy serializes and parses them automatically.

```python
from pydantic import BaseModel

class TranslationResult(BaseModel):
    translated_text: str
    glossary_terms_used: list[str]

class TranslationWithQuality(BaseModel):
    translated_text: str
    confidence: float        # 0.0–1.0
    needs_review: bool
    review_reason: str       # empty string if needs_review is False

class SupportTranslationResult(BaseModel):
    translated_text: str
    detected_source_language: str
    confidence: float
    needs_review: bool
    review_reason: str
```

Declare the Pydantic model as the output field type on the signature:

```python
result: TranslationWithQuality = dspy.OutputField()
```

Access fields via `pred.result.translated_text`, not `pred.translated_text`.

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, prompt_model=None, task_model=None, teacher_settings=None,
             max_bootstrapped_demos=4, max_labeled_demos=4,
             auto='light', num_candidates=None, num_threads=None,
             max_errors=None, seed=9, init_temperature=1.0,
             verbose=False, track_stats=True, log_dir=None,
             metric_threshold=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function — receives `(example, prediction, trace)` |
| `auto` | `'light' \| 'medium' \| 'heavy' \| None` | `'light'` | Optimization intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max auto-generated few-shot demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos pulled from trainset |
| `teacher_settings` | `dict \| None` | `None` | Settings for the teacher model |
| `init_temperature` | `float` | `1.0` | Initial sampling temperature |
| `metric_threshold` | `float \| None` | `None` | Min metric score to accept a candidate |
| `max_errors` | `int \| None` | `None` | Max errors before aborting optimization |

Key method: `.compile(module, trainset=list[dspy.Example])` — returns an optimized module.

## dspy.Example

[API docs](https://dspy.ai/api/primitives/)

```python
dspy.Example(**fields).with_inputs(*field_names)
```

`with_inputs()` is required on every training example — it tells DSPy which fields are inputs vs. expected outputs.

```python
dspy.Example(
    source_text="Upgrade your plan today.",
    target_language="German",
    glossary=["Pro", "Dashboard"],
).with_inputs("source_text", "target_language", "glossary")
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

## Quick reference

| Goal | Signature inputs | Module | Notes |
|------|-----------------|--------|-------|
| Basic translation | `source_text`, `target_language` | `dspy.Predict` | Use full locale label, not BCP-47 tag alone |
| Glossary enforcement | + `glossary: list[str]` | `dspy.Predict` | State "verbatim" in docstring |
| Tone / formality control | + `tone: str` | `dspy.Predict` | Values - casual, neutral, formal |
| Quality + review flag | + `confidence: float`, `needs_review: bool` | `dspy.Predict` or `dspy.ChainOfThought` | ChainOfThought for idiom-heavy content |
| i18n JSON batch | per-key loop, `glossary: list[str]` | `dspy.Predict` | One call per string; preserve `{placeholders}` |
| Meaning preservation judge | `source_text`, `translated_text`, `target_language` | `dspy.Predict` | Output `score: float`; threshold 0.8 |
| Optimization | trainset of labeled examples | `dspy.MIPROv2` | Metric combines glossary compliance + meaning score |
| Retry failed glossary | reward function checks term presence | `dspy.Refine` | See `/dspy-refine` |
