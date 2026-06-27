> Condensed from [dspy.ai/api/](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for PII Redaction

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Input/output contract for the detection task |

Use for contextual PII detection — wrap a `dspy.Signature` that returns `list[dict]` of detected entities. No reasoning step; `dspy.ChainOfThought` adds latency without improving recall on entity extraction tasks.

## dspy.Signature, InputField, OutputField

[API docs](https://dspy.ai/api/signatures/)

```python
class DetectContextualPII(dspy.Signature):
    """Docstring becomes the task instruction."""
    text: str = dspy.InputField(desc="description")
    pii_entities: list[dict] = dspy.OutputField(
        desc='JSON list - [{"pii_type": "PERSON_NAME", "value": "..."}]'
    )
```

### InputField / OutputField options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str` | `""` | Description injected into the prompt |
| `prefix` | `str` | field name | Label prefix shown in the prompt |

### Output types used in redaction signatures

| Type | Usage |
|---|---|
| `list[dict]` | Entity list from detection signatures (`pii_entities`, `phi_entities`, `names`) |
| `bool` | `is_clean` in validation signatures |
| `list[str]` | `leaked_examples` in `ValidateRedaction` |

## dspy.Module and dspy.Prediction

[API docs](https://dspy.ai/api/primitives/)

```python
class PIIRedactor(dspy.Module):
    def __init__(self, strategy: str = "placeholder"):
        self.detect = dspy.Predict(DetectContextualPII)

    def forward(self, text: str) -> dspy.Prediction:
        ...
        return dspy.Prediction(redacted_text=final, entities_found=seen)
```

`dspy.Prediction` fields are accessed as attributes: `result.redacted_text`, `result.entities_found`. Fields not set on the Prediction raise `AttributeError` — guard with `or []` when iterating LM output lists.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to wrap with retry logic |
| `N` | `int` | required | Max attempts before returning best result |
| `reward_fn` | `Callable[[args, pred], float]` | required | Returns 0.0–1.0; higher = better |
| `threshold` | `float \| None` | `None` | Stop early when reward meets or exceeds this value |

Use to validate that no PII survived redaction. `dspy.Assert` and `dspy.Suggest` were removed in DSPy 3.x; `dspy.Refine` is the replacement.

```python
def redaction_reward(example, prediction, trace=None) -> float:
    validator = dspy.Predict(ValidateRedaction)
    result = validator(original_text=example.text, redacted_text=prediction.redacted_text)
    return 1.0 if result.is_clean else 0.0
```

## dspy.context

[API docs](https://dspy.ai/api/models/LM/)

```python
with dspy.context(lm=other_lm):
    result = module(...)
```

Use in the pre-LLM sanitizer pattern to route detection to a trusted (internal) LM and the downstream task to an external LM — without changing the global `dspy.configure`. The context reverts when the `with` block exits.

## module.batch()

Available on all DSPy modules (DSPy 3.2.1+).

```python
results = redactor.batch(
    examples,                     # list[dspy.Example]
    num_threads=4,                # parallel threads
    max_errors=5,                 # abort after N consecutive errors
    return_failed_examples=False,
)
```

`examples` must be `dspy.Example` objects. Call `.with_inputs(...)` to mark which fields are inputs. Results are returned in input order.

## Regex Pattern Reference

Standard patterns — all compiled with `re.compile()` before the LM pass:

| PII Type | Pattern notes |
|---|---|
| `EMAIL` | `\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b` |
| `PHONE` | US format with optional country code and separators |
| `SSN` | `\b\d{3}-\d{2}-\d{4}\b` |
| `CREDIT_CARD` | Four groups of four digits with optional separators |
| `IP_ADDRESS` | `\b\d{1,3}(?:\.\d{1,3}){3}\b` |
| `DATE_OF_BIRTH` | Prefixed with `DOB`, `Date of Birth`, or `born` |
| `ZIP_CODE` | Five-digit with optional four-digit extension |

HIPAA Safe Harbor adds: `MRN`, `NPI`, `DEVICE_ID`, `URL`, `ACCOUNT`, `DATE` (month-name formats), `AGE_OVER_89`. Pre-mask structured PII with regex before sending text to the LM so the LM never sees raw PII values.

## Quick-Reference

| Task | DSPy API | Key notes |
|---|---|---|
| Contextual PII detection | `dspy.Predict(DetectContextualPII)` | Returns `list[dict]` with `pii_type`, `value` |
| Full pipeline | `dspy.Module` with `forward()` | Regex pass then LM pass then replacement pass |
| Validate redaction | `dspy.Refine(module, N, reward_fn)` | LM judge checks `is_clean`; replaces `dspy.Assert` |
| Per-request LM switching | `with dspy.context(lm=other_lm)` | Trusted LM for detection, external LM for task |
| Batch processing | `module.batch(num_threads=N)` | All DSPy modules support `.batch()` |
| Strategy - category | `f"[{pii_type}]"` | Readable; best for compliance audits |
| Strategy - indexed | `f"[{pii_type}_{n}]"` | Preserves co-references; deduplicate via `seen` dict |
| Strategy - hash | `hashlib.sha256(val).hexdigest()[:8]` | Pseudonymization; re-linkable with key |
