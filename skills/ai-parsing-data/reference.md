> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Data Parsing

## Output type quick-reference

| Shape | Type annotation | When to use |
|-------|----------------|-------------|
| Flat fields | `name: str = dspy.OutputField()` | Simple extraction, 3-6 fields |
| Pydantic model | `person: Person = dspy.OutputField()` | Complex or nested structure |
| List of models | `entities: list[Entity] = dspy.OutputField()` | Variable-count items |
| Dict | `fields: dict[str, str] = dspy.OutputField()` | Unknown key set (forms) |
| YAML string | `person_yaml: str = dspy.OutputField()` | Sub-4B parameter models |

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines inputs/outputs |
| `rationale_field` | `FieldInfo \| None` | `None` | Custom reasoning field |
| `rationale_field_type` | `type` | `str` | Type for the rationale |

Adds a `reasoning` field automatically before outputs — do not declare it in your signature. Improves accuracy 5-15% over bare `Predict` on ambiguous field mapping.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

No reasoning step. Use for YAML-output signatures on sub-4B models, or simple key-value extraction where reasoning adds cost without benefit.

## dspy.Signature — field declarations

[API docs](https://dspy.ai/api/signatures/)

```python
dspy.InputField(desc=None, prefix=None, **kwargs)
dspy.OutputField(desc=None, prefix=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str \| None` | `None` | Natural-language description shown to the LM |
| `prefix` | `str \| None` | `None` | Deprecated in DSPy 3.2 — emits a warning; use `desc` instead |

### Pydantic output types

DSPy serializes/deserializes Pydantic models automatically:

```python
from pydantic import BaseModel
from typing import Optional

class Address(BaseModel):
    street: str
    city: str

class Person(BaseModel):
    name: str
    age: Optional[int] = None   # Optional — model returns None if absent
    address: Address             # nested model
    skills: list[str]

class ParsePerson(dspy.Signature):
    """Extract person details. Return None for missing optional fields."""
    text: str = dspy.InputField()
    person: Person = dspy.OutputField()
```

Stick to `str`, `int`, `float`, `bool`, `list`, `dict`, and nested Pydantic models — avoid custom types or datetime objects that are not JSON-serializable.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=None, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to wrap |
| `N` | `int` | required | Max retry attempts |
| `reward_fn` | `Callable[[args, pred], float]` | required | Returns 0.0-1.0 quality score |
| `threshold` | `float \| None` | `None` | Stop early when score reaches this value |

Retries up to N times and returns the highest-scoring attempt. Use to enforce format constraints (email contains `@`, phone has 10+ digits).

## Batch processing

All modules support `.batch()` for parallel document processing:

```python
results = parser.batch(
    examples,                     # list of dspy.Example
    num_threads=4,
    max_errors=5,
    return_failed_examples=False,
)
```

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Field-level accuracy function |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `16` | Max labeled demos from trainset |

Start here (~50 labeled examples). Upgrade to `dspy.MIPROv2(metric, auto="medium")` if accuracy plateaus.

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, failure_score=0.0)
```

Call the instance with your module: `score = evaluator(parser)`. Use partial-credit metrics (score each field independently) — all-or-nothing scoring hides which specific fields fail.

## Save / load

```python
optimized.save("parser.json")
parser.load("parser.json")      # call on a fresh instance of the same module class
```
