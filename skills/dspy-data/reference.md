> Condensed from [dspy.ai/api/primitives/Example/](https://dspy.ai/api/primitives/Example/) and [dspy.ai/api/primitives/Prediction/](https://dspy.ai/api/primitives/Prediction/). Verify against upstream for latest.

# dspy.Example — API Reference

## Constructor

```python
dspy.Example(base=None, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base` | `dict | Example | None` | `None` | Copy fields from an existing dict or Example before applying kwargs |
| `**kwargs` | any | — | Field names and values to store |

## Methods

### Input/output management

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `with_inputs` | `with_inputs(*keys)` | `Example` | Marks specified fields as inputs. Non-input fields become labels. Returns a new Example. |
| `inputs` | `inputs()` | `Example` | Returns new Example containing only the input fields (set by `with_inputs`). |
| `labels` | `labels()` | `Example` | Returns new Example containing only the non-input (label) fields. |

### Data access

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `get` | `get(key, default=None)` | any | Retrieves field value or default if missing. |
| `keys` | `keys(include_dspy=False)` | `list` | Returns field names. Set `include_dspy=True` to include internal `dspy_` fields. |
| `values` | `values(include_dspy=False)` | `list` | Returns field values. |
| `items` | `items(include_dspy=False)` | `list[tuple]` | Returns `(name, value)` pairs. |
| `toDict` | `toDict()` | `dict` | Converts to plain dictionary with recursive serialization of nested objects (including Pydantic models). |

### Data manipulation

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `copy` | `copy(**kwargs)` | `Example` | Creates shallow copy, optionally overriding fields. |
| `without` | `without(*keys)` | `Example` | Returns copy with specified fields removed. |

### Access patterns

```python
ex = dspy.Example(question="What?", answer="That")

# Dot notation
ex.question           # "What?"

# Dict-style access
ex["question"]        # "What?"

# Membership test
"question" in ex      # True
```

---

# dspy.Prediction — API Reference

## Constructor

```python
dspy.Prediction(*args, **kwargs)
```

Inherits from `Example`. Returned by all DSPy modules (`Predict`, `ChainOfThought`, etc.). Adds completion tracking and LM usage metadata. Supports comparison (`<`, `>`, `<=`, `>=`) and arithmetic (`+`, `/`) on predictions with a `score` field.

## Additional methods (beyond Example)

| Method | Signature | Returns | Description |
|--------|-----------|---------|-------------|
| `from_completions` | `from_completions(list_or_dict, signature=None)` (classmethod) | `Prediction` | Creates a Prediction from raw completion data. |
| `get_lm_usage` | `get_lm_usage()` | `dict` | Returns language model usage metadata (tokens, etc.). |
| `set_lm_usage` | `set_lm_usage(value)` | — | Sets language model usage tracking information. |

All `Example` methods (`with_inputs`, `inputs`, `labels`, `keys`, `values`, `items`, `get`, `toDict`, `copy`, `without`) are also available on Prediction.

## Key differences from Example

- Prediction strips `_demos` and `_input_keys` from internal state
- Prediction adds `_completions` and `_lm_usage` tracking
- Use `toDict()` to serialize Prediction output to a plain dict (handles nested Pydantic models)

---

# Built-in datasets

```python
from dspy.datasets import HotPotQA, GSM8k, Colors
```

| Dataset | Task | Constructor |
|---------|------|-------------|
| `HotPotQA` | Multi-hop QA over Wikipedia | `HotPotQA(train_seed=1, train_size=200, dev_size=50, test_size=0)` |
| `GSM8k` | Grade-school math word problems | `GSM8k(train_seed=1, train_size=200, dev_size=50, test_size=0)` |
| `Colors` | Simple color identification | `Colors(train_seed=1, train_size=200, dev_size=50, test_size=0)` |

All constructors accept: `train_seed`, `train_size`, `dev_size`, `test_size` (all optional).

Access splits via `.train`, `.dev`, `.test` attributes. Returns lists of `dspy.Example` objects.
