> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Task Decomposition

## Package

```bash
pip install -U dspy  # DSPy 3.2.1+
```

## dspy.Module

[API docs](https://dspy.ai/api/primitives/Module/)

```python
class MyPipeline(dspy.Module):
    def __init__(self):
        self.step1 = dspy.ChainOfThought(Step1Signature)
        self.step2 = dspy.ChainOfThought(Step2Signature)

    def forward(self, **inputs) -> dspy.Prediction:
        ...
        return dspy.Prediction(field=value)
```

- Declare sub-modules as `self.<name>` attributes in `__init__`. DSPy optimizers auto-discover the full tree.
- `forward()` is called when you invoke the module: `result = pipeline(document=...)`.
- Return `dspy.Prediction(field=value, ...)` — dot-access any key as `result.field`.
- Modules are composable: a custom module can hold other custom modules as sub-modules.

## dspy.Signature

[API docs](https://dspy.ai/api/signatures/)

```python
class MySignature(dspy.Signature):
    """Docstring becomes the task instruction sent to the LM."""
    input_a: str = dspy.InputField(desc="description")
    input_b: str = dspy.InputField(desc="description")
    output: list[str] = dspy.OutputField(desc="description")
```

### dspy.InputField / dspy.OutputField

```python
dspy.InputField(desc="description", prefix="Label:")
dspy.OutputField(desc="description", prefix="Label:")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str` | `""` | Human-readable description of the field |
| `prefix` | `str \| None` | `None` | Override the auto-generated field label |

Supported output types: `str`, `int`, `float`, `bool`, `list[str]`, `list[MyModel]`, Pydantic `BaseModel` subclass.

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

Adds a `reasoning` field before the output automatically. Do not add `reasoning` to your signature — DSPy injects it.

Default choice for decomposition steps. The reasoning trace helps identify and extract stages avoid conflation.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

No reasoning step. Use for an identify stage when the task is a simple enumeration and the reasoning trace adds cost without improving recall.

## Typed Outputs with Pydantic

```python
from pydantic import BaseModel, Field

class ExtractedItem(BaseModel):
    name: str
    value: str
    source_text: str = Field(description="Exact text this was extracted from")

class MySignature(dspy.Signature):
    """..."""
    document: str = dspy.InputField()
    items: list[ExtractedItem] = dspy.OutputField()
```

Use Pydantic models for structured sub-items in list outputs. DSPy enforces the schema via its adapter layer.

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset, metric=None, num_threads=None,
                     display_progress=False, display_table=False,
                     max_errors=None, failure_score=0.0)
score = evaluator(module)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples |
| `metric` | `Callable` | `None` | Scoring function `(example, pred, trace) -> float` |
| `num_threads` | `int \| None` | `None` | Parallel threads |
| `display_table` | `bool \| int` | `False` | Rows to show (int) or toggle (bool) |

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto="light", prompt_model=None, task_model=None,
             max_bootstrapped_demos=4, max_labeled_demos=4,
             num_candidates=None, num_threads=None, seed=9, verbose=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function |
| `auto` | `'light' \| 'medium' \| 'heavy' \| None` | `'light'` | Optimization budget |
| `max_bootstrapped_demos` | `int` | `4` | Generated demo examples per stage |
| `max_labeled_demos` | `int` | `4` | Labeled examples drawn from trainset |

Key method: `.compile(module, trainset=list[Example])` — optimizes all sub-modules together in one pass.

## Per-stage LM assignment

```python
module.identify.set_lm(dspy.LM("openai/gpt-4o-mini"))  # cheap for easy stage
module.extract.set_lm(dspy.LM("openai/gpt-4o"))         # quality for complex stage
```

`set_lm()` is available on any DSPy module instance. Overrides the globally configured LM for that sub-module only.

## dspy.Example and with_inputs()

```python
example = dspy.Example(document="...", item_names=["a", "b"]).with_inputs("document")
```

`.with_inputs(*field_names)` marks which fields are inputs vs. labels. Required when passing examples to optimizers — without it, optimizers treat all fields as labels and optimization breaks.

## Quick reference — decomposition patterns

| Failure mode | Pattern | Modules |
|-------------|---------|---------|
| Accuracy drops on long text | Chunk-then-process | `ChainOfThought` per chunk, deduplicate results |
| AI conflates similar things | Sequential extraction | `ChainOfThought` identify → `ChainOfThought` extract per group |
| AI misses items in lists | Identify-then-process | `ChainOfThought` identify → `ChainOfThought` extract per item |
| Different input types fail | Classify-then-specialize | See `/ai-building-pipelines` |

## Chunking parameters (chunk-then-process)

| Parameter | Typical value | Purpose |
|-----------|--------------|---------|
| `chunk_size` | 1500–2500 words | Tokens per chunk; adjust to model context limit |
| `overlap` | 100–300 words | Prevents boundary items from being split and missed |

Split at natural boundaries (`\n\n`, section headers) rather than fixed character positions.
