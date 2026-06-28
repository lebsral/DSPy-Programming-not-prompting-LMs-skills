> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Building Pipelines

## dspy.Module

[API docs](https://dspy.ai/api/primitives/Module/)

```python
class MyPipeline(dspy.Module):
    def __init__(self):
        self.step1 = dspy.Predict(SomeSignature)
        self.step2 = dspy.ChainOfThought(AnotherSignature)

    def forward(self, **inputs) -> dspy.Prediction:
        mid = self.step1(...)
        return self.step2(...)
```

Declare sub-modules as `self.*` attributes in `__init__`. DSPy's optimizer traces through `forward()` automatically — every `self.*` module call is visible to compilers. `forward()` is plain Python; use `if/else`, loops, and `dict.get()` freely.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

```python
dspy.Predict(signature, **config)
```

No reasoning step. Use for simple pipeline stages — extraction, formatting, classification where the answer is obvious. Avoids the cost of a reasoning trace when reasoning does not improve accuracy.

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Defines inputs/outputs |
| `rationale_field` | `FieldInfo \| None` | `None` | Custom reasoning field |

Adds a `reasoning` field before the output. Do not declare `reasoning` in your signature — DSPy injects it.

## dspy.Retrieve

[API docs](https://dspy.ai/api/retrieve/Retrieve/)

```python
self.retrieve = dspy.Retrieve(k=3)
passages = self.retrieve(query_string).passages  # list[str]
```

Use for retrieval stages in RAG pipelines. Requires a retriever backend (e.g., ColBERTv2) configured before use.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to wrap |
| `N` | `int` | required | Max refinement attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores output; returns float; higher is better |
| `threshold` | `float` | required | Stop when reward exceeds this value |
| `fail_count` | `int \| None` | `None` | Max failures before raising an error; defaults to N if None |

Wraps a module to retry when output quality is low. Replaces `dspy.Assert` / `dspy.Suggest`, which were removed in DSPy 3.x.

## dspy.BestOfN

[API docs](https://dspy.ai/api/modules/BestOfN/)

```python
dspy.BestOfN(module, N, reward_fn, threshold, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to run N times |
| `N` | `int` | required | Number of independent runs |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores each candidate; higher is better |
| `threshold` | `float` | required | Return early if any candidate exceeds this score |
| `fail_count` | `int \| None` | `None` | Max failures before raising an error; defaults to N if None |

Runs the module `N` times independently (no feedback between runs) and returns the candidate with the highest `reward_fn` score. Use when output diversity matters more than iterative improvement (e.g., code generation, creative writing).

## Assigning LMs per stage

```python
cheap_lm   = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-haiku-4-5-20251001"
quality_lm = dspy.LM("openai/gpt-4o")        # or "anthropic/claude-sonnet-4-5-20250929"

# Option 1 — direct attribute
pipeline.classify.lm = cheap_lm
pipeline.draft.lm    = quality_lm

# Option 2 — method call (equivalent)
pipeline.classify.set_lm(cheap_lm)
pipeline.draft.set_lm(quality_lm)
```

Stages without an assigned LM fall back to `dspy.settings.lm` (the global default set by `dspy.configure`).

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, max_bootstrapped_demos=4, max_labeled_demos=16, max_rounds=1)
```

Generates few-shot examples for all pipeline stages together. Start here with ~50 labeled examples. Key method: `.compile(module, trainset=...)`.

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto="light", max_bootstrapped_demos=4, max_labeled_demos=4)
```

| `auto` value | Intensity | Examples needed | Note |
|-------------|-----------|-----------------|------|
| `"light"` | Fastest, fewest candidates | ~50 | Default |
| `"medium"` | Balanced | ~100–200 | Good starting point for most cases |
| `"heavy"` | Most thorough | ~200+ | |

Optimizes instructions AND few-shot examples across all pipeline stages end-to-end. Key method: `.compile(module, trainset=...)`.

## Debugging

```python
dspy.inspect_history(n=3)  # last N LM calls across all pipeline stages
print(pipeline)             # module structure
print(result)               # all prediction fields
print(result.field_name)    # specific field
```

## Save and load

```python
optimized.save("my_pipeline.json")

fresh = MyPipeline()
fresh.load("my_pipeline.json")
# Re-assign per-stage LMs after loading — they are not persisted
fresh.classify.lm = cheap_lm
fresh.draft.lm    = quality_lm
```
