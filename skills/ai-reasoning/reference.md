# AI Reasoning — API Reference

> Condensed from [dspy.ai/api/modules/](https://dspy.ai/api/modules/ChainOfThought/). Verify against upstream for latest.

## dspy.ChainOfThought

Adds step-by-step reasoning before producing the final answer.

```python
cot = dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | The task signature |
| `rationale_field` | `FieldInfo \| None` | `None` | Custom reasoning field (auto-generated if None) |
| `rationale_field_type` | `type` | `str` | Type for the rationale field |
| `**config` | `dict` | — | Passed to internal `dspy.Predict` |

**Key methods:**

| Method | Description |
|--------|-------------|
| `forward(**kwargs)` | Run prediction with reasoning |
| `aforward(**kwargs)` | Async version |
| `batch(examples, ...)` | Process multiple inputs in parallel |
| `set_lm(lm)` | Override the language model |
| `save(path)` / `load(path)` | Persist/restore optimized state |

**Output:** Returns a `Prediction` with a `.reasoning` field (the step-by-step trace) plus all signature output fields.

---

## dspy.ProgramOfThought

Generates and executes Python code to compute the answer.

```python
pot = dspy.ProgramOfThought(signature, max_iters=3, interpreter=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | The task signature |
| `max_iters` | `int` | `3` | Max retries for code generation/execution |
| `interpreter` | `PythonInterpreter \| None` | `None` | Custom sandbox (auto-created if None) |

**Key methods:** Same as ChainOfThought (`forward`, `aforward`, `batch`, `set_lm`, `save`, `load`).

**How it works:** Generates Python code, executes it in a sandboxed interpreter, and returns the output. Retries up to `max_iters` times if execution fails.

---

## dspy.MultiChainComparison

Generates multiple reasoning chains and selects the best answer.

```python
mcc = dspy.MultiChainComparison(signature, M=3, temperature=0.7, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | The task signature |
| `M` | `int` | `3` | Number of reasoning chains to generate |
| `temperature` | `float` | `0.7` | Sampling temperature for diversity |
| `**config` | `dict` | — | Additional config |

**Key methods:**

| Method | Description |
|--------|-------------|
| `forward(completions, **kwargs)` | Compare multiple completions and pick the best |

**How it works:** Internally generates M chains of thought at higher temperature, then passes all rationales to a comparison step that selects the best answer.

---

## dspy.BestOfN

Generate N completions and return the highest-scoring one.

```python
bon = dspy.BestOfN(module, N=3, reward_fn=my_reward, threshold=1.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to run N times |
| `N` | `int` | `3` | Number of completions to generate |
| `reward_fn` | `callable` | required | `(args, pred) -> float` |
| `threshold` | `float` | required | minimum reward to accept early |

Simpler than MultiChainComparison when you have a programmatic metric. No internal comparison LM call — just runs the metric on each completion.

---

## dspy.Refine

Iteratively improve an answer using feedback.

```python
refine = dspy.Refine(module, N=3, reward_fn=my_reward, threshold=0.7)
```

Runs the module up to N times, scores each output with `reward_fn(args, pred) -> float`, and returns the first output meeting `threshold` (or the best seen). Useful for tasks where first-draft quality is acceptable but refinement improves results (writing, code generation).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | The module to run |
| `N` | `int` | required | Max number of attempts |
| `reward_fn` | `callable` | required | `(args, pred) -> float` |
| `threshold` | `float` | required | minimum reward to accept |

---

## dspy.RLM

Reasoning Language Model — test-time compute scaling for verified reasoning.

```python
rlm = dspy.RLM(signature)
```

Uses extended generation with verification steps. Best for tasks with verifiable answers (math proofs, code that must pass tests). Higher latency but stronger correctness guarantees.

---

## Common patterns

### Accessing reasoning traces

```python
result = cot(question="Why did the deploy fail?")
print(result.reasoning)  # Step-by-step trace (auto-injected by ChainOfThought)
print(result.answer)     # Final answer from your signature
```

### Composing reasoning modules

```python
class Pipeline(dspy.Module):
    def __init__(self):
        self.plan = dspy.ChainOfThought("task -> steps: list[str]")
        self.execute = dspy.ProgramOfThought("steps, data -> result")

    def forward(self, task, data):
        steps = self.plan(task=task).steps
        return self.execute(steps=steps, data=data)
```

### Optimization

All reasoning modules support the standard DSPy optimization flow:

```python
optimizer = dspy.BootstrapFewShot(metric=my_metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(my_module, trainset=trainset)

# Or for instruction tuning:
optimizer = dspy.MIPROv2(metric=my_metric, auto="medium")
optimized = optimizer.compile(my_module, trainset=trainset)
```
