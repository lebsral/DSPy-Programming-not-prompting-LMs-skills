> Condensed from [dspy.ai/api](https://dspy.ai/api/) and [faker.readthedocs.io](https://faker.readthedocs.io/). Verify against upstream for latest.

# DSPy API Reference for Generating Data

## Packages

```bash
pip install -U dspy   # DSPy 3.2.1+
pip install faker     # v26+, optional — structured fields only
```

## Generator Signatures

The generator's **output fields become the task's input fields** — design them together.

```python
class GenerateTicketExample(dspy.Signature):
    """Generate a realistic support ticket for the given category."""
    category: str = dspy.InputField(desc="the target category")
    sindex: str = dspy.InputField(desc="unique seed index for diversity")
    ticket_text: str = dspy.OutputField(desc="a realistic support ticket")
```

Use `dspy.InputField(desc=...)` and `dspy.OutputField(desc=...)`. For boolean assessment fields, add `type_=bool` to the OutputField. Signature docstrings become the LM's task instruction — write them as direct imperatives with quality constraints (e.g., "Never use real patient data").

## dspy.Example and with_inputs()

```python
dspy.Example(**fields)
example.with_inputs(*field_names) -> Example
```

**Required on every synthetic Example.** Without it, optimizers pass expected outputs back into the program and scores are inflated. Returns a new Example — does not mutate.

```python
ex = dspy.Example(ticket_text="Charged twice.", category="billing").with_inputs("ticket_text")
ex.inputs()  # {"ticket_text": "Charged twice."}
ex.labels()  # {"category": "billing"}
```

## dspy.Predict

```python
dspy.Predict(signature, **config)
```

Use for the generation loop. Cheaper than `ChainOfThought` when the docstring constrains quality well enough. **The `n=N` batch parameter is not supported by all providers (Anthropic included) — use an explicit loop.**

## dspy.Refine

```python
dspy.Refine(module, N, reward_fn, threshold)
```

Runs `module` up to `N` times, scores each output with `reward_fn`, returns the first output meeting `threshold`. Returns the best attempt if none qualify. Use as an in-loop quality gate.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `dspy.Module` | The generator module to wrap |
| `N` | `int` | Max generation attempts per call |
| `reward_fn` | `Callable[[dict, Prediction], float]` | Returns 0.0–1.0; receives `(args, pred)` |
| `threshold` | `float` | Minimum score to accept (0.0–1.0) |

`args` is the input dict passed to the module; `pred` is the Prediction object.

```python
def quality_reward(args, pred):
    r = assessor(ticket_text=pred.ticket_text, category=args["category"])
    return float(r.is_realistic and r.is_correctly_labeled)

generator = dspy.Refine(module=MyGenerator(), N=3, reward_fn=quality_reward, threshold=0.75)
```

## dspy.BootstrapFewShot

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

Use to **meta-optimize the generator's prompt**. Pass seed examples as `trainset`; the metric checks whether generated outputs are correctly handled by the downstream task.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | `(example, prediction, trace=None) -> bool \| float` |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demonstrations per predictor |
| `max_labeled_demos` | `int` | `16` | Max labeled demos from trainset |

Key method: `.compile(student, *, trainset) -> Module`

## dspy.MIPROv2

```python
dspy.MIPROv2(metric, auto='light', max_bootstrapped_demos=4,
             max_labeled_demos=4, num_candidates=None, seed=9)
```

Optimizes the downstream task program after synthetic data is ready. `metric` (required) is the scoring function; `auto` (`'light'` / `'medium'` / `'heavy'`) controls intensity — use `'medium'` or `'heavy'` when accuracy matters more than speed.

Key method: `.compile(module, trainset=...) -> Module`

## dspy.Evaluate

```python
from dspy.evaluate import Evaluate
evaluator = Evaluate(devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(module)
```

Use real examples as `devset` when available — synthetic eval scores are inflated when both training and evaluation data come from the same generator. Set `display_table=20` to inspect failures.

## Faker (programmatic generation)

```python
from faker import Faker
from faker.providers import BaseProvider

fake = Faker()
fake.name(); fake.email(); fake.address()   # built-in generators

class OrderProvider(BaseProvider):
    def order_id(self):
        return f"ORD-{self.random_int(1000, 99999)}"

fake.add_provider(OrderProvider)
```

Use Faker for fields with known formats (names, emails, dates, IDs) — fast, free, zero LM cost. Use LM generation for open-ended text and domain-specific tone. Combine both: Faker for structured scaffolding, LM for surrounding context.

## Quick-reference

| Goal | Tool | Notes |
|------|------|-------|
| Generate in a loop | `dspy.Predict(GeneratorSig)` | Avoid `n=N`; not all providers support it |
| Quality gate per example | `dspy.Refine(module, N=3, ...)` | Returns best attempt if no call clears threshold |
| Bulk filter post-hoc | `dspy.Predict(AssessorSig)` | Keep `is_realistic and is_correctly_labeled` |
| Structured fields at scale | `Faker` | Names, emails, IDs — instant, no LM cost |
| Meta-optimize generator | `dspy.BootstrapFewShot` | `trainset` = seed examples (5–10) |
| Optimize downstream task | `dspy.MIPROv2(metric, auto=...)` | `trainset` = filtered synthetic data |
| Evaluate | `dspy.Evaluate(devset, metric)` | Prefer real examples as devset |
