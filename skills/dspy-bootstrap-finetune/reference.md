> Condensed from [dspy.ai/api/optimizers/BootstrapFinetune/](https://dspy.ai/api/optimizers/BootstrapFinetune/). Verify against upstream for latest.

# dspy.BootstrapFinetune — API Reference

## Constructor

```python
dspy.BootstrapFinetune(
    metric=None,            # Callable | None
    multitask=True,         # bool
    train_kwargs=None,      # dict | dict[LM, dict] | None
    adapter=None,           # Adapter | dict[LM, Adapter] | None
    exclude_demos=False,    # bool
    num_threads=None,       # int | None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable | None` | `None` | Scoring function `(example, prediction, trace=None) -> bool | float`. Only traces where metric passes become training data. |
| `multitask` | `bool` | `True` | When `True`, shares training data across all predictors. When `False`, each predictor gets separate fine-tuning data. |
| `train_kwargs` | `dict | dict[LM, dict] | None` | `None` | Fine-tuning hyperparameters passed to the provider (e.g., `{"n_epochs": 2}`). Can be LM-specific: `{lm_instance: {"n_epochs": 3}}`. |
| `adapter` | `Adapter | dict[LM, Adapter] | None` | `None` | Adapter for formatting training data. Can be LM-specific. |
| `exclude_demos` | `bool` | `False` | If `True`, clears few-shot demonstration examples after fine-tuning. Use when the fine-tuned model has internalized the patterns. |
| `num_threads` | `int | None` | `None` | Thread count for parallel fine-tuning jobs. Must be >= total number of fine-tuning jobs. Falls back to `dspy.settings.num_threads`. |

## Inheritance

`BootstrapFinetune` extends `FinetuneTeleprompter`.

## Methods

### `compile()`

```python
optimizer.compile(
    student,                # dspy.Module (required)
    trainset,               # list[Example] (required)
    teacher=None,           # Module | list[Module] | None
) -> Module
```

Bootstraps training data from teacher (or student) execution traces and fine-tunes the student model's weights.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `dspy.Module` | required | The program whose backing LM will be fine-tuned. |
| `trainset` | `list[Example]` | required | Labeled training examples (500+ recommended). |
| `teacher` | `Module | list[Module] | None` | `None` | Teacher program(s) for distillation. If `None`, student generates its own traces. |

**Returns:** Optimized `dspy.Module` with fine-tuned model(s) replacing the original LM(s).

**Raises:** `ValueError` if predictors lack assigned LMs or `num_threads` is insufficient for the number of fine-tuning jobs.

**Behavior:**
1. Runs teacher (or student) on each training example
2. Keeps traces where the metric passes
3. Formats passing traces as fine-tuning data
4. Calls `lm.kill()` on each model to free resources
5. Submits fine-tuning jobs to the provider
6. Returns the student with fine-tuned model references

### `get_params()`

```python
optimizer.get_params() -> dict[str, Any]
```

Returns all configuration parameters as a dictionary.

### `convert_to_lm_dict()` (static)

```python
BootstrapFinetune.convert_to_lm_dict(arg) -> dict[LM, Any]
```

Converts an argument to an LM-keyed dictionary. If already LM-indexed, returns unchanged; otherwise applies value uniformly across all LMs.

### `finetune_lms()` (static)

```python
BootstrapFinetune.finetune_lms(finetune_dict) -> dict[Any, LM]
```

Executes parallel fine-tuning jobs across all LMs in the dict.

## Supported providers

| Provider | Model format | Notes |
|----------|-------------|-------|
| OpenAI | `openai/gpt-4o-mini`, `openai/gpt-4o` | DSPy handles the fine-tuning API calls automatically |
| Together AI | `together_ai/meta-llama/Llama-3-70b-chat-hf` | Open-source models, competitive pricing |
| Local | Any HuggingFace model | Full control, requires GPU(s) |

## Key behaviors

- All student predictors must have LMs assigned before `compile()` is called
- `num_threads` must be >= the number of unique LMs across all predictors (one fine-tuning job per unique LM)
- Calls `lm.kill()` on each model to release resources before starting fine-tuning
- The returned program stores the fine-tuned model ID (e.g., `ft:gpt-4o-mini-2024-07-18:org::abc123`)
- `save()` and `load()` preserve fine-tuned model references
