# Fine-Tuning API Reference

> Condensed from [dspy.ai/api/optimizers/](https://dspy.ai/api/). Verify against upstream for latest.

## dspy.BootstrapFinetune

Fine-tunes model weights using bootstrapped reasoning traces.

```python
optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
finetuned = optimizer.compile(student, trainset=trainset, teacher=teacher_optimized)
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable or None` | `None` | Evaluation metric function |
| `multitask` | `bool` | `True` | Whether to use multitask training |
| `train_kwargs` | `dict or None` | `None` | Training arguments passed to fine-tuning API |
| `adapter` | `Adapter or None` | `None` | Output adapter configuration |
| `exclude_demos` | `bool` | `False` | Whether to exclude demonstration examples |
| `num_threads` | `int or None` | `None` | Number of threads for parallel processing |

### compile()

```python
optimizer.compile(student, trainset, teacher=None) -> Module
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `student` | `Module` | The program to fine-tune |
| `trainset` | `list[Example]` | Training examples |
| `teacher` | `Module or list[Module] or None` | Optional teacher for distillation |

## dspy.BetterTogether

Alternates prompt optimization and weight optimization for maximum quality.

```python
optimizer = dspy.BetterTogether(
    metric=metric,
    p=dspy.MIPROv2(metric=metric),
    w=dspy.BootstrapFinetune(metric=metric),
)
best = optimizer.compile(program, trainset=trainset, strategy="p -> w -> p")
```

### Constructor

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | Required | Evaluation metric function |
| `**optimizers` | `Teleprompter` | See below | Named optimizers as kwargs |

Default optimizers (if none provided): `p=BootstrapFewShotWithRandomSearch(metric=metric)`, `w=BootstrapFinetune(metric=metric)`.

### compile()

```python
optimizer.compile(student, trainset, strategy="p -> w -> p", valset=None) -> Module
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `Module` | Required | The program to optimize |
| `trainset` | `list[Example]` | Required | Training examples |
| `strategy` | `str` | `"p -> w"` | Optimizer sequence using keys from constructor |
| `valset` | `list[Example] or None` | `None` | Validation set (auto-split from trainset if None) |
| `valset_ratio` | `float` | `0.1` | Fraction of trainset to use for validation |

Strategy strings: `"p -> w"` (prompt then weights), `"p -> w -> p"` (cyclic), `"w -> p"` (reverse).

## dspy.MIPROv2

Best prompt optimizer. Optimizes instructions and few-shot examples.

```python
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized = optimizer.compile(program, trainset=trainset)
```

### Constructor (key parameters)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | Required | Evaluation metric function |
| `auto` | `Literal["light", "medium", "heavy"] or None` | `"light"` | Preset search intensity |
| `max_bootstrapped_demos` | `int` | `4` | Max few-shot demos from bootstrapping |
| `max_labeled_demos` | `int` | `4` | Max few-shot demos from labeled data |
| `num_threads` | `int or None` | `None` | Number of threads |
| `verbose` | `bool` | `False` | Show detailed progress |

## Links

- [BootstrapFinetune API docs](https://dspy.ai/api/optimizers/BootstrapFinetune/)
- [BetterTogether API docs](https://dspy.ai/api/optimizers/BetterTogether/)
- [MIPROv2 API docs](https://dspy.ai/api/optimizers/MIPROv2/)
