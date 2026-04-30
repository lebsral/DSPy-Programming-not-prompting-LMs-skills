> Condensed from [dspy.ai/api/optimizers/InferRules/](https://dspy.ai/api/optimizers/InferRules/). Verify against upstream for latest.

# dspy.InferRules — API Reference

## Constructor

```python
dspy.InferRules(
    num_candidates=10,
    num_rules=10,
    num_threads=None,
    teacher_settings=None,
    **kwargs,  # metric, max_errors, etc.
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_candidates` | `int` | `10` | Number of candidate rule-enhanced programs to generate and evaluate |
| `num_rules` | `int` | `10` | Number of rules to induce per predictor |
| `num_threads` | `int \| None` | `None` | Threads for parallel evaluation. Falls back to `dspy.settings.num_threads` |
| `teacher_settings` | `dict \| None` | `None` | Configuration for the teacher model used during bootstrapping |
| `metric` | `Callable` | required | Evaluation function `(example, prediction, trace) -> float\|bool`. Passed via `**kwargs` |
| `max_errors` | `int \| None` | `None` | Maximum errors before stopping evaluation. Passed via `**kwargs` |

**Inherits from:** `BootstrapFewShot` — inherits `max_bootstrapped_demos`, `max_labeled_demos`, `max_rounds` parameters.

## Methods

### `compile()`

```python
optimizer.compile(
    student,
    *,
    teacher=None,
    trainset,
    valset=None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `dspy.Module` | required | The program to optimize |
| `teacher` | `dspy.Module \| None` | `None` | Optional teacher module for bootstrapping |
| `trainset` | `list[Example]` | required | Training examples |
| `valset` | `list[Example] \| None` | `None` | Validation examples. If `None`, splits `trainset` 50/50 |

**Returns:** Optimized `dspy.Module` with discovered rules appended to each predictor's signature instructions.

### `induce_natural_language_rules(predictor, trainset)`

Extracts rules from training examples for a specific predictor. Progressively reduces demo count if context window limits are exceeded.

**Returns:** `str` — formatted rules as natural language.

### `update_program_instructions(predictor, natural_language_rules)`

Appends induced rules to the predictor's signature instructions with the prefix: "Please adhere to the following rules when making your prediction:"

### `evaluate_program(program, dataset)`

Scores a program on a dataset using the configured metric.

**Returns:** `float` — the evaluation score.

### `format_examples(demos, signature)`

Converts demo examples into formatted text separating input and output fields for rule induction.

**Returns:** `str` — formatted examples text.

### `get_predictor_demos(trainset, predictor)`

Filters training examples to include only fields matching the predictor's signature inputs/outputs.

**Returns:** `list[dict]` — filtered demos.

## Compilation stages

| Stage | What happens |
|-------|-------------|
| 1. Data splitting | Splits `trainset` 50/50 into train/val (unless `valset` provided) |
| 2. Bootstrap demos | Runs parent `BootstrapFewShot.compile()` to collect successful demonstrations |
| 3. Rule induction | Feeds bootstrapped demos into `RulesInductionProgram` to generate natural-language rules |
| 4. Candidate generation | Repeats rule induction `num_candidates` times with different samples |
| 5. Validation | Scores each candidate on valset, returns highest-scoring program |

## Output format

The induced rules are appended to each predictor's signature instructions as plain English statements. Example:

```
Please adhere to the following rules when making your prediction:
1. If the text mentions system failures, outages, or data loss, classify as urgent.
2. If the text is a routine account or billing question, classify as normal.
3. ...
```

Access the enhanced instructions after compilation:

```python
for name, predictor in compiled.named_predictors():
    print(f"{name}: {predictor.signature.instructions}")
```
