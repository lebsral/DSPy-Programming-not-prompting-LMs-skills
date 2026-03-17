---
name: dspy-simba
description: "Use DSPy's SIMBA optimizer for small-step optimization. Use when you want to use dspy.SIMBA, optimize programs incrementally with small improvements, or need a conservative optimization strategy that makes targeted changes."
---

# Small-Step Optimization with dspy.SIMBA

Guide the user through using `dspy.SIMBA` (Stochastic Introspective Mini-Batch Ascent) to optimize DSPy programs through incremental, targeted improvements rather than large sweeping changes.

## What is dspy.SIMBA

`dspy.SIMBA` is a DSPy optimizer that improves programs by analyzing mini-batches of examples, identifying where the program struggles most, and making small targeted fixes -- either adding demonstrations or generating self-reflective rules. Instead of rewriting the entire prompt at once, SIMBA takes conservative steps, focusing on the examples with the highest output variability.

Key properties:

- **Mini-batch driven** -- samples small batches from the training set each iteration, rather than evaluating the entire dataset
- **Variability-focused** -- identifies the hardest examples by measuring output variability (gap between best and worst scores)
- **Two improvement strategies** -- adds few-shot demonstrations or generates introspective rules based on failure analysis
- **Maintains a program pool** -- keeps multiple candidate programs and probabilistically selects from the best performers
- **Incremental by design** -- each step makes a small, targeted change rather than overhauling the entire program

## When to use SIMBA

Use `dspy.SIMBA` when:

- You want conservative, incremental optimization that avoids regressions
- Your program already works reasonably well and you want to push accuracy higher
- You have a moderate dataset (50-500 examples) and want efficient optimization
- You need stability -- production systems where large prompt changes are risky
- You want to understand which examples are hardest for your program

Do **not** use SIMBA when:

- You are starting from scratch with no working program -- use `dspy.BootstrapFewShot` first
- You want maximum prompt optimization in one shot -- use `dspy.MIPROv2` instead
- You need to fine-tune model weights -- use `dspy.BootstrapFinetune`
- Your dataset is very small (fewer than 30 examples) -- mini-batch sampling needs enough data

## Basic usage

Three things are needed: a DSPy program, a metric function, and a training set.

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 1. Define your program
classify = dspy.ChainOfThought("text -> label")

# 2. Define a metric
def metric(example, prediction, trace=None):
    return prediction.label.lower() == example.label.lower()

# 3. Build training data
trainset = [
    dspy.Example(text="Great product!", label="positive").with_inputs("text"),
    dspy.Example(text="Terrible service.", label="negative").with_inputs("text"),
    # ... more examples
]

# 4. Optimize with SIMBA
optimizer = dspy.SIMBA(metric=metric)
optimized = optimizer.compile(classify, trainset=trainset)

# 5. Use the optimized program
result = optimized(text="This exceeded my expectations!")
print(result.label)

# 6. Save for later
optimized.save("optimized_classifier.json")
```

## How small-step optimization works

SIMBA's optimization loop proceeds through repeated small steps:

### Step 1: Trajectory sampling

SIMBA runs the current program pool on a mini-batch of examples. Each program runs with distinct LM configurations to produce diverse outputs, scored by your metric.

### Step 2: Bucket analysis

Examples are grouped and sorted by **output variability** -- the gap between the best and worst scores across runs. High-variability examples are where the program is inconsistent and has the most room for improvement.

### Step 3: Strategy application

For each high-variability example, SIMBA applies one of two strategies:

- **Demonstration injection** -- takes a successful output for a hard example and adds it as a few-shot demonstration, teaching the program by example
- **Introspective rules** -- uses the LM to analyze why certain examples fail, then generates natural-language rules (instructions) that address the failure patterns

### Step 4: Candidate evaluation

New candidate programs (with the added demos or rules) are evaluated on a fresh mini-batch. This prevents overfitting to the examples used for rule generation.

### Step 5: Pool registration

The best-performing candidates are added to the program pool. Future iterations select source programs using softmax sampling weighted by average scores -- favoring better programs while still exploring alternatives.

This cycle repeats for `max_steps` iterations, with each step making a small, targeted improvement.

## Constructor parameters

```python
dspy.SIMBA(
    metric,                         # Scoring function (required)
    bsize=32,                       # Mini-batch size
    num_candidates=6,               # New candidates per iteration
    max_steps=8,                    # Number of optimization iterations
    max_demos=4,                    # Max demonstrations per predictor
    prompt_model=None,              # LM for generating rules (defaults to global LM)
    temperature_for_sampling=0.2,   # Temperature for trajectory sampling
    temperature_for_candidates=0.2, # Temperature for source program selection
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | (required) | Function `(example, prediction, trace=None) -> float` that scores outputs |
| `bsize` | `int` | `32` | Number of examples per mini-batch. Larger batches give more stable estimates but cost more LM calls |
| `num_candidates` | `int` | `6` | Candidate programs generated per iteration. More candidates explore more strategies but cost more |
| `max_steps` | `int` | `8` | Total optimization iterations. Each step samples a fresh mini-batch and produces new candidates |
| `max_demos` | `int` | `4` | Maximum few-shot demonstrations added to any predictor. Keeps prompts from growing too large |
| `prompt_model` | `dspy.LM` | `None` | LM used for generating introspective rules. Falls back to the globally configured LM if not set |
| `temperature_for_sampling` | `float` | `0.2` | Temperature when running programs on mini-batches. Lower values produce more deterministic outputs |
| `temperature_for_candidates` | `float` | `0.2` | Temperature for softmax selection of source programs from the pool. Lower values favor the top performers |

### Choosing parameter values

**`bsize` (mini-batch size):**

| Value | Use case |
|-------|----------|
| 16 | Small datasets (50-100 examples), faster iterations |
| 32 | Default, good balance for most tasks |
| 64 | Larger datasets, more stable gradient estimates |

**`max_steps`:**

| Value | Use case |
|-------|----------|
| 4-6 | Quick optimization pass, limited budget |
| 8 | Default, enough steps for meaningful improvement |
| 12-16 | Longer optimization for complex programs or larger datasets |

**`num_candidates`:**

| Value | Use case |
|-------|----------|
| 3-4 | Budget-conscious, smaller search space |
| 6 | Default, reasonable exploration |
| 8-10 | Wider search when you have LM budget to spare |

## Key methods

### compile()

Runs the optimization loop and returns the best program found.

```python
optimized = optimizer.compile(program, trainset=trainset)
```

The returned program includes two additional attributes:

- **`candidate_programs`** -- list of scored alternative programs discovered during optimization. Useful for ensemble strategies or analyzing what SIMBA tried.
- **`trial_logs`** -- per-batch metrics from each optimization step. Useful for understanding how performance evolved.

### get_params()

Returns the optimizer's configuration as a dictionary. Useful for logging and experiment tracking.

```python
params = optimizer.get_params()
print(params)
# {'bsize': 32, 'num_candidates': 6, 'max_steps': 8, ...}
```

## Inspecting optimization results

After optimization, examine what SIMBA found:

```python
optimizer = dspy.SIMBA(metric=metric)
optimized = optimizer.compile(program, trainset=trainset)

# Check the candidate pool
for i, (prog, score) in enumerate(optimized.candidate_programs):
    print(f"Candidate {i}: score={score:.3f}")

# Review trial logs to see improvement over time
for step, log in enumerate(optimized.trial_logs):
    print(f"Step {step}: {log}")
```

## Comparison with other optimizers

| Aspect | `dspy.SIMBA` | `dspy.MIPROv2` | `dspy.BootstrapFewShot` |
|--------|-------------|----------------|------------------------|
| **Strategy** | Small incremental steps on mini-batches | Full instruction + demo optimization | Bootstrap few-shot examples |
| **Change size** | Small, targeted per iteration | Can rewrite entire instructions | Adds demonstrations only |
| **Risk of regression** | Low -- changes are conservative | Higher -- rewrites can miss edge cases | Low -- additive only |
| **Data needed** | 50-500 examples | 200+ examples | 50+ examples |
| **Cost** | Moderate (mini-batch sampling) | Higher (full search) | Lower (single pass) |
| **Best for** | Incremental improvement, production stability | Maximum prompt quality | Quick first optimization |
| **Introspection** | Yes -- analyzes failures | Yes -- generates instructions | No |

### Optimization workflow

A common approach is to layer optimizers:

1. **Start with `BootstrapFewShot`** to get a working baseline with good demonstrations
2. **Then run SIMBA** to incrementally improve by targeting the hardest examples
3. **Optionally run `MIPROv2`** if you need maximum quality and can tolerate larger changes

```python
# Step 1: Bootstrap baseline
bootstrap = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
baseline = bootstrap.compile(program, trainset=trainset)

# Step 2: Incrementally improve with SIMBA
simba = dspy.SIMBA(metric=metric, max_steps=8)
improved = simba.compile(baseline, trainset=trainset)
```

## Tips

- **Start with defaults** -- `bsize=32`, `max_steps=8`, `num_candidates=6` work well for most tasks
- **Use graduated metrics** -- return floats between 0.0 and 1.0 rather than binary 0/1, so SIMBA can measure variability meaningfully
- **Bootstrap first** -- running `BootstrapFewShot` before SIMBA gives it a better starting point
- **Monitor trial logs** -- if scores plateau early, you may need more data or a different strategy
- **Keep `max_demos` reasonable** -- more than 4-6 demos can bloat prompts and hurt performance on some LMs
- **Use a separate `prompt_model`** for introspective rules if your main LM is small -- rule generation benefits from a stronger model

## Cross-references

- **Quick-start optimization** with few-shot examples -- see `/ai-improving-accuracy`
- **Evaluating your program** before and after optimization -- see `/dspy-evaluate`
- **Building the program to optimize** -- see `/dspy-chain-of-thought` or `/dspy-modules`
- **Preparing training data** -- see `/dspy-data`
- For worked examples, see [examples.md](examples.md)
