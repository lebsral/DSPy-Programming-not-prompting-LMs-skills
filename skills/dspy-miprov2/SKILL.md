---
name: dspy-miprov2
description: Use when you want the highest-quality prompt optimization DSPy offers — jointly optimizes instructions and few-shot demos, with auto=light/medium/heavy presets. Common scenarios: you want the best possible accuracy from prompt optimization, jointly tuning instructions and few-shot demonstrations, using auto presets for different compute budgets, or when COPRO or BootstrapFewShot alone are not reaching your accuracy target. Related: ai-improving-accuracy, dspy-copro, dspy-bootstrap-few-shot. Also: dspy.MIPROv2, best DSPy optimizer, highest quality optimization, auto=light medium heavy, joint instruction and demo optimization, most powerful prompt optimizer, MIPROv2 vs COPRO vs BootstrapFewShot, which optimizer should I use, state of the art prompt optimization, when to use MIPROv2, optimize both instructions and examples, heavy optimization for production, best optimizer for accuracy.
---

# Optimize Prompts with MIPROv2

Guide the user through using `dspy.MIPROv2`, DSPy's most powerful prompt optimizer. MIPROv2 jointly optimizes instructions and few-shot demonstrations to maximize a metric on your training data.

## What is MIPROv2

MIPROv2 (Multi-prompt Instruction PRoposal Optimizer v2) is DSPy's recommended optimizer for prompt optimization. Unlike simpler optimizers that only tune few-shot examples, MIPROv2 jointly optimizes:

1. **Instructions** — the natural-language task descriptions in each module's prompt
2. **Few-shot demonstrations** — the input-output examples included in each module's prompt

It works by proposing candidate instructions, bootstrapping demonstrations, and searching over combinations using Bayesian optimization. The result is a program with better prompts that produce higher-quality outputs.

## When to use MIPROv2

- **Production optimization** — you want the best prompt quality DSPy can deliver
- **50+ training examples** — MIPROv2 needs enough data to search effectively
- **Both instructions and demos matter** — you want the optimizer to tune everything, not just examples
- **You have budget for multiple LM calls** — MIPROv2 is more expensive than BootstrapFewShot but produces better results

If you have fewer than 50 examples or need a quick first pass, start with `BootstrapFewShot` (see `/dspy-bootstrap-few-shot`), then upgrade to MIPROv2.

## Basic usage

```python
import dspy
from dspy.evaluate import Evaluate

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Your program
qa = dspy.ChainOfThought("question -> answer")

# 2. Your data (mark which fields are inputs)
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is 2+2?", answer="4").with_inputs("question"),
    # 50-200+ examples recommended
]

devset = [
    dspy.Example(question="Who wrote Hamlet?", answer="Shakespeare").with_inputs("question"),
    # 20-50 held-out examples for evaluation
]

# 3. Your metric
def metric(example, prediction, trace=None):
    return prediction.answer.strip().lower() == example.answer.strip().lower()

# 4. Optimize with MIPROv2
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized = optimizer.compile(qa, trainset=trainset)

# 5. Evaluate improvement
evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_progress=True)
score = evaluator(optimized)
print(f"Optimized score: {score:.1f}%")

# 6. Save
optimized.save("optimized_qa.json")
```

## The auto parameter

The `auto` parameter controls how much computation MIPROv2 uses. It sets the number of instruction candidates, demo candidates, and search trials automatically:

| Level | What it does | Typical cost | When to use |
|-------|-------------|-------------|-------------|
| `"light"` | Fewer candidates, fewer trials | ~$1-2 | Quick experiments, early iteration |
| `"medium"` | Balanced search | ~$5-10 | Default choice for most tasks |
| `"heavy"` | More candidates, more trials | ~$15-30 | Production, maximum quality |

```python
# Quick experiment
optimizer = dspy.MIPROv2(metric=metric, auto="light")

# Balanced (recommended starting point)
optimizer = dspy.MIPROv2(metric=metric, auto="medium")

# Maximum quality
optimizer = dspy.MIPROv2(metric=metric, auto="heavy")
```

**Start with `"medium"`**. Only move to `"heavy"` if you have a large trainset (200+), a meaningful metric, and the budget for it. Use `"light"` for quick sanity checks during development.

## What MIPROv2 tunes

MIPROv2 optimizes every `dspy.Predict` (or `dspy.ChainOfThought`, etc.) module in your program. For each module, it tunes:

### Instructions

MIPROv2 generates candidate instructions by analyzing your training data and the task structure. It proposes multiple phrasings, then searches for the combination that maximizes your metric.

### Few-shot demonstrations

MIPROv2 bootstraps demonstrations by running your program on training examples and keeping successful traces (where the metric passes). It then selects which demos to include in each module's prompt.

### Joint optimization

The key advantage over simpler optimizers: MIPROv2 searches over **combinations** of instructions and demos together. Good instructions may need different demos than mediocre instructions, and MIPROv2 finds the best pairing.

## Key parameters

```python
optimizer = dspy.MIPROv2(
    metric=metric,          # Required: your metric function
    auto="medium",          # "light", "medium", "heavy" — controls search budget
)

optimized = optimizer.compile(
    my_program,             # Required: the program to optimize
    trainset=trainset,      # Required: list of dspy.Example with .with_inputs()
)
```

### Manual configuration (advanced)

If `auto` doesn't give you enough control, you can set parameters directly:

```python
optimizer = dspy.MIPROv2(
    metric=metric,
    num_candidates=10,          # Number of instruction candidates to generate per module
    max_bootstrapped_demos=4,   # Max bootstrapped demos per module
    max_labeled_demos=4,        # Max labeled demos per module
    num_trials=30,              # Number of Bayesian optimization trials
)
```

Most users should stick with `auto`. Manual configuration is useful when you want to fine-tune the search budget or when you have domain-specific constraints (e.g., limiting demo count to keep prompts short).

## Computational cost

MIPROv2 makes many LM calls during optimization. The cost depends on:

- **auto level** — `"heavy"` makes roughly 5-10x more calls than `"light"`
- **Number of modules** — programs with multiple Predict/ChainOfThought modules cost more
- **Trainset size** — more examples means more bootstrapping runs
- **Model cost** — using GPT-4o costs more per call than GPT-4o-mini

### Cost management tips

1. **Develop with `"light"`, ship with `"medium"` or `"heavy"`** — iterate cheaply, then invest in the final optimization
2. **Use a cheaper model for optimization, then evaluate on the target model** — if your production model is expensive, optimize with a cheaper one first to validate the approach
3. **Start with fewer training examples** — 50-100 examples is enough for `"light"` and `"medium"`; scale up for `"heavy"`
4. **Set `num_threads`** in your evaluator to parallelize evaluation calls

### Typical wall-clock time

| auto level | 50 examples | 200 examples |
|-----------|------------|-------------|
| `"light"` | 2-5 min | 5-15 min |
| `"medium"` | 10-20 min | 20-40 min |
| `"heavy"` | 30-60 min | 1-3 hours |

Times vary significantly based on model latency, number of modules, and thread count.

## Comparison with other optimizers

| | MIPROv2 | BootstrapFewShot | BootstrapFewShotWithRandomSearch | GEPA |
|---|---------|-----------------|--------------------------------|------|
| **Tunes instructions** | Yes | No | No | Yes |
| **Tunes demos** | Yes | Yes | Yes | No |
| **Joint optimization** | Yes | No | No | No |
| **Min examples** | ~50 | ~10 | ~50 | ~10 |
| **Typical improvement** | 15-35% | 5-20% | 10-25% | 5-15% |
| **Cost** | Medium-High | Low | Medium | Low |
| **Best for** | Production | Quick start | Better than bootstrap | Few examples |

### When to use what

- **BootstrapFewShot** — first optimization pass, quick iteration, small datasets
- **BootstrapFewShotWithRandomSearch** — better than BootstrapFewShot when you have 50+ examples and more budget
- **MIPROv2** — best prompt optimization, production use, 50+ examples
- **GEPA** — instruction-only tuning, very few examples
- **BootstrapFinetune** — fine-tuning model weights (different category entirely)

### Stacking optimizers

A common pattern is to run BootstrapFewShot first, then MIPROv2 on the result. Bootstrap finds good demonstrations quickly, then MIPRO refines the instructions around them:

```python
# Step 1: Quick bootstrap
bootstrap = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
bootstrapped = bootstrap.compile(my_program, trainset=trainset)

# Step 2: Refine with MIPROv2
mipro = dspy.MIPROv2(metric=metric, auto="medium")
final = mipro.compile(bootstrapped, trainset=trainset)
```

This often beats running either optimizer alone.

## Save and load

```python
# Save the optimized program
optimized.save("optimized_program.json")

# Load later
from my_module import MyProgram  # your program class
loaded = MyProgram()
loaded.load("optimized_program.json")

# Use it
result = loaded(question="What is DSPy?")
```

Optimized prompts are model-specific. If you switch LM providers or models, re-run the optimizer. See `/ai-switching-models`.

## Common patterns

### Evaluate before and after

Always measure the baseline before optimizing so you know the improvement:

```python
from dspy.evaluate import Evaluate

evaluator = Evaluate(devset=devset, metric=metric, num_threads=4, display_table=5)

# Baseline
baseline_score = evaluator(my_program)

# Optimize
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized = optimizer.compile(my_program, trainset=trainset)

# Compare
optimized_score = evaluator(optimized)
print(f"Baseline:  {baseline_score:.1f}%")
print(f"Optimized: {optimized_score:.1f}%")
print(f"Delta:     {optimized_score - baseline_score:+.1f}%")
```

### Trace-aware metric for better demos

Use the `trace` parameter to require stricter quality during optimization. This makes MIPROv2 select higher-quality demonstrations:

```python
def metric(example, prediction, trace=None):
    correct = prediction.answer.strip().lower() == example.answer.strip().lower()
    if trace is not None:
        # During optimization: require reasoning too
        has_reasoning = len(getattr(prediction, "reasoning", "")) > 50
        return correct and has_reasoning
    return correct
```

### Multi-module programs

MIPROv2 optimizes all modules in your program. For a multi-step pipeline, each module gets its own optimized instructions and demos:

```python
class RAGPipeline(dspy.Module):
    def __init__(self):
        self.retrieve = dspy.Retrieve(k=3)
        self.generate = dspy.ChainOfThought("context, question -> answer")

    def forward(self, question):
        context = self.retrieve(question).passages
        return self.generate(context=context, question=question)

rag = RAGPipeline()
optimizer = dspy.MIPROv2(metric=metric, auto="medium")
optimized_rag = optimizer.compile(rag, trainset=trainset)
```

## Cross-references

> Install any skill: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill <name>`

- Need to prepare training data? Use `/dspy-data`
- Want to write and run metrics? Use `/dspy-evaluate`
- Starting with a simpler optimizer first? Use `/dspy-bootstrap-few-shot`
- Want random search over few-shot demos? Use `/dspy-bootstrap-rs`
- For the full measure-improve-verify loop, see `/ai-improving-accuracy`
- For worked examples, see [examples.md](examples.md)
- **Install `/ai-do` if you do not have it** — it routes any AI problem to the right skill and is the fastest way to work: `npx skills add lebsral/DSPy-Programming-not-prompting-LMs-skills --skill ai-do`
