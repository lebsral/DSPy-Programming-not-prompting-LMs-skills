---
name: dspy-ensemble
description: "Use when you have multiple optimized versions of a program and want to combine them — voting, averaging, or routing across program variants for more robust outputs."
---

# Combine Programs with dspy.Ensemble

Guide the user through using DSPy's `Ensemble` optimizer to combine multiple optimized programs into a single ensemble that aggregates their outputs. This is useful when you've run several optimization passes (different optimizers, different hyperparameters, different random seeds) and want to combine them for more robust predictions.

## What is Ensemble

`dspy.Ensemble` is an optimizer (teleprompter) that takes a list of DSPy programs and returns a single `EnsembledProgram`. When you call the ensembled program, it runs each constituent program on the same inputs and aggregates the results using a reduce function you provide.

```
Program A ──┐
Program B ──┼──> Run all ──> reduce_fn ──> Single output
Program C ──┘
```

Unlike other optimizers that tune prompts or weights, Ensemble doesn't change the programs themselves. It combines their outputs at inference time.

## When to use Ensemble

- **You ran multiple optimization passes** (e.g., several BootstrapFewShot runs with different seeds) and want to combine the best of each
- **You want majority voting** -- run several programs and pick the most common answer for higher reliability
- **You want to average numeric outputs** -- combine scores or probabilities from multiple models
- **Different optimizers produced different strengths** -- one program is good at precision, another at recall, and you want both
- **You need a quick reliability boost** -- ensembling is a well-known technique to reduce variance

Do **not** use Ensemble when:
- You only have one program (nothing to ensemble)
- Latency is critical -- ensembling runs every program, multiplying your inference time
- Cost is a hard constraint -- you pay for every program in the ensemble
- Your programs produce complex structured outputs that are hard to aggregate

## Basic usage

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Define your base program
qa = dspy.ChainOfThought("question -> answer")

# 2. Create a training set and metric
trainset = [
    dspy.Example(question="What is the capital of France?", answer="Paris").with_inputs("question"),
    dspy.Example(question="What is 2 + 2?", answer="4").with_inputs("question"),
    # ... more examples
]

def exact_match(example, pred, trace=None):
    return pred.answer.strip().lower() == example.answer.strip().lower()

# 3. Run multiple optimization passes to get different programs
programs = []
for i in range(3):
    optimizer = dspy.BootstrapFewShot(
        metric=exact_match,
        max_bootstrapped_demos=4,
        max_labeled_demos=4,
    )
    optimized = optimizer.compile(qa, trainset=trainset)
    programs.append(optimized)

# 4. Combine with Ensemble using majority voting
ensemble_optimizer = dspy.Ensemble(reduce_fn=dspy.majority, size=None)
ensemble_program = ensemble_optimizer.compile(programs)

# 5. Use the ensemble like any module
result = ensemble_program(question="What is the capital of Germany?")
print(result.answer)
```

## Constructor parameters

```python
dspy.Ensemble(
    reduce_fn=None,     # Function to aggregate outputs from all programs
    size=None,          # How many programs to sample (None = use all)
    deterministic=False,  # Must be False (deterministic mode not yet implemented)
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `reduce_fn` | `Callable \| None` | Aggregation function applied to the list of outputs. If `None`, returns the raw list of predictions. |
| `size` | `int \| None` | Number of programs to randomly sample from the ensemble. `None` means use all programs. |
| `deterministic` | `bool` | Reserved for future use. Must be `False`. |

### compile method

```python
ensemble_optimizer.compile(programs)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `programs` | `list[dspy.Module]` | List of DSPy programs to ensemble |

Returns an `EnsembledProgram` that runs the selected programs and applies `reduce_fn`.

## Reduce functions

The reduce function determines how outputs from multiple programs are combined into a single result.

### dspy.majority (built-in)

The most common reduce function. It picks the most frequent output value across all programs -- majority voting.

```python
ensemble = dspy.Ensemble(reduce_fn=dspy.majority)
```

Use `dspy.majority` when:
- Outputs are categorical (classification labels, short factual answers, yes/no)
- You want the most robust answer -- the one most programs agree on

### Custom reduce: averaging numeric outputs

```python
def average_scores(predictions):
    """Average a numeric output field across all predictions."""
    scores = [float(p.score) for p in predictions]
    avg = sum(scores) / len(scores)
    # Return a Prediction-like object with the averaged score
    return predictions[0].__class__(score=str(avg))

ensemble = dspy.Ensemble(reduce_fn=average_scores)
```

### Custom reduce: weighted voting

```python
def weighted_vote(predictions):
    """Pick the answer backed by the most programs, with confidence weighting."""
    from collections import Counter
    votes = Counter(p.answer for p in predictions)
    winner = votes.most_common(1)[0][0]
    # Return a prediction with the winning answer
    return predictions[0].__class__(answer=winner)

ensemble = dspy.Ensemble(reduce_fn=weighted_vote)
```

### No reduce function

If you pass `reduce_fn=None`, the ensembled program returns the raw list of predictions from all programs. This is useful when you want to implement custom post-processing logic outside the ensemble.

```python
ensemble = dspy.Ensemble(reduce_fn=None)
ensemble_program = ensemble.compile(programs)

# Returns a list of predictions
all_predictions = ensemble_program(question="What is DSPy?")
# Process them yourself
for pred in all_predictions:
    print(pred.answer)
```

## Combining different optimizers

One of the most powerful uses of Ensemble is combining programs from different optimization strategies. Each optimizer may find different strengths.

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

qa = dspy.ChainOfThought("question -> answer")

# Program 1: Optimized with BootstrapFewShot
opt1 = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
prog1 = opt1.compile(qa, trainset=trainset)

# Program 2: Optimized with MIPROv2
opt2 = dspy.MIPROv2(metric=metric, auto="light")
prog2 = opt2.compile(qa, trainset=trainset)

# Program 3: Optimized with BootstrapFewShotWithRandomSearch
opt3 = dspy.BootstrapFewShotWithRandomSearch(
    metric=metric,
    max_bootstrapped_demos=4,
    num_candidate_programs=5,
)
prog3 = opt3.compile(qa, trainset=trainset)

# Ensemble all three
ensemble = dspy.Ensemble(reduce_fn=dspy.majority)
combined = ensemble.compile([prog1, prog2, prog3])

result = combined(question="What is the tallest mountain?")
print(result.answer)
```

This approach works because different optimizers explore different parts of the prompt space. BootstrapFewShot finds good demonstrations, MIPROv2 finds good instructions, and combining them via voting smooths out individual weaknesses.

## Sampling with size

When you have many optimized programs (e.g., from a large random search), you can use `size` to randomly sample a subset at inference time. This reduces cost while still benefiting from diversity.

```python
# You have 10 programs from BootstrapFewShotWithRandomSearch
programs = [...]  # 10 optimized programs

# Only run 3 of them per inference call (randomly sampled)
ensemble = dspy.Ensemble(reduce_fn=dspy.majority, size=3)
ensemble_program = ensemble.compile(programs)
```

Each call to `ensemble_program` randomly picks 3 of the 10 programs, runs them, and applies majority voting. This balances diversity against cost.

## Cost and latency considerations

Ensemble multiplies your inference cost and latency by the number of programs (or `size` if set):

| Programs | Cost multiplier | Latency (sequential) |
|----------|-----------------|----------------------|
| 3 | 3x | 3x |
| 5 | 5x | 5x |
| 10 | 10x | 10x |

Ways to manage this:
- **Use `size` to cap the number of programs** run per inference call
- **Use cheaper models** for the ensemble members and reserve expensive models for critical paths
- **Ensemble at evaluation time only** to pick the single best program, then deploy that one program in production
- **Parallelize** if your infrastructure supports concurrent LM calls -- the programs are independent

## Ensemble vs BestOfN

Both combine multiple outputs, but they work differently:

| | Ensemble | BestOfN |
|---|---------|---------|
| **What it combines** | Different optimized programs | Multiple runs of the same program |
| **Selection method** | Voting / averaging across programs | Reward function picks the best single run |
| **Diversity source** | Different prompts/demos from optimization | Temperature sampling of the same prompt |
| **When to use** | You have multiple optimized programs | You have one program and a scoring metric |
| **Optimizer type** | Combines at the program level | Combines at the inference level |

You can even stack them: ensemble multiple optimized programs, then wrap the ensemble with BestOfN for additional quality.

## Cross-references

- **BestOfN** for picking the best from multiple runs of a single program -- see `/dspy-best-of-n`
- **BootstrapFewShot** for generating the programs to ensemble -- see `/ai-improving-accuracy`
- **MIPROv2** for instruction optimization -- see `/ai-improving-accuracy`
- **Evaluate** for measuring ensemble quality with metrics -- see `/dspy-evaluate`
- For worked examples, see [examples.md](examples.md)
