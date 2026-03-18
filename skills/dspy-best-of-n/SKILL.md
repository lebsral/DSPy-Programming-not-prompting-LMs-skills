---
name: dspy-best-of-n
description: "Use when output quality varies across runs and you want to sample multiple completions and pick the best — trading latency for reliability on high-stakes outputs."
---

# Pick the Best Output with dspy.BestOfN

Guide the user through using DSPy's `BestOfN` module to run a program multiple times and keep the highest-scoring result. This is rejection sampling -- generate N candidates, score each one, return the winner.

## What is BestOfN

`dspy.BestOfN` wraps any DSPy module and calls it up to N times with `temperature=1.0` (each attempt uses a different rollout ID to get diverse outputs). A reward function scores every result, and BestOfN returns the single best prediction.

If any attempt hits a score threshold you set, execution stops early -- no need to burn through all N attempts when you already have a great result.

```
Your module ──> Run N times ──> Score each with reward_fn ──> Return best
```

## When to use BestOfN

- **You have a cheap, fast metric** that can score outputs automatically (test suite passes, regex match, word count check, etc.)
- **Quality variance is high** -- the same prompt sometimes produces great output and sometimes doesn't
- **You'd rather spend tokens than engineering time** -- BestOfN is the simplest way to boost quality without optimization
- **You need a quick quality boost** before investing in full prompt optimization with MIPROv2 or BootstrapFewShot

Do **not** use BestOfN when:
- You have no way to automatically score outputs (you need a metric)
- Latency matters more than quality (N calls take N times longer, unless you can parallelize)
- Cost is a hard constraint and N is large

## Basic usage

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Define your module
qa = dspy.ChainOfThought("question -> answer")

# 2. Define a reward function
def short_answer(args, pred):
    """Prefer concise single-word answers."""
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

# 3. Wrap with BestOfN
best_qa = dspy.BestOfN(
    module=qa,
    N=3,
    reward_fn=short_answer,
    threshold=1.0,
)

# 4. Call it like any module
result = best_qa(question="What is the capital of Belgium?")
print(result.answer)
```

## Constructor parameters

```python
dspy.BestOfN(
    module,       # Any dspy.Module to run repeatedly
    N,            # Number of attempts (int)
    reward_fn,    # Scoring function: (args_dict, prediction) -> float
    threshold,    # Early-stop threshold: stop as soon as a score >= threshold
    fail_count=None,  # Max failures before raising an error (defaults to N)
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `dspy.Module` | The module to run N times |
| `N` | `int` | Maximum number of attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | Scores each prediction; higher is better |
| `threshold` | `float` | If any attempt scores >= this value, return immediately |
| `fail_count` | `int \| None` | How many attempts can fail (raise exceptions) before BestOfN itself raises. Defaults to N (all can fail before error) |

## The reward function

The reward function is the core of BestOfN. It receives two arguments:

```python
def reward_fn(args: dict, prediction: dspy.Prediction) -> float:
    # args: the keyword arguments you passed to the BestOfN call
    # prediction: the output from one attempt of the wrapped module
    # Return: a scalar score (higher = better)
    ...
```

Key differences from a `dspy.Evaluate` metric:
- **Signature**: `(args_dict, prediction)` not `(example, prediction, trace)`
- **No gold labels**: `args` contains only the inputs you passed, not expected outputs
- **No trace parameter**: BestOfN doesn't use traces

### Reward function examples

**Binary pass/fail:**

```python
def passes_tests(args, pred):
    """Score 1.0 if generated code passes all tests, 0.0 otherwise."""
    try:
        exec(pred.code)
        return 1.0
    except Exception:
        return 0.0
```

**Graded score:**

```python
def quality_score(args, pred):
    """Score summaries on length and keyword coverage."""
    score = 0.0
    # Prefer summaries under 100 words
    if len(pred.summary.split()) <= 100:
        score += 0.5
    # Reward covering key topics
    keywords = ["revenue", "growth", "forecast"]
    covered = sum(1 for kw in keywords if kw in pred.summary.lower())
    score += 0.5 * (covered / len(keywords))
    return score
```

**Using an LM as judge inside the reward:**

```python
class JudgeQuality(dspy.Signature):
    """Rate the answer quality from 0.0 to 1.0."""
    question: str = dspy.InputField()
    answer: str = dspy.InputField()
    score: float = dspy.OutputField(desc="Quality score from 0.0 to 1.0")

judge = dspy.Predict(JudgeQuality)

def llm_reward(args, pred):
    result = judge(question=args["question"], answer=pred.answer)
    return result.score
```

Note: Using an LM as judge inside the reward function costs additional tokens per attempt. Reserve this for cases where programmatic scoring isn't feasible.

## Tuning N

| N | Trade-off |
|---|-----------|
| 2-3 | Low cost, modest quality gain. Good starting point. |
| 5 | Solid improvement for tasks with high variance. Sweet spot for most uses. |
| 10+ | Diminishing returns unless your metric is very selective (e.g., <10% pass rate). |

**Rule of thumb**: if your base module succeeds ~50% of the time, N=3 gives you a ~87.5% chance of at least one success. If it succeeds ~20% of the time, you need N=8 for ~83%.

The math: probability of at least one success in N tries = `1 - (1 - p)^N` where `p` is the single-attempt success rate.

## How selection works internally

1. BestOfN calls your module with `temperature=1.0` and a unique rollout ID for each attempt
2. Each attempt produces a `dspy.Prediction`
3. The reward function scores the prediction
4. If the score >= `threshold`, return immediately (early stopping)
5. If the attempt raises an exception, increment the failure counter
6. After all N attempts (or early stopping), return the prediction with the highest score
7. If failures exceed `fail_count`, raise an exception

The unique rollout IDs ensure the LM produces diverse outputs even with the same input. Temperature is fixed at 1.0 to maximize diversity.

## Cost considerations

BestOfN multiplies your token usage by up to N times (fewer if early stopping kicks in). Budget accordingly:

| Base cost per call | N | Max cost |
|--------------------|---|----------|
| $0.01 | 3 | $0.03 |
| $0.01 | 5 | $0.05 |
| $0.01 | 10 | $0.10 |

Ways to manage cost:
- **Set a tight threshold** so good results stop early (often after 1-2 attempts)
- **Use a cheap model** as the base module and a stronger model only for the reward function
- **Start with N=3** and increase only if your metric shows it helps
- **Use programmatic reward functions** (regex, test execution, length checks) instead of LM-based judges to avoid extra LM calls per attempt

## BestOfN vs MultiChainComparison

Both BestOfN and `dspy.MultiChainComparison` aim to pick the best output from multiple candidates, but they work differently:

| | BestOfN | MultiChainComparison |
|---|---------|---------------------|
| **Selection method** | Your reward function scores each candidate | An LM reads all candidates and picks the best |
| **Metric required** | Yes -- you must provide a `reward_fn` | No -- the LM decides what "best" means |
| **Token cost** | N calls to your module (+ reward fn) | Multiple chain calls + one comparison call |
| **Best when** | You have a clear, automatable scoring criterion | Quality is subjective or hard to score programmatically |
| **Optimizable** | The wrapped module can be optimized | The comparison module can be optimized |

Use BestOfN when you can write a reward function. Use MultiChainComparison when you want the LM to judge quality using its own understanding.

## Combining BestOfN with optimization

BestOfN works well as a complement to DSPy optimizers. Optimize your module first, then wrap the optimized version with BestOfN for an additional quality boost:

```python
# Optimize the base module
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized_qa = optimizer.compile(qa, trainset=trainset)

# Wrap the optimized module with BestOfN
best_qa = dspy.BestOfN(
    module=optimized_qa,
    N=3,
    reward_fn=my_reward,
    threshold=1.0,
)
```

This stacks two quality improvements: better prompts from the optimizer, and rejection sampling from BestOfN.

## Cross-references

- **MultiChainComparison** for LM-based candidate selection -- see `/dspy-multi-chain-comparison`
- **Evaluate** for measuring quality with metrics and devsets -- see `/dspy-evaluate`
- **Improving accuracy** for the full optimization workflow -- see `/ai-improving-accuracy`
- For worked examples, see [examples.md](examples.md)
