---
name: dspy-refine
description: "Use when you want outputs to improve iteratively through self-critique — generating a draft, evaluating it against criteria, and revising until quality thresholds are met. Common scenarios: iteratively improving essay quality through self-critique, generating then evaluating and revising answers, tasks requiring multiple drafts, quality-sensitive content generation, or any task where a first draft can be systematically improved. Related: ai-writing-content, ai-improving-accuracy, dspy-best-of-n. Also: \"dspy.Refine\", \"self-critique and revise\", \"iterative improvement loop\", \"generate then evaluate then fix\", \"AI self-editing\", \"multi-draft generation\", \"quality through iteration\", \"revise until good enough\", \"critique-driven refinement\", \"LLM improves its own output\", \"reflective generation\", \"edit loop for AI outputs\", \"when first draft isn't good enough\", \"self-improving AI output\"."
---

# Iterative Self-Improvement with dspy.Refine

Guide the user through using `dspy.Refine` to build pipelines that automatically retry and improve outputs until they meet a quality threshold.

## What is dspy.Refine

`dspy.Refine` is a DSPy module wrapper that runs another module up to N times, scoring each attempt with a reward function. It returns the first output that meets a threshold -- or the best output if none do. When an attempt fails to meet the threshold, Refine generates feedback that gets fed into the next attempt, enabling genuine iterative improvement rather than just random retries.

Key properties:

- **Wraps any DSPy module** -- ChainOfThought, Predict, ReAct, or your custom modules
- **Scores each attempt** with a reward function you define
- **Generates feedback** when an attempt falls short, improving subsequent tries
- **Returns early** as soon as an output meets the threshold (saves LM calls)
- **Falls back gracefully** -- returns the best attempt even if none hit the threshold

## When to use Refine

Use `dspy.Refine` when:

- Outputs must meet measurable quality criteria (format, length, accuracy)
- You can write a function that scores output quality as a number
- You want the LM to learn from its mistakes within a single request
- Quality is worth the extra LM calls (2-5x cost for N attempts)

Do **not** use Refine when:

- You have no clear way to score outputs -- use `dspy.ChainOfThought` instead
- You need human-in-the-loop feedback -- build a custom module with `dspy.Suggest`
- Speed matters more than quality -- use a single `dspy.Predict` call
- You just want multiple independent attempts without feedback -- use `dspy.BestOfN` (see comparison below)

## Basic usage

Three things are needed: a module to wrap, a reward function, and a threshold.

```python
import dspy

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

# 1. Define the module to refine
qa = dspy.ChainOfThought("question -> answer")

# 2. Define a reward function
# Takes (args_dict, prediction) -> float
def concise_answer(args, pred):
    """Reward one-word answers."""
    return 1.0 if len(pred.answer.split()) == 1 else 0.0

# 3. Wrap with Refine
refined_qa = dspy.Refine(
    module=qa,
    N=3,
    reward_fn=concise_answer,
    threshold=1.0,
)

# Use it -- same interface as the wrapped module
result = refined_qa(question="What is the capital of Belgium?")
print(result.answer)  # "Brussels"
```

## Constructor parameters

```python
dspy.Refine(
    module,       # The DSPy module to refine (required)
    N,            # Max number of attempts (required, int)
    reward_fn,    # Callable(args_dict, prediction) -> float (required)
    threshold,    # Target reward score to accept an output (required, float)
    fail_count,   # Max failures before raising an error (optional, defaults to N)
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | `dspy.Module` | The module whose outputs you want to refine |
| `N` | `int` | Maximum number of attempts. Each attempt uses temperature=1.0 with a different rollout ID |
| `reward_fn` | `Callable` | Scores a prediction. Receives `(args, pred)` where `args` is the input kwargs dict and `pred` is the module's output. Must return a `float` |
| `threshold` | `float` | Target score. Refine returns immediately when an attempt meets or exceeds this value |
| `fail_count` | `int` | Optional. Maximum allowed failures before raising an error. Defaults to `N` |

## Writing reward functions

The reward function is the core of Refine. It receives two arguments:

1. **`args`** -- a dict of the inputs passed to the module (e.g., `{"question": "What is..."}`)
2. **`pred`** -- the module's prediction object (access fields like `pred.answer`, `pred.reasoning`)

It must return a `float`. Higher is better.

### Simple binary reward

```python
def valid_json(args, pred):
    """Accept only valid JSON outputs."""
    import json
    try:
        json.loads(pred.output)
        return 1.0
    except (json.JSONDecodeError, TypeError):
        return 0.0
```

### Graduated reward

Return partial scores to help Refine pick the best attempt even when none fully succeed:

```python
def quality_score(args, pred):
    """Score answer quality on multiple criteria."""
    score = 0.0
    answer = pred.answer

    # Criterion 1: not empty
    if answer.strip():
        score += 0.3

    # Criterion 2: reasonable length (20-200 words)
    word_count = len(answer.split())
    if 20 <= word_count <= 200:
        score += 0.4

    # Criterion 3: addresses the question
    if args["question"].split()[0].lower() in answer.lower():
        score += 0.3

    return score
```

### Using external validation

```python
import re

def valid_email_extraction(args, pred):
    """Reward valid email addresses extracted from text."""
    emails = pred.emails if isinstance(pred.emails, list) else []
    if not emails:
        return 0.0
    email_pattern = r'^[\w.-]+@[\w.-]+\.\w+$'
    valid_count = sum(1 for e in emails if re.match(email_pattern, e))
    return valid_count / len(emails)
```

## How iteration count (N) works

Each attempt runs the wrapped module at temperature=1.0 with a different rollout ID, producing diverse outputs. Refine's selection logic:

1. Run the module and score the output with `reward_fn`
2. If the score meets or exceeds `threshold`, return immediately
3. If not, generate feedback from the failure and try again
4. After N attempts, return the attempt with the highest reward score

**Choosing N:**

| N value | Use case | Cost |
|---------|----------|------|
| 2-3 | Format validation, simple constraints | Low overhead |
| 3-5 | Quality criteria, multi-factor scoring | Moderate |
| 5-10 | High-stakes outputs, strict requirements | Higher cost, better results |

The sweet spot for most use cases is **N=3 to N=5**. Beyond 5, diminishing returns are common unless the reward function is very specific.

## The feedback mechanism

What makes Refine different from random retries is **feedback generation**. When an attempt fails to meet the threshold:

1. Refine examines why the attempt scored below the threshold
2. It generates natural-language feedback describing the shortcoming
3. This feedback is included in the prompt for the next attempt
4. The LM uses this feedback to produce a better output

This means later attempts are informed by earlier failures. Attempt 3 knows what went wrong in attempts 1 and 2.

You do not write the feedback logic -- Refine handles it automatically based on your reward function's scores.

## Refine vs BestOfN -- when to use which

Both modules run a wrapped module multiple times and select the best output, but they work differently:

| Aspect | `dspy.Refine` | `dspy.BestOfN` |
|--------|--------------|----------------|
| **Feedback** | Generates feedback from failures, improving subsequent attempts | No feedback -- each attempt is independent |
| **Attempts** | Sequential (each informed by previous) | Can be parallel (independent) |
| **Early stopping** | Returns on first success meeting threshold | Runs all N, picks best |
| **Best for** | Iterative improvement, complex quality criteria | Sampling diversity, simple pass/fail |
| **Cost pattern** | Often fewer LM calls (stops early) | Always N calls |

**Use Refine when** the LM can improve with feedback -- writing tasks, format compliance, multi-criteria quality.

**Use BestOfN when** attempts are independent and feedback would not help -- creative generation, sampling diverse options, simple binary checks.

## Wrapping custom modules

Refine works with any `dspy.Module`, not just built-in ones:

```python
class Summarizer(dspy.Module):
    def __init__(self):
        self.summarize = dspy.ChainOfThought("article -> summary")

    def forward(self, article):
        return self.summarize(article=article)


def good_summary(args, pred):
    """Score summary quality."""
    summary = pred.summary
    article = args["article"]
    score = 0.0

    # Shorter than original
    if len(summary) < len(article) * 0.3:
        score += 0.5

    # At least 2 sentences
    if summary.count('.') >= 2:
        score += 0.5

    return score


refined_summarizer = dspy.Refine(
    module=Summarizer(),
    N=3,
    reward_fn=good_summary,
    threshold=0.8,
)

result = refined_summarizer(article="Long article text here...")
print(result.summary)
```

## Tips

- **Start with N=3** and increase only if outputs consistently miss the threshold
- **Use graduated rewards** (0.0 to 1.0) rather than binary (0 or 1) so Refine can pick the best near-miss
- **Keep reward functions fast** -- they run on every attempt, so avoid expensive operations like LM calls inside them
- **Set threshold realistically** -- if your reward function rarely returns 1.0, set the threshold to 0.8 or similar
- **Use `fail_count`** to limit retries on genuinely impossible inputs rather than burning through all N attempts

## Cross-references

- **Chain of thought reasoning** as the inner module -- see `/dspy-chain-of-thought`
- **Checking and validating outputs** with assertions -- see `/ai-checking-outputs`
- **Improving accuracy** with optimization -- see `/ai-improving-accuracy`
- For worked examples, see [examples.md](examples.md)
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
