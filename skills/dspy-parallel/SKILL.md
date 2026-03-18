---
name: dspy-parallel
description: "Use when you have independent LM calls that can run concurrently — batch processing, fan-out patterns, or speeding up pipelines with no data dependencies between steps."
---

# Run LM Calls in Parallel with dspy.Parallel

Guide the user through using DSPy's `Parallel` module to execute multiple LM calls concurrently. `dspy.Parallel` is the built-in way to speed up batch processing and fan-out patterns without writing threading code yourself.

## What is dspy.Parallel

`dspy.Parallel` takes a list of (module, inputs) pairs and executes them concurrently using a thread pool. It handles threading, progress bars, error limits, and timeouts so you don't have to.

Use it when you have:

- **A batch of inputs** to run through the same module (classify 500 tickets, summarize 100 articles)
- **Multiple independent modules** to run on the same input (sentiment + topics + entities at once)
- **Any set of LM calls** that don't depend on each other

If call B depends on the result of call A, those two calls must be sequential. Everything else can be parallel.

## Basic usage

Pass a list of `(module, inputs)` pairs. Each pair is one unit of work:

```python
import dspy

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# A module to run on every input
classify = dspy.Predict("text -> label: str")

# A batch of inputs
texts = [
    "I love this product!",
    "Terrible experience, want a refund.",
    "It's okay, nothing special.",
    "Best purchase I've made this year.",
]

# Build execution pairs: (module, inputs_dict)
exec_pairs = [(classify, {"text": t}) for t in texts]

# Run them all in parallel
parallel = dspy.Parallel(num_threads=4)
results = parallel(exec_pairs)

for text, result in zip(texts, results):
    print(f"{text[:30]:30s} -> {result.label}")
```

`results` is a list in the same order as `exec_pairs`, so `results[i]` corresponds to `exec_pairs[i]`.

## Constructor options

```python
dspy.Parallel(
    num_threads=4,               # number of concurrent threads (default: settings.num_threads)
    max_errors=5,                # stop after this many failures (default: None = no limit)
    return_failed_examples=False,# if True, return failures separately instead of raising
    provide_traceback=False,     # include tracebacks in error output
    disable_progress_bar=False,  # suppress the tqdm progress bar
    timeout=120,                 # max seconds per task before timeout
)
```

| Parameter | Type | Default | Purpose |
|-----------|------|---------|---------|
| `num_threads` | `int \| None` | `None` | Number of concurrent threads. Falls back to `dspy.settings.num_threads`. |
| `max_errors` | `int \| None` | `None` | Stop execution after this many errors. `None` means no limit. |
| `return_failed_examples` | `bool` | `False` | When `True`, return failed examples separately instead of raising. |
| `provide_traceback` | `bool \| None` | `None` | Include Python tracebacks for failed examples. |
| `disable_progress_bar` | `bool` | `False` | Suppress the progress bar. |
| `timeout` | `int` | `120` | Max seconds per individual task. |
| `straggler_limit` | `int` | `3` | Threshold for flagging slow-running tasks. |

## Configuring concurrency

Start with a thread count that matches your rate limits, not your CPU cores. LM calls are I/O-bound (waiting on HTTP responses), so you can safely use many threads:

```python
# Conservative -- good starting point
parallel = dspy.Parallel(num_threads=4)

# Aggressive -- if your provider allows high concurrency
parallel = dspy.Parallel(num_threads=16)

# Match your provider's rate limit
# e.g., 60 requests/min = ~1/sec, so 4-8 threads keeps the pipeline full
parallel = dspy.Parallel(num_threads=8)
```

If you hit rate-limit errors (HTTP 429), reduce `num_threads` or add retry logic in your LM configuration.

## Input formats

`Parallel` accepts inputs as dictionaries, `dspy.Example` objects, or tuples:

```python
module = dspy.Predict("question -> answer")

# Dict inputs (most common)
pairs = [(module, {"question": "What is DSPy?"})]

# dspy.Example inputs
example = dspy.Example(question="What is DSPy?").with_inputs("question")
pairs = [(module, example)]

# Both work the same way
parallel = dspy.Parallel(num_threads=2)
results = parallel(pairs)
```

## Aggregating results

Results come back as a list. Aggregate however your application needs:

```python
import dspy

classify = dspy.Predict("text -> label: str, confidence: float")
texts = ["Great!", "Terrible.", "Meh.", "Amazing!", "Awful."]

parallel = dspy.Parallel(num_threads=4)
results = parallel([(classify, {"text": t}) for t in texts])

# Count labels
from collections import Counter
label_counts = Counter(r.label for r in results)
print(label_counts)  # Counter({'positive': 2, 'negative': 2, 'neutral': 1})

# Filter by confidence
high_confidence = [
    (text, r.label)
    for text, r in zip(texts, results)
    if r.confidence > 0.8
]

# Build a summary dict
output = [
    {"text": t, "label": r.label, "confidence": r.confidence}
    for t, r in zip(texts, results)
]
```

## Error handling

By default, `Parallel` raises an exception after `max_errors` failures. To handle errors gracefully, use `return_failed_examples=True`:

```python
parallel = dspy.Parallel(
    num_threads=4,
    max_errors=10,
    return_failed_examples=True,
    provide_traceback=True,
)

results, failed_examples, exceptions = parallel(exec_pairs)
```

When `return_failed_examples=True`, the return value is a 3-tuple:

- **`results`** -- list of successful predictions (same length as successes)
- **`failed_examples`** -- list of `(module, inputs)` pairs that failed
- **`exceptions`** -- list of exceptions corresponding to each failure

Handle failures after the batch completes:

```python
results, failed, errors = parallel(exec_pairs)

print(f"Succeeded: {len(results)}, Failed: {len(failed)}")

# Retry failures with a fallback module
if failed:
    fallback = dspy.ChainOfThought("text -> label: str")
    retry_pairs = [(fallback, inputs) for _, inputs in failed]
    retry_results = parallel(retry_pairs)
```

### Setting an error budget

Use `max_errors` to fail fast when too many calls are failing (e.g., provider outage):

```python
# Stop the whole batch if more than 5 calls fail
parallel = dspy.Parallel(num_threads=4, max_errors=5)

try:
    results = parallel(exec_pairs)
except Exception as e:
    print(f"Batch aborted: {e}")
```

### Timeouts

The `timeout` parameter sets a per-task time limit in seconds. Tasks that exceed this are terminated:

```python
# Give each task up to 60 seconds
parallel = dspy.Parallel(num_threads=4, timeout=60)
```

## Using different modules per item

Each pair can use a different module. This is useful for fan-out patterns where you run multiple analyses on the same input:

```python
import dspy

sentiment = dspy.Predict("text -> sentiment: str")
topics = dspy.Predict("text -> topics: list[str]")
summary = dspy.ChainOfThought("text -> summary: str")

text = "DSPy is a framework for programming language models..."

# Fan out: three different modules, same input
exec_pairs = [
    (sentiment, {"text": text}),
    (topics, {"text": text}),
    (summary, {"text": text}),
]

parallel = dspy.Parallel(num_threads=3)
results = parallel(exec_pairs)

combined = {
    "sentiment": results[0].sentiment,
    "topics": results[1].topics,
    "summary": results[2].summary,
}
```

## When to use Parallel vs a sequential loop

| Scenario | Use | Why |
|----------|-----|-----|
| Process 100+ items through the same module | `Parallel` | Massive speedup from concurrent HTTP requests |
| Run 3 independent analyses on one input | `Parallel` | All three calls happen at once |
| Pipeline where step 2 needs step 1's output | Sequential loop | There's a data dependency |
| Single LM call | Neither | No benefit from parallelism |
| Processing 2-3 items | Either works | Overhead is negligible either way |

### Sequential loop (before)

```python
# Slow: each call waits for the previous one to finish
results = []
for text in texts:
    result = classify(text=text)
    results.append(result)
```

### Parallel (after)

```python
# Fast: all calls run concurrently
parallel = dspy.Parallel(num_threads=8)
results = parallel([(classify, {"text": t}) for t in texts])
```

For a batch of 100 items with ~1 second per LM call:
- Sequential: ~100 seconds
- Parallel (8 threads): ~13 seconds

## Parallel inside a module

Wrap `Parallel` usage inside a `dspy.Module` for clean composition:

```python
class BatchClassifier(dspy.Module):
    def __init__(self, num_threads=4):
        self.classify = dspy.Predict("text -> label: str, confidence: float")
        self.num_threads = num_threads

    def forward(self, texts: list[str]):
        parallel = dspy.Parallel(num_threads=self.num_threads)
        exec_pairs = [(self.classify, {"text": t}) for t in texts]
        results = parallel(exec_pairs)

        return dspy.Prediction(
            labels=[r.label for r in results],
            confidences=[r.confidence for r in results],
        )

# Usage
classifier = BatchClassifier(num_threads=8)
result = classifier(texts=["Great!", "Terrible.", "Meh."])
print(result.labels)  # ["positive", "negative", "neutral"]
```

This keeps the parallelism as an implementation detail. Callers don't need to know about threading -- they just pass a list and get a list back.

## Cross-references

- **Modules** are the building blocks you pass to Parallel -- see `/dspy-modules`
- **Multi-step pipelines** that combine sequential and parallel stages -- see `/ai-building-pipelines`
- **Evaluation** uses its own threading via `num_threads` -- see `/dspy-evaluate`
- For worked examples (batch classification, multi-aspect analysis), see [examples.md](examples.md)
