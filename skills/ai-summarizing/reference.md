> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Summarization

## Setup

```bash
pip install -U dspy       # DSPy 3.2.1+
```

```python
import dspy
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

## Approach quick reference

| Approach | Input length | LM calls | Best for |
|----------|-------------|----------|----------|
| `dspy.ChainOfThought` | Under ~4K words | 1 | Articles, emails, threads |
| Pydantic structured output | Under ~4K words | 1 | Meetings, support threads — action items and decisions |
| Parallel multi-aspect | Under ~4K words | 3-4 | When extraction quality matters more than cost |
| Map-reduce | 4K-50K words | N + 1 | Reports, long transcripts |
| Hierarchical | 50K+ words | N + log(N) | Books, legal documents |

## dspy.Signature

[API docs](https://dspy.ai/api/signatures/)

```python
class Summarize(dspy.Signature):
    """Summarize for a technical PM who needs to decide whether to escalate."""
    text: str = dspy.InputField(desc="The text to summarize")
    max_words: int = dspy.InputField(desc="Maximum number of words for the summary")
    summary: str = dspy.OutputField(desc="A concise summary within the word limit")
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `desc` | `str` | Field description passed to the LM |
| `prefix` | `str` | Override label in the prompt |

Do not add a `reasoning` field — `dspy.ChainOfThought` injects it automatically. Pass `max_words` as an `InputField` (not hardcoded in the docstring) so it is accessible in reward functions via `args["max_words"]`. Use `Literal["brief", "standard", "detailed"]` for constrained detail-level inputs.

For structured outputs (meetings, support threads), use a Pydantic `BaseModel` as the output type — DSPy enforces the schema and retries on parse failures. Access nested fields as `pred.summary.tldr`, `pred.summary.action_items`.

## dspy.ChainOfThought

[API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `signature` | `str \| type[Signature]` | required | Inputs/outputs contract |
| `rationale_field` | `FieldInfo \| None` | `None` | Custom reasoning field |
| `rationale_field_type` | `type` | `str` | Type for the rationale |

Default choice for summarization. Adds a `reasoning` field before the output.

## dspy.Predict

[API docs](https://dspy.ai/api/modules/Predict/)

`dspy.Predict(signature, **config)` — no reasoning step. Use for LM-as-judge steps (`JudgeFaithfulness`, `JudgeCoverage`) where direct scoring is preferred.

## dspy.Refine

[API docs](https://dspy.ai/api/modules/Refine/)

```python
dspy.Refine(module, N, reward_fn, threshold=0.8, fail_count=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `dspy.Module` | required | Module to refine |
| `N` | `int` | required | Max retry attempts |
| `reward_fn` | `Callable[[dict, Prediction], float]` | required | Scores output; 1.0 = perfect |
| `threshold` | `float` | `0.8` | Stop retrying once score >= threshold |

Reward function signature - `reward_fn(args: dict, pred: Prediction) -> float`. `args` holds the module's input keyword arguments. Use `dspy.Refine` (not `dspy.Assert`/`dspy.Suggest`, removed in DSPy 3.x) for all length enforcement.

### Reward function patterns

```python
def length_reward(args, pred):
    word_count = len(pred.summary.split())
    max_words = args["max_words"]
    if word_count <= max_words:
        return 1.0
    return max(0.0, 1.0 - (word_count - max_words) / max_words)

def meeting_reward(args, pred):
    score = 1.0
    if len(pred.output.action_items) == 0:
        score -= 0.2
    if len(pred.output.decisions) == 0:
        score -= 0.2
    return score
```

## Map-reduce chunking (long documents)

```python
class LongDocSummarizer(dspy.Module):
    def __init__(self, chunk_size=2000):
        self.chunk_size = chunk_size
        self.map_step = dspy.ChainOfThought(SummarizeChunk)
        self.reduce_step = dspy.ChainOfThought(CombineSummaries)

    def forward(self, text):
        words = text.split()
        chunks = [" ".join(words[i:i+self.chunk_size]) for i in range(0, len(words), self.chunk_size)]
        summaries = [self.map_step(chunk=c).chunk_summary for c in chunks]
        return self.reduce_step(section_summaries=summaries, original_length=len(words))
```

Wrap `dspy.Refine` around the full module instance when the reward depends on the composed output. Add 50-100 word overlap between chunks when cross-chunk context matters.

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, max_bootstrapped_demos=4, max_labeled_demos=16)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `16` | Max labeled demos from trainset |

Key method - `.compile(module, trainset=...)` - returns the optimized module. Metric signature - `metric(example, prediction, trace=None) -> float`. For summarization, combine faithfulness (0.4), coverage (0.4), and conciseness (0.2) — return `0.0` immediately if faithfulness fails.
