> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Recommending

## Quick-reference config

```bash
pip install -U dspy numpy
```

```python
import dspy
lm = dspy.LM("openai/gpt-4o-mini")  # or "anthropic/claude-sonnet-4-5-20250929", etc.
dspy.configure(lm=lm)
```

## Re-ranking signature

Ranked outputs require parallel `list[str]` output fields. The docstring must say "Reorder... placing the most relevant first" and "Do not reveal internal scoring" — without these, the LM echoes items in their original order and leaks score values into explanations.

```python
class ReRankRecommendations(dspy.Signature):
    """Reorder the candidate items below, placing the most relevant first for this user.
    Do not reveal internal scoring. Write friendly explanations referencing why each item fits the user."""

    user_profile: str      = dspy.InputField(desc="Anonymized preference summary — interests, activity, categories")
    candidate_items: str   = dspy.InputField(desc="Numbered list: index. [ID] Title - Description")
    num_results: int       = dspy.InputField(desc="Number of top items to return")
    ranked_items: list[str]  = dspy.OutputField(desc="Item IDs in ranked order, most relevant first")
    explanations: list[str]  = dspy.OutputField(desc="One friendly sentence per item")
```

### dspy.InputField / dspy.OutputField

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `desc` | `str` | `""` | Field description injected into the LM prompt |
| `prefix` | `str` | `None` | Custom label replacing the field name in the prompt |

## dspy.Predict vs dspy.ChainOfThought

[Predict API docs](https://dspy.ai/api/modules/Predict/) | [ChainOfThought API docs](https://dspy.ai/api/modules/ChainOfThought/)

| Module | Adds reasoning | Latency | Use when |
|--------|---------------|---------|----------|
| `dspy.Predict(sig)` | No | Lower | Latency is tight; ranking logic is straightforward |
| `dspy.ChainOfThought(sig)` | Yes (`reasoning` field auto-injected) | Higher | Explanation quality matters more than speed |

Do not add a `reasoning` field to the signature — `dspy.ChainOfThought` injects it automatically.

## Candidate retrieval (numpy cosine similarity)

The two-stage pipeline retrieves candidates by cosine similarity before passing them to the LM.

```python
import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

candidate_ids = sorted(
    item_embeddings,                                          # {item_id: np.ndarray}
    key=lambda iid: cosine_sim(user_embedding, item_embeddings[iid]),
    reverse=True,
)[:top_n]                                                     # typically 20–50
```

Precompute item embeddings at catalog ingestion — do not re-embed on every request. Pass at most 20–50 candidates to the LM re-ranker; larger sets exceed context windows and reduce re-ranking quality.

For DSPy-integrated retrieval backends (ColBERTv2, etc.), use `dspy.Retrieve(k=n)` instead — `self.retrieve(query).passages` returns the top-k passages directly.

## dspy.Refine and dspy.BestOfN

[Refine API docs](https://dspy.ai/api/modules/Refine/) | [BestOfN API docs](https://dspy.ai/api/modules/BestOfN/)

Use these for re-ranking quality control. `dspy.Assert` and `dspy.Suggest` were removed in DSPy 3.x.

```python
def ranking_reward(args, pred) -> float:
    if not pred.ranked_items:
        return 0.0
    if len(pred.explanations) != len(pred.ranked_items):
        return 0.5
    return 1.0

reranker = dspy.Predict(ReRankRecommendations)
refined   = dspy.Refine(module=reranker, N=3, reward_fn=ranking_reward, threshold=0.8)
best      = dspy.BestOfN(module=reranker, N=5, reward_fn=ranking_reward, threshold=0.8)
```

| Module | Strategy | When to use |
|--------|----------|-------------|
| `dspy.Refine` | Iterative — passes feedback to LM on failure | Re-ranking quality loops |
| `dspy.BestOfN` | N independent samples, returns highest-scoring | High-stakes recommendation slots |

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | Scoring function — use precision@k for recommendations |
| `max_bootstrapped_demos` | `int` | `4` | Max generated few-shot demos |
| `max_labeled_demos` | `int` | `16` | Max labeled demos pulled from trainset |
| `max_rounds` | `int` | `1` | Bootstrap iterations |

Key method: `.compile(module, trainset=...)` — returns an optimized module. Save with `.save("path.json")`.

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, failure_score=0.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function |
| `num_threads` | `int \| None` | `None` | Parallel evaluation threads |
| `display_progress` | `bool` | `False` | Show progress bar |
| `display_table` | `bool \| int` | `False` | Show results table (int = row count) |

Call the evaluator with a module: `score = evaluator(pipeline)`.

Precision@k is the standard offline metric for recommendations:

```python
def precision_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int) -> float:
    return len([iid for iid in recommended_ids[:k] if iid in relevant_ids]) / k
```
