---
name: dspy-knn-few-shot
description: Use when you want few-shot demos that are dynamically selected per input based on similarity — better than fixed demos when inputs vary widely. Common scenarios: inputs vary widely and fixed examples don't cover enough cases, dynamically selecting the most relevant demos per input, building a retrieval-augmented prompt with similar examples, or when static few-shot examples work for some inputs but fail on others. Related: ai-improving-accuracy, dspy-labeled-few-shot, dspy-bootstrap-few-shot. Also: dspy.KNNFewShot, dynamic few-shot selection, similar examples per input, retrieval-augmented few-shot, adaptive demonstrations, nearest neighbor example selection, dynamic prompt construction, different examples for different inputs, embedding-based demo retrieval, when fixed examples don't generalize, per-input demo selection, contextual few-shot examples, smart example selection.
---

# Dynamic Few-Shot with dspy.KNN and dspy.KNNFewShot

Guide the user through using DSPy's KNN-based retrieval to dynamically select the most relevant few-shot demonstrations for each input at inference time, rather than using the same static examples for every query.

## What KNN and KNNFewShot are

`dspy.KNN` is an in-memory nearest-neighbor retriever. Given a training set and an embedding function, it converts every training example into a vector. At query time, it embeds the new input, computes dot-product similarity against all stored vectors, and returns the k most similar training examples.

`dspy.KNNFewShot` is an optimizer (teleprompter) that wraps `KNN` and `BootstrapFewShot` together. It compiles a student program so that every forward call first retrieves the k nearest training examples, then uses them as the few-shot demonstrations for the underlying module. The demonstrations change per input -- each query gets the examples most relevant to it.

```
New input ──> Embed ──> Find k nearest training examples ──> Use as demos ──> Run module
```

## When to use

- **Your training examples cover diverse subtasks** and you want the LM to see only the most relevant ones for each input (e.g., a classifier that handles many categories, a QA system across different domains)
- **Static few-shot examples hurt more than they help** because irrelevant demos confuse the model on certain inputs
- **You have enough labeled examples** (at least 20-50) to make similarity-based retrieval meaningful
- **You want the simplicity of few-shot prompting** but with per-query adaptation

Do **not** use KNNFewShot when:
- You have very few training examples (< 10) -- static few-shot or BootstrapFewShot is simpler and sufficient
- All your inputs are nearly identical -- retrieval adds overhead without benefit
- You need optimized instructions, not just better demo selection -- use MIPROv2 instead

## Basic usage with KNNFewShot

```python
import dspy
from sentence_transformers import SentenceTransformer

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# 1. Prepare training data
trainset = [
    dspy.Example(question="What causes rain?", answer="Condensation of water vapor in clouds").with_inputs("question"),
    dspy.Example(question="What is photosynthesis?", answer="The process plants use to convert sunlight into energy").with_inputs("question"),
    # ... more examples
]

# 2. Set up an embedding function
encoder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = dspy.Embedder(encoder.encode)

# 3. Create the optimizer
knn_optimizer = dspy.KNNFewShot(
    k=3,
    trainset=trainset,
    vectorizer=embedder,
)

# 4. Compile your module
qa = dspy.ChainOfThought("question -> answer")
optimized_qa = knn_optimizer.compile(qa)

# 5. Use it -- each call retrieves relevant demos automatically
result = optimized_qa(question="How do volcanoes form?")
print(result.answer)
```

Each call to `optimized_qa` now dynamically selects the 3 training examples most similar to the input question and includes them as few-shot demonstrations in the prompt.

## Using KNN directly

If you only need the retrieval step (without the BootstrapFewShot compilation), use `dspy.KNN` on its own:

```python
import dspy
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = dspy.Embedder(encoder.encode)

trainset = [
    dspy.Example(question="What is gravity?", answer="A fundamental force of attraction between masses").with_inputs("question"),
    dspy.Example(question="What is friction?", answer="A force that opposes the relative motion of surfaces").with_inputs("question"),
    # ... more examples
]

knn = dspy.KNN(
    k=3,
    trainset=trainset,
    vectorizer=embedder,
)

# Retrieve the 3 most similar examples to a new query
similar = knn(question="What is inertia?")
# similar is a list of dspy.Example objects, ranked by similarity
```

This is useful when you want to plug KNN retrieval into a custom module or pipeline.

## Embedding configuration

KNNFewShot and KNN require a `dspy.Embedder` wrapping any function that takes text (or a list of texts) and returns vectors.

### Using sentence-transformers (recommended default)

```python
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = dspy.Embedder(encoder.encode)
```

`all-MiniLM-L6-v2` is fast, small (~80MB), and works well for general-purpose similarity. For domain-specific tasks, consider models from the [MTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard).

### Using OpenAI embeddings

```python
import openai

client = openai.OpenAI()

def openai_embed(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]

embedder = dspy.Embedder(openai_embed)
```

### Using any callable

Any function with the signature `(str | list[str]) -> list[list[float]]` works:

```python
embedder = dspy.Embedder(my_custom_embed_function)
```

## How retrieval works internally

1. **Indexing (at init time)**: KNN concatenates all input fields of each training example into a single string, then calls the embedder to produce a vector per example. These vectors are stored in memory as a matrix.

2. **Querying (at call time)**: The new input's fields are concatenated and embedded the same way. KNN computes dot-product similarity between the query vector and all stored vectors, then returns the k examples with the highest scores.

3. **Demo injection (KNNFewShot only)**: The retrieved examples are set as the `demos` on each `Predict` module inside the compiled student program. This happens on every forward call, so demonstrations adapt per query.

The dot-product similarity means vectors should ideally be normalized (most sentence-transformer models do this by default). If your embedding function does not normalize, cosine similarity and dot-product may diverge.

## Constructor parameters

### dspy.KNN

```python
dspy.KNN(
    k,           # int -- number of nearest neighbors to retrieve
    trainset,    # list[dspy.Example] -- examples to search through
    vectorizer,  # dspy.Embedder -- embedding function wrapper
)
```

### dspy.KNNFewShot

```python
dspy.KNNFewShot(
    k,                        # int -- number of nearest neighbors to retrieve
    trainset,                 # list[dspy.Example] -- examples to search through
    vectorizer,               # dspy.Embedder -- embedding function wrapper
    **few_shot_bootstrap_args # passed to BootstrapFewShot (e.g., metric, max_bootstrapped_demos)
)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `k` | `int` | Number of nearest neighbors to retrieve per query |
| `trainset` | `list[dspy.Example]` | Training examples to index and search |
| `vectorizer` | `dspy.Embedder` | Wraps any embedding function for vectorization |
| `**few_shot_bootstrap_args` | `dict` | Forwarded to `BootstrapFewShot` (e.g., `metric`, `max_bootstrapped_demos`, `max_labeled_demos`) |

### Key method

**`compile(student, *, teacher=None)`**: Returns a copy of the student program whose forward method retrieves k nearest demos per call. Accepts an optional teacher program (passed through to BootstrapFewShot).

## Choosing k

| k | Trade-off |
|---|-----------|
| 1-2 | Minimal prompt overhead. Works when examples are very similar to queries. |
| 3-5 | Good default range. Enough diversity without bloating the prompt. |
| 7-10 | Use with short examples or large context windows. Diminishing returns beyond this. |

Keep in mind that each retrieved demo adds to the prompt length. If your examples are long (multi-paragraph), use a smaller k to stay within context limits.

## Comparison with static few-shot

| | Static few-shot (BootstrapFewShot / LabeledFewShot) | Dynamic few-shot (KNNFewShot) |
|---|---|---|
| **Demo selection** | Same demos for every input | Per-input demos based on similarity |
| **Best when** | Inputs are homogeneous, few examples available | Inputs are diverse, many examples available |
| **Setup cost** | Lower -- no embedding model needed | Higher -- requires an embedder and more training data |
| **Prompt relevance** | May include irrelevant demos for some inputs | Demos are always relevant to the current input |
| **Latency** | No retrieval overhead | Small overhead for embedding + similarity search |
| **Scales with data** | More data doesn't help (fixed demo slots) | More data improves retrieval quality |

## Passing BootstrapFewShot arguments

Since KNNFewShot wraps BootstrapFewShot internally, you can pass any BootstrapFewShot parameter via `**few_shot_bootstrap_args`:

```python
knn_optimizer = dspy.KNNFewShot(
    k=5,
    trainset=trainset,
    vectorizer=embedder,
    metric=my_metric,
    max_bootstrapped_demos=2,
    max_labeled_demos=3,
)
optimized = knn_optimizer.compile(my_program, teacher=teacher_program)
```

This retrieves 5 nearest neighbors per query and then applies BootstrapFewShot logic (with the given metric and demo limits) over those neighbors.

## Cross-references

- **BootstrapFewShot** for static few-shot optimization -- see `/dspy-bootstrap-few-shot`
- **LabeledFewShot** for simple static demo selection without bootstrapping -- see `/dspy-labeled-few-shot`
- **Improving accuracy** for the full optimization workflow -- see `/ai-improving-accuracy`
- For worked examples, see [examples.md](examples.md)
- Not sure which skill to use next? Try `/ai-do` to get routed to the right one
