> Condensed from [dspy.ai/api/optimizers/KNN/](https://dspy.ai/api/optimizers/KNN/) and [dspy.ai/api/optimizers/KNNFewShot/](https://dspy.ai/api/optimizers/KNNFewShot/). Verify against upstream for latest.

# dspy.KNN and dspy.KNNFewShot — API Reference

## dspy.KNN

In-memory nearest-neighbor retriever over training examples.

### Constructor

```python
dspy.KNN(
    k: int,
    trainset: list[Example],
    vectorizer: Embedder,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | required | Number of nearest neighbors to retrieve |
| `trainset` | `list[Example]` | required | Training examples to index and search |
| `vectorizer` | `Embedder` | required | Embedding function wrapper for vectorization |

### Methods

#### `__call__(**kwargs) -> list[Example]`

Retrieves the k nearest training examples for the given input.

```python
similar = knn(question="What is inertia?")
# Returns list of k Example objects, ranked by descending similarity
```

**How it works internally:**
1. Concatenates all input fields of each training example with `" | "` delimiter
2. Pre-computes embeddings for all training examples as `float32` vectors at init time
3. At query time, embeds the input kwargs the same way
4. Computes dot-product similarity between query vector and all stored vectors
5. Returns the k examples with highest similarity scores

## dspy.KNNFewShot

Optimizer that wraps KNN + BootstrapFewShot for dynamic per-input demo selection.

**Inherits from:** `Teleprompter`

### Constructor

```python
dspy.KNNFewShot(
    k: int,
    trainset: list[Example],
    vectorizer: Embedder,
    **few_shot_bootstrap_args,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | required | Number of nearest neighbors to retrieve per query |
| `trainset` | `list[Example]` | required | Training examples to index and search |
| `vectorizer` | `Embedder` | required | Embedding function wrapper |
| `**few_shot_bootstrap_args` | `dict` | `{}` | Forwarded to `BootstrapFewShot` (e.g., `metric`, `max_bootstrapped_demos`, `max_labeled_demos`, `max_rounds`) |

### Methods

#### `compile(student, *, teacher=None)`

Returns a compiled copy of the student program whose forward method retrieves k nearest demos per call.

```python
optimized = knn_optimizer.compile(qa)
# or with a teacher:
optimized = knn_optimizer.compile(qa, teacher=teacher_program)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `student` | `dspy.Module` | required | The program to optimize |
| `teacher` | `dspy.Module \| None` | `None` | Optional teacher for bootstrapping |

**Returns:** Compiled `dspy.Module` with dynamic demo retrieval on every forward call.

#### `get_params() -> dict[str, Any]`

Returns the optimizer's parameters as a dictionary.

## dspy.Embedder

Wraps any embedding function for use with KNN.

```python
dspy.Embedder(embed_function)
```

The `embed_function` must accept `str | list[str]` and return `list[list[float]]`.

### Common configurations

```python
# sentence-transformers (local, free)
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = dspy.Embedder(encoder.encode)

# OpenAI embeddings (API, paid)
import openai
client = openai.OpenAI()
def openai_embed(texts):
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [item.embedding for item in response.data]
embedder = dspy.Embedder(openai_embed)

# Ollama embeddings (local, free)
embedder = dspy.Embedder("ollama/nomic-embed-text", api_base="http://localhost:11434", api_key="")
```

## Similarity computation

KNN uses **dot-product similarity** between the query vector and all stored vectors. This means:

- Normalized vectors (most sentence-transformer models): dot-product = cosine similarity
- Unnormalized vectors: results may be biased toward longer vectors

If using a custom embedding function, ensure it normalizes output or use `normalize_embeddings=True` in sentence-transformers.
