# DSPy Retrieval API Reference

> Condensed from [dspy.ai/api/tools/](https://dspy.ai/api/tools/ColBERTv2/) and [dspy.ai/api/tools/Embeddings/](https://dspy.ai/api/tools/Embeddings/). Verify against upstream for latest.

## dspy.Embeddings (built-in retriever)

Embedding-based similarity search with automatic FAISS indexing for large corpora. No external vector DB needed.

```python
retriever = dspy.Embeddings(corpus=texts, embedder=embedder, k=5)
result = retriever("search query")  # returns Prediction(passages=..., indices=...)
```

### Constructor

```python
dspy.Embeddings(
    corpus: list[str],
    embedder,                       # dspy.Embedder instance or compatible callable
    k: int = 5,
    callbacks: list[Any] | None = None,
    cache: bool = False,
    brute_force_threshold: int = 20_000,  # FAISS above this, brute-force below
    normalize: bool = True,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus` | `list[str]` | required | List of text passages to search over |
| `embedder` | `Embedder` | required | Embedding model to encode texts |
| `k` | `int` | `5` | Number of passages to retrieve |
| `cache` | `bool` | `False` | Cache embeddings to avoid recomputation |
| `brute_force_threshold` | `int` | `20000` | Use FAISS index above this corpus size |
| `normalize` | `bool` | `True` | L2-normalize embeddings before search |

### Key methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `(query: str)` | `dspy.Prediction(passages=..., indices=...)` |
| `save` | `(path: str)` | Persists index and embeddings to disk |
| `load` | `(path: str, embedder)` | Restores from saved state |
| `from_saved` | `@classmethod (path: str, embedder)` | Creates instance without recomputing embeddings |

## dspy.Embedder

Compute embeddings for text using hosted or custom models.

```python
embedder = dspy.Embedder("openai/text-embedding-3-small")
vectors = embedder(["text 1", "text 2"])  # returns np.ndarray
```

### Constructor

```python
dspy.Embedder(
    model: str | Callable,
    batch_size: int = 200,
    caching: bool = True,
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| Callable` | required | Model name (e.g., `"openai/text-embedding-3-small"`) or custom function |
| `batch_size` | `int` | `200` | Batch size for processing inputs |
| `caching` | `bool` | `True` | Cache responses for hosted models |

### Key methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `__call__` | `(inputs: str \| list[str], batch_size=None, caching=None)` | `np.ndarray` (1D for single, 2D for list) |
| `acall` | `(inputs, batch_size=None, caching=None)` | Same, async |

## dspy.ColBERTv2

Neural retriever using ColBERTv2 server.

```python
colbert = dspy.ColBERTv2(url="http://your-server:port/endpoint")
results = colbert("search query", k=10)
```

### Constructor

```python
dspy.ColBERTv2(
    url: str = "http://0.0.0.0",
    port: str | int | None = None,
    post_requests: bool = False,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"http://0.0.0.0"` | ColBERTv2 server URL |
| `port` | `str \| int \| None` | `None` | Server port |
| `post_requests` | `bool` | `False` | Use POST instead of GET |

### Key methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `__call__` | `(query: str, k: int = 10, simplify: bool = False)` | `list[str]` if simplify else `list[dotdict]` |

## dspy.Retrieve (base class)

Base class for custom retrievers. Subclass this to wrap any search backend.

```python
class MyRetriever(dspy.Retrieve):
    def __init__(self, k=3):
        super().__init__(k=k)

    def forward(self, query, k=None):
        k = k or self.k
        results = your_search(query, top_k=k)
        return dspy.Prediction(passages=[r["text"] for r in results])
```

### Constructor

```python
dspy.Retrieve(k: int = 3, callbacks=None)
```

### Key methods

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `(query: str, k: int = None)` | `dspy.Prediction(passages=...)` |
| `dump_state` | `()` | `dict` with `k` value |
| `load_state` | `(state: dict)` | Restores state |

## Experimental: Citations and Document

For citation-enabled RAG with source references (experimental API, may change).

```python
from dspy.experimental import Citations, Document

class CitedAnswer(dspy.Signature):
    """Answer with citations from the provided documents."""
    documents: list[Document] = dspy.InputField()
    question: str = dspy.InputField()
    answer: Citations = dspy.OutputField()
```

`Document(data="text content", title="optional title")` wraps retrieved text for citation-aware models. `Citations` provides structured citation output with `cited_text`, `document_index`, and character positions. Works best with Anthropic models that support native citations.
