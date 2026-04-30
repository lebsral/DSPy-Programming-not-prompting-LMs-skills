# Qdrant + DSPy API Reference

> Condensed from [github.com/qdrant/dspy-qdrant](https://github.com/qdrant/dspy-qdrant) and [dspy.ai/api/tools/Embeddings](https://dspy.ai/api/tools/Embeddings/). Verify against upstream for latest.

## QdrantRM

```python
from qdrant_client import QdrantClient
from dspy_qdrant import QdrantRM

retriever = QdrantRM(
    qdrant_collection_name,   # str -- required
    qdrant_client,            # QdrantClient -- required
    k=3,                      # int -- top passages to retrieve
    document_field="document",# str -- payload field with document text
    vectorizer=None,          # BaseSentenceVectorizer -- default: FastEmbedVectorizer
    vector_name=None,         # str -- named vector to search (default: first available)
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `qdrant_collection_name` | `str` | required | Name of the Qdrant collection |
| `qdrant_client` | `QdrantClient` | required | An initialized `qdrant_client.QdrantClient` instance |
| `k` | `int` | `3` | Number of top passages to retrieve per query |
| `document_field` | `str` | `"document"` | Payload field containing the document text |
| `vectorizer` | `BaseSentenceVectorizer` | `FastEmbedVectorizer` | Embedding model for vectorizing queries. Uses `BAAI/bge-small-en-v1.5` via FastEmbed by default |
| `vector_name` | `str \| None` | `None` | Named vector in collection to search. Defaults to first available |

### Key Methods

- `forward(query_or_queries, k=None, filter=None)` — search Qdrant for top-k passages. Accepts a single query string or list of strings. Optional `filter` parameter takes a `qdrant_client.models.Filter` for payload filtering.
- Inherits `__call__` from `dspy.Retrieve` — calls `forward()` with callbacks.

### Return Value

Returns a list of `dotdict({"long_text": passage})` objects. When used via `dspy.Retrieve()` after `dspy.configure(rm=retriever)`, these are automatically converted to `dspy.Prediction(passages=[...])`.

## dspy.Retrieve

```python
dspy.Retrieve(k=3)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `3` | Number of top passages to retrieve |

Thin proxy that delegates to whatever retriever is configured via `dspy.configure(rm=retriever)`. Call `search = dspy.Retrieve(k=5); result = search(query)` to get `dspy.Prediction(passages=[...])`.

## dspy.Embeddings

```python
dspy.Embeddings(
    corpus,                    # list[str] -- required, passages to search
    embedder,                  # Callable -- required, embedding function
    k=5,                       # int -- top results to return
    brute_force_threshold=20000,  # int -- FAISS activates above this
    normalize=True,            # bool -- normalize embeddings
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus` | `list[str]` | required | Text passages to index and search |
| `embedder` | `Callable` | required | Function that converts text to embeddings (e.g., `dspy.Embedder(...)`) |
| `k` | `int` | `5` | Number of top results |
| `brute_force_threshold` | `int` | `20000` | Corpus size above which FAISS indexing activates |
| `normalize` | `bool` | `True` | Whether to normalize embeddings |

### Key Methods

- `forward(query)` — returns `dspy.Prediction(passages=[...], indices=[...])`
- `save(path)` — persist embeddings and FAISS index
- `Embeddings.from_saved(path, embedder)` — load without recomputing

Use `dspy.Embeddings` for simple in-memory retrieval without a vector DB. Use QdrantRM when you need persistence, filtering, hybrid search, or scale.

## dspy.Embedder

```python
dspy.Embedder(
    model,           # str | Callable -- required (e.g., "openai/text-embedding-3-small")
    batch_size=200,  # int
    caching=True,    # bool
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| Callable` | required | LiteLLM model string or custom callable |
| `batch_size` | `int` | `200` | Batch size for processing |
| `caching` | `bool` | `True` | Cache responses from hosted models |

### Key Methods

- `__call__(inputs)` — compute embeddings. Returns 1D `np.ndarray` for single string, 2D for list of strings.
- `acall(inputs)` — async variant.

## BaseSentenceVectorizer

Base class for custom vectorizers to use with QdrantRM:

```python
from dspy_qdrant import BaseSentenceVectorizer
import numpy as np

class MyVectorizer(BaseSentenceVectorizer):
    def __call__(self, inp_examples) -> np.ndarray:
        texts = self._extract_text_from_examples(inp_examples)
        # Return embeddings as numpy array
        return np.array([your_embed_fn(t) for t in texts])
```

## FastEmbedVectorizer

Default vectorizer used by QdrantRM when no `vectorizer` is provided:

```python
from dspy_qdrant import FastEmbedVectorizer

vectorizer = FastEmbedVectorizer(
    model_name="BAAI/bge-small-en-v1.5",  # default
    batch_size=256,
)
```

Requires `pip install fastembed`.
