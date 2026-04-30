# Retrieval API Reference

> Condensed from [dspy.ai/api](https://dspy.ai/api/). Verify against upstream for latest.

## dspy.Retrieve

```python
dspy.Retrieve(k=3)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `k` | `int` | `3` | Number of passages to retrieve |

**Returns:** `dspy.Prediction` with `.passages: list[str]`

Requires `dspy.configure(rm=...)` or a subclass implementing `forward()`.

## dspy.ColBERTv2

```python
dspy.ColBERTv2(url="http://0.0.0.0", port=None, post_requests=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `url` | `str` | `"http://0.0.0.0"` | ColBERTv2 server endpoint |
| `port` | `str \| int \| None` | `None` | Port to append to URL |
| `post_requests` | `bool` | `False` | Use POST instead of GET |

## dspy.Embedder

```python
dspy.Embedder(model, batch_size=200, caching=True, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| Callable` | required | LiteLLM model string or custom callable |
| `batch_size` | `int` | `200` | Batch size for multi-text embedding |
| `caching` | `bool` | `True` | Cache responses for hosted models |
| `**kwargs` | | | Model-specific args (e.g., `dimensions=512`) |

**Returns:** 1D numpy array (single text) or 2D array (multiple texts).

## dspy.retrievers.Embeddings

```python
dspy.retrievers.Embeddings(corpus, embedder, k=5, brute_force_threshold=20000, normalize=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `corpus` | `list[str]` | required | Documents to index |
| `embedder` | `Embedder` | required | Embedder instance |
| `k` | `int` | `5` | Default number of results |
| `brute_force_threshold` | `int` | `20000` | Corpus size threshold for FAISS indexing |
| `normalize` | `bool` | `True` | Normalize embeddings |

**Key methods:**
- `save(path)` -- persist embeddings to disk
- `Embeddings.from_saved(path, embedder=embedder)` -- load without re-embedding
