> Condensed from [dspy.ai/api/models/LM/](https://dspy.ai/api/models/LM/) and [dspy.ai/api/models/Embedder/](https://dspy.ai/api/models/Embedder/). Verify against upstream for latest.

# dspy.LM and dspy.Embedder — API Reference (Ollama focus)

## dspy.LM

### Constructor

```python
dspy.LM(
    model: str,                          # e.g., "ollama_chat/llama3.1"
    model_type: Literal['chat', 'text', 'responses'] = 'chat',
    temperature: float | None = None,
    max_tokens: int | None = None,
    cache: bool = True,
    callbacks: list[BaseCallback] | None = None,
    num_retries: int = 3,
    provider: Provider | None = None,
    finetuning_model: str | None = None,
    launch_kwargs: dict[str, Any] | None = None,
    train_kwargs: dict[str, Any] | None = None,
    use_developer_role: bool = False,
    **kwargs,                            # Provider-specific args like num_ctx
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Model identifier. For Ollama: `"ollama_chat/<model_name>"` |
| `model_type` | `str` | `'chat'` | `'chat'`, `'text'`, or `'responses'` |
| `temperature` | `float \| None` | `None` | Sampling temperature |
| `max_tokens` | `int \| None` | `None` | Max output tokens |
| `cache` | `bool` | `True` | Enable response caching |
| `num_retries` | `int` | `3` | Retry count on failures |
| `**kwargs` | `dict` | `{}` | Passed to LiteLLM. For Ollama: `api_base`, `api_key`, `num_ctx` |

#### Ollama-specific kwargs

| Parameter | Required | Description |
|-----------|----------|-------------|
| `api_base` | Yes | Ollama server URL, e.g. `"http://localhost:11434"` |
| `api_key` | Yes | Must be `""` (empty string). LiteLLM requires it, Ollama ignores it |
| `num_ctx` | Recommended | Context window size. Ollama defaults to 4096 regardless of model capacity |

### Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(prompt=None, messages=None, **kwargs) -> list` | Send a prompt or messages |
| `acall` | `async (prompt=None, messages=None, **kwargs) -> list` | Async version |
| `inspect_history` | `(n=1, file=None) -> None` | Print last n LM interactions |
| `copy` | `(**kwargs) -> LM` | Copy with overrides |

### Module-level methods (on dspy.Predict, dspy.Module, etc.)

| Method | Description |
|--------|-------------|
| `set_lm(lm)` | Assign a specific LM to a predictor (for multi-model pipelines) |
| `save(path)` | Save optimized program to JSON |
| `load(path)` | Load a saved program |

## dspy.Embedder

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
| `model` | `str \| Callable` | required | Model name or custom function. For Ollama: `"ollama/nomic-embed-text"` |
| `batch_size` | `int` | `200` | Batch processing size |
| `caching` | `bool` | `True` | Enable response caching for hosted models |
| `**kwargs` | `dict` | `{}` | Additional args. For Ollama: `api_base`, `api_key` |

### Ollama embedding usage

```python
embedder = dspy.Embedder(
    "ollama/nomic-embed-text",
    api_base="http://localhost:11434",
    api_key="",
)

# Single string
vector = embedder("What is DSPy?")   # returns 1D numpy array

# List of strings
vectors = embedder(["query 1", "query 2"])  # returns 2D numpy array
```

### Key methods

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(inputs, batch_size=None, caching=None, **kwargs) -> ndarray` | Compute embeddings |
| `acall` | `async (inputs, batch_size=None, caching=None, **kwargs) -> ndarray` | Async version |
