> Condensed from [dspy.ai/api/models/LM/](https://dspy.ai/api/models/LM/). Verify against upstream for latest.

# dspy.LM — API Reference

## Constructor

```python
dspy.LM(
    model,
    model_type="chat",
    temperature=None,
    max_tokens=None,
    cache=True,
    callbacks=None,
    num_retries=3,
    provider=None,
    finetuning_model=None,
    launch_kwargs=None,
    train_kwargs=None,
    use_developer_role=False,
    **kwargs,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Provider/model string, e.g. `"openai/gpt-4o-mini"` (LiteLLM format) |
| `model_type` | `Literal["chat", "text", "responses"]` | `"chat"` | `"chat"` for conversation endpoints, `"text"` for completion-only, `"responses"` for OpenAI responses API |
| `temperature` | `float \| None` | `None` | Sampling temperature. Reasoning models (o1, o3, o4, gpt-5) require `1.0` or `None` |
| `max_tokens` | `int \| None` | `None` | Max output tokens. Reasoning models require `>= 16000` or `None` |
| `cache` | `bool` | `True` | Enable built-in response caching |
| `callbacks` | `list[BaseCallback] \| None` | `None` | Callback handlers for instrumentation |
| `num_retries` | `int` | `3` | Retries with exponential backoff on transient failures |
| `provider` | `Provider \| None` | `None` | Provider implementation (auto-inferred from model string) |
| `finetuning_model` | `str \| None` | `None` | Model ID for fine-tuning jobs |
| `launch_kwargs` | `dict \| None` | `None` | Arguments for launching model servers |
| `train_kwargs` | `dict \| None` | `None` | Arguments for training/fine-tuning |
| `use_developer_role` | `bool` | `False` | Use developer/system role in messages |
| `**kwargs` | | | Extra arguments passed to LiteLLM (e.g., `api_base`, `api_key`, `num_ctx`) |

## Key methods

### Calling the LM

| Method | Signature | Description |
|--------|-----------|-------------|
| `__call__` | `(prompt=None, messages=None, **kwargs) -> list[str]` | Call with callbacks and usage tracking |
| `forward` | `(prompt=None, messages=None, **kwargs) -> list[str]` | Direct synchronous completion |
| `acall` | `async (prompt=None, messages=None, **kwargs) -> list[str]` | Async call with callbacks |
| `aforward` | `async (prompt=None, messages=None, **kwargs) -> list[str]` | Async direct completion |

Both `prompt` (string) and `messages` (list of dicts with `role`/`content`) formats are supported. Returns a list of strings.

### Configuration and state

| Method | Signature | Description |
|--------|-----------|-------------|
| `copy` | `(**kwargs) -> LM` | Shallow copy with updated parameters |
| `dump_state` | `() -> dict` | Serialize config (excludes API keys) |
| `inspect_history` | `(n=1, file=None) -> None` | Print last n LM interactions |

### Fine-tuning

| Method | Signature | Description |
|--------|-----------|-------------|
| `finetune` | `(train_data, train_data_format, train_kwargs) -> str` | Provider-specific fine-tuning |
| `reinforce` | `(train_kwargs) -> str` | Reinforcement learning job |

## Global configuration

```python
# Set the default LM for all modules
dspy.configure(lm=lm)

# Temporary LM override
with dspy.context(lm=other_lm):
    result = module(...)

# Configure caching
dspy.configure_cache(enable=True)

# View recent LM history
dspy.inspect_history(n=1)
```

## Provider string format

The model string follows LiteLLM's `"provider/model-name"` convention:

| Provider | Format | Example |
|----------|--------|---------|
| OpenAI | `"openai/model"` | `"openai/gpt-4o-mini"` |
| Anthropic | `"anthropic/model"` | `"anthropic/claude-sonnet-4-5-20250929"` |
| Google | `"gemini/model"` | `"gemini/gemini-2.0-flash"` |
| Together AI | `"together_ai/model"` | `"together_ai/meta-llama/Llama-3-70b-chat-hf"` |
| Groq | `"groq/model"` | `"groq/llama-3.1-70b-versatile"` |
| Ollama | `"ollama_chat/model"` | `"ollama_chat/llama3.1"` (requires `api_base`) |
| Azure | `"azure/deployment"` | `"azure/my-gpt4-deployment"` (requires `api_base` + `api_key`) |
| OpenAI-compatible | `"openai/model"` | Any server with `api_base` |

## Key behaviors

- **Caching**: Enabled by default. Same prompt + same params + same model returns cached result with no API call. Use `rollout_id` parameter to bypass cache for stochastic sampling without disabling future caching.
- **History**: Auto-tracked unless `settings.disable_history` is set. Accessible via `inspect_history()`.
- **API keys**: Read from environment variables automatically via LiteLLM. Never serialized in `dump_state()`.
