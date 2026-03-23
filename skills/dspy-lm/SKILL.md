---
name: dspy-lm
description: "Use when you need to configure which language model DSPy uses — setting up providers, API keys, model parameters, or assigning different models to different pipeline stages. Common scenarios: setting up OpenAI or Anthropic API keys, configuring model parameters like temperature and max_tokens, using different models for different pipeline stages, switching between providers, using local models with Ollama or vLLM, or setting up Azure OpenAI. Related: ai-switching-models, ai-cutting-costs, ai-kickoff. Also: "dspy.LM", "dspy.configure", "configure language model in DSPy", "OpenAI API key setup DSPy", "Anthropic Claude with DSPy", "use Ollama with DSPy", "local model DSPy", "Azure OpenAI DSPy setup", "model temperature and max_tokens", "different models per module", "multi-model DSPy pipeline", "vLLM with DSPy", "change provider without changing code", "model configuration DSPy"."
---

# Configure Language Models with dspy.LM

`dspy.LM` is DSPy's unified interface for calling language models. It wraps [LiteLLM](https://docs.litellm.ai/docs/providers) so any provider -- OpenAI, Anthropic, Google, Together AI, Ollama, vLLM, and 100+ others -- works through one consistent API. You configure a model once, then every DSPy module uses it automatically.

## Basic setup

```python
import dspy

# Create an LM instance with a provider/model string
lm = dspy.LM("openai/gpt-4o-mini")

# Set it as the default for all DSPy modules
dspy.configure(lm=lm)

# Now any module uses this LM automatically
classify = dspy.ChainOfThought("text -> label")
result = classify(text="DSPy is great")
print(result.label)
```

The pattern is always: `dspy.LM("provider/model")` then `dspy.configure(lm=lm)`.

## Provider strings

DSPy uses the LiteLLM `"provider/model-name"` format. Here are the most common providers:

| Provider | Example string | Notes |
|----------|---------------|-------|
| OpenAI | `"openai/gpt-4o"` | Default provider, auto-detected |
| OpenAI | `"openai/gpt-4o-mini"` | Cheaper, faster |
| Anthropic | `"anthropic/claude-sonnet-4-5-20250929"` | |
| Anthropic | `"anthropic/claude-haiku-4-5-20251001"` | Fast and cheap |
| Google | `"gemini/gemini-2.0-flash"` | |
| Together AI | `"together_ai/meta-llama/Llama-3-70b-chat-hf"` | Open-source models |
| Groq | `"groq/llama-3.1-70b-versatile"` | Fast inference |
| Ollama (local) | `"ollama_chat/llama3.1"` | Requires `api_base` |
| Azure OpenAI | `"azure/my-gpt4-deployment"` | Requires `api_base` + `api_key` |
| OpenAI-compatible | `"openai/my-model"` | Any server with `api_base` |

See [LiteLLM provider docs](https://docs.litellm.ai/docs/providers) for the full list.

## Constructor parameters

```python
lm = dspy.LM(
    model="openai/gpt-4o",           # Required: "provider/model-name"
    model_type="chat",                # "chat" (default), "text", or "responses"
    temperature=0.7,                  # Sampling temperature (default: provider default)
    max_tokens=1000,                  # Max output tokens (default: provider default)
    cache=True,                       # Enable built-in caching (default: True)
    num_retries=3,                    # Retry on transient failures (default: 3)
    use_developer_role=False,         # Use developer/system role (default: False)
    # Plus any extra kwargs passed to LiteLLM
)
```

### Key parameters

- **`model`** (required) -- The provider/model string. This is the only required argument.
- **`temperature`** -- Controls randomness. Lower = more deterministic. Set to `0.0` for reproducible outputs. Reasoning models (o1, o3) require `temperature=1.0` or `None`.
- **`max_tokens`** -- Maximum tokens in the response. Reasoning models require `max_tokens >= 16000` or `None`.
- **`cache`** -- When `True` (the default), DSPy caches LM responses to reduce costs and speed up repeated calls. Set to `False` to disable.
- **`num_retries`** -- Number of retries with exponential backoff on transient failures.
- **`model_type`** -- Usually leave as `"chat"`. Use `"text"` for completion-only models. Use `"responses"` for OpenAI responses API.

## Per-module LM assignment

You don't have to use the same model for every step. Assign different LMs to different modules with `set_lm()`:

```python
expensive_lm = dspy.LM("openai/gpt-4o")
cheap_lm = dspy.LM("openai/gpt-4o-mini")

# Set a default
dspy.configure(lm=cheap_lm)

class MyPipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> category")
        self.generate = dspy.ChainOfThought("text, category -> summary")

    def forward(self, text):
        category = self.classify(text=text)
        return self.generate(text=text, category=category.category)

pipeline = MyPipeline()

# Route: cheap model for classification, expensive for generation
pipeline.classify.set_lm(cheap_lm)
pipeline.generate.set_lm(expensive_lm)
```

### Temporary LM override with `dspy.context`

Use `dspy.context` to temporarily switch LMs for a block of code:

```python
with dspy.context(lm=expensive_lm):
    # Everything inside uses expensive_lm
    result = pipeline(text="important document")

# Back to the default LM outside the block
```

## Direct LM calls

You can call an `LM` instance directly for one-off prompts outside of DSPy modules:

```python
lm = dspy.LM("openai/gpt-4o-mini")

# Pass a string prompt
response = lm("What is the capital of France?")
print(response)  # returns a list of strings

# Pass a messages list (chat format)
response = lm(messages=[
    {"role": "user", "content": "What is the capital of France?"}
])
print(response)  # returns a list of strings
```

Direct calls are useful for quick tests, but for structured tasks use DSPy modules and signatures -- they give you type checking, optimization, and caching.

## Environment variables

Set API keys as environment variables. Never hardcode them.

```bash
# OpenAI
export OPENAI_API_KEY=sk-...

# Anthropic
export ANTHROPIC_API_KEY=sk-ant-...

# Together AI
export TOGETHER_API_KEY=...

# Google
export GEMINI_API_KEY=...

# Groq
export GROQ_API_KEY=...

# Azure OpenAI
export AZURE_API_KEY=...
export AZURE_API_BASE=https://your-resource.openai.azure.com/
```

DSPy (via LiteLLM) reads these automatically. You can also pass `api_key` directly to `dspy.LM()` if needed, but environment variables are preferred.

## Caching

DSPy caches LM responses by default. This means:

- **Repeated identical calls are free** -- same prompt, same parameters, same model returns a cached result instantly with no API call.
- **Development is faster** -- re-running your script doesn't re-call the LM for already-seen inputs.
- **Optimization is cheaper** -- optimizers that re-evaluate examples benefit from cached results.

### Controlling caching

```python
# Caching enabled (default)
lm = dspy.LM("openai/gpt-4o-mini", cache=True)

# Disable caching for this LM
lm = dspy.LM("openai/gpt-4o-mini", cache=False)

# Configure cache settings globally
dspy.configure_cache(
    enable=True,          # Toggle caching on/off
)
```

Cache is stored locally. If you need different responses for the same prompt (e.g., generating diverse examples), disable caching or use different `temperature` values.

## Useful methods

| Method | Purpose |
|--------|---------|
| `lm("prompt")` | Direct call -- returns list of strings |
| `lm.copy(**kwargs)` | Deep copy with updated parameters |
| `lm.inspect_history()` | View recent request/response history |
| `lm.dump_state()` | Serialize config (excludes API keys) |

### Inspecting history

```python
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

classify = dspy.Predict("text -> label")
classify(text="Hello world")

# See what was sent to the LM
dspy.inspect_history(n=1)
```

## Connecting to local models

### Ollama

```python
# Start Ollama: ollama serve
# Pull a model: ollama pull llama3.1
lm = dspy.LM(
    "ollama_chat/llama3.1",
    api_base="http://localhost:11434",
    temperature=0.7,
)
dspy.configure(lm=lm)
```

### vLLM or any OpenAI-compatible server

```python
# Start vLLM: python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8b-chat-hf
lm = dspy.LM(
    "openai/meta-llama/Llama-3-8b-chat-hf",
    api_base="http://localhost:8000/v1",
    api_key="none",
)
dspy.configure(lm=lm)
```

For any server that exposes an OpenAI-compatible `/v1/chat/completions` endpoint, use the `"openai/model-name"` provider string with `api_base` pointing to your server.

## Cross-references

- **`/dspy-signatures`** -- Define what your LM should do (inputs, outputs, types)
- **`/dspy-modules`** -- Wrap signatures with inference strategies (Predict, ChainOfThought, ReAct)
- **`/ai-switching-models`** -- Safely migrate between providers with re-optimization
- **`/ai-cutting-costs`** -- Reduce LM costs with per-module assignment and cheaper models
