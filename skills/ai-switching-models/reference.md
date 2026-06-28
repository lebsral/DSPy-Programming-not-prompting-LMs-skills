> Condensed from [dspy.ai/api/models/LM/](https://dspy.ai/api/models/LM/) and [LiteLLM provider docs](https://docs.litellm.ai/docs/providers). Verify against upstream for latest.

# DSPy API Reference for Switching Models

## Provider string quick-reference

DSPy uses the LiteLLM `"provider/model-name"` format. All providers are interchangeable — swap the string, nothing else changes.

| Provider | Example string | Notes |
|----------|---------------|-------|
| OpenAI | `"openai/gpt-4o"` | |
| OpenAI | `"openai/gpt-4o-mini"` | Cheaper, faster |
| Anthropic | `"anthropic/claude-sonnet-4-5-20250929"` | |
| Anthropic | `"anthropic/claude-haiku-4-5-20251001"` | Fast and cheap |
| Google | `"gemini/gemini-2.0-flash"` | |
| Together AI | `"together_ai/meta-llama/Llama-3-70b-chat-hf"` | Open-source models |
| Groq | `"groq/llama-3.1-70b-versatile"` | Fast inference |
| Ollama (local) | `"ollama_chat/llama3.1"` | Requires `api_base` |
| Azure OpenAI | `"azure/my-deployment-name"` | Requires `api_base` + `api_key` |
| OpenAI-compatible | `"openai/my-model"` | Any vLLM/TGI server via `api_base` |

## dspy.LM

[API docs](https://dspy.ai/api/models/LM/)

```python
dspy.LM(model, model_type='chat', temperature=None, max_tokens=None, cache=True,
        num_retries=3, callbacks=None, provider=None, **kwargs)
# Pass api_base and api_key via **kwargs (forwarded to LiteLLM)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | `"provider/model-name"` string (LiteLLM format) |
| `model_type` | `'chat' \| 'text' \| 'responses'` | `'chat'` | Interaction type. Use `'responses'` for OpenAI Responses API models |
| `temperature` | `float \| None` | `None` | Randomness. `0.0` = deterministic. Reasoning models require `1.0` or `None` |
| `max_tokens` | `int \| None` | `None` | Max output tokens. Reasoning models require `>= 16000` or `None` |
| `cache` | `bool` | `True` | Cache responses — set `False` to disable |
| `num_retries` | `int` | `3` | Retries with exponential backoff on transient failures |
| `callbacks` | `list[BaseCallback] \| None` | `None` | Pre/post-request callback hooks (for tracing, logging) |
| `api_base` | `str` (via `**kwargs`) | — | Override API endpoint (local servers, Azure, vLLM) |
| `api_key` | `str` (via `**kwargs`) | — | Override API key (use env vars instead when possible) |

## dspy.configure

```python
dspy.configure(lm=lm)
```

Sets the global default LM for all DSPy modules. Call once at startup. Every `dspy.Predict`, `dspy.ChainOfThought`, etc. call uses this LM unless overridden by `dspy.context` or `set_lm`.

## dspy.context

Temporarily overrides the global LM for the duration of a `with` block. Does not persist after the block exits. Use for per-call overrides and to pin the judge LM in evaluation metrics.

```python
with dspy.context(lm=cheap_lm):
    result = my_module(text="...")  # uses cheap_lm only inside this block
```

## module.set_lm

Permanently assigns a specific LM to one module instance. Persists through optimization. Use when a module should always use a particular model regardless of the global config.

```python
pipeline.classify.set_lm(cheap_lm)
pipeline.generate.set_lm(expensive_lm)
```

Use `set_lm` (not `dspy.context`) when the assignment must survive optimizer `.compile()` calls.

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | `None` | `(example, prediction, trace) -> float \| bool` |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos per predictor |
| `max_labeled_demos` | `int` | `16` | Max labeled demos from trainset |

Key method: `.compile(module, trainset=trainset)` — always pass a fresh (uncompiled) program.

## dspy.MIPROv2

[API docs](https://dspy.ai/api/optimizers/MIPROv2/)

```python
dspy.MIPROv2(metric, auto="light", max_bootstrapped_demos=4, max_labeled_demos=4,
             num_candidates=None, num_threads=None, seed=9, verbose=False)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `metric` | `Callable` | required | Scoring function |
| `auto` | `"light" \| "medium" \| "heavy" \| None` | `"light"` | Optimization intensity; use `"medium"` or `"heavy"` for smaller/local models |
| `max_bootstrapped_demos` | `int` | `4` | Max generated demos |
| `max_labeled_demos` | `int` | `4` | Max labeled demos |

Key method: `.compile(module, trainset=trainset)` — returns an optimized module.

## dspy.Evaluate

[API docs](https://dspy.ai/api/evaluation/Evaluate/)

```python
dspy.Evaluate(devset, metric=None, num_threads=None, display_progress=False,
              display_table=False, max_errors=None, failure_score=0.0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `devset` | `list[Example]` | required | Held-out evaluation examples |
| `metric` | `Callable \| None` | `None` | Scoring function |
| `num_threads` | `int \| None` | `None` | Parallel threads |
| `display_table` | `bool \| int` | `False` | Show results table (int = row count) |

Call the evaluator with a module: `score = evaluator(module)`. Returns a float.

## Program save / load

```python
optimized.save("optimized_gpt4o.json")   # save compiled prompts to disk

program = MyProgram()
program.load("optimized_gpt4o.json")     # load back — instantiate first, then load
```

Compiled prompts are model-specific. Save a separate `.json` per model.

## Environment variables

Set API keys as env vars; do not hardcode them in `dspy.LM(api_key=...)`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
TOGETHER_API_KEY=...
AZURE_API_KEY=...
AZURE_API_BASE=https://your-resource.openai.azure.com/
```
