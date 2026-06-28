> Condensed from [dspy.ai/api/](https://dspy.ai/api/). Verify against upstream for latest.

# DSPy API Reference for Cutting Costs

## dspy.LM

[API docs](https://dspy.ai/api/models/LM/)

```python
dspy.LM(model, model_type="chat", temperature=None, max_tokens=None,
        cache=True, num_retries=3, use_developer_role=False, **kwargs)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str` | required | Provider/model string, e.g. `"openai/gpt-4o-mini"` |
| `temperature` | `float \| None` | `None` | Randomness. Set `0.0` for deterministic caching. Reasoning models require `1.0` or `None`. |
| `max_tokens` | `int \| None` | `None` | Max output tokens. Reasoning models require `>= 16000` or `None`. |
| `cache` | `bool` | `True` | Cache responses. Same prompt + params = instant free replay. |
| `num_retries` | `int` | `3` | Retries with exponential backoff on transient failures. |
| `model_type` | `str` | `"chat"` | `"chat"`, `"text"`, or `"responses"`. Leave as `"chat"` for all modern models. |

Uses [LiteLLM](https://docs.litellm.ai/docs/providers) under the hood — any supported provider works with the `"provider/model"` string.

Common cheap model strings:

| Provider | Cheap model | Notes |
|----------|------------|-------|
| OpenAI | `"openai/gpt-4o-mini"` | ~30x cheaper than gpt-4o |
| Anthropic | `"anthropic/claude-haiku-4-5-20251001"` | Fastest Anthropic model |
| Google | `"gemini/gemini-2.0-flash"` | Low cost, high throughput |
| Together AI | `"together_ai/meta-llama/Llama-3-70b-chat-hf"` | Open-source option |
| Ollama (local) | `"ollama_chat/llama3.1"` | Free at inference time |

## dspy.configure / dspy.context / set_lm

```python
dspy.configure(lm=lm)                  # Set global default LM
dspy.configure_cache(enable_disk_cache=True, enable_memory_cache=True)  # Toggle caching globally

with dspy.context(lm=other_lm):        # Temporary override for a block
    result = module(input=...)

module.step.set_lm(other_lm)           # Permanent assignment to one module step
```

| Method | Scope | Survives optimizer? | Use for |
|--------|-------|---------------------|---------|
| `dspy.configure(lm=...)` | Global default | Yes | Initial setup |
| `dspy.context(lm=...)` | `with` block only | No | Per-call overrides, routing |
| `module.step.set_lm(lm)` | One module permanently | Yes | Per-stage cost assignment |

## dspy.Predict vs dspy.ChainOfThought

[Predict API docs](https://dspy.ai/api/modules/Predict/) — [ChainOfThought API docs](https://dspy.ai/api/modules/ChainOfThought/)

```python
dspy.Predict(signature, **config)
dspy.ChainOfThought(signature, rationale_field=None, rationale_field_type=str, **config)
```

| Module | Token cost | Output | Best for |
|--------|-----------|--------|---------|
| `dspy.Predict` | Low | Direct answer | Simple extraction, routing, confidence checks |
| `dspy.ChainOfThought` | Medium (adds `reasoning`) | Reasoning + answer | Multi-step reasoning, nuanced tasks |

Always use `dspy.Predict` for the router or confidence-check step in cascading/routing patterns — adding `ChainOfThought` reasoning to a routing step wastes the tokens you are trying to save.

## dspy.BootstrapFewShot

[API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)

```python
dspy.BootstrapFewShot(metric=None, metric_threshold=None, teacher_settings=None,
                      max_bootstrapped_demos=4, max_labeled_demos=16,
                      max_rounds=1, max_errors=None)
```

| Parameter | Type | Default | Cost-relevant guidance |
|-----------|------|---------|------------------------|
| `metric` | `Callable` | `None` | Required scoring function |
| `max_bootstrapped_demos` | `int` | `4` | Lower = shorter prompts = cheaper. Use `2` as a starting point when cutting costs. Never go to `0` — quality collapses. |
| `max_labeled_demos` | `int` | `16` | Lower = shorter prompts. Use `2–4`. |

Key method: `.compile(module, trainset=...)` — returns optimized module.

## dspy.BootstrapFinetune

[API docs](https://dspy.ai/api/optimizers/BootstrapFinetune/)

```python
dspy.BootstrapFinetune(metric=None, num_threads=None, **kwargs)
```

Distills an expensive teacher program into a cheaper fine-tuned student model.

```python
optimizer = dspy.BootstrapFinetune(metric=metric, num_threads=24)
finetuned = optimizer.compile(my_program, trainset=trainset, teacher=teacher_optimized)
```

Requirements: 500+ training examples, a fine-tunable target model. Typical outcome: 10-50x cost reduction with 85-95% quality retention. See `/ai-fine-tuning` for the full decision framework.

## dspy.Retrieve

[API docs](https://dspy.ai/api/retrieval/)

```python
dspy.Retrieve(k=3)
```

`k` controls how many passages are fetched per query. Reducing `k` (e.g., from `5` to `2`) cuts context length and lowers cost in RAG pipelines with minimal quality impact for high-precision retrievers.

## dspy.inspect_history

```python
dspy.inspect_history(n=3)  # Print last N LM calls with token counts
```

Use after any call to verify token usage and confirm caching hits. A cached call shows no token counts.

## Links

- [dspy.LM API docs](https://dspy.ai/api/models/LM/)
- [dspy.Predict API docs](https://dspy.ai/api/modules/Predict/)
- [dspy.ChainOfThought API docs](https://dspy.ai/api/modules/ChainOfThought/)
- [dspy.BootstrapFewShot API docs](https://dspy.ai/api/optimizers/BootstrapFewShot/)
- [dspy.BootstrapFinetune API docs](https://dspy.ai/api/optimizers/BootstrapFinetune/)
- [LiteLLM provider docs](https://docs.litellm.ai/docs/providers)
