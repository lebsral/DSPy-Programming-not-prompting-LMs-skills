---
name: dspy-ollama
description: "Run DSPy with local models using Ollama — no API key needed. Use when you want to run DSPy locally, use Ollama, set up a local LLM, run offline, or configure local model parameters. Also: 'ollama', 'local model', 'run LLM locally', 'llama local', 'self-hosted LLM', 'ollama serve', 'ollama_chat', 'local inference', 'run DSPy offline', 'no API key needed', 'ollama pull', 'num_ctx', 'ollama context window', 'ollama GPU', 'OLLAMA_NUM_GPU', 'which local model', 'best model for ollama', 'ollama too slow', 'ollama vs vllm', 'develop locally deploy remotely'."
---

# Ollama — Run DSPy with Local Models

Guide the user through running DSPy with local models via Ollama. No API keys, no cloud costs, full privacy.

## What is Ollama

[Ollama](https://ollama.com/) is a local LLM runner (166k+ GitHub stars) that wraps llama.cpp. It downloads, manages, and serves models locally with a simple CLI. DSPy connects to it through LiteLLM's `ollama_chat/` provider.

## Setup

### Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows: download from ollama.com
```

### Start the server and pull a model

```bash
# Start the Ollama server (runs in background)
ollama serve

# Pull a model (one-time download)
ollama pull llama3.1

# Quick test
ollama run llama3.1 "What is DSPy?"
```

### Connect DSPy to Ollama

```python
import dspy

lm = dspy.LM(
    "ollama_chat/llama3.1",
    api_base="http://localhost:11434",
    api_key="",  # required but ignored
    temperature=0.7,
    num_ctx=8192,  # IMPORTANT: set context window explicitly
)
dspy.configure(lm=lm)

# Test it
classify = dspy.Predict("text -> sentiment")
result = classify(text="DSPy makes AI development easier")
print(result.sentiment)
```

**Note:** `dspy.OllamaLocal` is deprecated. Use `dspy.LM("ollama_chat/...")` instead.

## Model selection guide

| Model | Sizes | Context | Good for | Notes |
|-------|-------|---------|----------|-------|
| **Llama 3.1** | 8B, 70B | 128K | General purpose, instruction following | Best all-rounder |
| **Llama 3.2** | 1B, 3B | 128K | Edge, mobile, lightweight tasks | Very fast, less capable |
| **Qwen 2.5** | 0.5B–72B | 128K | Multilingual, coding, math | Strong on benchmarks |
| **Qwen 3** | 0.6B–32B | 128K | Reasoning, multilingual | Latest, thinking mode |
| **Mistral** | 7B | 32K | Fast general purpose | Good speed/quality tradeoff |
| **Phi-4** | 14B | 16K | Reasoning, STEM, code | Small but capable |
| **Gemma 2** | 2B, 9B, 27B | 8K | Lightweight, fast | Google, good quality/size ratio |
| **DeepSeek-R1** | 1.5B–70B | 128K | Complex reasoning | Distilled reasoning chains |
| **CodeLlama** | 7B, 13B, 34B | 16K | Code generation | Fine-tuned for code |

### Quick recommendations

```
Prototyping (fast iteration, good quality):
  → llama3.1:8b or qwen2.5:7b

Best quality on consumer hardware (16GB+ RAM):
  → llama3.1:8b or phi4:14b

Complex reasoning:
  → deepseek-r1:14b or qwen3:14b

Coding tasks:
  → qwen2.5-coder:7b or codellama:13b

Minimal resources (8GB RAM):
  → llama3.2:3b or gemma2:2b or qwen2.5:3b
```

## Context window gotcha (critical)

**Ollama defaults to 4096 tokens regardless of the model's actual capacity.** This is the #1 source of issues when running DSPy with Ollama. DSPy prompts with few-shot demos can easily exceed 4096 tokens.

Always set `num_ctx` explicitly:

```python
# BAD — defaults to 4096 tokens, will silently truncate
lm = dspy.LM("ollama_chat/llama3.1", api_base="http://localhost:11434", api_key="")

# GOOD — set context window to match model capability
lm = dspy.LM(
    "ollama_chat/llama3.1",
    api_base="http://localhost:11434",
    api_key="",
    num_ctx=8192,  # 8K is a safe default for most tasks
)
```

**Larger context = more VRAM.** If you get OOM errors, reduce `num_ctx`:

| num_ctx | VRAM overhead (approx) | When to use |
|---------|----------------------:|-------------|
| 4096 | Baseline | Simple classification, short prompts |
| 8192 | +2-4 GB | Most DSPy tasks, few-shot demos |
| 16384 | +4-8 GB | RAG with long contexts |
| 32768 | +8-16 GB | Long document processing |

## Performance tuning

### GPU acceleration

Ollama automatically uses GPU if available. Check with:

```bash
ollama ps  # shows which models are loaded and GPU usage
```

Control GPU usage with environment variables:

```bash
# Use all GPU layers (default if GPU detected)
export OLLAMA_NUM_GPU=999

# CPU only (useful for testing or shared machines)
export OLLAMA_NUM_GPU=0

# Partial offload (when model doesn't fully fit in VRAM)
export OLLAMA_NUM_GPU=20
```

### Concurrent requests

```bash
# Allow multiple parallel requests (default: 1)
export OLLAMA_NUM_PARALLEL=4

# Keep multiple models loaded (for multi-model pipelines)
export OLLAMA_MAX_LOADED_MODELS=2
```

### Apple Silicon optimization

Ollama runs natively on Apple Silicon using Metal. Performance tips:

- **M1/M2 (8GB):** 8B models work well with `num_ctx=4096`
- **M1/M2 Pro (16GB):** 8B models with `num_ctx=8192`, or 14B with `num_ctx=4096`
- **M1/M2 Max (32GB+):** 70B quantized models with `num_ctx=4096`
- **M3/M4 Max (64GB+):** 70B models with `num_ctx=8192`

## Per-module model assignment

Use a big model for hard tasks and a small model for simple ones:

```python
import dspy

big = dspy.LM("ollama_chat/llama3.1:8b", api_base="http://localhost:11434",
              api_key="", num_ctx=8192)
small = dspy.LM("ollama_chat/llama3.2:3b", api_base="http://localhost:11434",
                api_key="", num_ctx=4096)

dspy.configure(lm=small)  # default: cheap model

class Pipeline(dspy.Module):
    def __init__(self):
        self.classify = dspy.Predict("text -> category")
        self.analyze = dspy.ChainOfThought("text, category -> analysis")

    def forward(self, text):
        cat = self.classify(text=text)
        return self.analyze(text=text, category=cat.category)

pipeline = Pipeline()
pipeline.classify.set_lm(small)   # simple task → small model
pipeline.analyze.set_lm(big)      # complex task → big model
```

## Running DSPy optimization with Ollama

Optimization works with local models but is significantly slower than cloud APIs. Tips:

```python
import dspy

lm = dspy.LM("ollama_chat/llama3.1:8b", api_base="http://localhost:11434",
             api_key="", num_ctx=8192)
dspy.configure(lm=lm)

# Tip 1: Start with BootstrapFewShot (fastest optimizer)
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
optimized = optimizer.compile(program, trainset=trainset)

# Tip 2: For MIPROv2, use auto="light" (fewest trials)
optimizer = dspy.MIPROv2(metric=metric, auto="light")
optimized = optimizer.compile(program, trainset=trainset)

# Tip 3: Use a bigger model as teacher, smaller as student
teacher_lm = dspy.LM("ollama_chat/llama3.1:70b", api_base="http://localhost:11434",
                     api_key="", num_ctx=8192)
optimizer = dspy.BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
with dspy.context(lm=teacher_lm):
    optimized = optimizer.compile(program, trainset=trainset)
# Deploy optimized program with the smaller model
```

**Expect hours, not minutes** for optimization with local models. A MIPROv2 `auto="medium"` run that takes 5 minutes with GPT-4o-mini might take 2-4 hours with a local 8B model.

## Ollama vs vLLM

| | Ollama | vLLM |
|---|--------|------|
| **Setup** | `brew install ollama` | `pip install vllm` (NVIDIA only) |
| **Platform** | macOS, Linux, Windows | Linux (NVIDIA GPU required) |
| **Apple Silicon** | Yes (Metal) | No |
| **Throughput** | Single-user | High concurrency (10+ users) |
| **Multi-GPU** | No | Yes (tensor parallelism) |
| **Best for** | Development, prototyping | Production serving |

**Recommended workflow:** Develop with Ollama locally, deploy with vLLM in production. The DSPy code is identical — only the LM config line changes:

```python
# Development (Ollama)
lm = dspy.LM("ollama_chat/llama3.1:8b", api_base="http://localhost:11434", api_key="")

# Production (vLLM)
lm = dspy.LM("openai/meta-llama/Llama-3.1-8B-Instruct", api_base="http://gpu-server:8000/v1", api_key="none")
```

## Gotchas

1. **Context window defaults to 4096** — always set `num_ctx` explicitly. DSPy optimized prompts with few-shot demos easily exceed 4096 tokens.
2. **`api_key=""` is required** — even though Ollama doesn't use it, LiteLLM requires the parameter.
3. **First request is slow** — Ollama loads the model into memory on the first call. Subsequent calls are fast.
4. **OOM errors** — reduce `num_ctx` or switch to a smaller model. Check VRAM with `ollama ps`.
5. **`dspy.OllamaLocal` is deprecated** — use `dspy.LM("ollama_chat/...")` instead.

## Cross-references

- **LM configuration basics** (providers, parameters, caching) — `/dspy-lm`
- **Production serving with vLLM** — `/dspy-vllm`
- **Reducing costs** (model routing, caching) — `/ai-cutting-costs`
- **Switching models** without breaking things — `/ai-switching-models`
- For worked examples, see [examples.md](examples.md)
